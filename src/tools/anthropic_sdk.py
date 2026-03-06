import base64
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from config.anthropic import AnthropicConfig
from config.logger import get_logger
from langchain.tools import tool
from anthropic import AsyncAnthropic

logger = get_logger(__name__)
anthropic_config = AnthropicConfig()

BETA_FLAGS = [
    "files-api-2025-04-14",
    "skills-2025-10-02",
    "code-execution-2025-08-25",
]

_anthropic_client = None

def _get_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(
            api_key=anthropic_config.anthropic_api_key,
            timeout=anthropic_config.timeout,
            max_retries=anthropic_config.max_retries,
        )
    return _anthropic_client


def _build_message_content(
    prompt: Optional[str],
    file_ids: Optional[List[str]],
    files_to_exclude: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    content: List[Dict[str, Any]] = []
    files_for_prompt: List[str] = []

    for fid in (file_ids or []):
        if fid in files_to_exclude:
            content.append({"type": "container_upload", "file_id": fid})
            files_for_prompt.append(fid)
            continue

        content.append(
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": fid,
                },
            }
        )

    prompt_text = prompt or ""
    if files_for_prompt:
        count = len(files_for_prompt)
        prompt_text = (
            f"{prompt_text}\n\n"
            f"(Note: {count} file{'s' if count > 1 else ''} uploaded to container "
            "and accessible via code execution)"
        )
    content.append({"type": "text", "text": prompt_text})

    return content, files_for_prompt


def _merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


def _normalize_message_types(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []

    for msg in messages:
        new_content = []
        for block in msg.get("content", []):
            new_block = block.copy()
            btype = new_block.get("type")

            if btype in ["tool_use", "server_tool_use"]:
                new_block["type"] = "tool_use"
                if new_block.get("name") in ["code_execution", "text_editor_code_execution"]:
                    new_block["name"] = "text_editor_code_execution"
            elif btype in [
                "tool_result",
                "code_execution_tool_result",
                "text_editor_code_execution_tool_result",
            ]:
                new_block["type"] = "tool_result"
                content = new_block.get("content")

                if isinstance(content, dict):
                    extracted_text = ""
                    if "stdout" in content:
                        extracted_text = content.get("stdout", "")
                        if content.get("stderr"):
                            extracted_text += f"\n[Error]\n{content['stderr']}"
                    elif "content" in content:
                        extracted_text = content["content"]
                    else:
                        extracted_text = json.dumps(content)
                    new_block["content"] = extracted_text
                elif "stdout" in new_block:
                    new_block["content"] = new_block["stdout"]

            new_content.append(new_block)

        normalized_messages.append({"role": msg["role"], "content": new_content})

    return normalized_messages


def _bridge_missing_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_history: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        final_history.append(msg)

        if msg["role"] != "assistant":
            continue

        tool_use_ids = [b["id"] for b in msg["content"] if b.get("type") == "tool_use"]
        if not tool_use_ids:
            continue

        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        if not next_msg or next_msg["role"] != "user":
            continue

        result_ids = {
            b.get("tool_use_id")
            for b in next_msg["content"]
            if b.get("type") == "tool_result"
        }
        missing_ids = [tid for tid in tool_use_ids if tid not in result_ids]
        if not missing_ids:
            continue

        injected_blocks = []
        for tid in missing_ids:
            injected_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": (
                        "Error: Tool execution was interrupted by user input before completion. "
                        "Please retry the previous action."
                    ),
                    "is_error": True,
                }
            )
        next_msg["content"] = injected_blocks + next_msg["content"]

    return final_history


def _extract_session_id(result: Dict[str, Any]) -> Optional[str]:
    container = result.get("container")
    if isinstance(container, dict):
        return container.get("id") or container.get("session_id")
    return None


def _is_file_format_error(error_message: str) -> bool:
    lower_msg = error_message.lower()
    return (
        "unsupported extension" in lower_msg and "document content block" in lower_msg
    ) or ("unsupported document file format" in lower_msg)


def _extract_status_and_message(exc: Exception) -> Tuple[int, str]:
    status_code = int(getattr(exc, "status_code", 0) or 0)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("error", {}).get("message") or str(exc)
    else:
        message = str(exc)
    return status_code, message


@tool
async def anthopic_ask_claude(
    prompt: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    skill_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-sonnet-4-5-20250929",
) -> Dict[str, Any]:
    """
    Send a prompt or conversation history to Claude via the official Anthropic SDK.
    """
    if session_id and (
        not session_id.startswith("container_") or len(session_id) <= len("container_")
    ):
        logger.warning("Invalid session_id format '%s'. Ignoring.", session_id)
        session_id = None

    if not history and not prompt:
        return {"error": True, "message": "Neither prompt nor history provided"}

    client = _get_client()
    files_to_exclude: Set[str] = set()
    max_routing_attempts = len(file_ids) + 1 if file_ids else 1

    for _ in range(max_routing_attempts):
        content, _ = _build_message_content(prompt, file_ids, files_to_exclude)
        final_messages = (history or []) + [{"role": "user", "content": content}]
        final_messages = _merge_consecutive_messages(final_messages)
        final_messages = _normalize_message_types(final_messages)
        final_messages = _bridge_missing_tool_results(final_messages)

        container: Dict[str, Any] = {}
        if session_id:
            container["id"] = session_id
        if skill_ids:
            container["skills"] = [
                {"type": "anthropic", "skill_id": sid, "version": "latest"} for sid in skill_ids
            ]

        try:
            response = await client.beta.messages.create(
                model=model,
                max_tokens=16384,
                system=anthropic_config.default_system_prompt,
                messages=final_messages,
                tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
                betas=BETA_FLAGS,
                container=container if container else None,
            )

            result = response.model_dump(mode="json")
            extracted_session_id = _extract_session_id(result)
            if extracted_session_id:
                result["session_id"] = extracted_session_id
            return result

        except Exception as exc:
            status_code, error_message = _extract_status_and_message(exc)
            if status_code == 400 and _is_file_format_error(error_message):
                match = re.search(r"id '(file_[^']+)'", error_message)
                if match:
                    unsupported_fid = match.group(1)
                    if unsupported_fid not in files_to_exclude:
                        files_to_exclude.add(unsupported_fid)
                        logger.info("Rerouting %s as container_upload and retrying", unsupported_fid)
                        continue

                # fallback: reroute first document file that isn't excluded yet
                for fid in file_ids or []:
                    if fid not in files_to_exclude:
                        files_to_exclude.add(fid)
                        logger.info("Rerouting %s as container_upload and retrying", fid)
                        break
                continue

            return {
                "error": True,
                "status_code": status_code,
                "message": f"{type(exc).__name__}: {error_message}",
            }

    return {"error": True, "message": "Failed to route files correctly."}


@tool
async def anthopic_upload_file(
    path: str,
    session_id: Optional[str] = None,
    base64_content: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload file to Anthropic using the official SDK.
    """
    if base64_content:
        if not filename:
            raise ValueError("filename is required when providing base64_content")
        file_content = base64.b64decode(base64_content)
        file_ext = os.path.splitext(filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(filename)
    else:
        if path.startswith("base64_"):
            raise ValueError(
                "PARAMETER ERROR: path appears to be a base64 reference. "
                "Use base64_content with filename."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        filename = os.path.basename(path)
        file_ext = os.path.splitext(filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            file_content = f.read()

    mime_type = mime_type or "application/octet-stream"

    if session_id and (
        not session_id.startswith("container_") or len(session_id) <= len("container_")
    ):
        logger.warning("Invalid session_id format '%s'. Ignoring.", session_id)
        session_id = None

    if file_ext not in [".txt", ".pdf"] and not session_id:
        logger.info("File type %s requires container. Auto-creating...", file_ext)
        response = await anthopic_ask_claude(
            prompt="Execute: import os; print('Container initialized')",
            model="claude-sonnet-4-5-20250929",
        )
        session_id = response.get("session_id")

    client = _get_client()

    try:
        extra_body: Dict[str, Any] = {}
        if session_id:
            extra_body["container"] = {"id": session_id}

        upload_result = await client.beta.files.upload(
            file=(filename, file_content, mime_type),
            betas=BETA_FLAGS,
            timeout=anthropic_config.upload_timeout,
            extra_body=extra_body or None,
        )
    except Exception as exc:
        status_code, error_message = _extract_status_and_message(exc)
        return {
            "error": True,
            "status_code": status_code,
            "message": f"{type(exc).__name__}: {error_message}",
        }

    data = upload_result.model_dump(mode="json")
    response_session_id = None
    if isinstance(data.get("container"), dict):
        response_session_id = (
            data["container"].get("id") or data["container"].get("session_id")
        )
    final_session_id = response_session_id or session_id

    return {
        "file_id": data.get("id"),
        "session_id": final_session_id,
    }


@tool
async def anthopic_download_file(file_id: str, output_dir: Optional[str] = None) -> str:
    """
    Download a file from Anthropic using the official SDK and save it locally.
    """
    output_dir = output_dir or anthropic_config.download_dir
    os.makedirs(output_dir, exist_ok=True)

    client = _get_client()

    try:
        binary_response = await client.beta.files.download(
            file_id=file_id,
            betas=BETA_FLAGS,
            timeout=anthropic_config.upload_timeout,
        )
        content = await binary_response.read()

        metadata = await client.beta.files.retrieve_metadata(
            file_id=file_id,
            betas=BETA_FLAGS,
            timeout=anthropic_config.upload_timeout,
        )
        filename = metadata.filename or f"{file_id}.bin"
    except Exception as exc:
        status_code, error_message = _extract_status_and_message(exc)
        raise RuntimeError(
            f"Failed to download file {file_id} (status={status_code}): {error_message}"
        ) from exc

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(content)
    return output_path
