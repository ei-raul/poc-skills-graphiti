import os
import re
import json
import mimetypes
import httpx
import base64
from typing import Dict, Any, Optional, List, Set, Tuple
from utils.http import request_with_retry
from config.logger import get_logger
from config.config import Config
from config.anthropic import AnthropicConfig
from langchain.tools import tool

logger = get_logger(__name__)

config = Config()
anthropic_config = AnthropicConfig()


def _build_message_content(
    prompt: Optional[str], file_ids: Optional[List[str]], files_to_exclude: Set[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build message content blocks with file routing.

    Args:
        prompt: The user prompt text
        file_ids: List of file IDs to include
        files_to_exclude: Set of file IDs to exclude from document blocks

    Returns:
        Tuple of (content blocks, files routed to prompt)
    """
    content: List[Dict[str, Any]] = []
    files_for_prompt: List[str] = []

    for fid in file_ids or []:
        if fid in files_to_exclude:
            # Use container_upload for files that can't be document blocks
            content.append({"type": "container_upload", "file_id": fid})
            logger.debug(
                f"File {fid} added as container_upload (accessible via code execution)"
            )
            files_for_prompt.append(fid)
        else:
            # Try as document block first (for .txt/.pdf direct reading)
            content.append(
                {"type": "document", "source": {"type": "file", "file_id": fid}}
            )
            logger.debug(f"File {fid} added as document block")

    # Add prompt text
    prompt_text = prompt
    if files_for_prompt:
        # When using container_upload, inform Claude the files are accessible
        file_count = len(files_for_prompt)
        prompt_text = f"{prompt}\n\n(Note: {file_count} file{'s' if file_count > 1 else ''} uploaded to container and accessible via code execution)"
        logger.info(
            f"Routed {file_count} file(s) to container_upload (accessible in container filesystem)"
        )

    content.append({"type": "text", "text": prompt_text})

    return content, files_for_prompt


def _merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages with the same role.

    This fixes 400 errors when consecutive user turns occur.

    Args:
        messages: List of message dictionaries

    Returns:
        List of merged messages
    """
    merged_messages = []
    for msg in messages:
        if merged_messages and merged_messages[-1]["role"] == msg["role"]:
            merged_messages[-1]["content"].extend(msg["content"])
        else:
            merged_messages.append(msg)
    return merged_messages


def _normalize_message_types(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize message content block types to be API-compliant.

    Converts various tool_use and tool_result types to standard forms.

    Args:
        messages: List of message dictionaries

    Returns:
        List of normalized messages
    """
    normalized_messages = []

    for msg in messages:
        new_content = []
        for block in msg.get("content", []):
            new_block = block.copy()
            btype = new_block.get("type")

            # Normalize tool_use (standard API block type)
            if btype in ["tool_use", "server_tool_use"]:
                new_block["type"] = "tool_use"
                if new_block.get("name") in [
                    "code_execution",
                    "text_editor_code_execution",
                ]:
                    new_block["name"] = "text_editor_code_execution"

            # Normalize tool_result (standard API block type)
            elif btype in [
                "tool_result",
                "code_execution_tool_result",
                "text_editor_code_execution_tool_result",
            ]:
                new_block["type"] = "tool_result"
                content = new_block.get("content")

                # Extract raw text from specialized beta dicts
                if isinstance(content, dict):
                    extracted_text = ""
                    # Code Execution (run)
                    if "stdout" in content:
                        extracted_text = content.get("stdout", "")
                        if content.get("stderr"):
                            extracted_text += f"\n[Error]\n{content['stderr']}"
                    # Text Editor (view/create/str_replace)
                    elif "content" in content:
                        extracted_text = content["content"]
                    # Fallback for other dict-based content
                    else:
                        extracted_text = json.dumps(content)
                    new_block["content"] = extracted_text
                elif "stdout" in new_block:
                    # Handle cases where stdout is at the top level
                    new_block["content"] = new_block["stdout"]

            new_content.append(new_block)

        normalized_messages.append({"role": msg["role"], "content": new_content})

    return normalized_messages


def _bridge_missing_tool_results(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add bridging tool results for dangling tool_use blocks.

    Any tool_use in an Assistant turn MUST be followed by its result in the next User turn.
    If missing, inject a minimally valid error result.

    Args:
        messages: List of normalized messages

    Returns:
        List of messages with bridging results injected
    """
    final_history = []

    for i, msg in enumerate(messages):
        final_history.append(msg)

        if msg["role"] == "assistant":
            tool_use_ids = [
                b["id"] for b in msg["content"] if b.get("type") == "tool_use"
            ]
            if tool_use_ids:
                # Check next message (must be User)
                next_msg = messages[i + 1] if i + 1 < len(messages) else None

                # Ensure we have a User msg to attach results to
                if next_msg and next_msg["role"] == "user":
                    result_ids = set(
                        b.get("tool_use_id")
                        for b in next_msg["content"]
                        if b.get("type") == "tool_result"
                    )

                    missing_ids = [tid for tid in tool_use_ids if tid not in result_ids]
                    if missing_ids:
                        logger.debug(f"Bridging dangling tool_uses: {missing_ids}")
                        injected_blocks = []

                        for tid in missing_ids:
                            # Create interrupted error result
                            content = (
                                "Error: Tool execution was interrupted by user input "
                                "before completion. Please retry the previous action."
                            )
                            injected_blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tid,
                                    "content": content,
                                    "is_error": True,
                                }
                            )

                        next_msg["content"] = injected_blocks + next_msg["content"]

    return final_history


def _build_payload(
    model: str,
    messages: List[Dict[str, Any]],
    skill_ids: Optional[List[str]],
    session_id: Optional[str],
) -> Dict[str, Any]:
    """
    Build the API request payload.

    Args:
        model: Model name
        messages: List of messages
        skill_ids: Optional list of skill IDs
        session_id: Optional session ID

    Returns:
        Complete payload dictionary
    """
    payload = {
        "model": model,
        "max_tokens": 16384,  # Increased to 16K for complex PDF operations with skills
        "system": anthropic_config.default_system_prompt,
        "messages": messages,
        "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
    }

    # Include container only if we have session_id or skills
    # Empty containers don't trigger container creation in the API
    container = {}
    if session_id:
        container["id"] = session_id
    if skill_ids:
        container["skills"] = [
            {"type": "anthropic", "skill_id": sid, "version": "latest"}
            for sid in skill_ids
        ]

    # Only add container to payload if it has content
    if container:
        payload["container"] = container

    return payload


def _log_payload_debug(messages: List[Dict[str, Any]]) -> None:
    """
    Log payload structure for debugging.

    Args:
        messages: List of messages to log
    """
    logger.debug(f"Outgoing Payload (Turn Count: {len(messages)})")
    for i, m in enumerate(messages):
        contents = []
        for b in m.get("content", []):
            binfo = f"type={b.get('type')}"
            if "name" in b:
                binfo += f", name={b['name']}"
            if "id" in b:
                binfo += f", id={b['id']}"
            if "tool_use_id" in b:
                binfo += f", tu_id={b['tool_use_id']}"
            contents.append(f"[{binfo}]")
        logger.debug(f"  {i}. {m['role']}: {' '.join(contents)}")


def _dump_payload_on_error(payload: Dict[str, Any]) -> None:
    """
    Dump payload to file for debugging on error.

    Args:
        payload: The payload dictionary to dump
    """
    dump_path = os.path.expanduser("~/.claude_request_dump.json")
    try:
        with open(dump_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug(f"Request payload dumped to: {dump_path}")
    except Exception as de:
        logger.warning(f"Could not dump payload: {de}")


def _handle_api_error(
    response: Any, payload: Dict[str, Any], files_to_exclude: Set[str]
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Handle API error responses.

    Args:
        response: HTTP response object
        payload: Request payload
        files_to_exclude: Set to add excluded files to

    Returns:
        Tuple of (should_retry, file_to_exclude, error_response)
    """
    logger.error(f"API error {response.status_code}")
    _dump_payload_on_error(payload)

    if response.status_code == 400:
        try:
            error_body = response.json()
            error_msg = error_body.get("error", {}).get("message", "")

            # Check for unsupported file format/extension errors
            # Match both "unsupported extension" and "Unsupported document file format"
            is_file_format_error = (
                "unsupported extension" in error_msg.lower()
                and "document content block" in error_msg.lower()
            ) or ("unsupported document file format" in error_msg.lower())

            if is_file_format_error:
                # Try to extract file_id from error message
                match = re.search(r"id '(file_[^']+)'", error_msg)
                if match:
                    unsupported_fid = match.group(1)
                    if unsupported_fid not in files_to_exclude:
                        files_to_exclude.add(unsupported_fid)
                        logger.info(
                            f"Rerouting {unsupported_fid} (unsupported as document block) and retrying..."
                        )
                        return True, unsupported_fid, None
                else:
                    # If we can't extract file_id from message, check the payload
                    # to find which file was in the content blocks
                    if "messages" in payload:
                        for msg in payload["messages"]:
                            if isinstance(msg.get("content"), list):
                                for block in msg["content"]:
                                    if block.get("type") == "document":
                                        file_id = block.get("source", {}).get("file_id")
                                        if file_id and file_id not in files_to_exclude:
                                            files_to_exclude.add(file_id)
                                            logger.info(
                                                f"Rerouting {file_id} (detected from payload) and retrying..."
                                            )
                                            return True, file_id, None
        except Exception:
            pass

    # Build error response
    try:
        error_body = response.json()
        error_msg = error_body.get("error", {}).get("message", response.text)
    except Exception:
        error_msg = response.text

    # For 400 errors, include the payload for debugging
    if response.status_code == 400:
        error_msg += f"\nDEBUG_PAYLOAD: {json.dumps(payload, indent=2)}"

    error_response = {
        "error": True,
        "status_code": response.status_code,
        "message": f"API error {response.status_code}: {error_msg}",
    }

    return False, None, error_response


def _extract_session_id(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract session ID from API response.

    Args:
        result: API response dictionary

    Returns:
        Session ID if found, None otherwise
    """
    if "container" in result and isinstance(result["container"], dict):
        container_id = result["container"].get("id") or result["container"].get(
            "session_id"
        )
        if container_id:
            return container_id
    return None


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
    Send a prompt or conversation history to Claude via the Anthropic API.

    Args:
        prompt: User prompt text
        file_ids: List of file IDs to include
        skill_ids: List of skill IDs to enable
        session_id: Container ID from a previous API response. Must start with 'container_'.
                   DO NOT invent custom strings - only use IDs returned by the API.
        history: Optional conversation history
        model: Model name to use

    Returns:
        API response dictionary with 'session_id' for reuse
    """

    # Validate session_id format - must be container_* with actual content after prefix
    # Ignore invalid formats to prevent API errors
    if session_id:
        if not session_id.startswith("container_") or len(session_id) <= len(
            "container_"
        ):
            logger.warning(
                f"Invalid session_id format '{session_id}'. Must be 'container_' followed by ID. Ignoring."
            )
            session_id = None

    if not history and not prompt:
        return {"error": True, "message": "Neither prompt nor history provided"}

    headers = anthropic_config.get_headers_with_content_type()
    files_to_exclude_from_doc_blocks: Set[str] = set()

    # We may need to retry if some files are rejected as document content blocks
    max_routing_attempts = len(file_ids) + 1 if file_ids else 1

    for route_attempt in range(max_routing_attempts):
        # Build message content
        content, files_for_prompt = _build_message_content(
            prompt, file_ids, files_to_exclude_from_doc_blocks
        )

        # Combine previous history with current turn
        final_messages = (history or []) + [{"role": "user", "content": content}]

        # Merge consecutive messages with same role
        merged_messages = _merge_consecutive_messages(final_messages)

        # Normalize message types
        normalized_messages = _normalize_message_types(merged_messages)

        # Bridge missing tool results
        final_history = _bridge_missing_tool_results(normalized_messages)

        # Build payload
        payload = _build_payload(model, final_history, skill_ids, session_id)

        # Log for debugging
        _log_payload_debug(normalized_messages)

        try:
            async with httpx.AsyncClient(timeout=anthropic_config.timeout) as client:
                response = await request_with_retry(
                    client,
                    "post",
                    f"{anthropic_config.base_url}/messages",
                    headers=headers,
                    json=payload,
                )

                if response.status_code != 200:
                    should_retry, excluded_file, error_response = _handle_api_error(
                        response, payload, files_to_exclude_from_doc_blocks
                    )

                    if should_retry:
                        continue

                    return error_response

                result = response.json()

                # Extract and add session_id to response
                session_id_extracted = _extract_session_id(result)
                if session_id_extracted:
                    result["session_id"] = session_id_extracted
                    logger.debug(f"Extracted session_id: {session_id_extracted}")
                else:
                    logger.debug("No session_id found in API response")

                return result

        except httpx.TimeoutException:
            return {
                "error": True,
                "status_code": 408,
                "message": f"Request timed out after {anthropic_config.timeout} seconds. "
                "If processing many files or using skills, this may take longer. "
                "Consider using E2B to generate content and upload via Anthropic instead.",
            }
        except httpx.ConnectError as e:
            return {
                "error": True,
                "status_code": 0,
                "message": f"Connection error: {e}",
            }
        except Exception as e:
            return {
                "error": True,
                "status_code": 0,
                "message": f"{type(e).__name__}: {e}",
            }

    # Should not be reached but just in case
    return {"error": True, "message": "Failed to route files correctly."}


@tool
async def anthopic_upload_file(
    path: str,
    session_id: Optional[str] = None,
    base64_content: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uploads a file to Anthropic API and returns a dictionary with file_id and optional session_id.

    Smart Upload Feature:
    - For .txt and .pdf files: Can upload without container (document content blocks)
    - For other file types: Auto-creates container if needed (enables ANY file type)

    This allows uploading .xlsx, .docx, .csv, .png, etc. by automatically creating
    a container where code execution can access these files.

    Args:
        path: Path to the file to upload (use this OR base64_content, not both)
        session_id: Optional container ID from a previous API response. Must start with 'container_'.
                   Leave empty to auto-create. DO NOT invent custom strings.
        base64_content: Optional base64-encoded file content (alternative to path)
        filename: Required when using base64_content, optional otherwise

    Returns:
        Dict with 'file_id' and 'session_id' (container ID for reuse)

    Retries on transient server errors (502, 503, 429).
    """
    # Handle base64 content input
    if base64_content:
        if not filename:
            raise ValueError("filename is required when providing base64_content")

        file_content = base64.b64decode(base64_content)
        file_ext = os.path.splitext(filename)[1].lower()

        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"

    # Handle file path input
    else:
        # Check if path looks like a base64 ref (common mistake)
        if path.startswith("base64_"):
            raise ValueError(
                f"PARAMETER ERROR: You used path='{path}' but this looks like a base64 reference!\n"
                f"The 'base64_ref' parameter exists for this purpose.\n"
                f"\n"
                f"✅ CORRECT: upload_file(base64_ref='{path}', filename='file.pdf')\n"
                f"❌ WRONG:   upload_file(path='{path}')\n"
                f"\n"
                f"Please retry with the base64_ref parameter."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        filename = os.path.basename(path)
        file_ext = os.path.splitext(filename)[1].lower()

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "application/octet-stream"

        with open(path, "rb") as f:
            file_content = f.read()

    # Smart Upload: Auto-create container for non-.txt/.pdf files
    needs_container = file_ext not in [".txt", ".pdf"]

    # Validate session_id format - must be container_* with actual content after prefix
    # If agent passes an invalid format, treat as if no session_id was provided
    if session_id:
        if not session_id.startswith("container_") or len(session_id) <= len(
            "container_"
        ):
            logger.warning(
                f"Invalid session_id format '{session_id}'. Must be 'container_' followed by ID. Ignoring."
            )
            session_id = None

    if needs_container and not session_id:
        logger.info(f"File type {file_ext} requires container. Auto-creating...")

        # Import here to avoid circular imports at module load time
        try:
            # Create a container by making a request that triggers code execution
            # This ensures a container is actually created and returned
            response = await anthopic_ask_claude(
                prompt="Execute: import os; print('Container initialized')",
                model="claude-sonnet-4-5-20250929",
            )

            session_id = response.get("session_id")

            if session_id:
                logger.info(
                    f"📦 Auto-created container {session_id} for {file_ext} upload"
                )
            else:
                logger.warning(
                    "Container creation did not return session_id. Upload may fail for non-.txt/.pdf files."
                )

        except Exception as e:
            logger.error(f"Failed to auto-create container: {e}")
            logger.warning(
                "Attempting upload anyway - may fail for non-.txt/.pdf files"
            )

    headers = anthropic_config.get_headers()

    files = {"file": (filename, file_content, mime_type)}

    async with httpx.AsyncClient(timeout=anthropic_config.upload_timeout) as client:
        data_payload = {}
        if session_id:
            # Send container ID if we have one
            data_payload["container"] = json.dumps({"id": session_id})

        response = await request_with_retry(
            client,
            "post",
            f"{anthropic_config.base_url}/files",
            headers=headers,
            files=files,
            data=data_payload,
        )
        if response.status_code != 200:
            logger.error(
                f"Upload failed with status {response.status_code}: {response.text}"
            )

        response.raise_for_status()
        data = response.json()

        # Return both file ID and session ID
        # Prefer session_id from response, but keep auto-created one if response doesn't have it
        response_session_id = data.get("session_id")
        if (
            not response_session_id
            and "container" in data
            and isinstance(data["container"], dict)
        ):
            response_session_id = data["container"].get("id") or data["container"].get(
                "session_id"
            )

        # Use response session_id if available, otherwise keep the auto-created one
        final_session_id = response_session_id or session_id

        if final_session_id:
            logger.info(
                f"Upload successful. File: {data.get('id')}, Container: {final_session_id}"
            )
        else:
            logger.info(f"Upload successful. File: {data.get('id')}, No container")

        return {"file_id": data.get("id"), "session_id": final_session_id}


@tool
async def anthopic_download_file(file_id: str, output_dir: Optional[str] = None) -> str:
    """
    Downloads a file from Anthropic API using its file_id.
    Returns the local path where the file was saved.
    Retries on transient server errors (502, 503, 429).
    """
    if output_dir is None:
        output_dir = anthropic_config.download_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    headers = anthropic_config.get_headers()

    async with httpx.AsyncClient(timeout=anthropic_config.upload_timeout) as client:
        # Get file content with retry
        response = await request_with_retry(
            client,
            "get",
            f"{anthropic_config.base_url}/files/{file_id}/content",
            headers=headers,
        )
        response.raise_for_status()
        content = response.content

        # Get file metadata for filename
        meta_response = await request_with_retry(
            client,
            "get",
            f"{anthropic_config.base_url}/files/{file_id}",
            headers=headers,
        )
        filename = f"{file_id}.bin"
        if meta_response.status_code == 200:
            meta = meta_response.json()
            if "filename" in meta:
                filename = meta["filename"]

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "wb") as f:
            f.write(content)

        return output_path
