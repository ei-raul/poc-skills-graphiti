import json
import re
import uuid

from langchain_core.messages import ToolMessage


def _update_state_from_tool(state, tool_name, tool_input, result):
    if tool_name.startswith("anthropic_") or tool_name in {
        "ask_claude",
        "upload_file",
        "download_file",
    }:
        if "upload_file" in tool_name or tool_name == "upload_file":
            file_match = re.search(r'"file_id":\s*"([^"]+)"', result)
            if file_match:
                state["uploaded_files"].append(file_match.group(1))

        session_match = re.search(r'"session_id":\s*"([^"]+)"', result)
        if session_match:
            state["session_id"] = session_match.group(1)
            print(f"📦 Container session updated: {session_match.group(1)[:20]}...")

    elif tool_name.startswith("e2b_") or tool_name == "run_code":
        state["e2b_session_id"] = "active"

    return state


def build_tool_node(tools):
    tools_by_name = {
        getattr(tool, "name", getattr(tool, "__name__", "")): tool for tool in tools
    }

    async def tool_node(state):
        messages = state["messages"]
        last_message = messages[-1]
        tool_results = []
        extracted_data = state.get("extracted_data", {})

        tool_calls = getattr(last_message, "tool_calls", []) or []

        for tool_call in tool_calls:
            tool_name = (
                tool_call.get("name")
                if isinstance(tool_call, dict)
                else tool_call.name
            )
            tool_input = (
                dict(tool_call.get("args", {}) or {})
                if isinstance(tool_call, dict)
                else dict(tool_call.args or {})
            )
            tool_id = (
                tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
            )

            print(f"\n🔧 Executing: {tool_name}")
            print(f"📋 Tool Input: {json.dumps(tool_input, indent=2)[:500]}")

            tool = tools_by_name.get(tool_name)
            if not tool:
                error_text = json.dumps(
                    {"error": True, "message": f"Tool not found: {tool_name}"}
                )
                tool_results.append(
                    ToolMessage(content=error_text, tool_call_id=tool_id)
                )
                continue

            # Backward compatibility for prompts that pass base64_ref.
            if tool_name == "upload_file" and "base64_ref" in tool_input:
                ref_id = tool_input.pop("base64_ref")
                if ref_id in extracted_data:
                    tool_input["base64_content"] = extracted_data[ref_id]
                    tool_input.setdefault("path", "")
                    print(
                        f"📦 Injected stored base64 data ({len(extracted_data[ref_id])} chars)"
                    )

            # Reuse Anthropic container when available.
            if tool_name in {"ask_claude", "upload_file"} and state.get("session_id"):
                tool_input.setdefault("session_id", state["session_id"])

            try:
                result_obj = await tool.ainvoke(tool_input)
            except Exception as e:
                result_obj = {"error": True, "message": str(e)}

            if isinstance(result_obj, str):
                result_text = result_obj
            else:
                result_text = json.dumps(result_obj, ensure_ascii=False, default=str)

            display_result = result_text
            result_preview = result_text[:300] + ("..." if len(result_text) > 300 else "")
            print(f"📤 Result preview: {result_preview}")

            base64_match = re.search(r"BASE64:([A-Za-z0-9+/=]{100,})", result_text)
            if base64_match:
                base64_data = base64_match.group(1)
                ref_id = f"base64_{uuid.uuid4().hex[:8]}"
                extracted_data[ref_id] = base64_data
                display_result = result_text.replace(
                    base64_data,
                    f"<BASE64_DATA_EXTRACTED: {len(base64_data)} chars, ref={ref_id}>",
                )
                print(f"📦 Extracted large base64 ({len(base64_data)} chars) → {ref_id}")

            tool_results.append(
                ToolMessage(content=display_result, tool_call_id=tool_id)
            )
            state = _update_state_from_tool(state, tool_name, tool_input, result_text)

        return {
            "messages": messages + tool_results,
            "uploaded_files": state.get("uploaded_files", []),
            "enabled_skills": state.get("enabled_skills", []),
            "session_id": state.get("session_id"),
            "e2b_session_id": state.get("e2b_session_id"),
            "extracted_data": extracted_data,
        }

    return tool_node
