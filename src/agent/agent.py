import json
import re
import uuid
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config
from tools.anthropic import anthopic_ask_claude, anthopic_download_file, anthopic_upload_file
from tools.e2b import e2b_run_code
from tools.graphiti import (
    graphiti_add_event,
    graphiti_remove_event,
    graphiti_get_entity_edges,
    graphiti_search_events,
    graphiti_list_recent_episodes
)


config = Config()

class GraphState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    uploaded_files: List[str]
    enabled_skills: List[str]
    session_id: Optional[str]
    e2b_session_id: Optional[str]
    extracted_data: Dict[str, str]

class Agent:

    def __init__(self):
        self.graph = None
        self.tools = [
            # ask_claude, 
            # download_file, 
            # upload_file,
            e2b_run_code,
            graphiti_add_event,
            graphiti_remove_event,
            graphiti_get_entity_edges,
            graphiti_search_events,
            graphiti_list_recent_episodes
        ]
        # self.llm = ChatAnthropic(
        #     model=config.get("CLAUDE_MODEL"),
        #     temperature=0,
        #     streaming=True
        # )
        self.llm = ChatGoogleGenerativeAI(
            model=config.get("GEMINI_MODEL", "gemini-2.5-pro"),
            temperature=0,
        )

    def _build_system_message(self) -> BaseMessage:
        """Build system message with context about all MCP servers."""
        context = (
            "You are an AI assistant with access to THREE different groups of tools:",
            "",
            "## 1. ANTHROPIC SKILLS (anthropic_* tools)",
            "  - Purpose: File storage, document processing, persistent downloads",
            "  - Tools: anthropic_upload_file, anthropic_ask_claude, anthropic_download_file",
            "  - File Persistence: YES - uploaded files persist and can be downloaded",
            "  - Use for: Processing existing documents, storing generated files for download",
            "",
            "## 2. E2B PYTHON SANDBOX (e2b_* tools)",
            "  - Purpose: Custom Python code execution, data analysis, computations",
            "  - Tools: e2b_run_code",
            "  - File Persistence: NO - sandbox is stateless, files are EPHEMERAL",
            "  - Use for: Computations, data processing, generating content (return as data/base64)",
            "",
            "## 3. GRAPHITI TEMPORAL EVENTS (graphiti_* tools)",
            "  - Purpose: Temporal event tracking and knowledge graph",
            "  - Tools: graphiti_add_event, graphiti_search_events, graphiti_get_entity_edges, graphiti_list_recent_episodes, graphiti_remove_event",
            "  - Use for: Storing and retrieving temporal events, tracking what happened when",
            "  - Examples: Recording meetings, tasks, observations; querying past events",
            "",
            "  **CRITICAL: Use graphiti_search_events for temporal queries!**",
            "  - To find events AFTER a time: graphiti_search_events(after_timestamp='2025-05-01T14:00:00')",
            "  - To find events BEFORE a time: graphiti_search_events(before_timestamp='2025-06-01T00:00:00')",
            "  - To find events in a range: graphiti_search_events(after_timestamp='...', before_timestamp='...')",
            "  - To find entity events after a time: graphiti_search_events(entity_name='Chelsea apartment', after_timestamp='...')",
            "  - DO NOT use list_recent_episodes and filter manually - use search_events with temporal filters!",
            "",
            "## CRITICAL: E2B LIMITATIONS (READ THIS BEFORE USING E2B)",
            "- E2B sandbox is STATELESS - files do NOT persist between executions",
            "- There is ABSOLUTELY NO WAY to download files from E2B - this is impossible",
            "- If you try to create a PDF/image in E2B and save it, it will NOT be downloadable",
            "- If you try to read a file saved in a previous E2B execution, it will NOT exist",
            "- The ONLY way to get data out is: print to stdout as text or base64",
            "- E2B does NOT have access to files uploaded via Anthropic tools",
            "",
            "THEREFORE: For downloadable files, you MUST use Pattern A (generate base64 + upload)",
            "",
            "## DECISION FRAMEWORK",
            "",
            "**Single-step tasks:**",
            "- Document processing → Use ANTHROPIC tools",
            "- Python computation (result as text/data) → Use E2B",
            "- List available skills → Use anthropic_list_skills",
            "",
            "**Multi-step workflows (IMPORTANT):**",
            "",
            "For requests like 'create X and let me download it', choose the right pattern:",
            "",
            "**Pattern A: Upload to Anthropic (when file needs processing)**",
            "   Use when file will be:",
            "   - Processed further with ask_claude or skills",
            "   - Referenced in future API calls",
            "   - Shared or persisted in Anthropic's system",
            "   Steps:",
            "   1. E2B: Generate content, convert to base64, print with 'BASE64:' prefix",
            "   2. System automatically extracts: <BASE64_DATA_EXTRACTED: ... ref=base64_xyz>",
            "   3. Upload: anthropic_upload_file(base64_ref='base64_xyz', filename='file.pdf')",
            "   4. Download: anthropic_download_file(file_id=...) to get download path",
            "   CRITICAL: Use base64_ref parameter, NOT path parameter!",
            "",
            "**Pattern C: Direct Save (simple files, no further processing)**",
            "   Use when file is:",
            "   - Ready to download immediately",
            "   - NOT needed for Anthropic skills or API calls",
            "   - Just a simple image/PDF to save locally",
            "   Steps:",
            "   1. E2B: Generate content, convert to base64, print with 'BASE64:' prefix",
            "   2. System automatically extracts: <BASE64_DATA_EXTRACTED: ... ref=base64_xyz>",
            "   3. Save: anthropic_save_base64_file(base64_ref='base64_xyz', filename='chart.png')",
            "   Result: File saved directly to ~/Downloads, faster and simpler than Pattern A",
            "",
            "**Pattern B: Anthropic skill generates file (for PDFs from code, formatted documents)**",
            "   - Step 1: Use E2B to run/validate code if needed",
            "   - Step 2: Use anthropic_list_skills to find PDF generation skill",
            "   - Step 3: Use anthropic_ask_claude with skill_ids=['pdf'] and prompt containing code/text",
            "   - Step 4: Extract file_id from response",
            "   - Step 5: Use anthropic_download_file to provide download",
            "",
            "2. If processing uploaded documents:",
            "   - Use anthropic_upload_file first",
            "   - Then use anthropic_ask_claude with file_ids",
            "   - E2B cannot access these files",
            "",
            "**Examples:**",
            "",
            "Example 1: Simple chart for immediate download (use Pattern C)",
            "   Request: 'Create a matplotlib chart and let me download it'",
            "   ✅ BEST: Pattern C (direct save - faster, simpler)",
            "      1. e2b_run_code: Generate chart → base64 → print with BASE64: prefix",
            "      2. anthropic_save_base64_file(base64_ref='base64_abc', filename='chart.png')",
            "      Result: File saved to ~/Downloads immediately",
            "",
            "Example 1b: Chart that needs Anthropic processing (use Pattern A)",
            "   Request: 'Create a chart, then use Claude to add a title'",
            "   ✅ CORRECT: Pattern A (upload for processing)",
            "      1. e2b_run_code: Generate chart → base64 → print",
            "      2. anthropic_upload_file(base64_ref='base64_abc', filename='chart.png')",
            "      3. anthropic_ask_claude(file_ids=['file_123'], prompt='Add title')",
            "      4. anthropic_download_file(file_id=...)",
            "",
            "Example 2: Generate BFS code, run it, create PDF with code and output",
            "❌ WRONG: e2b_run_code (creates PDF) → expect download",
            "   Problem: Better to use Anthropic's skills for formatted PDFs",
            "",
            "✅ CORRECT: 'BFS code → run → PDF with code and output':",
            "   1. e2b_run_code → generate and run BFS code, capture output",
            "   2. anthropic_list_skills → find PDF generation skill (e.g., 'pdf' skill)",
            "   3. anthropic_ask_claude(skill_ids=['pdf'], prompt='Create PDF with code: [code] and output: [output]')",
            "      NOTE: Use skill_ids parameter (list), not skill (string)",
            "   4. Extract file_id from response",
            "   5. anthropic_download_file(file_id) → download PDF",
            "",
            "Example 3: Analyze uploaded file",
            "❌ WRONG: 'Analyze this Excel file' → e2b_run_code",
            "   Problem: E2B cannot access files uploaded to Anthropic",
            "",
            "✅ CORRECT: 'Analyze this Excel file':",
            "   1. anthropic_upload_file → upload Excel",
            "   2. anthropic_ask_claude → analyze with file_id",
            "",
            "## WORKFLOW STRATEGY",
            "",
            "Always ask yourself:",
            "1. Does the user need a downloadable file?",
            "   - Chart/plot/visualization → Pattern A: E2B (generate base64) + Anthropic (store)",
            "   - Formatted PDF from code/text → Pattern B: E2B (run code) + Anthropic skill (create PDF)",
            "2. Is this document processing? → Anthropic only",
            "3. Is this pure computation (result as text)? → E2B only",
            "4. Are there uploaded files involved? → Anthropic only (E2B can't access them)",
            "5. Does the task need code formatting/styling? → Use Anthropic skills, not E2B PDF generation",
            "",
            "Think step-by-step, identify if this requires multiple tools, and execute them in order.",
            "",
            "## IMPORTANT: RETRY LIMITS",
            "- Maximum 2 retry attempts for any failing operation",
            "- If first approach fails, try ONE alternative approach",
            "- If that also fails, STOP immediately and explain to user",
            "- DO NOT keep retrying E2B when files can't be downloaded - this NEVER works",
            "- Empty results (stdout=[], results=[]) indicate code failure - stop retrying",
            "",
            "## CRITICAL: Using Base64 References",
            "When you see extracted base64 like: <BASE64_DATA_EXTRACTED: 26864 chars, ref=base64_abc123>",
            "You MUST use the 'base64_ref' parameter, NOT the 'path' parameter:",
            "  ✅ CORRECT: anthropic_upload_file(base64_ref='base64_abc123', filename='file.pdf')",
            "  ❌ WRONG: anthropic_upload_file(path='base64_abc123')  # This will fail!",
            "The system will automatically inject the real base64 content when you use base64_ref."
        )

        return AIMessage(content="\n".join(context))

    def _update_state_from_tool(self, state: Dict, tool_name: str, tool_input: Dict, result: str) -> dict:
        """Update state based on tool execution."""
        import re

        if tool_name.startswith("anthropic_") or tool_name in {"ask_claude", "upload_file", "download_file"}:
            # Extract file_id from upload_file
            if "upload_file" in tool_name or tool_name == "upload_file":
                file_match = re.search(r'"file_id":\s*"([^"]+)"', result)
                if file_match:
                    state["uploaded_files"].append(file_match.group(1))

            # Extract session_id from ANY Anthropic tool (upload_file, ask_claude, etc.)
            session_match = re.search(r'"session_id":\s*"([^"]+)"', result)
            if session_match:
                state["session_id"] = session_match.group(1)
                print(f"📦 Container session updated: {session_match.group(1)[:20]}...")

        elif tool_name.startswith("e2b_") or tool_name == "run_code":
            state["e2b_session_id"] = "active"

        return state

    async def _interactive_loop(self):
        """Run the interactive command loop."""
        print("="*60)
        print("Multi-MCP LangGraph Client")
        print("="*60)
        print("Commands: /quit, /status, or type your message")
        print("="*60 + "\n")

        state = {
            "messages": [],
            "uploaded_files": [],
            "enabled_skills": [],
            "session_id": None,
            "e2b_session_id": None,
            "extracted_data": {}
        }

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input in ["/quit", "/exit"]:
                    print("👋 Goodbye!")
                    break
                elif user_input == "/status":
                    print(f"\nFiles: {state['uploaded_files']}")
                    print(f"Anthropic session: {state.get('session_id', 'None')}")
                    print(f"E2B session: {state.get('e2b_session_id', 'None')}")
                    continue

                state["messages"].append(HumanMessage(content=user_input))

                print("\n🤖 Agent thinking...\n")

                final_state = state
                async for event in self.graph.astream(state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if "messages" in node_output:
                            final_state["messages"] = node_output["messages"]
                            last_msg = node_output["messages"][-1]

                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                if isinstance(last_msg.content, str):
                                    print(f"💭 {last_msg.content}")
                                elif isinstance(last_msg.content, list):
                                    for item in last_msg.content:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            if item.get('text'):
                                                print(f"💭 {item['text']}")

                            elif isinstance(last_msg, ToolMessage):
                                result_preview = str(last_msg.content)[:150]
                                if len(str(last_msg.content)) > 150:
                                    result_preview += "..."
                                print(f"✅ {result_preview}")

                state = final_state
                print("\n✅ Done")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    async def _build_graph(self):
        """Build the LangGraph agent workflow."""
        print("🔧 Building LangGraph agent...")

        # Define agent node
        async def agent_node(state: GraphState):
            """Agent reasoning node - decides what to do."""
            messages = state["messages"]
            system_message = self._build_system_message()
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke([system_message] + messages)
            return {"messages": messages + [response]}

        # Define tool execution node
        async def tool_node(state: GraphState):
            """Execute tools requested by the agent using self.tools."""
            messages = state["messages"]
            last_message = messages[-1]
            tool_results = []
            extracted_data = state.get("extracted_data", {})

            tools_by_name = {
                getattr(tool, "name", getattr(tool, "__name__", "")): tool
                for tool in self.tools
            }

            tool_calls = getattr(last_message, "tool_calls", []) or []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
                tool_input = dict(tool_call.get("args", {}) or {}) if isinstance(tool_call, dict) else dict(tool_call.args or {})
                tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

                print(f"\n🔧 Executing: {tool_name}")
                print(f"📋 Tool Input: {json.dumps(tool_input, indent=2)[:500]}")

                tool = tools_by_name.get(tool_name)
                if not tool:
                    error_text = json.dumps({"error": True, "message": f"Tool not found: {tool_name}"})
                    tool_results.append(ToolMessage(content=error_text, tool_call_id=tool_id))
                    continue

                # Backward compatibility for prompts that pass base64_ref.
                if tool_name == "upload_file" and "base64_ref" in tool_input:
                    ref_id = tool_input.pop("base64_ref")
                    if ref_id in extracted_data:
                        tool_input["base64_content"] = extracted_data[ref_id]
                        tool_input.setdefault("path", "")
                        print(f"📦 Injected stored base64 data ({len(extracted_data[ref_id])} chars)")

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

                base64_match = re.search(r'BASE64:([A-Za-z0-9+/=]{100,})', result_text)
                if base64_match:
                    base64_data = base64_match.group(1)
                    ref_id = f"base64_{uuid.uuid4().hex[:8]}"
                    extracted_data[ref_id] = base64_data
                    display_result = result_text.replace(
                        base64_data,
                        f"<BASE64_DATA_EXTRACTED: {len(base64_data)} chars, ref={ref_id}>"
                    )
                    print(f"📦 Extracted large base64 ({len(base64_data)} chars) → {ref_id}")

                tool_results.append(ToolMessage(content=display_result, tool_call_id=tool_id))
                state = self._update_state_from_tool(state, tool_name, tool_input, result_text)

            return {
                "messages": messages + tool_results,
                "uploaded_files": state.get("uploaded_files", []),
                "enabled_skills": state.get("enabled_skills", []),
                "session_id": state.get("session_id"),
                "e2b_session_id": state.get("e2b_session_id"),
                "extracted_data": extracted_data
            }

        # Define routing function
        def should_continue(state: GraphState):
            """Determine if we should continue to tools or end."""
            messages = state["messages"]
            last_message = messages[-1]

            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            if hasattr(last_message, 'additional_kwargs'):
                tool_use = last_message.additional_kwargs.get('tool_use', [])
                if tool_use:
                    return "tools"

            return END

        # Build the graph
        workflow = StateGraph(GraphState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile()

    async def run(self):
        print(f"\n✅ Initializing...")

        # Build the agent graph
        await self._build_graph()

        print("✨ LangGraph agent ready\n")

        # Run interactive loop (connections stay open throughout)
        await self._interactive_loop()
