import asyncio

from agent.agent import Agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


async def interactive_loop(agent: Agent):
    """Run the interactive command loop."""
    print("=" * 60)
    print("Multi-MCP LangGraph Client")
    print("=" * 60)
    print("Commands: /quit, /status, or type your message")
    print("=" * 60 + "\n")

    state = {
        "messages": [],
        "uploaded_files": [],
        "enabled_skills": [],
        "session_id": None,
        "e2b_session_id": None,
        "extracted_data": {},
    }

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input in ["/quit", "/exit"]:
                print("👋 Goodbye!")
                break
            if user_input == "/status":
                print(f"\nFiles: {state['uploaded_files']}")
                print(f"Anthropic session: {state.get('session_id', 'None')}")
                print(f"E2B session: {state.get('e2b_session_id', 'None')}")
                continue

            state["messages"].append(HumanMessage(content=user_input))

            print("\n🤖 Agent thinking...\n")

            final_state = state
            async for event in agent.astream(state, stream_mode="updates"):
                for node_output in event.values():
                    if "messages" in node_output:
                        final_state["messages"] = node_output["messages"]
                        last_msg = node_output["messages"][-1]

                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            if isinstance(last_msg.content, str):
                                print(f"💭 {last_msg.content}")
                            elif isinstance(last_msg.content, list):
                                for item in last_msg.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        if item.get("text"):
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


async def main():
    print("\n✅ Initializing...")
    agent = Agent()
    agent.build_graph()
    print("✨ LangGraph agent ready\n")
    await interactive_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
