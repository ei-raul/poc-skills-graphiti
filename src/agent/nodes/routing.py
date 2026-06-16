from langgraph.graph import END


def build_routing_node():
    def should_continue(state):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        if hasattr(last_message, "additional_kwargs"):
            tool_use = last_message.additional_kwargs.get("tool_use", [])
            if tool_use:
                return "tools"

        return END

    return should_continue
