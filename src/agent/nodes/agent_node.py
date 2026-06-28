from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import AIMessage

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_jinja_env = Environment(loader=FileSystemLoader(_PROMPTS_DIR), keep_trailing_newline=True)


def _render_system_prompt(**kwargs) -> str:
    template = _jinja_env.get_template("agent_node.md")
    return template.render(**kwargs)


def build_agent_node(llm, tools):
    async def agent_node(state):
        messages = state["messages"]
        system_message = AIMessage(content=_render_system_prompt())
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke([system_message] + messages)
        return {"messages": messages + [response]}

    return agent_node
