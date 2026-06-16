from typing import Any, Dict, List, Optional, Sequence, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import Config
from tools.e2b import e2b_run_code
from tools.graphiti import (
    graphiti_add_event,
    graphiti_remove_event,
    graphiti_get_entity_edges,
    graphiti_search_events,
    graphiti_list_recent_episodes,
)
from agent.nodes import build_agent_node, build_tool_node, build_routing_node


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
        self._graph = None
        self.tools = [
            e2b_run_code,
            graphiti_add_event,
            graphiti_remove_event,
            graphiti_get_entity_edges,
            graphiti_search_events,
            graphiti_list_recent_episodes,
        ]
        self.llm = ChatGoogleGenerativeAI(
            model=config.get("GEMINI_MODEL", "gemini-2.5-pro"),
            temperature=0,
        )

    def build_graph(self):
        print("🔧 Building LangGraph agent...")

        agent_node = build_agent_node(self.llm, self.tools)
        tool_node = build_tool_node(self.tools)
        should_continue = build_routing_node()

        workflow = StateGraph(GraphState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        self._graph = workflow.compile()

    def _ensure_graph(self):
        if self._graph is None:
            raise RuntimeError("Graph is not built. Call build_graph() first.")
        return self._graph

    def invoke(self, *args: Any, **kwargs: Any):
        return self._ensure_graph().invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any):
        return await self._ensure_graph().ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any):
        return self._ensure_graph().stream(*args, **kwargs)

    def astream(self, *args: Any, **kwargs: Any):
        return self._ensure_graph().astream(*args, **kwargs)
