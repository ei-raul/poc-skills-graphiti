from datetime import datetime
from typing import Any, Dict, Optional
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config.config import Config
from config.logger import get_logger

config = Config()
logger = get_logger(__name__)


class GraphitiAddEventInput(BaseModel):
    content: str = Field(
        ...,
        description=(
            "Natural language description of the event "
            "(e.g., 'Met with John at the office at 2pm to discuss the project')"
        ),
    )
    timestamp: Optional[str] = Field(
        None,
        description=(
            "ISO 8601 timestamp (optional, defaults to now). "
            "Example: '2026-02-20T14:00:00'"
        ),
    )
    source: str = Field(
        "user",
        description="Source of the event (optional, e.g., 'user', 'system', 'email')",
    )
    group_id: str = Field("default", description="Group ID for organizing events (optional)")


class GraphitiEntityEdgesInput(BaseModel):
    entity_name: str = Field(
        ...,
        description="Name of the entity to explore (e.g., 'John', 'Project Alpha')",
    )
    group_id: Optional[str] = Field(None, description="Filter by group ID (optional)")


class GraphitiRemoveEpisodeInput(BaseModel):
    episode_uuid: str = Field(
        ...,
        description="UUID of the episode to remove from Graphiti.",
    )


class GraphitiRecentEpisodesInput(BaseModel):
    limit: int = Field(10, description="Number of recent episodes to retrieve (default: 10)")
    group_id: Optional[str] = Field(None, description="Filter by group ID (optional)")


class GraphitiSearchEventsInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description=(
            "Optional search query (e.g., 'travel to London', 'wedding events')"
        ),
    )
    entity_name: Optional[str] = Field(
        None,
        description="Optional entity to search for (e.g., 'Chelsea apartment', 'John')",
    )
    after_timestamp: Optional[str] = Field(
        None,
        description=(
            "ISO 8601 timestamp - only return events AFTER this time "
            "(e.g., '2025-05-01T14:00:00')"
        ),
    )
    before_timestamp: Optional[str] = Field(
        None,
        description=(
            "ISO 8601 timestamp - only return events BEFORE this time "
            "(e.g., '2025-05-10T00:00:00')"
        ),
    )
    group_id: Optional[str] = Field(None, description="Filter by group ID (optional)")
    limit: int = Field(20, description="Maximum number of events to return (default: 20)")


_graphiti_instance = None


def _parse_iso_timestamp(timestamp_str: Optional[str], field_name: str) -> Optional[datetime]:
    if not timestamp_str:
        return None
    try:
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError:
        logger.warning(f"Invalid {field_name} format: {timestamp_str}")
        return None


def _format_episode(episode: Any) -> Dict[str, Any]:
    source_value = getattr(episode, "source", "")
    if hasattr(source_value, "value"):
        source_value = source_value.value

    return {
        "name": episode.name,
        "content": getattr(episode, "content", ""),
        "source": str(source_value) if source_value else "",
        "created_at": episode.created_at.isoformat() if hasattr(episode, "created_at") else None,
        "valid_at": episode.valid_at.isoformat() if hasattr(episode, "valid_at") else None,
        "group_id": getattr(episode, "group_id", ""),
        "uuid": getattr(episode, "uuid", ""),
    }


async def get_graphiti() -> Graphiti:
    global _graphiti_instance

    if _graphiti_instance is None:
        logger.info("Initializing Graphiti instance...")

        llm_config = LLMConfig(
            api_key=config.get("GOOGLE_API_KEY"),
            model="gemini-2.5-flash",
        )
        llm_client = GeminiClient(config=llm_config)

        _graphiti_instance = Graphiti(
            uri=config.get("NEO4J_URI"),
            user=config.get("NEO4J_USER"),
            password=config.get("NEO4J_PASSWORD"),
            llm_client=llm_client,
        )

        logger.info("Building Neo4j indices and constraints...")
        await _graphiti_instance.build_indices_and_constraints()
        logger.info("Graphiti instance initialized successfully")

    return _graphiti_instance


@tool(args_schema=GraphitiAddEventInput)
async def graphiti_add_event(
    content: str,
    timestamp: Optional[str] = None,
    source: str = "user",
    group_id: str = "default",
) -> Dict[str, Any]:
    """
    Add a new temporal event to the knowledge graph.
    """
    graphiti = await get_graphiti()

    parsed_timestamp = _parse_iso_timestamp(timestamp, "timestamp")
    if timestamp and not parsed_timestamp:
        parsed_timestamp = datetime.now()
    if not parsed_timestamp:
        parsed_timestamp = datetime.now()

    logger.info(f"Adding event: {content[:100]}... at {parsed_timestamp}")

    await graphiti.add_episode(
        name=f"Event at {parsed_timestamp.isoformat()}",
        episode_body=content,
        source_description=source,
        reference_time=parsed_timestamp,
        group_id=group_id,
    )

    result = {
        "success": True,
        "message": "Event added successfully",
        "timestamp": parsed_timestamp.isoformat(),
        "content": content,
        "source": source,
        "group_id": group_id,
    }

    logger.info(f"Event added successfully: {result}")
    return result


@tool(args_schema=GraphitiRemoveEpisodeInput)
async def graphiti_remove_event(episode_uuid: str) -> Dict[str, Any]:
    """
    Remove an existing episode from the knowledge graph by UUID.
    """
    graphiti = await get_graphiti()

    logger.info(f"Removing episode: {episode_uuid}")
    await graphiti.remove_episode(episode_uuid=episode_uuid)

    result = {
        "success": True,
        "message": "Episode removed successfully",
        "episode_uuid": episode_uuid,
    }

    logger.info(f"Episode removed successfully: {result}")
    return result


@tool(args_schema=GraphitiEntityEdgesInput)
async def graphiti_get_entity_edges(entity_name: str, group_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for relationships and facts about entities in the knowledge graph.
    """
    graphiti = await get_graphiti()

    logger.info(f"Getting edges for entity: {entity_name}")

    edges = await graphiti.search(
        query=entity_name,
        group_ids=[group_id] if group_id else None,
        num_results=20,
    )

    if not edges:
        return {
            "success": False,
            "error": f"No relationships found for entity '{entity_name}'",
        }

    relationships = []
    for edge in edges:
        relationships.append(
            {
                "fact": getattr(edge, "fact", edge.name),
                "source_id": edge.source_node_uuid,
                "target_id": edge.target_node_uuid,
                "created_at": edge.created_at.isoformat() if hasattr(edge, "created_at") else None,
            }
        )

    result = {
        "success": True,
        "entity_query": entity_name,
        "relationships_found": len(relationships),
        "relationships": relationships,
    }

    logger.info(f"Found {len(relationships)} relationships for {entity_name}")
    return result


@tool(args_schema=GraphitiRecentEpisodesInput)
async def graphiti_list_recent_episodes(limit: int = 10, group_id: Optional[str] = None) -> Dict[str, Any]:
    """
    List the most recent temporal episodes (events) from the knowledge graph.
    """
    graphiti = await get_graphiti()
    logger.info(f"Listing {limit} recent episodes")

    episodes = await graphiti.retrieve_episodes(
        reference_time=datetime.now(),
        last_n=limit,
        group_ids=[group_id] if group_id else None,
    )

    episodes_list = [_format_episode(episode) for episode in episodes]
    result = {
        "success": True,
        "episodes_found": len(episodes_list),
        "episodes": episodes_list,
    }

    logger.info(f"Retrieved {len(episodes_list)} episodes")
    return result


@tool(args_schema=GraphitiSearchEventsInput)
async def graphiti_search_events(
    query: Optional[str] = None,
    entity_name: Optional[str] = None,
    after_timestamp: Optional[str] = None,
    before_timestamp: Optional[str] = None,
    group_id: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search for events with temporal and entity filtering.
    """
    graphiti = await get_graphiti()

    after_dt = _parse_iso_timestamp(after_timestamp, "after_timestamp")
    before_dt = _parse_iso_timestamp(before_timestamp, "before_timestamp")
    effective_query = query or entity_name or "events"

    logger.info(
        "Searching events: "
        f"query={query}, entity={entity_name}, after={after_dt}, before={before_dt}, limit={limit}"
    )

    episodes = await graphiti.retrieve_episodes(
        reference_time=before_dt or datetime.now(),
        last_n=100,
        group_ids=[group_id] if group_id else None,
    )

    search_text = (entity_name or query or "").lower()
    filtered_episodes = []

    for episode in episodes:
        episode_time = getattr(episode, "valid_at", None)
        if not episode_time:
            continue

        if after_dt and episode_time <= after_dt:
            continue
        if before_dt and episode_time >= before_dt:
            continue

        if search_text:
            content = getattr(episode, "content", "").lower()
            name = getattr(episode, "name", "").lower()
            if search_text not in content and search_text not in name:
                continue

        filtered_episodes.append(episode)
        if len(filtered_episodes) >= limit:
            break

    events = [_format_episode(episode) for episode in filtered_episodes]
    result = {
        "success": True,
        "query": effective_query,
        "after_timestamp": after_dt.isoformat() if after_dt else None,
        "before_timestamp": before_dt.isoformat() if before_dt else None,
        "events_found": len(events),
        "events": events,
    }

    logger.info(f"Found {len(events)} events matching criteria")
    return result
