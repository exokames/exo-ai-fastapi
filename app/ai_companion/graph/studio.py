from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.ai_companion.graph import graph_builder
from app.core.config import settings


async def get_graph():
    """Asynchronously creates and returns the LangGraph agent's graph."""
    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)

    return graph
