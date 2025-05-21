from app.core.langgraph.retrieval_graph.graph import LangGraphAgent

agent = LangGraphAgent()


async def get_graph():
    """Asynchronously creates and returns the LangGraph agent's graph."""
    # The graph instance needs to be created and returned by this async function
    graph_instance = await agent.create_graph()
    return graph_instance


# LangGraph Studio typically allows specifying a module and a factory function
# (e.g., studio:get_graph) or expects a variable named 'graph' or 'app'.
# If Studio expects a 'graph' variable directly:
# Option 1 (graph is a coroutine, Studio awaits it):
#   graph = get_graph()
# Option 2 (graph is resolved, blocks import - use with care):
#   import asyncio
#   graph = asyncio.run(get_graph())
# The factory pattern (get_graph) is generally preferred for async setups.
