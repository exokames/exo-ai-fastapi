from app.core.langgraph.retrieval_graph.graph import LangGraphAgent

agent = LangGraphAgent()


async def get_graph():
    """Asynchronously creates and returns the LangGraph agent's graph."""
    # The graph instance needs to be created and returned by this async function
    graph_instance = await agent.create_graph()
    return graph_instance
