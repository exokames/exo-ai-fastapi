"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    TypedDict,
    cast,
)

from asgiref.sync import sync_to_async
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import Send
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.retrieval_graph.configuration import AgentConfiguration
from app.core.langgraph.retrieval_graph.researcher_graph import retrieval
from app.core.langgraph.retrieval_graph.researcher_graph.state import QueryState, ResearcherState
from app.core.logging import logger
from app.schemas import (
    Message,
)
from app.utils import dump_messages


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use environment-specific LLM model
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            **self._get_model_kwargs(),
        )
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

        logger.info("llm_initialized", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                self._connection_pool = AsyncConnectionPool(
                    settings.POSTGRES_URL,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _generate_queries(self, state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
        """Generate search queries based on the question (a step in the research plan).

        This function uses a language model to generate diverse search queries to help answer the question.

        Args:
            state (ResearcherState): The current state of the researcher, including the user's question.
            config (RunnableConfig): Configuration with the model used to generate queries.

        Returns:
            dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
        """

        class Response(TypedDict):
            queries: list[str]

        configuration = AgentConfiguration.from_runnable_config(config)
        structured_output_kwargs = {"method": "function_calling"} if "openai" in configuration.query_model else {}
        model = self.llm.with_structured_output(Response, **structured_output_kwargs)
        messages = [
            {"role": "system", "content": configuration.generate_queries_system_prompt},
            {"role": "human", "content": state.question},
        ]
        response = cast(Response, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]}))
        return {"queries": response["queries"]}

    async def _retrieve_documents(self, state: QueryState, *, config: RunnableConfig) -> dict[str, list[Document]]:
        """Retrieve documents based on a given query.

        This function uses a retriever to fetch relevant documents for a given query.

        Args:
            state (QueryState): The current state containing the query string.
            config (RunnableConfig): Configuration with the retriever used to fetch documents.

        Returns:
            dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
        """
        with retrieval.make_retriever(config) as retriever:
            response = await retriever.ainvoke(state.query, config)
            return {"documents": response}

    def _retrieve_in_parallel(self, state: ResearcherState) -> list[Send]:
        """Create parallel retrieval tasks for each generated query.

        This function prepares parallel document retrieval tasks for each query in the researcher's state.

        Args:
            state (ResearcherState): The current state of the researcher, including the generated queries.

        Returns:
            Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

        Behavior:
            - Creates a Send object for each query in the state.
            - Each Send object targets the "retrieve_documents" node with the corresponding query.
        """
        return [Send("retrieve_documents", QueryState(query=query)) for query in state.queries]

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(ResearcherState)
                graph_builder.add_node("generate_queries", self._generate_queries)
                graph_builder.add_node("retrieve_documents", self._retrieve_documents)

                graph_builder.add_edge(START, "generate_queries")
                graph_builder.add_conditional_edges(
                    "generate_queries",
                    self._retrieve_in_parallel,  # type: ignore
                    path_map=["retrieve_documents"],
                )
                graph_builder.add_edge("retrieve_documents", END)

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(checkpointer=checkpointer, name="ResearcherGraph")

                logger.info(
                    "graph_created",
                    graph_name="ResearcherGraph",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for tracking.
            user_id (Optional[str]): The user ID for tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
        }
        try:
            response = await self._graph.ainvoke(
                {"messages": dump_messages(messages), "session_id": session_id}, config
            )
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise e

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        try:
            async for data in self._graph.astream_events(
                {"messages": dump_messages(messages), "session_id": session_id}, config, version="v2"
            ):
                try:
                    yield data
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
