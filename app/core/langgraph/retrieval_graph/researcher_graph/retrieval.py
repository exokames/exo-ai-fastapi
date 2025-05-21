from contextlib import contextmanager
from typing import Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.embeddings import make_text_encoder
from app.schemas.configuration import BaseConfiguration


@contextmanager
def make_qdrant_retriever(
    configuration: BaseConfiguration,
    embedding_model: Embeddings,
) -> Iterator[BaseRetriever]:
    # with QdrantClient(url=QDRANT_URL) as vs_client:
    store = QdrantVectorStore.from_existing_collection(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        # prefer_grpc=True,
        embedding=embedding_model,
    )
    search_kwargs = {**configuration.search_kwargs}
    yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "qdrant":
            with make_qdrant_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
