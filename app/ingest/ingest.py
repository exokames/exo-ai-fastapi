"""Load html from files, clean up, split, ingest into Qdrant."""

import logging
import os

from langchain.indexes import index
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
)
from langchain_community.indexes._document_manager import MongoDocumentManager
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.embeddings import make_text_encoder
from app.ingest.utils import (
    product_metadata_func,
    restaurant_metadata_func,
)
from app.ingest.web_loaders import (
    load_web_docs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_restaurant_docs(file_path: str) -> list[Document]:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",
        content_key="text",
        json_lines=True,
        metadata_func=restaurant_metadata_func,
    )

    return loader.load()


def load_product_docs(file_path: str) -> list[Document]:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",
        content_key="text",
        json_lines=True,
        metadata_func=product_metadata_func,
    )

    return loader.load()


def load_csv_docs(file_path: str) -> list[Document]:
    loader = CSVLoader(file_path=file_path)

    return loader.load()


class CustomMongoDocumentManager(MongoDocumentManager):
    def get_time(self):
        # Alternative to avoid hostInfo on MongoDB Atlas
        from datetime import datetime

        return datetime.now()


def ingest_docs():
    vs_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    embedding_model = make_text_encoder(settings.EMBEDDING_MODEL)
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        # prefer_grpc=True,
        embedding=embedding_model,
    )

    record_manager = CustomMongoDocumentManager(
        namespace=f"qdrant/{settings.QDRANT_COLLECTION_NAME}",
        mongodb_url=settings.ATLAS_URI,
        db_name=settings.DBNAME,
        collection_name="qdrant_rm",
    )

    record_manager.create_schema()

    # docs = load_restaurant_docs(settings.RESTAURANT_DOCS_PATH)
    # docs = load_product_docs(settings.PRODUCT_DOCS_PATH)

    # logger.info(f"Loaded {len(docs)} docs")

    web_docs = load_web_docs()
    logger.info(f"Loaded {len(web_docs)} web docs.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    docs_transformed = text_splitter.split_documents(
        # docs_from_documentation
        # + docs_from_api
        web_docs
        # + docs_from_langgraph
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]
    docs = docs_transformed

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Qdrant will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing Stats: {indexing_stats}")

    vs_stats = vs_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
    logger.info(f"Collection Stats (points_count): {vs_stats.points_count}")


if __name__ == "__main__":
    ingest_docs()
