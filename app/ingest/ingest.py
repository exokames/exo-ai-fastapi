"""Load html from files, clean up, split, ingest into Qdrant."""

import logging
import os

from bs4 import BeautifulSoup, SoupStrainer
from langchain.indexes import index
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.document_loaders import (
    JSONLoader,
    RecursiveUrlLoader,
    SitemapLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.indexes._document_manager import MongoDocumentManager
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.embeddings import make_text_encoder
from app.ingest.parser import langchain_docs_extractor
from app.ingest.utils import (
    metadata_extractor,
    product_metadata_func,
    restaurant_metadata_func,
    simple_extractor,
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


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(name=("article", "title", "html", "lang", "content")),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(meta, soup, title_suffix=" | 🦜🕸️LangGraph"),
    ).load()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


class CustomMongoDocumentManager(MongoDocumentManager):
    def get_time(self):
        # Alternative to avoid hostInfo on MongoDB Atlas
        from datetime import datetime

        return datetime.now()


def ingest_docs():
    vs_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        # prefer_grpc=True,
        embedding=make_text_encoder(settings.EMBEDDING_MODEL),
    )

    record_manager = CustomMongoDocumentManager(
        namespace=f"qdrant/{settings.QDRANT_COLLECTION_NAME}",
        mongodb_url=settings.ATLAS_URI,
        db_name=settings.DBNAME,
        collection_name="qdrant_rm",
    )

    record_manager.create_schema()

    docs = load_restaurant_docs(settings.RESTAURANT_DOCS_PATH)
    # docs = load_product_docs(settings.PRODUCT_DOCS_PATH)

    logger.info(f"Loaded {len(docs)} docs")

    # docs_from_fruitsandroots = load_fruitsandroots_docs()
    # logger.info(f"Loaded {len(docs_from_fruitsandroots)} docs from Fruits & Roots")

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    # docs_transformed = text_splitter.split_documents(
    #     # docs_from_documentation
    #     # + docs_from_api
    #     docs_from_fruitsandroots
    #     # + docs_from_langgraph
    # )
    # docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

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
