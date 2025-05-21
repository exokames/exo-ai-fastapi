from langchain_core.embeddings import Embeddings
from langchain_google_vertexai.embeddings import VertexAIEmbeddings


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)

        case "vertexai":
            from langchain_google_vertexai.embeddings import VertexAIEmbeddings

            return VertexAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")
