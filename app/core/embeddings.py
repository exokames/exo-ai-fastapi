from langchain_core.embeddings import Embeddings


def make_text_encoder(model: str, task_type: str = "RETRIEVAL_DOCUMENT") -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)

        case "vertexai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(model=f"models/{model}", task_type=task_type)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")
