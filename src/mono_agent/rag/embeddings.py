"""Embedding models behind LangChain's Embeddings interface.

"local:BAAI/bge-m3" runs sentence-transformers on this machine; anything else
("google_genai:gemini-embedding-001", "openai:text-embedding-3-small", ...) goes through
langchain's init_embeddings and needs that provider's package and API key.
"""

from functools import cached_property

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model: str) -> None:
        self.model_name = model

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer  # heavy import: only when used

        return load_cached(SentenceTransformer, self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_cached(cls, name: str):
    """Load from the local Hugging Face cache without asking the Hub for updates (faster
    start, works offline); download only if the model isn't cached yet."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        # A local folder path never touches the network (local_files_only alone still did).
        return cls(snapshot_download(name, local_files_only=True))
    except LocalEntryNotFoundError:
        return cls(name)


def get_embeddings(spec: str) -> Embeddings:
    provider, _, model = spec.partition(":")
    if provider == "local":
        return LocalEmbeddings(model)
    from langchain.embeddings import init_embeddings

    return init_embeddings(spec)
