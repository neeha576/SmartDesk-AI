"""Embedding models. Provider and model names come from .env via config.settings:

Dense:  EMBEDDING_PROVIDER = openai | huggingface,  EMBEDDING_MODEL = <model name>
Sparse: SPARSE_MODEL = <fastembed sparse model>  (BM25, runs locally, no API key)
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class OpenAIEmbedder:
    def __init__(self, model: str):
        from openai import OpenAI
        settings.required("OPENAI_API_KEY")   # fail fast with a clear message
        self.client = OpenAI()
        self.model = model

    @retry(stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), settings.EMBEDDING_BATCH_SIZE):
            out.extend(self._embed_batch(texts[i:i + settings.EMBEDDING_BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


class HuggingFaceEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE,
                                 normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


_PROVIDERS = {
    "openai": OpenAIEmbedder,
    "huggingface": HuggingFaceEmbedder,
    "hf": HuggingFaceEmbedder,
    "sentence-transformers": HuggingFaceEmbedder,
}


def get_dense_embedder():
    """Build the dense embedder named by EMBEDDING_PROVIDER / EMBEDDING_MODEL in .env."""
    provider = settings.embedding_provider()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{provider}'. Use one of: {', '.join(_PROVIDERS)}")
    return _PROVIDERS[provider](settings.embedding_model())


def get_sparse_embedder():
    """Build the sparse (BM25) embedder named by SPARSE_MODEL in .env."""
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=settings.sparse_model())
