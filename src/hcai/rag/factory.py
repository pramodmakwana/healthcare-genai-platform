from __future__ import annotations

from hcai.config import Settings, get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.ingestion import pipeline
from hcai.llm.factory import get_llm
from hcai.rag.chain import RAGChain
from hcai.retrieval.retriever import Retriever
from hcai.vectorstores.stores import get_vector_store


def build_rag_chain(settings: Settings | None = None, ingest_if_empty: bool = True) -> RAGChain:
    """Wire embedder -> vector store -> retriever -> LLM into a RAGChain."""
    settings = settings or get_settings()
    embedder = get_embedder(settings)
    store = get_vector_store(settings, embedder.dim)
    if ingest_if_empty and store.count() == 0:
        pipeline.run(settings)
        store = get_vector_store(settings, embedder.dim)
    retriever = Retriever(embedder, store, settings.retrieval_mode, patient_scoping=settings.patient_scoping)
    return RAGChain(get_llm(settings), retriever, settings)
