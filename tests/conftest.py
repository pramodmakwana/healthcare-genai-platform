import pytest

from hcai.config import get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.ingestion import pipeline
from hcai.llm.providers import MockProvider
from hcai.rag.chain import RAGChain
from hcai.retrieval.retriever import Retriever
from hcai.vectorstores.stores import get_vector_store


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return get_settings(llm_provider="mock", vector_store="memory", embedding_provider="hashing",
                        index_path=str(tmp_path_factory.mktemp("index")))


@pytest.fixture(scope="session")
def chain(settings):
    pipeline.run(settings)
    emb = get_embedder(settings)
    retriever = Retriever(emb, get_vector_store(settings, emb.dim), "hybrid", patient_scoping=True)
    return RAGChain(MockProvider(model="mock"), retriever, settings)
