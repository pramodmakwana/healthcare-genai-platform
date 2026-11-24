"""Ingestion pipeline: load -> chunk -> embed -> index.

Run directly (`python -m hcai.ingestion.pipeline`) or from the Airflow DAG in dags/.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from hcai.config import Settings, get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.ingestion.chunking import chunk_documents
from hcai.ingestion.documents import Document, load_fhir_dir, load_fhir_server, load_markdown_dir
from hcai.vectorstores.stores import get_vector_store

log = logging.getLogger("hcai.ingest")


def load_all(settings: Settings, fhir_server: str | None = None) -> list[Document]:
    raw = Path(settings.raw_data_dir)
    docs = load_markdown_dir(raw / "guidelines")
    docs += load_fhir_server(fhir_server) if fhir_server else load_fhir_dir(raw / "fhir")
    return docs


def validate(docs: list[Document]) -> None:
    """Basic data-quality gates before anything is indexed."""
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs in ingestion batch")
    empty = [d.doc_id for d in docs if len(d.text.split()) < 10]
    if empty:
        raise ValueError(f"Documents with almost no text: {empty[:5]}")


def run(settings: Settings | None = None, fhir_server: str | None = None, reset: bool = True) -> dict:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    docs = load_all(settings, fhir_server)
    validate(docs)
    chunks = chunk_documents(docs, settings.chunking, settings.chunk_size, settings.chunk_overlap)

    embedder = get_embedder(settings)
    store = get_vector_store(settings, embedder.dim)
    if reset:
        store.reset()
    batch = 64
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        store.add(part, embedder.embed([c.text for c in part]))

    stats = {
        "documents": len(docs),
        "patient_records": sum(d.source_type == "patient_record" for d in docs),
        "guidelines": sum(d.source_type == "guideline" for d in docs),
        "chunks": len(chunks),
        "vector_store": settings.vector_store,
        "embedding_provider": settings.embedding_provider,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    log.info("Ingestion complete: %s", stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest FHIR records and guidelines into the vector store")
    ap.add_argument("--fhir-server", help="FHIR base URL, e.g. http://localhost:8080/fhir (default: read data/raw/fhir)")
    args = ap.parse_args()
    print(run(fhir_server=args.fhir_server))


if __name__ == "__main__":
    main()
