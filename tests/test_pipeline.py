import json
from pathlib import Path

import numpy as np

from hcai.config import PROJECT_ROOT
from hcai.ingestion.chunking import chunk_documents
from hcai.ingestion.documents import fhir_bundle_to_documents, load_markdown_dir
from hcai.llm.base import Usage, estimate_cost_usd
from hcai.vectorstores.stores import FaissStore, MemoryStore


def _bundle() -> dict:
    f = sorted((PROJECT_ROOT / "data/raw/fhir").glob("*.json"))[0]
    return json.loads(f.read_text())


def test_fhir_bundle_renders_codes_and_sections():
    doc = fhir_bundle_to_documents(_bundle())[0]
    assert doc.source_type == "patient_record"
    assert "## Recent observations" in doc.text and "LOINC 4548-4" in doc.text
    assert doc.metadata["mrn"].startswith("MRN")


def test_section_chunks_carry_title_and_skip_disclaimers():
    docs = load_markdown_dir(PROJECT_ROOT / "data/raw/guidelines")
    chunks = chunk_documents(docs, "section", 120, 20)
    assert all(" > " in c.text.splitlines()[0] for c in chunks)
    assert not any("Synthetic document" in c.text for c in chunks)


def test_memory_and_faiss_stores_agree(tmp_path: Path):
    docs = load_markdown_dir(PROJECT_ROOT / "data/raw/guidelines")
    chunks = chunk_documents(docs, "section", 120, 20)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(chunks), 32)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    mem, fai = MemoryStore(str(tmp_path / "m")), FaissStore(str(tmp_path / "f"), 32)
    mem.add(chunks, vecs)
    fai.add(chunks, vecs)
    q = vecs[3]
    assert [c.chunk_id for c, _ in mem.search(q, 3)] == [c.chunk_id for c, _ in fai.search(q, 3)]
    # persisted index reloads
    assert MemoryStore(str(tmp_path / "m")).count() == len(chunks)


def test_cost_estimate_uses_pricing_table():
    pricing = str(PROJECT_ROOT / "config/pricing.yaml")
    cost = estimate_cost_usd("claude-haiku-5-5", Usage(input_tokens=1_000_000, output_tokens=0), pricing)
    assert cost is not None and cost > 0
    assert estimate_cost_usd("unknown-model", Usage(), pricing) is None
