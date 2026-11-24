"""Retrieval: dense (vector), sparse (BM25), or hybrid via Reciprocal Rank Fusion.

Why hybrid matters in healthcare: questions often hinge on exact tokens - a patient name,
an MRN, a LOINC/ICD code, a drug name - which keyword search nails and embeddings can blur.
Semantic search covers paraphrases ("kidney function" vs "eGFR"). RRF combines both rankings
without having to calibrate their very different score scales.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from hcai.embeddings.embedders import Embedder
from hcai.ingestion.documents import Chunk
from hcai.vectorstores.stores import VectorStore

_TITLE = re.compile(r"Patient record: (.+?) \((MRN\d+)\)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


# Function words and words present in nearly every chunk; left in, they make BM25 favour
# short chunks that merely repeat the header ("Patient record: ... > Encounters").
_STOP = frozenset(
    "a an the of to and or in on for is are was were be been with by at as from that this it its "
    "what which who whom when where how why does do did should can could would will may might must "
    "patient patients record most recent current currently".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, mode: str = "hybrid", rrf_k: int = 60,
                 patient_scoping: bool = False):
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.rrf_k = rrf_k
        self.patient_scoping = patient_scoping
        self.last_confidence: float = 0.0  # top dense cosine of the last query
        self._build_sparse()

    def _build_sparse(self) -> None:
        self._chunks = self.store.all_chunks()
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks]) if self._chunks else None
        # Entity index for patient-scoped retrieval: MRN / full name -> patient doc_id.
        self._patients: dict[str, str] = {}
        for c in self._chunks:
            if c.source_type != "patient_record":
                continue
            m = _TITLE.match(c.title)
            if m:
                self._patients[m.group(1).lower()] = c.doc_id
                self._patients[m.group(2).lower()] = c.doc_id

    def patient_names(self) -> list[str]:
        return [k for k in self._patients if not k.startswith("mrn")]

    def detect_patient(self, query: str) -> str | None:
        """Return the patient doc_id if the query names exactly one known patient (by MRN or full name)."""
        q = query.lower()
        found = {doc for key, doc in self._patients.items() if re.search(rf"\b{re.escape(key)}\b", q)}
        return found.pop() if len(found) == 1 else None

    def _scoped(self, query: str, doc_id: str, k: int) -> list[tuple[Chunk, float]]:
        """Rank only this patient's chunks (BM25 + dense via RRF), then add the best non-patient
        context (guidelines). Other patients' records never enter the prompt."""
        own = [c for c in self._chunks if c.doc_id == doc_id]
        q_tokens = tokenize(query)
        sparse_scores = {c.chunk_id: sum(t in tokenize(c.text) for t in q_tokens) for c in own}
        vecs = self.embedder.embed([c.text for c in own])
        dense_scores = dict(zip([c.chunk_id for c in own], (vecs @ self.embedder.embed_query(query)).tolist(), strict=True))
        fused: dict[str, float] = {}
        for scores in (sparse_scores, dense_scores):
            for rank, cid in enumerate(sorted(scores, key=lambda x: -scores[x])):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        by_id = {c.chunk_id: c for c in own}
        n_own = max(1, k - 1)
        ranked = [(by_id[cid], sc) for cid, sc in sorted(fused.items(), key=lambda kv: -kv[1])[:n_own]]
        general = [(c, sc) for c, sc in self._hybrid(query, k * 4) if c.source_type != "patient_record"]
        return ranked + general[: k - len(ranked)]

    def refresh(self) -> None:
        """Call after re-ingestion so BM25 sees the new corpus."""
        self._build_sparse()

    def _dense(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed_query(query), k)

    def _sparse(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def _hybrid(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        pool = max(k * 4, 20)
        dense = self._dense(query, pool)
        self.last_confidence = dense[0][1] if dense else 0.0
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranking in (dense, self._sparse(query, pool)):
            for rank, (chunk, _) in enumerate(ranking):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.chunk_id] = chunk
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(by_id[cid], score) for cid, score in top]

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        patient_doc = self.detect_patient(query) if self.patient_scoping else None
        if patient_doc:
            hits = self._scoped(query, patient_doc, k)
            self.last_confidence = 1.0  # a named, known patient is always in scope
        elif self.mode == "bm25":
            hits = self._sparse(query, k)
            self.last_confidence = 1.0 if hits else 0.0
        elif self.mode == "dense":
            hits = self._dense(query, k)
            self.last_confidence = hits[0][1] if hits else 0.0
        else:
            hits = self._hybrid(query, k)
        return [RetrievedChunk(c, s, i + 1) for i, (c, s) in enumerate(hits)]
