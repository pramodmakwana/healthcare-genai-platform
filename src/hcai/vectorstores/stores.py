"""Vector store backends behind one interface.

memory   : numpy brute-force cosine; persisted as .npz + json. Default, zero dependencies.
faiss    : FAISS IndexFlatIP (exact inner product on normalised vectors = cosine).
chroma   : ChromaDB persistent collection (cosine space).
pgvector : PostgreSQL + pgvector with an HNSW index. Fits next to the Postgres you already run.
pinecone : Pinecone serverless index (managed; needs PINECONE_API_KEY).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hcai.config import Settings
from hcai.ingestion.documents import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...

    @abstractmethod
    def reset(self) -> None: ...

    def count(self) -> int:
        return len(self.all_chunks())


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(**d)


class MemoryStore(VectorStore):
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self._chunks: list[Chunk] = []
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        if self.path and (self.path / "vectors.npz").exists():
            self._vecs = np.load(self.path / "vectors.npz")["v"]
            self._chunks = [_chunk_from_dict(d) for d in json.loads((self.path / "chunks.json").read_text())]

    def add(self, chunks, vectors):
        self._vecs = vectors if self._vecs.size == 0 else np.vstack([self._vecs, vectors])
        self._chunks.extend(chunks)
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path / "vectors.npz", v=self._vecs)
            (self.path / "chunks.json").write_text(json.dumps([asdict(c) for c in self._chunks]))

    def search(self, query_vec, k):
        if not self._chunks:
            return []
        scores = self._vecs @ query_vec
        idx = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    def all_chunks(self):
        return list(self._chunks)

    def reset(self):
        self._chunks, self._vecs = [], np.zeros((0, 0), dtype=np.float32)
        if self.path:
            for f in ("vectors.npz", "chunks.json"):
                (self.path / f).unlink(missing_ok=True)


class FaissStore(VectorStore):
    def __init__(self, path: str, dim: int):
        import faiss

        self.faiss, self.dim, self.path = faiss, dim, Path(path)
        self._chunks: list[Chunk] = []
        self.index = faiss.IndexFlatIP(dim)
        if (self.path / "faiss.index").exists():
            self.index = faiss.read_index(str(self.path / "faiss.index"))
            self._chunks = [_chunk_from_dict(d) for d in json.loads((self.path / "faiss_chunks.json").read_text())]

    def add(self, chunks, vectors):
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)
        self.path.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.path / "faiss.index"))
        (self.path / "faiss_chunks.json").write_text(json.dumps([asdict(c) for c in self._chunks]))

    def search(self, query_vec, k):
        if self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [(self._chunks[i], float(s)) for s, i in zip(scores[0], idx[0], strict=False) if i >= 0]

    def all_chunks(self):
        return list(self._chunks)

    def reset(self):
        self.index = self.faiss.IndexFlatIP(self.dim)
        self._chunks = []
        for f in ("faiss.index", "faiss_chunks.json"):
            (self.path / f).unlink(missing_ok=True)


class ChromaStore(VectorStore):
    COLLECTION = "hcai_chunks"

    def __init__(self, path: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})

    def add(self, chunks, vectors):
        for start in range(0, len(chunks), 500):
            batch = chunks[start:start + 500]
            self.col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start:start + 500].tolist(),
                documents=[c.text for c in batch],
                metadatas=[{"payload": json.dumps(asdict(c))} for c in batch],
            )

    def search(self, query_vec, k):
        res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=k)
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0], strict=False):
            out.append((_chunk_from_dict(json.loads(meta["payload"])), 1.0 - float(dist)))
        return out

    def all_chunks(self):
        got = self.col.get(include=["metadatas"])
        return [_chunk_from_dict(json.loads(m["payload"])) for m in got["metadatas"]]

    def reset(self):
        self.client.delete_collection(self.COLLECTION)
        self.col = self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})


class PgVectorStore(VectorStore):
    """Postgres + pgvector. Detects an embedding-dimension change (e.g. switching from hashing
    to bge-small) and rebuilds the table on the next re-ingestion instead of failing on insert."""

    def __init__(self, dsn: str, dim: int, table: str = "rag_chunks"):
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        self.dim, self.table = dim, table
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        existing = self._existing_dim()
        self.dim_mismatch = existing is not None and existing != dim
        if existing is None:
            self._create()

    def _existing_dim(self) -> int | None:
        row = self.conn.execute(
            "SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding'",
            (self.table,),
        ).fetchone()
        return row[0] if row else None

    def _create(self) -> None:
        t = self.table
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {t} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                payload JSONB NOT NULL,
                embedding vector({self.dim}) NOT NULL
            )"""
        )
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)")

    def _check(self) -> None:
        if self.dim_mismatch:
            raise ValueError(f"Table {self.table} holds vectors of a different dimension than the current "
                             f"embedder ({self.dim}). Re-run ingestion to rebuild the index.")

    def add(self, chunks, vectors):
        self._check()
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (chunk_id, doc_id, source_type, payload, embedding)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET payload = EXCLUDED.payload, embedding = EXCLUDED.embedding""",
                [(c.chunk_id, c.doc_id, c.source_type, json.dumps(asdict(c)), v) for c, v in zip(chunks, vectors, strict=True)],
            )

    def search(self, query_vec, k):
        self._check()
        rows = self.conn.execute(
            f"SELECT payload, 1 - (embedding <=> %s) AS score FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
        return [(_chunk_from_dict(p), float(s)) for p, s in rows]

    def all_chunks(self):
        if self.dim_mismatch:
            return []
        return [_chunk_from_dict(p) for (p,) in self.conn.execute(f"SELECT payload FROM {self.table}").fetchall()]

    def count(self) -> int:
        if self.dim_mismatch:
            return 0
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def reset(self):
        if self.dim_mismatch:
            self.conn.execute(f"DROP TABLE {self.table}")
            self._create()
            self.dim_mismatch = False
        else:
            self.conn.execute(f"TRUNCATE {self.table}")


class PineconeStore(VectorStore):
    def __init__(self, dim: int, index_name: str | None = None):
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        name = index_name or os.environ.get("PINECONE_INDEX", "hcai-chunks")
        if name not in [i["name"] for i in pc.list_indexes()]:
            pc.create_index(name=name, dimension=dim, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_REGION", "us-east-1")))
        self.index = pc.Index(name)
        self._cache: list[Chunk] = []

    def add(self, chunks, vectors):
        self._cache.extend(chunks)
        for start in range(0, len(chunks), 100):
            self.index.upsert(vectors=[
                {"id": c.chunk_id, "values": v.tolist(), "metadata": {"payload": json.dumps(asdict(c))}}
                for c, v in zip(chunks[start:start + 100], vectors[start:start + 100], strict=True)
            ])

    def search(self, query_vec, k):
        res = self.index.query(vector=query_vec.tolist(), top_k=k, include_metadata=True)
        return [(_chunk_from_dict(json.loads(m["metadata"]["payload"])), float(m["score"])) for m in res["matches"]]

    def all_chunks(self):
        return list(self._cache)  # Pinecone has no cheap "scan"; BM25 uses the ingest-time cache.

    def reset(self):
        self.index.delete(delete_all=True)
        self._cache = []


def get_vector_store(settings: Settings, dim: int) -> VectorStore:
    kind = settings.vector_store.lower()
    if kind == "memory":
        return MemoryStore(settings.index_path)
    if kind == "faiss":
        return FaissStore(settings.index_path, dim)
    if kind == "chroma":
        return ChromaStore(settings.chroma_path)
    if kind == "pgvector":
        return PgVectorStore(settings.pg_dsn, dim)
    if kind == "pinecone":
        return PineconeStore(dim)
    raise ValueError(f"Unknown VECTOR_STORE: {settings.vector_store}")
