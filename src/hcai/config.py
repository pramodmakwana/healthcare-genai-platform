"""Central configuration, read from environment variables (and a .env file if present).

Every setting has a default that works offline, so the platform runs end to end
with no API keys: mock LLM, hashing embeddings, in-memory vector store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# HCAI_HOME lets an installed (non-editable) package, e.g. inside Docker, find config/ and data/.
PROJECT_ROOT = Path(os.environ.get("HCAI_HOME", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5-5"))
    llm_fast_model: str = field(default_factory=lambda: _env("LLM_FAST_MODEL", "claude-haiku-5-5"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "700")))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.0")))
    # Route short, simple questions to the cheaper model (cost optimisation).
    llm_routing: bool = field(default_factory=lambda: _env("LLM_ROUTING", "true").lower() == "true")

    # Embeddings
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "hashing"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "768")))

    # Vector store
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "memory"))
    # In AWS the task gets PGHOST/PGUSER/PGPASSWORD/PGDATABASE/PGSSLMODE (password injected from
    # Secrets Manager); an empty DSN tells libpq to use those variables.
    pg_dsn: str = field(
        default_factory=lambda: _env("PG_DSN", "" if os.environ.get("PGHOST") else "postgresql://hcai:hcai@localhost:5433/hcai")
    )
    chroma_path: str = field(default_factory=lambda: _env("CHROMA_PATH", str(PROJECT_ROOT / ".chroma")))
    index_path: str = field(default_factory=lambda: _env("INDEX_PATH", str(PROJECT_ROOT / ".index")))

    # Retrieval
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))  # dense|bm25|hybrid
    # If a question names one known patient (MRN or full name), retrieve only from that patient's
    # record plus non-patient guidance. Improves precision and enforces "minimum necessary".
    patient_scoping: bool = field(default_factory=lambda: _env("PATIENT_SCOPING", "true").lower() == "true")
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "120")))  # words
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "20")))
    chunking: str = field(default_factory=lambda: _env("CHUNKING", "section"))  # fixed|section

    # Guardrails
    redact_phi_in_logs: bool = field(
        default_factory=lambda: _env("REDACT_PHI_IN_LOGS", "true").lower() == "true"
    )
    # Abstain (no LLM call) when the best dense similarity is below this. Calibrate per embedder:
    # run `python -m hcai.eval.calibrate` after changing EMBEDDING_PROVIDER. 0 disables.
    abstain_min_score: float = field(default_factory=lambda: float(_env("ABSTAIN_MIN_SCORE", "0.27")))
    min_groundedness: float = field(default_factory=lambda: float(_env("MIN_GROUNDEDNESS", "0.5")))

    # Data
    raw_data_dir: str = field(default_factory=lambda: _env("RAW_DATA_DIR", str(PROJECT_ROOT / "data" / "raw")))
    pricing_file: str = field(
        default_factory=lambda: _env("PRICING_FILE", str(PROJECT_ROOT / "config" / "pricing.yaml"))
    )


def get_settings(**overrides) -> Settings:
    """Return settings, optionally overriding individual fields (handy in tests and evals)."""
    base = Settings()
    if not overrides:
        return base
    values = {f: getattr(base, f) for f in base.__dataclass_fields__}
    values.update(overrides)
    return Settings(**values)
