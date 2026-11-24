"""Production-path tests: Postgres agent checkpoints, pgvector store, logging and metrics.

The Postgres tests run when TEST_PG_DSN points at a database with pgvector (CI starts one
as a service container); otherwise they are skipped.
"""

import json
import logging
import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hcai.config import PROJECT_ROOT
from hcai.observability import JsonFormatter, emf_record, request_id_var

PG = os.environ.get("TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not PG, reason="set TEST_PG_DSN to run Postgres tests")


@needs_pg
def test_paused_review_survives_restart(chain):
    """Start a prior-auth run, throw the graph away (simulated restart/deploy), resume from Postgres."""
    from hcai.agents.checkpoint import get_checkpointer
    from hcai.agents.prior_auth import build_prior_auth_graph, start_review, submit_review

    case = json.loads((PROJECT_ROOT / "data/eval/prior_auth_cases.jsonl").read_text().splitlines()[0])
    g1 = build_prior_auth_graph(chain, checkpointer=get_checkpointer("postgres", PG))
    tid, st = start_review(g1, case["request"], thread_id=f"test-{uuid.uuid4().hex}")
    del g1
    g2 = build_prior_auth_graph(chain, checkpointer=get_checkpointer("postgres", PG))  # "new process"
    out = submit_review(g2, tid, {"action": "approve_note", "reviewer": "after-restart"})
    assert out["final_status"] == st["status"]
    assert out["audit"][-2]["reviewer"] == "after-restart"


@needs_pg
def test_pgvector_store_round_trip():
    from hcai.ingestion.documents import Chunk
    from hcai.vectorstores.stores import PgVectorStore

    store = PgVectorStore(PG, dim=8, table="rag_chunks_test")
    store.reset()
    chunks = [Chunk(f"c{i}", f"d{i}", "t", "s", f"text {i}", "guideline") for i in range(5)]
    vecs = np.eye(5, 8, dtype=np.float32)
    store.add(chunks, vecs)
    hits = store.search(vecs[3], 2)
    assert hits[0][0].chunk_id == "c3" and hits[0][1] == pytest.approx(1.0)
    assert store.count() == 5

    # Switching embedder dimension: detected, then rebuilt on re-ingestion (reset)
    wider = PgVectorStore(PG, dim=16, table="rag_chunks_test")
    assert wider.dim_mismatch and wider.count() == 0
    with pytest.raises(ValueError):
        wider.search(np.zeros(16, dtype=np.float32), 1)
    wider.reset()
    wider.add(chunks[:1], np.ones((1, 16), dtype=np.float32) / 4)
    assert wider.count() == 1
    wider.conn.execute("DROP TABLE rag_chunks_test")


def test_emf_record_is_valid_cloudwatch_format():
    token = request_id_var.set("abc123")
    rec = emf_record("/ask:rag", {"LatencyMs": 12.5, "InputTokens": 400})
    request_id_var.reset(token)
    meta = rec["_aws"]["CloudWatchMetrics"][0]
    assert meta["Dimensions"] == [["Service", "Endpoint"]]
    assert {m["Name"] for m in meta["Metrics"]} == {"LatencyMs", "InputTokens"}
    assert rec["LatencyMs"] == 12.5 and rec["Endpoint"] == "/ask:rag" and rec["request_id"] == "abc123"


def test_json_logs_carry_request_id():
    token = request_id_var.set("req-42")
    line = JsonFormatter().format(logging.LogRecord("hcai", logging.INFO, "", 0, "hello", None, None))
    request_id_var.reset(token)
    assert json.loads(line)["request_id"] == "req-42"


def test_request_id_header_and_emf_lines(monkeypatch, settings, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_PATH", settings.index_path)
    monkeypatch.setenv("LOG_FORMAT", "json")
    from hcai.api.main import app

    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "Which drug is first-line therapy for type 2 diabetes?"},
                        headers={"X-Request-ID": "trace-me"})
    assert r.headers["X-Request-ID"] == "trace-me"
    emf = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"_aws"')]
    endpoints = {e["Endpoint"] for e in emf if e["request_id"] == "trace-me"}
    assert {"/ask", "/ask:rag"} <= endpoints
