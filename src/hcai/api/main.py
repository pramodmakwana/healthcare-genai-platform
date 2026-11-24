"""FastAPI service.

  GET  /health   liveness + index stats
  POST /ask      question -> grounded answer with citations, guardrail verdict, tokens, cost, latency
  POST /search   retrieval only (debugging / UI "show sources")
  POST /ingest   re-run ingestion (protect with ADMIN_TOKEN)
  GET  /metrics  in-process counters (requests, guardrail blocks, avg latency, total cost)

  Agents (human-in-the-loop: each run pauses for a person, then resumes on their decision)
  POST /agents/prior-auth                        start triage -> returns thread_id + draft for pharmacist
  POST /agents/prior-auth/{thread_id}/review     pharmacist decision -> final status + audit trail
  POST /agents/discharge-summary                 start drafting -> returns thread_id + draft + open queries
  POST /agents/discharge-summary/{thread_id}/signoff   consultant answers + sign-off -> final summary
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from hcai.agents.checkpoint import get_checkpointer
from hcai.agents.discharge import build_discharge_graph, render, start_draft, submit_signoff
from hcai.agents.prior_auth import build_prior_auth_graph, start_review, submit_review
from hcai.config import get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.ingestion import pipeline
from hcai.llm.factory import get_llm
from hcai.observability import configure_logging, emit_metrics, request_id_var
from hcai.rag.chain import RAGChain
from hcai.retrieval.retriever import Retriever
from hcai.vectorstores.stores import get_vector_store

configure_logging()
log = logging.getLogger("hcai.api")

STATE: dict = {}
_METRICS = {"requests": 0, "blocked": 0, "flagged": 0, "latency_ms_sum": 0.0, "cost_usd_sum": 0.0, "tokens_sum": 0}
_LOCK = threading.Lock()


def _build() -> None:
    settings = get_settings()
    embedder = get_embedder(settings)
    store = get_vector_store(settings, embedder.dim)
    if store.count() == 0:
        logging.getLogger("hcai.api").info("Empty index - running ingestion on startup")
        pipeline.run(settings)
        store = get_vector_store(settings, embedder.dim)
    retriever = Retriever(embedder, store, settings.retrieval_mode, patient_scoping=settings.patient_scoping)
    chain = RAGChain(get_llm(settings), retriever, settings)
    # Paused agent runs are checkpointed: in memory locally, in Postgres in production
    # (AGENT_CHECKPOINTER=postgres) so reviews survive restarts and work across replicas.
    if "checkpointer" not in STATE:
        STATE["checkpointer"] = get_checkpointer()
    cp = STATE["checkpointer"]
    STATE.update(settings=settings, store=store, retriever=retriever, chain=chain,
                 pa_graph=build_prior_auth_graph(chain, checkpointer=cp),
                 ds_graph=build_discharge_graph(chain.llm, checkpointer=cp))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _build()
    yield


app = FastAPI(title="Healthcare GenAI Platform", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Request ID on every log line + response header; one access log line per request."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000
        if request.url.path != "/health":
            log.info("%s %s %s %.0fms", request.method, request.url.path, status, ms)
            route = request.scope.get("route")
            emit_metrics(getattr(route, "path", request.url.path), {"Requests": 1, "Errors": int(status >= 500),
                                                                    "LatencyMs": ms}, status=status)
        request_id_var.reset(token)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class CitationOut(BaseModel):
    number: int
    doc_id: str
    title: str
    section: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    guardrail: str
    flags: list[str]
    groundedness: float | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: float


class SearchHit(BaseModel):
    rank: int
    score: float
    doc_id: str
    title: str
    section: str
    text: str


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Admin token required")


@app.get("/health")
def health() -> dict:
    s = STATE["settings"]
    return {"status": "ok", "chunks": STATE["store"].count(), "llm_provider": s.llm_provider,
            "vector_store": s.vector_store, "retrieval_mode": s.retrieval_mode}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    res = STATE["chain"].ask(req.question, req.top_k)
    with _LOCK:
        _METRICS["requests"] += 1
        _METRICS["blocked"] += res.guardrail != "ok"
        _METRICS["flagged"] += bool(res.flags)
        _METRICS["latency_ms_sum"] += res.latency_ms
        _METRICS["cost_usd_sum"] += res.cost_usd or 0.0
        _METRICS["tokens_sum"] += res.usage.total_tokens
    emit_metrics("/ask:rag", {
        "InputTokens": res.usage.input_tokens + res.usage.cache_read_tokens + res.usage.cache_write_tokens,
        "OutputTokens": res.usage.output_tokens, "CostUSD": res.cost_usd or 0.0, "RetrievalMs": res.retrieval_ms,
        "GuardrailBlocked": int(res.guardrail != "ok"), "Abstained": int("abstained_low_relevance" in res.flags),
        "LowGroundedness": int("low_groundedness" in res.flags),
    }, model=res.model, guardrail=res.guardrail)
    return AskResponse(
        answer=res.answer,
        citations=[CitationOut(number=c.number, doc_id=c.doc_id, title=c.title, section=c.section, snippet=c.snippet)
                   for c in res.citations],
        guardrail=res.guardrail, flags=res.flags, groundedness=res.groundedness, model=res.model,
        input_tokens=res.usage.input_tokens + res.usage.cache_read_tokens + res.usage.cache_write_tokens,
        output_tokens=res.usage.output_tokens, cost_usd=res.cost_usd, latency_ms=round(res.latency_ms, 1),
    )


@app.post("/search", response_model=list[SearchHit])
def search(req: AskRequest) -> list[SearchHit]:
    hits = STATE["retriever"].retrieve(req.question, req.top_k or STATE["settings"].top_k)
    return [SearchHit(rank=h.rank, score=round(h.score, 4), doc_id=h.chunk.doc_id, title=h.chunk.title,
                      section=h.chunk.section, text=h.chunk.text) for h in hits]


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest() -> dict:
    stats = pipeline.run(STATE["settings"])
    _build()
    return stats


@app.get("/metrics")
def metrics() -> dict:
    with _LOCK:
        n = max(_METRICS["requests"], 1)
        return {**_METRICS, "avg_latency_ms": round(_METRICS["latency_ms_sum"] / n, 1)}


# ---------------- Agents ----------------

class PriorAuthIn(BaseModel):
    mrn: str
    drug: str
    indication: str = ""
    urgent: bool = False
    prescriber_justification: str = ""
    received_at: str | None = None


class PharmacistReview(BaseModel):
    action: str = Field(..., pattern="^(approve_note|edit_note|reject_triage)$")
    reviewer: str
    edited_note: str | None = None


class DischargeIn(BaseModel):
    admission_id: str


class ConsultantSignoff(BaseModel):
    action: str = Field(..., pattern="^(sign_off|return)$")
    signed_by: str
    answers: dict[str, str] = {}


def _paused(graph, thread_id: str) -> bool:
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    return bool(snap and snap.next)


@app.post("/agents/prior-auth")
def prior_auth_start(req: PriorAuthIn) -> dict:
    body = req.model_dump(exclude_none=True)
    tid, st = start_review(STATE["pa_graph"], body)
    emit_metrics("agent:prior_auth", {"AgentRuns": 1}, triage_status=st.get("status", "unknown"))
    return {"thread_id": tid, "request_id": st["request"]["request_id"], "status": st.get("status"),
            "checklist": st.get("checklist", []), "missing": st.get("missing", []), "deadlines": st.get("deadlines", {}),
            "draft_note": st.get("draft_note"), "policy_citations": st.get("policy_citations", []),
            "awaiting": "pharmacist_review" if _paused(STATE["pa_graph"], tid) else None,
            "final_status": st.get("final_status")}


@app.post("/agents/prior-auth/{thread_id}/review")
def prior_auth_review(thread_id: str, review: PharmacistReview) -> dict:
    if not _paused(STATE["pa_graph"], thread_id):
        raise HTTPException(404, "No prior-auth run awaiting review for this thread_id")
    out = submit_review(STATE["pa_graph"], thread_id, review.model_dump(exclude_none=True))
    return {"final_status": out["final_status"], "note": out["draft_note"], "audit": out["audit"]}


@app.post("/agents/discharge-summary")
def discharge_start(req: DischargeIn) -> dict:
    tid, st = start_draft(STATE["ds_graph"], req.admission_id)
    if st.get("status") == "not_found":
        raise HTTPException(404, f"Admission {req.admission_id} not found")
    emit_metrics("agent:discharge", {"AgentRuns": 1}, open_queries=sum(not i["auto_fixable"] for i in st["issues"]))
    return {"thread_id": tid, "send_to_gp_by": st["due_by"], "draft": render(st),
            "open_queries": [i for i in st["issues"] if not i["auto_fixable"]],
            "auto_fixes": [a for a in st["audit"] if a["step"] == "revise"],
            "awaiting": "consultant_signoff"}


@app.post("/agents/discharge-summary/{thread_id}/signoff")
def discharge_signoff(thread_id: str, decision: ConsultantSignoff) -> dict:
    if not _paused(STATE["ds_graph"], thread_id):
        raise HTTPException(404, "No discharge summary awaiting sign-off for this thread_id")
    out = submit_signoff(STATE["ds_graph"], thread_id, decision.model_dump())
    return {"status": out["status"], "summary": render(out), "audit": out["audit"]}
