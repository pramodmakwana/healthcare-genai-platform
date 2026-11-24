"""Evaluation harness: compares RAG configurations on the same golden set.

Metrics
  Retrieval   hit@k (an expected doc in top-k), MRR, recall@k
  Answers     accuracy (expected facts present), citation rate, citation precision,
              mean groundedness, ungrounded-answer rate (hallucination proxy),
              correct-refusal rate on unanswerable questions
  Safety      guardrail accuracy on adversarial / emergency / personal-advice prompts
  Efficiency  tokens per query, p50/p95 latency, cost per 1,000 queries

With LLM_PROVIDER=mock, token counts are estimates and cost is *projected* at the Claude
prices in config/pricing.yaml. With LLM_PROVIDER=anthropic, all numbers are measured.

Usage:  python -m hcai.eval.run_eval                 # baseline vs improved, writes reports/
        python -m hcai.eval.run_eval --limit 20      # quick run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from hcai.config import PROJECT_ROOT, Settings, get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.eval.judge import judge_faithfulness
from hcai.guardrails.guards import check_input
from hcai.ingestion import pipeline
from hcai.llm.base import estimate_cost_usd
from hcai.llm.factory import get_llm
from hcai.rag.chain import RAGChain
from hcai.retrieval.retriever import Retriever
from hcai.vectorstores.stores import get_vector_store

EXPERIMENTS = {
    # Naive RAG: fixed-size chunks, vector search only, generous context, one model, no abstention.
    "baseline": dict(chunking="fixed", retrieval_mode="dense", top_k=8, llm_routing=False, abstain_min_score=0.0,
                     patient_scoping=False),
    "section_chunking": dict(chunking="section", retrieval_mode="dense", top_k=8, llm_routing=False, abstain_min_score=0.0,
                             patient_scoping=False),
    "hybrid_retrieval": dict(chunking="section", retrieval_mode="hybrid", top_k=5, llm_routing=False, abstain_min_score=0.0,
                             patient_scoping=False),
    # Full stack: + patient-scoped retrieval + relevance-gated abstention + fast-model routing.
    "improved": dict(chunking="section", retrieval_mode="hybrid", top_k=5, llm_routing=True,
                     patient_scoping=True),
}


@dataclass
class ExperimentResult:
    name: str
    config: dict
    n_questions: int
    hit_at_k: float
    mrr: float
    recall_at_k: float
    evidence_recall: float
    answer_accuracy: float
    citation_rate: float
    citation_precision: float
    mean_groundedness: float
    ungrounded_rate: float
    correct_refusal_rate: float
    avg_input_tokens: float
    avg_output_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_queries_usd: float
    cost_is_projected: bool
    judge_faithfulness: float | None = None
    judge_hallucination_rate: float | None = None


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def run_experiment(name: str, overrides: dict, rows: list[dict], base: Settings,
                   judge=None) -> tuple[ExperimentResult, list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings(**{**{f: getattr(base, f) for f in base.__dataclass_fields__},
                                   **overrides, "vector_store": "memory", "index_path": tmp})
        pipeline.run(settings)
        embedder = get_embedder(settings)
        store = get_vector_store(settings, embedder.dim)
        retriever = Retriever(embedder, store, settings.retrieval_mode, patient_scoping=settings.patient_scoping)
        chain = RAGChain(get_llm(settings), retriever, settings)
        projected = settings.llm_provider == "mock"

        details, hits, rr, recalls, evidence, acc, cite, cite_prec, grounded, ungrounded, refusals = ([] for _ in range(11))
        in_tok, out_tok, lat, costs, judged = [], [], [], [], []
        for row in rows:
            res = chain.ask(row["question"])
            expected = set(row["expected_doc_ids"])
            ranked_docs = list(dict.fromkeys(h.chunk.doc_id for h in res.retrieved))
            answer_l = res.answer.lower()

            if expected:
                first = next((i + 1 for i, d in enumerate(ranked_docs) if d in expected), None)
                hits.append(1.0 if first else 0.0)
                rr.append(1.0 / first if first else 0.0)
                recalls.append(len(expected & set(ranked_docs)) / len(expected))
                facts = row.get("expected_answer_contains", [])
                # Chunk-level check: did retrieval surface text that actually contains the answer?
                context = " ".join(h.chunk.text for h in res.retrieved if h.chunk.doc_id in expected).lower()
                evidence.append(1.0 if all(f.lower() in context for f in facts) else 0.0)
                acc.append(1.0 if all(f.lower() in answer_l for f in facts) else 0.0)
                cite.append(1.0 if res.citations else 0.0)
                if res.citations:
                    cite_prec.append(sum(c.doc_id in expected for c in res.citations) / len(res.citations))
                g = res.groundedness or 0.0
                grounded.append(g)
                ungrounded.append(1.0 if (g < settings.min_groundedness or "invalid_citations" in res.flags) else 0.0)
            else:
                refusals.append(1.0 if "don't have enough information" in answer_l else 0.0)

            verdict = None
            if judge is not None and expected and res.guardrail == "ok":
                verdict = judge_faithfulness(judge, row["question"], res.answer, [h.chunk.text for h in res.retrieved])
                if not verdict["refusal"]:
                    judged.append(verdict["faithfulness"])

            model_for_price = res.model if not projected else (chain.pick_model(row["question"]) if settings.llm_routing else settings.llm_model)
            cost = estimate_cost_usd(model_for_price, res.usage, settings.pricing_file) or 0.0
            in_tok.append(res.usage.input_tokens + res.usage.cache_read_tokens + res.usage.cache_write_tokens)
            out_tok.append(res.usage.output_tokens)
            lat.append(res.latency_ms)
            costs.append(cost)
            details.append({"experiment": name, "id": row["id"], "category": row["category"], "question": row["question"],
                            "answer": res.answer, "retrieved_docs": ranked_docs[:5], "expected_docs": sorted(expected),
                            "groundedness": res.groundedness, "flags": res.flags, "model": model_for_price,
                            "latency_ms": round(res.latency_ms, 1),
                            "judge": verdict})

    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else 0.0  # noqa: E731
    result = ExperimentResult(
        name=name, config=overrides, n_questions=len(rows),
        hit_at_k=mean(hits), mrr=mean(rr), recall_at_k=mean(recalls), evidence_recall=mean(evidence),
        answer_accuracy=mean(acc), citation_rate=mean(cite), citation_precision=mean(cite_prec),
        mean_groundedness=mean(grounded), ungrounded_rate=mean(ungrounded), correct_refusal_rate=mean(refusals),
        avg_input_tokens=round(statistics.fmean(in_tok), 1), avg_output_tokens=round(statistics.fmean(out_tok), 1),
        latency_p50_ms=_pct(lat, 0.5), latency_p95_ms=_pct(lat, 0.95),
        cost_per_1k_queries_usd=round(sum(costs) / len(costs) * 1000, 4), cost_is_projected=projected,
        judge_faithfulness=mean(judged) if judged else None,
        judge_hallucination_rate=round(1 - mean(judged), 4) if judged else None,
    )
    return result, details


def run_safety(path: Path) -> dict:
    rows = _load_jsonl(path)
    got = [(r, check_input(r["question"]).category) for r in rows]
    harmful = [(r, c) for r, c in got if r["expect"] != "ok"]
    benign = [(r, c) for r, c in got if r["expect"] == "ok"]
    return {
        "n": len(rows),
        "accuracy": round(sum(r["expect"] == c for r, c in got) / len(rows), 4) if rows else 0.0,
        "harmful_caught": round(sum(c != "ok" for _, c in harmful) / max(len(harmful), 1), 4),
        "benign_false_positive_rate": round(sum(c != "ok" for _, c in benign) / max(len(benign), 1), 4),
        "misses": [{"question": r["question"], "expected": r["expect"], "got": c} for r, c in got if r["expect"] != c],
    }


def _change(before, after, lower_is_better: bool = False) -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or before == 0:
        return "n/a"
    delta = (after - before) / before * 100
    better = delta < 0 if lower_is_better else delta > 0
    return f"{delta:+.0f}% ({'better' if better else 'worse' if delta else 'same'})"


def write_report(results: list[ExperimentResult], safety: dict, settings: Settings, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, best = results[0], results[-1]
    rows = [
        ("Retrieval hit@k", "hit_at_k", False), ("Retrieval MRR", "mrr", False), ("Recall@k", "recall_at_k", False),
        ("Evidence recall (answer text retrieved)", "evidence_recall", False),
        ("Answer accuracy", "answer_accuracy", False), ("Citation rate", "citation_rate", False),
        ("Citation precision", "citation_precision", False), ("Mean groundedness", "mean_groundedness", False),
        ("Ungrounded-answer rate", "ungrounded_rate", True), ("Correct refusals (unanswerable)", "correct_refusal_rate", False),
        ("Avg input tokens / query", "avg_input_tokens", True), ("Latency p50 (ms)", "latency_p50_ms", True),
        ("Latency p95 (ms)", "latency_p95_ms", True),
        ("Judge faithfulness (Claude)", "judge_faithfulness", False),
        ("Judge hallucination rate", "judge_hallucination_rate", True), ("Cost per 1k queries (USD)", "cost_per_1k_queries_usd", True),
    ]
    header = "| Metric | " + " | ".join(r.name for r in results) + f" | Change ({base.name} → {best.name}) |"
    lines = [
        "# Evaluation report", "",
        f"- LLM provider: `{settings.llm_provider}`  |  Embeddings: `{settings.embedding_provider}`  |  Questions: {base.n_questions}",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M')}",
    ]
    if base.cost_is_projected:
        lines.append("- **Mock LLM run.** Token counts are estimates and cost is projected at Claude prices "
                     "(config/pricing.yaml). Answer-quality metrics reflect retrieval + the extractive mock, "
                     "not a real model. Re-run with `LLM_PROVIDER=anthropic` for measured numbers.")
    lines += ["", "## Configurations", ""]
    for r in results:
        lines.append(f"- **{r.name}**: `{json.dumps(r.config)}`")
    lines += ["", "## Results", "", header, "|" + "---|" * (len(results) + 2)]
    for label, key, lower in rows:
        vals = [getattr(r, key) for r in results]
        if all(v is None for v in vals):
            continue
        vals = ["-" if v is None else v for v in vals]
        lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v}" for v in vals)
                     + f" | {_change(getattr(base, key), getattr(best, key), lower)} |")
    lines += ["", "## Guardrails", "",
              f"{safety['n']} prompts (injection, emergency, personal-advice, plus benign controls):", "",
              f"- Category accuracy: **{safety['accuracy']:.0%}**",
              f"- Harmful prompts caught: **{safety['harmful_caught']:.0%}**",
              f"- Benign prompts wrongly blocked: **{safety['benign_false_positive_rate']:.0%}**", ""]
    for m in safety["misses"]:
        lines.append(f"  - miss: \"{m['question']}\" expected `{m['expected']}`, got `{m['got']}`")
    path = out_dir / "eval_report.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--judge", action="store_true",
                    help="Score faithfulness with Claude as judge (needs ANTHROPIC_API_KEY; JUDGE_MODEL, default claude-sonnet-5-5)")
    args = ap.parse_args()

    settings = get_settings()
    rows = _load_jsonl(PROJECT_ROOT / "data" / "eval" / "golden.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    results, details = [], []
    judge = None
    if args.judge:
        from hcai.llm.providers import AnthropicProvider

        os.environ.setdefault("JUDGE_MODEL", "claude-sonnet-5-5")
        judge = AnthropicProvider(model=os.environ["JUDGE_MODEL"], max_tokens=1200, temperature=0.0)
    for name in args.experiments.split(","):
        res, det = run_experiment(name, EXPERIMENTS[name], rows, settings, judge=judge)
        results.append(res)
        details.extend(det)
        print(f"{name:18s} hit@k={res.hit_at_k:.3f} evid={res.evidence_recall:.3f} mrr={res.mrr:.3f} acc={res.answer_accuracy:.3f} "
              f"grounded={res.mean_groundedness:.3f} tokens_in={res.avg_input_tokens:.0f} cost/1k=${res.cost_per_1k_queries_usd:.3f}")

    safety = run_safety(PROJECT_ROOT / "data" / "eval" / "safety.jsonl")
    out = PROJECT_ROOT / "reports"
    report = write_report(results, safety, settings, out)
    (out / "eval_results.json").write_text(json.dumps({"experiments": [asdict(r) for r in results], "safety": safety}, indent=2))
    with (out / "eval_details.jsonl").open("w") as fh:
        for d in details:
            fh.write(json.dumps(d) + "\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
