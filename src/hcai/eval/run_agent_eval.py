"""Evaluate the two agents against their synthetic case sets.

Prior authorization (data/eval/prior_auth_cases.jsonl)
  - triage status accuracy, exact match on the list of missing documents
  - checklist items verified automatically per request (work a human no longer gathers by hand)
Discharge summaries (data/eval/discharge_cases.jsonl)
  - reviewer issue detection: precision / recall against planted gaps
  - auto-fix rate for issues provable from the record, send-by deadline accuracy
  - completeness: every SOP section present after sign-off

Human steps are simulated (pharmacist approves the note; consultant answers every query),
so the numbers measure the agents, not the reviewers.

Usage: python -m hcai.eval.run_agent_eval
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections import Counter

from hcai.agents.discharge import (
    REQUIRED_SECTIONS,
    build_discharge_graph,
    start_draft,
    submit_signoff,
)
from hcai.agents.prior_auth import build_prior_auth_graph, start_review, submit_review
from hcai.config import PROJECT_ROOT, get_settings
from hcai.rag.factory import build_rag_chain

EVAL = PROJECT_ROOT / "data" / "eval"
PLANTED = {"med_change_without_reason", "missing_follow_up", "missing_allergy_status", "diagnosis_missing_icd10"}


def _jsonl(name: str) -> list[dict]:
    return [json.loads(x) for x in (EVAL / name).read_text().splitlines() if x.strip()]


def eval_prior_auth(rag) -> dict:
    graph = build_prior_auth_graph(rag)
    rows, status_ok, missing_ok, auto_items, lat = [], [], [], [], []
    for case in _jsonl("prior_auth_cases.jsonl"):
        t0 = time.perf_counter()
        tid, st = start_review(graph, case["request"])
        if st.get("status") != "invalid_request":
            st = submit_review(graph, tid, {"action": "approve_note", "reviewer": "eval.pharmacist"})
        lat.append((time.perf_counter() - t0) * 1000)
        got_status = st.get("final_status") or st.get("status")
        got_missing = sorted(m for m in st.get("missing", []) if m != "valid patient MRN")
        status_ok.append(got_status == case["expected_status"])
        missing_ok.append(got_missing == case["expected_missing"])
        auto_items.append(len(st.get("checklist", [])))
        if not (status_ok[-1] and missing_ok[-1]):
            rows.append({"id": case["id"], "scenario": case["scenario"], "expected": case["expected_status"],
                         "got": got_status, "expected_missing": case["expected_missing"], "got_missing": got_missing})
    return {
        "cases": len(status_ok),
        "status_accuracy": round(sum(status_ok) / len(status_ok), 4),
        "missing_items_exact_match": round(sum(missing_ok) / len(missing_ok), 4),
        "avg_checks_automated_per_request": round(statistics.fmean(auto_items), 2),
        "latency_p50_ms": round(statistics.median(lat), 1),
        "failures": rows,
    }


def eval_discharge(llm) -> dict:
    graph = build_discharge_graph(llm)
    tp = fp = fn = 0
    per_type: Counter = Counter()
    auto_fixed = auto_expected = due_ok = complete = 0
    failures = []
    truth = _jsonl("discharge_cases.jsonl")
    for case in truth:
        tid, st = start_draft(graph, case["admission_id"])
        fixed_types = {"diagnosis_missing_icd10"} if any(
            "ICD-10" in f for a in st["audit"] if a["step"] == "revise" for f in a.get("fixed", [])) else set()
        found = {i["type"] for i in st["issues"]} | fixed_types
        found_planted = found & PLANTED
        expected = set(case["expected_issues"])
        tp += len(found_planted & expected)
        fp += len(found_planted - expected)
        fn += len(expected - found_planted)
        per_type.update(f"caught:{t}" for t in found_planted & expected)
        per_type.update(f"missed:{t}" for t in expected - found_planted)
        if found - PLANTED:
            per_type.update(f"other:{t}" for t in found - PLANTED)
        auto_expected += len(case["auto_fixable"])
        auto_fixed += len(set(case["auto_fixable"]) & fixed_types)
        due_ok += st["due_by"] == case["expected_due_by"]
        answers = {i["id"]: "Consultant answer (eval)" for i in st["issues"] if not i["auto_fixable"]}
        out = submit_signoff(graph, tid, {"action": "sign_off", "signed_by": "eval.consultant", "answers": answers})
        complete += out["status"] == "signed_off" and all(
            out["sections"].get(s) and "REQUIRED]" not in out["sections"][s] for s in REQUIRED_SECTIONS)
        if found_planted != expected:
            failures.append({"admission_id": case["admission_id"], "expected": sorted(expected), "found": sorted(found)})
    n = len(truth)
    return {
        "admissions": n,
        "issue_recall": round(tp / max(tp + fn, 1), 4),
        "issue_precision": round(tp / max(tp + fp, 1), 4),
        "auto_fix_rate": round(auto_fixed / max(auto_expected, 1), 4),
        "send_by_deadline_accuracy": round(due_ok / n, 4),
        "complete_after_signoff": round(complete / n, 4),
        "by_type": dict(sorted(per_type.items())),
        "failures": failures,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings(vector_store="memory", index_path=tmp)
        rag = build_rag_chain(settings)
        pa = eval_prior_auth(rag)
        ds = eval_discharge(rag.llm)
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "agent_eval_results.json").write_text(json.dumps({"prior_auth": pa, "discharge": ds}, indent=2))
    lines = [
        "# Agent evaluation report", "",
        f"- LLM provider: `{settings.llm_provider}`. Human review steps are simulated.",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M')}", "",
        "## Prior-authorization triage agent", "",
        "| Metric | Value |", "|---|---|",
        f"| Requests | {pa['cases']} |",
        f"| Triage status accuracy | {pa['status_accuracy']:.0%} |",
        f"| Missing-document list exact match | {pa['missing_items_exact_match']:.0%} |",
        f"| Checks completed automatically per request | {pa['avg_checks_automated_per_request']} |",
        f"| Latency p50 (to human-review pause) | {pa['latency_p50_ms']} ms |", "",
        "## Discharge-summary agent (drafter + reviewer)", "",
        "| Metric | Value |", "|---|---|",
        f"| Admissions | {ds['admissions']} |",
        f"| Planted documentation gaps caught (recall) | {ds['issue_recall']:.0%} |",
        f"| Reviewer precision on planted gap types | {ds['issue_precision']:.0%} |",
        f"| Gaps auto-fixed from the FHIR record | {ds['auto_fix_rate']:.0%} |",
        f"| Send-to-GP deadline correct | {ds['send_by_deadline_accuracy']:.0%} |",
        f"| Complete after consultant sign-off | {ds['complete_after_signoff']:.0%} |", "",
    ]
    if pa["failures"] or ds["failures"]:
        lines += ["## Failures", "", "```json", json.dumps({"prior_auth": pa["failures"], "discharge": ds["failures"]}, indent=1), "```"]
    (out / "agent_eval_report.md").write_text("\n".join(lines))
    print(json.dumps({"prior_auth": {k: v for k, v in pa.items() if k != "failures"},
                      "discharge": {k: v for k, v in ds.items() if k != "failures"}}, indent=1))


if __name__ == "__main__":
    main()
