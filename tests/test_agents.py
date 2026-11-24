import json

import pytest
from fastapi.testclient import TestClient

from hcai.agents.discharge import build_discharge_graph, start_draft, submit_signoff
from hcai.agents.prior_auth import (
    build_prior_auth_graph,
    normalise_drug,
    start_review,
    submit_review,
)
from hcai.agents.tools import add_business_days, default_repository
from hcai.config import PROJECT_ROOT
from hcai.eval.judge import judge_faithfulness
from hcai.llm.base import LLMProvider, LLMResponse


def _cases(name):
    return [json.loads(x) for x in (PROJECT_ROOT / "data/eval" / name).read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def pa_graph(chain):
    return build_prior_auth_graph(chain)


@pytest.mark.parametrize("text,expected", [("Ozempic", "semaglutide"), ("  semaglutide 1mg pen ", "semaglutide"),
                                           ("Jardiance 10mg", "empagliflozin"), ("unobtainium", "")])
def test_drug_normalisation(text, expected):
    assert normalise_drug(text) == expected


def test_business_days_skip_weekend():
    from datetime import date
    assert add_business_days(date(2026, 11, 20), 1) == date(2026, 11, 23)  # Fri -> Mon


def test_prior_auth_pauses_for_pharmacist_then_finalises(pa_graph):
    case = next(c for c in _cases("prior_auth_cases.jsonl") if c["expected_status"] == "pended")
    tid, st = start_review(pa_graph, case["request"])
    assert st["status"] == "pended" and sorted(st["missing"]) == case["expected_missing"]
    assert "notify_prescriber_by" in st["deadlines"] and st["draft_note"]
    assert "final_status" not in st  # nothing is final until a human acts
    out = submit_review(pa_graph, tid, {"action": "reject_triage", "reviewer": "test"})
    assert out["final_status"] == "returned_for_manual_triage"
    assert [a["step"] for a in out["audit"]][-2:] == ["human_review", "finalize"]


def test_unknown_drug_is_never_passed_as_complete(pa_graph):
    mrn = default_repository().all()[0].mrn
    _, st = start_review(pa_graph, {"mrn": mrn, "drug": "Unobtainium XR", "indication": "x",
                                    "prescriber_justification": "A long and detailed clinical justification."})
    assert st["status"] == "pended"


def test_discharge_reviewer_queries_clinician_and_blocks_unanswered_signoff(chain):
    graph = build_discharge_graph(chain.llm)
    truth = next(t for t in _cases("discharge_cases.jsonl") if "missing_follow_up" in t["expected_issues"])
    tid, st = start_draft(graph, truth["admission_id"])
    queries = [i for i in st["issues"] if not i["auto_fixable"]]
    assert any(q["type"] == "missing_follow_up" for q in queries)
    assert st["due_by"] == truth["expected_due_by"]
    out = submit_signoff(graph, tid, {"action": "sign_off", "signed_by": "Dr Test", "answers": {}})
    assert out["status"] == "returned"  # open queries -> cannot be signed off


def test_discharge_autofixes_icd10_from_record(chain):
    graph = build_discharge_graph(chain.llm)
    truth = next(t for t in _cases("discharge_cases.jsonl") if t["auto_fixable"])
    _, st = start_draft(graph, truth["admission_id"])
    assert "[CODE REQUIRED]" not in st["sections"]["Diagnoses"]
    assert any(a["step"] == "revise" and a["fixed"] for a in st["audit"])


class _FakeJudge(LLMProvider):
    name = "fake"

    def generate(self, system, prompt, model=None):
        return LLMResponse('Here: {"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}',
                           "fake", "fake")


def test_judge_parses_claims_and_scores():
    v = judge_faithfulness(_FakeJudge("fake"), "q", "Answer [1].", ["src"])
    assert v["total"] == 2 and v["faithfulness"] == 0.5
    assert judge_faithfulness(_FakeJudge("fake"), "q", "I don't have enough information.", [])["refusal"]


def test_agent_api_round_trip(monkeypatch, settings):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("INDEX_PATH", settings.index_path)
    from hcai.api.main import app

    case = next(c for c in _cases("prior_auth_cases.jsonl") if c["expected_status"] == "ready_for_pharmacist_review")
    adm = _cases("discharge_cases.jsonl")[0]["admission_id"]
    with TestClient(app) as client:
        r = client.post("/agents/prior-auth", json=case["request"]).json()
        assert r["awaiting"] == "pharmacist_review"
        done = client.post(f"/agents/prior-auth/{r['thread_id']}/review",
                           json={"action": "approve_note", "reviewer": "pharm"}).json()
        assert done["final_status"] == "ready_for_pharmacist_review"
        assert client.post(f"/agents/prior-auth/{r['thread_id']}/review",
                           json={"action": "approve_note", "reviewer": "pharm"}).status_code == 404

        d = client.post("/agents/discharge-summary", json={"admission_id": adm}).json()
        answers = {q["id"]: "Answered by consultant" for q in d["open_queries"]}
        s = client.post(f"/agents/discharge-summary/{d['thread_id']}/signoff",
                        json={"action": "sign_off", "signed_by": "Dr X", "answers": answers}).json()
        assert s["status"] == "signed_off"
        assert client.post("/agents/discharge-summary", json={"admission_id": "NOPE"}).status_code == 404


def test_dagster_definitions_load():
    pytest.importorskip("dagster")
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from orchestration.dagster_defs import defs

    assert {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()} >= {
        "source_documents", "rag_index", "rag_quality", "agent_quality"}
