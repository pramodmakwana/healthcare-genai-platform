"""Prior-authorization triage agent (LangGraph).

    intake ──► lookup_patient ──► classify_drug ──┬─► no_pa_required ─────────────┐
                                                   └─► check_documentation         │
                                                          │                        │
                                                    decide_route (rules + SLAs)    │
                                                          │                        │
                                                    cite_policy (RAG tool)  ◄──────┘
                                                          │
                                                    draft_note (LLM)
                                                          │
                                                    human_review  ◄── interrupt: pharmacist approves / edits / rejects
                                                          │
                                                       finalize (audit record)

Design rule: the coverage *decision* is never made by the model. Rules decide what is
required, what is missing and the deadlines; the model drafts the note; a clinical
pharmacist makes the call. Every step is recorded in `audit` for traceability.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hcai.agents.tools import (
    PatientRepository,
    add_business_days,
    default_repository,
    months_between,
    parse_dt,
)
from hcai.llm.base import LLMProvider
from hcai.rag.chain import RAGChain

# Drug -> class. In production this comes from the formulary service.
DRUG_CLASSES = {
    "semaglutide": "glp1", "liraglutide": "glp1", "dulaglutide": "glp1", "tirzepatide": "glp1",
    "empagliflozin": "sglt2", "dapagliflozin": "sglt2",
    "adalimumab": "biologic", "etanercept": "biologic",
    "metformin": "standard", "lisinopril": "standard", "atorvastatin": "standard", "salbutamol": "standard",
    "amlodipine": "standard",
}
# Brand -> generic. In production: a drug-terminology service (e.g. RxNorm / dm+d lookups).
BRANDS = {"ozempic": "semaglutide", "wegovy": "semaglutide", "rybelsus": "semaglutide", "victoza": "liraglutide",
          "saxenda": "liraglutide", "trulicity": "dulaglutide", "mounjaro": "tirzepatide", "zepbound": "tirzepatide",
          "jardiance": "empagliflozin", "forxiga": "dapagliflozin", "farxiga": "dapagliflozin", "humira": "adalimumab",
          "enbrel": "etanercept", "glucophage": "metformin", "lipitor": "atorvastatin", "ventolin": "salbutamol"}


def normalise_drug(text: str) -> str:
    """Map free text ("  Semaglutide 1mg pen ", "Ozempic") to a generic name; '' if unrecognised."""
    for token in re.findall(r"[a-z]+", (text or "").lower()):
        if token in DRUG_CLASSES:
            return token
        if token in BRANDS:
            return BRANDS[token]
    return ""


CLASS_LABEL = {"glp1": "GLP-1 receptor agonist", "sglt2": "SGLT2 inhibitor", "biologic": "biologic therapy"}
DIABETES_MEDS = ("metformin", "insulin", "empagliflozin", "dapagliflozin", "gliclazide")


class PARequest(TypedDict, total=False):
    request_id: str
    mrn: str
    drug: str
    indication: str           # e.g. "type 2 diabetes", "weight management"
    urgent: bool
    prescriber_justification: str
    received_at: str          # ISO datetime


class PAState(TypedDict, total=False):
    request: PARequest
    patient: dict | None
    drug_class: str
    requires_pa: bool
    checklist: list[dict]     # {item, satisfied, evidence}
    missing: list[str]
    status: str               # no_pa_required | pended | ready_for_pharmacist_review | invalid_request
    deadlines: dict
    policy_answer: str
    policy_citations: list[dict]
    draft_note: str
    review: dict
    final_status: str
    audit: list[dict]


def _log(state: PAState, step: str, **detail: Any) -> list[dict]:
    return [*state.get("audit", []), {"step": step, "at": datetime.now().isoformat(timespec="seconds"), **detail}]


def build_prior_auth_graph(rag: RAGChain, llm: LLMProvider | None = None,
                           repo: PatientRepository | None = None, checkpointer=None):
    repo = repo or default_repository()
    llm = llm or rag.llm

    def intake(state: PAState) -> dict:
        req = dict(state["request"])
        req.setdefault("request_id", f"PA-{uuid.uuid4().hex[:8]}")
        req.setdefault("received_at", datetime.now().isoformat(timespec="minutes"))
        req.setdefault("urgent", False)
        req["drug_as_written"] = req.get("drug", "")
        req["drug"] = normalise_drug(req["drug_as_written"])
        return {"request": req, "audit": _log(state, "intake", request_id=req["request_id"],
                                               drug_as_written=req["drug_as_written"], drug=req["drug"] or "UNRECOGNISED")}

    def lookup_patient(state: PAState) -> dict:
        p = repo.get(state["request"].get("mrn", ""))
        if p is None:
            return {"patient": None, "status": "invalid_request", "missing": ["valid patient MRN"],
                    "audit": _log(state, "lookup_patient", found=False)}
        patient = {"mrn": p.mrn, "name": p.name, "doc_id": p.doc_id,
                   "labs": {k: {"value": v.value, "unit": v.unit, "on": v.on.isoformat()} for k, v in p.labs.items()},
                   "medications": [m["text"] for m in p.medications],
                   "conditions": [c["text"] for c in p.conditions]}
        return {"patient": patient, "audit": _log(state, "lookup_patient", found=True)}

    def classify_drug(state: PAState) -> dict:
        req = state["request"]
        cls = DRUG_CLASSES.get(req["drug"], "unknown")
        weight_only = "weight" in req.get("indication", "").lower() and "diabet" not in req.get("indication", "").lower()
        requires = cls in ("glp1", "biologic", "unknown") or (cls == "sglt2" and weight_only)
        return {"drug_class": cls, "requires_pa": requires,
                "audit": _log(state, "classify_drug", drug_class=cls, requires_pa=requires)}

    def check_documentation(state: PAState) -> dict:
        req, p, cls = state["request"], state["patient"], state["drug_class"]
        received = parse_dt(req["received_at"]).date()
        labs = p["labs"]
        items: list[dict] = []

        def add(item: str, ok: bool, evidence: str) -> None:
            items.append({"item": item, "satisfied": ok, "evidence": evidence})

        justification = (req.get("prescriber_justification") or "").strip()
        add("prescriber clinical justification", len(justification) >= 20,
            justification[:80] if justification else "not provided")
        if cls == "unknown":
            # Fail safe: an unrecognised drug can never be passed as complete.
            add("drug classification by pharmacist (not on formulary list)", False,
                f"'{req.get('drug_as_written', '')}' not recognised")
        if cls == "glp1":
            h = labs.get("hba1c")
            if h:
                age = months_between(datetime.fromisoformat(h["on"]).date(), received)
                add("HbA1c within the last 3 months", age <= 3, f"HbA1c {h['value']}% on {h['on']} ({age:.1f} months old)")
            else:
                add("HbA1c within the last 3 months", False, "no HbA1c on record")
            b = labs.get("bmi")
            add("current BMI", bool(b), f"BMI {b['value']} on {b['on']}" if b else "no BMI on record")
            tried = [m for m in p["medications"] if any(d in m.lower() for d in DIABETES_MEDS)]
            add("diabetes medicines already tried", bool(tried), "; ".join(tried) if tried else "none on record")
        if cls == "sglt2":
            e = labs.get("egfr")
            add("latest eGFR", bool(e), f"eGFR {e['value']} on {e['on']}" if e else "no eGFR on record")
        missing = [i["item"] for i in items if not i["satisfied"]]
        return {"checklist": items, "missing": missing,
                "audit": _log(state, "check_documentation", missing=missing)}

    def decide_route(state: PAState) -> dict:
        req = state["request"]
        received = parse_dt(req["received_at"])
        if state["missing"]:
            status = "pended"
            deadlines = {"notify_prescriber_by": add_business_days(received.date(), 1).isoformat(),
                         "close_as_withdrawn_if_incomplete_by": (received.date() + timedelta(days=14)).isoformat()}
        else:
            status = "ready_for_pharmacist_review"
            due = received + timedelta(hours=24) if req.get("urgent") else parse_dt(add_business_days(received.date(), 5))
            deadlines = {"decision_due_by": due.isoformat(timespec="minutes"),
                         "sla": "24 hours (urgent)" if req.get("urgent") else "5 business days (standard)"}
        return {"status": status, "deadlines": deadlines, "audit": _log(state, "decide_route", status=status)}

    def no_pa_required(state: PAState) -> dict:
        return {"status": "no_pa_required", "missing": [], "deadlines": {},
                "audit": _log(state, "no_pa_required")}

    def cite_policy(state: PAState) -> dict:
        cls = state.get("drug_class", "unknown")
        if state.get("status") == "no_pa_required":
            q = "Which medicines do not require prior authorization?"
        elif state.get("status") == "pended":
            q = "What happens if a prior authorization request is missing documentation?"
        else:
            q = f"What documentation must be attached to a {CLASS_LABEL.get(cls, 'high-cost medicine')} prior authorization request?"
        ans = rag.ask(q)
        cites = [{"n": c.number, "doc_id": c.doc_id, "section": c.section} for c in ans.citations]
        return {"policy_answer": ans.answer, "policy_citations": cites,
                "audit": _log(state, "cite_policy", question=q, citations=len(cites))}

    def draft_note(state: PAState) -> dict:
        req, p = state["request"], state.get("patient") or {}
        facts = [f"Request {req['request_id']}: {req.get('drug')} for {req.get('indication', 'unspecified indication')}"
                 f" ({'urgent' if req.get('urgent') else 'standard'}).",
                 f"Triage status: {state['status']}."]
        for i in state.get("checklist", []):
            facts.append(f"{'OK' if i['satisfied'] else 'MISSING'}: {i['item']} ({i['evidence']}).")
        for k, v in state.get("deadlines", {}).items():
            facts.append(f"{k.replace('_', ' ')}: {v}.")
        facts.append(f"Policy basis: {state.get('policy_answer', '')}")
        header = f"To the prescriber of {p.get('name', 'the patient')} ({p.get('mrn', req.get('mrn'))})"
        if llm.name == "mock":
            body = "\n".join(facts)
        else:
            resp = llm.generate(
                "You draft concise, professional notes from a pharmacy prior-authorization team to prescribers. "
                "Use only the facts given. Keep every date, deadline and missing item exactly. Keep policy citations "
                "like [1]. Do not approve or deny the request; say it is pending pharmacist review or pended. "
                "Max 150 words.",
                "FACTS:\n" + "\n".join(facts) + "\n\nWrite the note.",
            )
            body = resp.text
        return {"draft_note": f"{header}\n\n{body}", "audit": _log(state, "draft_note", model=llm.name)}

    def human_review(state: PAState) -> dict:
        decision = interrupt({
            "task": "Clinical pharmacist review of prior-authorization triage",
            "request_id": state["request"]["request_id"],
            "status": state["status"],
            "missing": state.get("missing", []),
            "draft_note": state["draft_note"],
            "options": ["approve_note", "edit_note", "reject_triage"],
        })
        return {"review": decision, "audit": _log(state, "human_review", reviewer=decision.get("reviewer"),
                                                  action=decision.get("action"))}

    def finalize(state: PAState) -> dict:
        r = state.get("review", {})
        action = r.get("action")
        if action == "reject_triage":
            final = "returned_for_manual_triage"
        elif action in ("approve_note", "edit_note"):
            final = state["status"]
        else:
            final = state.get("status", "invalid_request")
        note = r.get("edited_note") or state.get("draft_note", "")
        return {"final_status": final, "draft_note": note, "audit": _log(state, "finalize", final_status=final)}

    def after_lookup(state: PAState) -> str:
        return "finalize" if state.get("status") == "invalid_request" else "classify_drug"

    def after_classify(state: PAState) -> str:
        return "check_documentation" if state["requires_pa"] else "no_pa_required"

    g = StateGraph(PAState)
    for name, fn in [("intake", intake), ("lookup_patient", lookup_patient), ("classify_drug", classify_drug),
                     ("check_documentation", check_documentation), ("decide_route", decide_route),
                     ("no_pa_required", no_pa_required), ("cite_policy", cite_policy), ("draft_note", draft_note),
                     ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "lookup_patient")
    g.add_conditional_edges("lookup_patient", after_lookup, ["classify_drug", "finalize"])
    g.add_conditional_edges("classify_drug", after_classify, ["check_documentation", "no_pa_required"])
    g.add_edge("check_documentation", "decide_route")
    g.add_edge("decide_route", "cite_policy")
    g.add_edge("no_pa_required", "cite_policy")
    g.add_edge("cite_policy", "draft_note")
    g.add_edge("draft_note", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def start_review(graph, request: PARequest, thread_id: str | None = None) -> tuple[str, dict]:
    """Run until the human-review pause. Returns (thread_id, state)."""
    thread_id = thread_id or uuid.uuid4().hex
    graph.invoke({"request": request, "audit": []}, {"configurable": {"thread_id": thread_id}})
    return thread_id, graph.get_state({"configurable": {"thread_id": thread_id}}).values


def submit_review(graph, thread_id: str, decision: dict) -> dict:
    return graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
