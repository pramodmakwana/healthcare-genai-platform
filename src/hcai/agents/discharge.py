"""Discharge-summary agent: drafter + reviewer + consultant sign-off (LangGraph).

    load_context ──► draft ──► review ──┬─► revise ──► review   (auto-fixable issues, max 2 rounds)
                                        └─► consultant_signoff  ◄── interrupt: sign off / answer queries / return
                                                     │
                                                  finalize ──► audit record

The reviewer is a separate step with its own checklist taken from the discharge SOP. It
can only auto-fix what it can *prove* from the record (e.g. an ICD-10 code that exists in the
patient's FHIR conditions). Anything clinical - a missing reason for a medication change,
follow-up plan, allergy status - becomes a query for the consultant. The agent never invents
those, and a summary cannot be finalised while they're open.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hcai.agents.tools import PatientRepository, default_repository, load_admission
from hcai.guardrails.guards import check_output
from hcai.llm.base import LLMProvider

REQUIRED_SECTIONS = ["Reason for admission", "Diagnoses", "Procedures", "Hospital course", "Medication on discharge",
                     "Medication changes", "Allergies", "Follow-up", "Pending results"]
MAX_REVISIONS = 2

NARRATIVE_SYSTEM = (
    "You write the 'Hospital course' section of a discharge summary for a GP. Use ONLY the numbered "
    "clinical notes. Cite each sentence with the note number, e.g. [2]. 2-4 sentences, plain clinical "
    "English, no new diagnoses, doses or plans that are not in the notes."
)


class DSState(TypedDict, total=False):
    admission_id: str
    admission: dict
    patient: dict
    sections: dict[str, str]
    narrative_grounding: float
    issues: list[dict]          # {id, type, detail, auto_fixable}
    revisions: int
    due_by: str
    signoff: dict
    status: str                 # draft | awaiting_signoff | signed_off | returned | not_found
    audit: list[dict]


def _log(state: DSState, step: str, **detail: Any) -> list[dict]:
    return [*state.get("audit", []), {"step": step, "at": datetime.now().isoformat(timespec="seconds"), **detail}]


def _due_by(discharged_at: str) -> str:
    d = datetime.fromisoformat(discharged_at)
    # SOP: within 48h; Friday discharges must be sent before the patient leaves the ward.
    return (d if d.weekday() == 4 else d + timedelta(hours=48)).isoformat(timespec="minutes")


def build_discharge_graph(llm: LLMProvider, repo: PatientRepository | None = None, checkpointer=None):
    repo = repo or default_repository()

    def load_context(state: DSState) -> dict:
        adm = load_admission(state["admission_id"])
        if not adm:
            return {"status": "not_found", "audit": _log(state, "load_context", found=False)}
        p = repo.get(adm["mrn"])
        patient = {"name": p.name, "mrn": p.mrn, "conditions": p.conditions,
                   "medications": [m["text"] for m in p.medications]} if p else {"mrn": adm["mrn"], "conditions": [], "medications": []}
        return {"admission": adm, "patient": patient, "revisions": 0, "due_by": _due_by(adm["discharged_at"]),
                "status": "draft", "audit": _log(state, "load_context", found=True)}

    def _narrative(adm: dict, strict: bool = False) -> tuple[str, float]:
        notes = adm["clinical_notes"]
        if llm.name == "mock":
            text = " ".join(f"{n.rstrip('.')} [{i}]." for i, n in enumerate(notes, 1))
        else:
            system = NARRATIVE_SYSTEM + (" Be strictly extractive: reuse the notes' wording." if strict else "")
            prompt = "SOURCES:\n" + "\n".join(f"[{i}] {n}" for i, n in enumerate(notes, 1)) + "\n\nQUESTION: Write the hospital course."
            text = llm.generate(system, prompt).text.strip()
        return text, check_output(text, notes).groundedness

    def draft(state: DSState) -> dict:
        adm, p = state["admission"], state["patient"]
        changes = []
        for m in adm["medication_changes"]:
            reason = m.get("reason") or "[REASON REQUIRED - query consultant]"
            changes.append(f"- {m['drug']}: {m['change'].replace('_', ' ')}. Reason: {reason}")
        diag = [f"- {d['text']} (ICD-10 {d['icd10'] or '[CODE REQUIRED]'})" for d in adm["diagnoses"]]
        allergies = adm.get("allergies")
        narrative, grounding = _narrative(adm)
        sections = {
            "Reason for admission": adm["reason_for_admission"],
            "Diagnoses": "\n".join(diag),
            "Procedures": ", ".join(adm["procedures"]) or "None",
            "Hospital course": narrative,
            "Medication on discharge": "\n".join(f"- {m}" for m in p["medications"]) or "None recorded",
            "Medication changes": "\n".join(changes) or "None",
            "Allergies": ", ".join(allergies) if allergies else "[ALLERGY STATUS REQUIRED]",
            "Follow-up": adm.get("follow_up") or "[FOLLOW-UP PLAN REQUIRED]",
            "Pending results": ", ".join(adm["pending_results"]) or "None",
        }
        return {"sections": sections, "narrative_grounding": grounding, "audit": _log(state, "draft", model=llm.name)}

    def review(state: DSState) -> dict:
        adm, s = state["admission"], state["sections"]
        issues: list[dict] = []

        def add(kind: str, detail: str, auto: bool = False) -> None:
            issues.append({"id": f"{kind}:{len(issues)}", "type": kind, "detail": detail, "auto_fixable": auto})

        for name in REQUIRED_SECTIONS:
            if not s.get(name, "").strip():
                add("missing_section", f"Section '{name}' is empty")
        for m in adm["medication_changes"]:
            if not m.get("reason"):
                add("med_change_without_reason", f"{m['drug']} ({m['change'].replace('_', ' ')}) has no documented reason")
        if not adm.get("follow_up"):
            add("missing_follow_up", "No follow-up arrangements recorded")
        if not adm.get("allergies"):
            add("missing_allergy_status", "Allergy status not recorded (state 'No known drug allergies' if none)")
        # Medication reconciliation: a dose change or stop must refer to a medicine on the record.
        on_record = [m.lower() for m in state["patient"]["medications"]]
        for m in adm["medication_changes"]:
            if m["change"] in ("dose_changed", "stopped"):
                key = m["drug"].lower().split()[0]
                if not any(key in r for r in on_record):
                    add("med_change_not_in_record", f"{m['drug']} is {m['change'].replace('_', ' ')} but is not on the medication record")
        known = {(c["text"] or "").lower(): c["icd10"] for c in state["patient"]["conditions"]}
        for d in adm["diagnoses"]:
            if not d.get("icd10"):
                add("diagnosis_missing_icd10", f"'{d['text']}' has no ICD-10 code", auto=(d["text"] or "").lower() in known)
        if state["narrative_grounding"] < 0.8:
            add("narrative_ungrounded", f"Hospital course groundedness {state['narrative_grounding']:.2f} < 0.80", auto=True)
        return {"issues": issues, "audit": _log(state, "review", issues=[i["type"] for i in issues],
                                                 revision=state.get("revisions", 0))}

    def revise(state: DSState) -> dict:
        adm = {**state["admission"], "diagnoses": [dict(d) for d in state["admission"]["diagnoses"]]}
        sections = dict(state["sections"])
        grounding = state["narrative_grounding"]
        known = {(c["text"] or "").lower(): c["icd10"] for c in state["patient"]["conditions"]}
        fixed = []
        for issue in [i for i in state["issues"] if i["auto_fixable"]]:
            if issue["type"] == "diagnosis_missing_icd10":
                for d in adm["diagnoses"]:
                    if not d.get("icd10") and (d["text"] or "").lower() in known:
                        d["icd10"] = known[d["text"].lower()]
                        fixed.append(f"ICD-10 {d['icd10']} for {d['text']} (from FHIR Condition)")
                sections["Diagnoses"] = "\n".join(f"- {d['text']} (ICD-10 {d['icd10'] or '[CODE REQUIRED]'})"
                                                  for d in adm["diagnoses"])
            elif issue["type"] == "narrative_ungrounded":
                sections["Hospital course"], grounding = _narrative(adm, strict=True)
                fixed.append("regenerated hospital course (strict extractive)")
        return {"admission": adm, "sections": sections, "narrative_grounding": grounding,
                "revisions": state.get("revisions", 0) + 1, "audit": _log(state, "revise", fixed=fixed)}

    def route_after_review(state: DSState) -> str:
        if any(i["auto_fixable"] for i in state["issues"]) and state.get("revisions", 0) < MAX_REVISIONS:
            return "revise"
        return "consultant_signoff"

    def consultant_signoff(state: DSState) -> dict:
        queries = [i for i in state["issues"] if not i["auto_fixable"]]
        decision = interrupt({
            "task": "Consultant review and sign-off of AI-drafted discharge summary",
            "admission_id": state["admission_id"],
            "send_to_gp_by": state["due_by"],
            "summary": render(state),
            "open_queries": queries,
            "options": ["sign_off (answers required for every open query)", "return"],
        })
        return {"signoff": decision, "audit": _log(state, "consultant_signoff", action=decision.get("action"),
                                                    signed_by=decision.get("signed_by"))}

    def finalize(state: DSState) -> dict:
        if state.get("status") == "not_found":
            return {"audit": _log(state, "finalize", status="not_found")}
        d = state.get("signoff", {})
        answers: dict = d.get("answers", {})
        sections = dict(state["sections"])
        open_queries = [i for i in state["issues"] if not i["auto_fixable"] and not answers.get(i["id"])]
        for i in state["issues"]:
            ans = answers.get(i["id"])
            if not ans:
                continue
            if i["type"] == "med_change_without_reason":
                drug = i["detail"].split(" (")[0]
                sections["Medication changes"] = "\n".join(
                    ln.replace("[REASON REQUIRED - query consultant]", ans) if ln.startswith(f"- {drug}") else ln
                    for ln in sections["Medication changes"].splitlines())
            elif i["type"] == "missing_follow_up":
                sections["Follow-up"] = ans
            elif i["type"] == "missing_allergy_status":
                sections["Allergies"] = ans
        if d.get("action") == "sign_off" and not open_queries and d.get("signed_by"):
            status = "signed_off"
        else:
            status = "returned"
        return {"sections": sections, "status": status,
                "audit": _log(state, "finalize", status=status, unanswered=[q["id"] for q in open_queries])}

    g = StateGraph(DSState)
    for name, fn in [("load_context", load_context), ("draft", draft), ("review", review), ("revise", revise),
                     ("consultant_signoff", consultant_signoff), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "load_context")
    g.add_conditional_edges("load_context", lambda s: "finalize" if s.get("status") == "not_found" else "draft",
                            ["draft", "finalize"])
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", route_after_review, ["revise", "consultant_signoff"])
    g.add_edge("revise", "review")
    g.add_edge("consultant_signoff", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def render(state: dict) -> str:
    adm = state.get("admission", {})
    head = (f"DISCHARGE SUMMARY (AI-drafted - requires consultant sign-off)\n"
            f"Patient: {state.get('patient', {}).get('name', '')} ({adm.get('mrn', '')}) | Admission {adm.get('admission_id', '')}\n"
            f"Admitted {adm.get('admitted_at', '')} | Discharged {adm.get('discharged_at', '')} | Send to GP by {state.get('due_by', '')}\n")
    body = "\n\n".join(f"{k.upper()}\n{v}" for k, v in state.get("sections", {}).items())
    return head + "\n" + body


def start_draft(graph, admission_id: str, thread_id: str | None = None) -> tuple[str, dict]:
    thread_id = thread_id or uuid.uuid4().hex
    graph.invoke({"admission_id": admission_id, "audit": []}, {"configurable": {"thread_id": thread_id}})
    return thread_id, graph.get_state({"configurable": {"thread_id": thread_id}}).values


def submit_signoff(graph, thread_id: str, decision: dict) -> dict:
    return graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
