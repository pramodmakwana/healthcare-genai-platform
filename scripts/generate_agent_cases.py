"""Generate synthetic test cases for the two agents.

1. Prior-authorization requests (data/eval/prior_auth_cases.jsonl) with the expected triage
   outcome written from the policy text, independently of the agent code.
2. Hospital admissions (data/raw/admissions/*.json) with documentation gaps planted on
   purpose, plus a ground-truth file (data/eval/discharge_cases.jsonl) listing the gaps the
   reviewer agent should catch. The admission files themselves don't reveal the answers.

Run after scripts/generate_sample_data.py (needs the FHIR bundles).
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcai.agents.tools import PatientRepository  # noqa: E402


def _bdays(d: date, n: int) -> date:
    while n:
        d += timedelta(days=1)
        n -= d.weekday() < 5
    return d


def prior_auth_cases(repo: PatientRepository, rng: random.Random) -> list[dict]:
    patients = sorted(repo.all(), key=lambda p: p.mrn)
    diabetics = [p for p in patients if any("diabetes" in (c["text"] or "").lower() for c in p.conditions)]
    others = [p for p in patients if p not in diabetics]
    just = "HbA1c above target despite at least 3 months of metformin; BMI over 30; cardiovascular risk."
    cases = []

    def case(p, drug, indication, received: datetime, urgent=False, justification=just, note=""):
        # Expected outcome from the policy, written independently of the agent's rules.
        cls = {"semaglutide": "glp1", "tirzepatide": "glp1", "empagliflozin": "sglt2", "adalimumab": "biologic",
               "metformin": "none", "atorvastatin": "none", "ozempic": "glp1", "mounjaro": "glp1",
               "jardiance": "sglt2", "  semaglutide 1mg pen ": "glp1"}[drug.lower()]
        if p is None:
            return {"expected_status": "invalid_request", "expected_missing": []}
        weight_only = indication == "weight management"
        if cls == "none" or (cls == "sglt2" and not weight_only):
            return {"expected_status": "no_pa_required", "expected_missing": []}
        missing = []
        if len(justification) < 20:
            missing.append("prescriber clinical justification")
        if cls == "glp1":
            h = p.labs.get("hba1c")
            if not h or (received.date() - h.on).days / 30.44 > 3:
                missing.append("HbA1c within the last 3 months")
            if "bmi" not in p.labs:
                missing.append("current BMI")
            if not any(k in m["text"].lower() for m in p.medications for k in ("metformin", "insulin", "empagliflozin")):
                missing.append("diabetes medicines already tried")
        if cls == "sglt2" and "egfr" not in p.labs:
            missing.append("latest eGFR")
        return {"expected_status": "pended" if missing else "ready_for_pharmacist_review", "expected_missing": sorted(missing)}

    def add(p, drug, indication, received, urgent=False, justification=just, scenario=""):
        exp = case(p, drug, indication, received, urgent, justification)
        cases.append({"id": f"pa{len(cases):02d}", "scenario": scenario,
                      "request": {"mrn": p.mrn if p else "MRN999999", "drug": drug, "indication": indication,
                                  "urgent": urgent, "prescriber_justification": justification,
                                  "received_at": received.isoformat(timespec="minutes")},
                      **exp})

    for p in diabetics[:8]:
        h = p.labs["hba1c"].on
        add(p, "semaglutide", "type 2 diabetes", datetime.combine(h + timedelta(days=30), datetime.min.time()).replace(hour=10),
            scenario="GLP-1, recent HbA1c")
        add(p, "tirzepatide", "type 2 diabetes", datetime.combine(h + timedelta(days=150), datetime.min.time()).replace(hour=10),
            urgent=rng.random() < 0.3, scenario="GLP-1, HbA1c too old")
    for p in diabetics[8:12]:
        h = p.labs["hba1c"].on
        add(p, "semaglutide", "type 2 diabetes", datetime.combine(h + timedelta(days=20), datetime.min.time()).replace(hour=15),
            justification="needs it", scenario="GLP-1, weak justification")
    for p in diabetics[12:15]:
        add(p, "empagliflozin", "type 2 diabetes with CKD", datetime(2026, 11, 20, 9), scenario="SGLT2 for diabetes: no PA")
    for p in others[:3]:
        add(p, "empagliflozin", "weight management", datetime(2026, 11, 18, 11), scenario="SGLT2 for weight only: PA")
    for p in others[3:5]:
        add(p, "adalimumab", "rheumatoid arthritis", datetime(2026, 11, 19, 14), urgent=True, scenario="biologic, urgent")
    for p in others[5:8]:
        add(p, "atorvastatin", "hyperlipidaemia", datetime(2026, 11, 17, 9), scenario="standard drug: no PA")
    add(None, "semaglutide", "type 2 diabetes", datetime(2026, 11, 20, 9), scenario="unknown MRN")

    # Harder, realistic inputs: brand names, free-text drug strings, the 3-month boundary.
    p = diabetics[0]
    h = p.labs["hba1c"].on
    add(p, "Ozempic", "type 2 diabetes", datetime.combine(h + timedelta(days=40), datetime.min.time()), scenario="brand name (GLP-1)")
    add(p, "Mounjaro", "type 2 diabetes", datetime.combine(h + timedelta(days=40), datetime.min.time()), scenario="brand name (GLP-1)")
    add(others[0], "Jardiance", "weight management", datetime(2026, 11, 18, 11), scenario="brand name (SGLT2, weight only)")
    add(p, "  semaglutide 1mg pen ", "type 2 diabetes", datetime.combine(h + timedelta(days=40), datetime.min.time()),
        scenario="free-text drug string")
    add(p, "semaglutide", "type 2 diabetes", datetime.combine(h + timedelta(days=91), datetime.min.time()),
        scenario="HbA1c 91 days old (within 3 months)")
    add(p, "semaglutide", "type 2 diabetes", datetime.combine(h + timedelta(days=95), datetime.min.time()),
        scenario="HbA1c 95 days old (over 3 months)")
    return cases


REASONS = {
    "hyperglycaemia": ("Hyperglycaemia with dehydration", "E11.65", [
        "Presented with polyuria, polydipsia and capillary glucose of 24 mmol/L; no ketosis.",
        "Treated with intravenous fluids and a variable-rate insulin infusion; glucose settled within 24 hours.",
        "Diabetes team review: transitioned to basal insulin and diabetes education completed before discharge.",
    ], [{"drug": "Insulin glargine 100 UNT/ML", "change": "started", "reason": "persistent hyperglycaemia on oral therapy"},
        {"drug": "Gliclazide 80 MG", "change": "stopped", "reason": "replaced by basal insulin"}]),
    "pneumonia": ("Community-acquired pneumonia", "J18.9", [
        "Presented with fever, productive cough and right basal crackles; chest X-ray showed right lower lobe consolidation.",
        "Treated with oral amoxicillin and clarithromycin; CURB-65 score 1.",
        "Afebrile for 48 hours with improving inflammatory markers; mobilising independently.",
    ], [{"drug": "Amoxicillin 500 MG", "change": "started", "reason": "community-acquired pneumonia, complete 5-day course"},
        {"drug": "Metformin 500 MG", "change": "dose_changed", "reason": "held during acute illness, restarted at reduced dose"}]),
    "hypertensive": ("Hypertensive urgency", "I16.0", [
        "Presented with blood pressure 196/112 mmHg and headache; no evidence of end-organ damage on ECG, bloods or fundoscopy.",
        "Treated with oral amlodipine; blood pressure reduced gradually over 24 hours.",
        "Blood pressure 148/88 mmHg at discharge; adherence counselling given.",
    ], [{"drug": "Amlodipine 5 MG", "change": "started", "reason": "blood pressure control"},
        {"drug": "Lisinopril 10 MG", "change": "dose_changed", "reason": "increased to 20 mg for blood pressure control"}]),
}
DEFECTS = ["med_change_without_reason", "missing_follow_up", "missing_allergy_status", "diagnosis_missing_icd10"]


def admissions(repo: PatientRepository, rng: random.Random) -> list[dict]:
    out_dir = ROOT / "data" / "raw" / "admissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.json"):
        f.unlink()
    patients = sorted(repo.all(), key=lambda p: p.mrn)[:12]
    truth = []
    for i, p in enumerate(patients):
        kind = list(REASONS)[i % 3]
        reason, icd, notes, changes = REASONS[kind]
        admitted = datetime(2026, 11, 2 + i, 8 + i % 6, 0)
        discharged = admitted + timedelta(days=2 + i % 3, hours=5)
        defects = [] if i % 4 == 0 else rng.sample(DEFECTS, k=1 + (i % 2))
        comorbid = [{"text": c["text"], "icd10": c["icd10"]} for c in p.conditions[:2]]
        diagnoses = [{"text": reason, "icd10": icd}, *comorbid]
        if "diagnosis_missing_icd10" in defects and comorbid:
            diagnoses[1] = {"text": comorbid[0]["text"], "icd10": None}  # recoverable from the FHIR record
        elif "diagnosis_missing_icd10" in defects:
            defects.remove("diagnosis_missing_icd10")
        meds = [dict(changes[0])]
        # Second change: adjust a medicine the patient is actually on, so admissions stay coherent
        # with the FHIR record (the reviewer reconciles changes against it).
        if p.medications:
            m = p.medications[0]["text"].split(" Oral")[0].split(" Injectable")[0]
            meds.append({"drug": m, "change": "dose_changed", "reason": "reviewed and adjusted during admission"})
        else:
            meds.append({"drug": "Paracetamol 500 MG", "change": "started", "reason": "symptomatic relief"})
        if "med_change_without_reason" in defects:
            meds[rng.randrange(len(meds))]["reason"] = None
        adm = {
            "admission_id": f"ADM{2600 + i}", "mrn": p.mrn, "ward": rng.choice(["St Brigid's", "Liffey", "Phoenix"]),
            "consultant": rng.choice(["Dr A. Example", "Dr B. Sample", "Dr C. Placeholder"]),
            "admitted_at": admitted.isoformat(timespec="minutes"), "discharged_at": discharged.isoformat(timespec="minutes"),
            "reason_for_admission": reason, "diagnoses": diagnoses,
            "procedures": ["Chest X-ray"] if kind == "pneumonia" else [],
            "clinical_notes": notes, "medication_changes": meds,
            "allergies": None if "missing_allergy_status" in defects else rng.choice([["No known drug allergies"], ["Penicillin (rash)"]]),
            "follow_up": None if "missing_follow_up" in defects else "GP review in 2 weeks; repeat bloods at GP.",
            "pending_results": rng.choice([[], ["Blood cultures (final report)"]]),
        }
        (out_dir / f"{adm['admission_id']}.json").write_text(json.dumps(adm, indent=1))
        due = discharged if discharged.weekday() == 4 else discharged + timedelta(hours=48)
        truth.append({"admission_id": adm["admission_id"], "expected_issues": sorted(defects),
                      "expected_due_by": due.isoformat(timespec="minutes"),
                      "auto_fixable": ["diagnosis_missing_icd10"] if "diagnosis_missing_icd10" in defects else []})
    return truth


def main() -> None:
    rng = random.Random(11)
    repo = PatientRepository(ROOT / "data" / "raw" / "fhir")
    eval_dir = ROOT / "data" / "eval"
    pa = prior_auth_cases(repo, rng)
    with (eval_dir / "prior_auth_cases.jsonl").open("w") as fh:
        fh.writelines(json.dumps(c) + "\n" for c in pa)
    truth = admissions(repo, rng)
    with (eval_dir / "discharge_cases.jsonl").open("w") as fh:
        fh.writelines(json.dumps(t) + "\n" for t in truth)
    from collections import Counter
    print(f"{len(pa)} prior-auth cases: {dict(Counter(c['expected_status'] for c in pa))}")
    print(f"{len(truth)} admissions; planted issues: {dict(Counter(i for t in truth for i in t['expected_issues']))}")


if __name__ == "__main__":
    main()
