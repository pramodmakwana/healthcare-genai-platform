"""Tools the agents call. Structured data comes from FHIR (exact values, dates, codes);
policy text comes from the RAG chain (cited). Agents never read facts out of free text
when a structured source exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from hcai.config import PROJECT_ROOT, get_settings

LAB_CODES = {"4548-4": "hba1c", "8480-6": "sbp", "8462-4": "dbp", "18262-6": "ldl", "33914-3": "egfr", "39156-5": "bmi"}


@dataclass
class LabResult:
    name: str
    value: float
    unit: str
    on: date


@dataclass
class PatientFacts:
    patient_id: str
    mrn: str
    name: str
    conditions: list[dict] = field(default_factory=list)  # {text, snomed, icd10, onset}
    medications: list[dict] = field(default_factory=list)  # {text, rxnorm, dosage, started}
    labs: dict[str, LabResult] = field(default_factory=dict)  # latest per lab

    @property
    def doc_id(self) -> str:
        return f"patient-{self.patient_id}"


def _code(concept: dict, system_suffix: str) -> str | None:
    return next((c["code"] for c in concept.get("coding", []) if c.get("system", "").endswith(system_suffix)), None)


def facts_from_bundle(bundle: dict) -> list[PatientFacts]:
    patients: dict[str, PatientFacts] = {}
    others = []
    for e in bundle.get("entry", []):
        r = e["resource"]
        if r["resourceType"] == "Patient":
            n = r["name"][0]
            mrn = next((i["value"] for i in r.get("identifier", []) if "Record" in i.get("type", {}).get("text", "")), "")
            patients[r["id"]] = PatientFacts(r["id"], mrn, " ".join(n.get("given", []) + [n.get("family", "")]))
        else:
            others.append(r)
    for r in others:
        pid = r.get("subject", {}).get("reference", "").split(":")[-1].split("/")[-1]
        p = patients.get(pid)
        if not p:
            continue
        if r["resourceType"] == "Condition":
            p.conditions.append({"text": r["code"].get("text"), "snomed": _code(r["code"], "sct"),
                                 "icd10": _code(r["code"], "icd-10-cm"), "onset": r.get("onsetDateTime", "")[:10]})
        elif r["resourceType"] == "MedicationRequest":
            c = r["medicationCodeableConcept"]
            p.medications.append({"text": c.get("text"), "rxnorm": _code(c, "rxnorm"),
                                  "dosage": (r.get("dosageInstruction") or [{}])[0].get("text", ""),
                                  "started": r.get("authoredOn", "")[:10]})
        elif r["resourceType"] == "Observation":
            key = LAB_CODES.get(_code(r["code"], "loinc.org") or "")
            if key:
                on = date.fromisoformat(r["effectiveDateTime"][:10])
                vq = r.get("valueQuantity", {})
                if key not in p.labs or on > p.labs[key].on:
                    p.labs[key] = LabResult(key, float(vq.get("value")), vq.get("unit", ""), on)
    return list(patients.values())


class PatientRepository:
    """Lookup by MRN. Backed by FHIR bundles on disk (swap for a FHIR server client in production)."""

    def __init__(self, fhir_dir: Path | None = None):
        fhir_dir = fhir_dir or Path(get_settings().raw_data_dir) / "fhir"
        self._by_mrn: dict[str, PatientFacts] = {}
        for f in sorted(fhir_dir.glob("*.json")):
            for p in facts_from_bundle(json.loads(f.read_text())):
                self._by_mrn[p.mrn.upper()] = p

    def get(self, mrn: str) -> PatientFacts | None:
        return self._by_mrn.get((mrn or "").upper())

    def all(self) -> list[PatientFacts]:
        return list(self._by_mrn.values())


@lru_cache(maxsize=1)
def default_repository() -> PatientRepository:
    return PatientRepository()


# ---------- date helpers (SLA calculations) ----------

def add_business_days(start: date, days: int) -> date:
    d = start
    while days > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days -= 1
    return d


def months_between(earlier: date, later: date) -> float:
    return (later - earlier).days / 30.44


def parse_dt(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, 9, 0)
    return datetime.fromisoformat(value)


ADMISSIONS_DIR = PROJECT_ROOT / "data" / "raw" / "admissions"


def load_admission(admission_id: str) -> dict | None:
    f = ADMISSIONS_DIR / f"{admission_id}.json"
    return json.loads(f.read_text()) if f.exists() else None
