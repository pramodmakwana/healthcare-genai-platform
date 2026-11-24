"""Document loaders: FHIR R4 bundles (Synthea / HAPI FHIR) and markdown guidelines.

A FHIR bundle is turned into one readable clinical summary per patient. Codes are kept in
the text (SNOMED CT, ICD-10, LOINC, RxNorm) so both keyword and semantic search can match them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_type: str  # "patient_record" | "guideline"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_type: str
    metadata: dict = field(default_factory=dict)


def _codes(concept: dict) -> str:
    systems = {
        "http://snomed.info/sct": "SNOMED",
        "http://hl7.org/fhir/sid/icd-10-cm": "ICD-10",
        "http://loinc.org": "LOINC",
        "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    }
    parts = [f"{systems.get(c.get('system'), 'code')} {c.get('code')}" for c in concept.get("coding", []) if c.get("code")]
    return f" ({', '.join(parts)})" if parts else ""


def _concept_text(concept: dict) -> str:
    if concept.get("text"):
        return concept["text"]
    coding = concept.get("coding") or [{}]
    return coding[0].get("display", "Unknown")


def fhir_bundle_to_documents(bundle: dict) -> list[Document]:
    """Group resources by patient and render each patient as one document."""
    patients: dict[str, dict] = {}
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype == "Patient":
            patients[res["id"]] = res
        else:
            ref = (res.get("subject") or res.get("patient") or {}).get("reference", "")
            pid = ref.split(":")[-1].split("/")[-1]
            if pid:
                by_patient[pid].append(res)

    docs = []
    for pid, p in patients.items():
        name = p.get("name", [{}])[0]
        full_name = " ".join(name.get("given", []) + [name.get("family", "")]).strip()
        mrn = next((i.get("value") for i in p.get("identifier", []) if "Record" in i.get("type", {}).get("text", "")), "")
        res = by_patient.get(pid, [])

        lines = [
            f"Patient: {full_name}",
            f"MRN: {mrn}",
            f"Gender: {p.get('gender', 'unknown')}. Date of birth: {p.get('birthDate', 'unknown')}.",
        ]
        if p.get("telecom"):
            lines.append(f"Phone: {p['telecom'][0].get('value', '')}")
        if p.get("address"):
            a = p["address"][0]
            lines.append(f"Address: {', '.join(a.get('line', []))}, {a.get('city', '')}")

        conds = [r for r in res if r["resourceType"] == "Condition"]
        meds = [r for r in res if r["resourceType"] == "MedicationRequest"]
        obs = [r for r in res if r["resourceType"] == "Observation"]
        encs = [r for r in res if r["resourceType"] == "Encounter"]

        lines.append("\n## Active conditions")
        for c in conds:
            lines.append(f"{full_name} has {_concept_text(c['code'])}{_codes(c['code'])}, onset {c.get('onsetDateTime', 'unknown')[:10]}.")
        if not conds:
            lines.append(f"{full_name} has no recorded active conditions.")

        lines.append("\n## Current medications")
        for m in meds:
            concept = m.get("medicationCodeableConcept", {})
            dose = (m.get("dosageInstruction") or [{}])[0].get("text", "")
            lines.append(f"{full_name} is prescribed {_concept_text(concept)}{_codes(concept)}. Dosage: {dose}. Started {m.get('authoredOn', '')[:10]}.")
        if not meds:
            lines.append(f"{full_name} has no active medication requests.")

        lines.append("\n## Recent observations")
        for o in sorted(obs, key=lambda r: r.get("effectiveDateTime", ""), reverse=True):
            vq = o.get("valueQuantity", {})
            label = _concept_text(o["code"])
            short = {"4548-4": "HbA1c", "8480-6": "Systolic BP", "8462-4": "Diastolic BP", "18262-6": "LDL cholesterol",
                     "33914-3": "eGFR", "39156-5": "BMI"}.get((o["code"].get("coding") or [{}])[0].get("code"), label)
            lines.append(f"{full_name}'s most recent {short} was {vq.get('value')} {vq.get('unit', '')} on {o.get('effectiveDateTime', '')[:10]} ({label}{_codes(o['code'])}).")

        if encs:
            last = max(e.get("period", {}).get("start", "") for e in encs)
            lines.append(f"\n## Encounters\nLast encounter for {full_name}: {last[:10]}.")

        docs.append(Document(
            doc_id=f"patient-{pid}",
            title=f"Patient record: {full_name} ({mrn})",
            text="\n".join(lines),
            source_type="patient_record",
            metadata={"patient_id": pid, "mrn": mrn},
        ))
    return docs


def load_fhir_dir(path: Path) -> list[Document]:
    docs: list[Document] = []
    for f in sorted(path.glob("*.json")):
        docs.extend(fhir_bundle_to_documents(json.loads(f.read_text())))
    return docs


def load_fhir_server(base_url: str, max_patients: int = 50) -> list[Document]:
    """Pull patients from a FHIR server (e.g. the HAPI FHIR in healthcare-fhir-data-pipeline)."""
    import httpx

    docs: list[Document] = []
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=60) as http:
        page = http.get("/Patient", params={"_count": max_patients, "_elements": "id"}).json()
        for entry in page.get("entry", [])[:max_patients]:
            pid = entry["resource"]["id"]
            everything = http.get(f"/Patient/{pid}/$everything", params={"_count": 500}).json()
            docs.extend(fhir_bundle_to_documents(everything))
    return docs


def load_markdown_dir(path: Path) -> list[Document]:
    docs = []
    for f in sorted(path.glob("*.md")):
        text = f.read_text()
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), f.stem)
        docs.append(Document(doc_id=f.stem, title=title, text=text, source_type="guideline", metadata={"file": f.name}))
    return docs
