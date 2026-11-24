"""Guardrails for a clinical RAG assistant.

Input side
  - prompt-injection / data-exfiltration detection
  - emergency-symptom detection (route to emergency services, never answer from documents)
  - personal medical-advice detection (dosing/diagnosis requests get a safe refusal + signpost)
Output side
  - citation validation: every [n] must point at a retrieved source; uncited answers are flagged
  - groundedness score: share of answer sentences whose content words are supported by the
    cited sources (a cheap, deterministic hallucination proxy; an LLM judge can be layered on)
Privacy
  - PHI redaction for logs and for de-identified output modes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------- PHI redaction ----------

_PHI_PATTERNS = [
    # Separators are required so clinical codes (SNOMED 714628002, LOINC 4548-4) are not masked.
    ("PHONE", re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]\d{3}[\s-]\d{3,4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("MRN", re.compile(r"\bMRN\s*:?\s*\d{4,}\b|\bMRN\d{4,}\b", re.I)),
    ("PPSN", re.compile(r"\b\d{7}[A-W][A-IW]?\b")),  # Irish PPS number shape
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("DOB", re.compile(r"(?i)(date of birth|dob)\s*:?\s*\d{4}-\d{2}-\d{2}")),
    ("ADDRESS", re.compile(r"(?i)address:\s*[^\n]+")),
]


def redact_phi(text: str, names: list[str] | None = None) -> str:
    """Replace direct identifiers with typed placeholders. Pass known patient names to mask them too."""
    out = text
    for label, pat in _PHI_PATTERNS:
        out = pat.sub(f"[{label}]", out)
    for name in names or []:
        if name:
            out = re.sub(re.escape(name), "[NAME]", out, flags=re.I)
    return out


# ---------- Input checks ----------

_INJECTION = re.compile(
    r"(?i)(ignore (all |any )?(previous|prior|above) (instructions|rules)|disregard (the )?(system|previous)"
    r"|you are now|reveal (your|the) (system )?prompt|print (every|all) patient|dump (the )?(database|records)"
    r"|list all (patients|phone numbers|addresses))"
)
_EMERGENCY = re.compile(
    r"(?i)(chest pain|can'?t breathe|cannot breathe|difficulty breathing|stroke|face (is )?drooping|"
    r"suicid|overdos|unconscious|severe bleeding|arm is numb|seizure)"
)
_PERSONAL = re.compile(r"(?i)\b(i|i'm|im|i've|my|me|we|our|he|she|his|her|they)\b")
_PERSONAL_ADVICE = re.compile(
    r"(?i)\b(should i|can i|do i need to|how much .* should i|what should i take|is it safe for me)\b"
    r".*\b(take|dose|dosage|double|stop|start|increase|decrease|insulin|medication|pill|tablet)"
)


@dataclass
class InputVerdict:
    allowed: bool
    category: str = "ok"  # ok | blocked_injection | emergency | medical_advice
    message: str = ""


def check_input(question: str) -> InputVerdict:
    if _INJECTION.search(question):
        return InputVerdict(False, "blocked_injection",
                            "This request was blocked: it asks the assistant to bypass its rules or bulk-export patient data.")
    # Emergency symptoms only trigger when someone is describing a person's current situation,
    # so staff questions like "how are chest pain referrals documented?" are not blocked.
    if _EMERGENCY.search(question) and _PERSONAL.search(question):
        return InputVerdict(False, "emergency",
                            "This may be a medical emergency. Call 112 or 999 (Ireland/EU) or your local emergency number now. "
                            "This assistant cannot help with emergencies.")
    if _PERSONAL_ADVICE.search(question):
        return InputVerdict(False, "medical_advice",
                            "I can't give personal dosing or treatment advice. Please contact your GP, pharmacist or the "
                            "clinician managing your care. I can explain what the clinical guidelines say in general terms.")
    return InputVerdict(True)


# ---------- Output checks ----------

_CITE = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set(
    "the a an of to and or in on for is are was were be been with by at as from that this it its "
    "should may can will must not no their his her they he she which who what when where how than "
    "also any all each per most more less other such these those into within".split()
)
_REFUSAL = re.compile(r"(?i)(don't|do not|cannot|can't) (have|find) enough information|not (covered|mentioned) in the (provided )?sources")


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class OutputReport:
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    has_citations: bool = False
    groundedness: float = 0.0
    is_refusal: bool = False
    flags: list[str] = field(default_factory=list)


def check_output(answer: str, sources: list[str], min_groundedness: float = 0.5) -> OutputReport:
    rep = OutputReport()
    rep.is_refusal = bool(_REFUSAL.search(answer))
    nums = [int(n) for n in _CITE.findall(answer)]
    rep.cited = sorted(set(nums))
    rep.invalid_citations = [n for n in rep.cited if n < 1 or n > len(sources)]
    rep.has_citations = bool(rep.cited)

    if rep.is_refusal:
        rep.groundedness = 1.0
        return rep

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(_content(s)) >= 2]
    if not sentences:
        rep.flags.append("empty_answer")
        return rep

    supported = 0
    for sent in sentences:
        refs = [int(n) for n in _CITE.findall(sent) if 1 <= int(n) <= len(sources)]
        # Uncited sentences are checked against all sources but still count against citation quality.
        pool = " ".join(sources[n - 1] for n in refs) if refs else " ".join(sources)
        words = _content(_CITE.sub("", sent))
        if words and len(words & _content(pool)) / len(words) >= 0.6:
            supported += 1
    rep.groundedness = round(supported / len(sentences), 3)

    if not rep.has_citations:
        rep.flags.append("no_citations")
    if rep.invalid_citations:
        rep.flags.append("invalid_citations")
    if rep.groundedness < min_groundedness:
        rep.flags.append("low_groundedness")
    return rep
