"""LLM-as-judge faithfulness scoring (Claude by default).

The judge splits an answer into atomic claims and marks each as supported or not by the
retrieved sources. Faithfulness = supported / total claims; hallucination rate = 1 - that.
This is stricter than the word-overlap groundedness check in hcai.guardrails, which stays
as the cheap always-on check; the judge is used offline in evaluation.

Good practice built in: the judge model is configurable and should differ from (or be
stronger than) the answering model, and it sees only the sources - never the gold answer.
"""

from __future__ import annotations

import json
import os
import re

from hcai.llm.base import LLMProvider

JUDGE_SYSTEM = """You are a strict fact-checker for a clinical knowledge assistant.
Given SOURCES and an ANSWER, split the ANSWER into atomic factual claims (ignore citation
markers and refusals). For each claim decide if it is fully supported by the SOURCES alone.
Numbers, units, drug names and time limits must match exactly to count as supported.
Respond with JSON only, no prose:
{"claims": [{"claim": "...", "supported": true, "source": 1}], "notes": "..."}"""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"Judge returned no JSON: {text[:200]}")
    return json.loads(m.group(0))


def judge_faithfulness(judge: LLMProvider, question: str, answer: str, sources: list[str]) -> dict:
    if "don't have enough information" in answer.lower():
        return {"claims": [], "supported": 0, "total": 0, "faithfulness": 1.0, "refusal": True}
    prompt = ("SOURCES:\n" + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
              + f"\n\nQUESTION: {question}\n\nANSWER: {answer}")
    resp = judge.generate(JUDGE_SYSTEM, prompt, model=os.environ.get("JUDGE_MODEL"))
    data = _parse(resp.text)
    claims = data.get("claims", [])
    supported = sum(bool(c.get("supported")) for c in claims)
    total = len(claims)
    return {"claims": claims, "supported": supported, "total": total,
            "faithfulness": round(supported / total, 4) if total else 1.0, "refusal": False,
            "judge_tokens": resp.usage.total_tokens}
