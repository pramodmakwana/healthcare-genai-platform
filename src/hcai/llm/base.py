"""Provider-agnostic LLM interface.

Every provider (Claude, OpenAI, Azure OpenAI, Bedrock, Gemini, mock) returns the same
LLMResponse, so the RAG chain, guardrails and evaluation never depend on a vendor SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, max_tokens: int = 700, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        """Single-turn generation. `system` is stable across calls (cacheable); `prompt` varies."""


@lru_cache(maxsize=4)
def _load_pricing(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return (yaml.safe_load(p.read_text()) or {}).get("models", {})


def estimate_cost_usd(model: str, usage: Usage, pricing_file: str) -> float | None:
    """Cost of one call from the pricing table; None if the model isn't priced."""
    rates = _load_pricing(pricing_file).get(model)
    if rates is None:
        return None
    per_m = 1_000_000
    return round(
        usage.input_tokens * rates.get("input", 0) / per_m
        + usage.output_tokens * rates.get("output", 0) / per_m
        + usage.cache_read_tokens * rates.get("cache_read", 0) / per_m
        + usage.cache_write_tokens * rates.get("cache_write", 0) / per_m,
        8,
    )
