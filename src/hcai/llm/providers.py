"""Concrete LLM providers.

SDKs are imported lazily inside each provider so you only need the ones you use
(e.g. `pip install anthropic` for Claude; `pip install boto3` only if you use Bedrock).
"""

from __future__ import annotations

import os
import re
import time

from hcai.llm.base import LLMProvider, LLMResponse, Usage


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic API. Uses prompt caching on the system prompt to cut input cost."""

    name = "anthropic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        model = model or self.model
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # The system prompt is identical on every call, so mark it cacheable.
            # (Caching only applies once the cached prefix passes the model's minimum length.)
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return LLMResponse(text=text, model=model, provider=self.name, usage=usage, latency_ms=latency)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, *args, azure: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if azure:
            from openai import AzureOpenAI

            self.name = "azure_openai"
            self.client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        # For Azure, `model` is the deployment name.
        model = model or self.model
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000
        u = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens),
            latency_ms=latency,
        )


class BedrockProvider(LLMProvider):
    """AWS Bedrock via the Converse API (works for Claude and other Bedrock models)."""

    name = "bedrock"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-west-1"))

    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        model = model or self.model  # a Bedrock model ID or inference profile ARN
        start = time.perf_counter()
        resp = self.client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        latency = (time.perf_counter() - start) * 1000
        text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=u.get("inputTokens", 0), output_tokens=u.get("outputTokens", 0)),
            latency_ms=latency,
        )


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from google import genai

        self.genai = genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        model = model or self.model
        start = time.perf_counter()
        resp = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=self.genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        latency = (time.perf_counter() - start) * 1000
        meta = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        )
        return LLMResponse(text=resp.text or "", model=model, provider=self.name, usage=usage, latency_ms=latency)


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "was", "what",
    "which", "who", "how", "does", "do", "with", "by", "at", "be", "this", "that", "patient",
    "patients", "their", "his", "her", "from", "as", "it", "should", "when", "has", "have",
}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


class MockProvider(LLMProvider):
    """Deterministic, offline, extractive "LLM".

    It reads the numbered sources in the prompt, picks the sentences that overlap most with
    the question, and returns them with [n] citations. That lets the whole pipeline - retrieval,
    citation checks, groundedness scoring, the eval harness and CI - run with no API key.
    It is a test double, not a model: real answer quality comes from a real provider.
    """

    name = "mock"
    _SOURCE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.S | re.M)

    def generate(self, system: str, prompt: str, model: str | None = None) -> LLMResponse:
        start = time.perf_counter()
        q_match = re.search(r"QUESTION:\s*(.+)", prompt)
        question = q_match.group(1) if q_match else prompt
        q_terms = _terms(question)

        sources_part = prompt.split("\n\nQUESTION:")[0]
        candidates: list[tuple[int, str, set[str]]] = []
        for num, body in self._SOURCE.findall(sources_part):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                sent = sent.strip()
                if len(sent) >= 15:
                    candidates.append((int(num), sent, _terms(sent)))

        # IDF-weight query terms across candidate sentences so distinctive words
        # ("urgent", "eGFR") outweigh words every sentence shares ("prior", "request").
        import math

        n = max(len(candidates), 1)
        df = {t: sum(t in c[2] for c in candidates) for t in q_terms}
        scored: list[tuple[float, int, str]] = []
        for num, sent, terms in candidates:
            score = sum(math.log(1 + n / (1 + df[t])) for t in q_terms & terms)
            if score:
                scored.append((score / (1 + 0.02 * len(sent.split())), num, sent))

        scored.sort(key=lambda x: -x[0])
        if not scored:
            text = "I don't have enough information in the provided sources to answer that."
        else:
            picked, seen = [], set()
            for _, num, sent in scored[:3]:
                if sent not in seen:
                    seen.add(sent)
                    picked.append(f"{sent.rstrip('.')} [{num}].")
            text = " ".join(picked)

        in_tok = int(len((system + prompt).split()) * 1.3)
        out_tok = int(len(text.split()) * 1.3)
        return LLMResponse(
            text=text,
            model="mock",
            provider=self.name,
            usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
