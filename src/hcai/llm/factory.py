from __future__ import annotations

from hcai.config import Settings
from hcai.llm.base import LLMProvider
from hcai.llm.providers import (
    AnthropicProvider,
    BedrockProvider,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
)


def get_llm(settings: Settings) -> LLMProvider:
    kwargs = dict(model=settings.llm_model, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature)
    provider = settings.llm_provider.lower()
    if provider in ("anthropic", "claude"):
        return AnthropicProvider(**kwargs)
    if provider == "openai":
        return OpenAIProvider(**kwargs)
    if provider in ("azure", "azure_openai"):
        return OpenAIProvider(azure=True, **kwargs)
    if provider == "bedrock":
        return BedrockProvider(**kwargs)
    if provider == "gemini":
        return GeminiProvider(**kwargs)
    if provider == "mock":
        return MockProvider(**kwargs)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
