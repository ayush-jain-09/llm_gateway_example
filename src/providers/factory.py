"""
Factory Pattern for LLM Provider instantiation.
Implements Creational Factory Pattern with dynamic registration.
"""

from typing import Dict, Type
from src.providers.base import BaseProvider
from src.providers.mock_adapter import MockProvider
from src.providers.openai_adapter import OpenAIAdapter
from src.providers.gemini_adapter import GeminiAdapter
from src.providers.perplexity_adapter import PerplexityAdapter
from src.providers.ollama_adapter import OllamaAdapter
from src.gateway.config import ProviderConfig


class ProviderFactory:
    _registry: Dict[str, Type[BaseProvider]] = {
        "mock": MockProvider,
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
        "perplexity": PerplexityAdapter,
        "ollama": OllamaAdapter,
    }

    @classmethod
    def register(cls, provider_type: str, provider_cls: Type[BaseProvider]) -> None:
        cls._registry[provider_type.lower()] = provider_cls

    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        p_type = config.type.lower()
        provider_cls = cls._registry.get(p_type)
        if not provider_cls:
            raise ValueError(f"Unknown provider type: '{config.type}'")
        return provider_cls(config)
