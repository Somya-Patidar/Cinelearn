"""
Providers package exposing available adapters and factory.
"""

from typing import Dict, Type
from ..base import BaseLLMProvider
from ..types import ProviderType, ModelConfig
from ..exceptions import LLMConfigurationError

from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai_provider import OpenAIProvider
from .mock_provider import MockProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    ProviderType.GEMINI.value: GeminiProvider,
    ProviderType.OLLAMA.value: OllamaProvider,
    ProviderType.OPENAI.value: OpenAIProvider,
    ProviderType.MOCK.value: MockProvider,
}


def register_provider(name: str, provider_cls: Type[BaseLLMProvider]) -> None:
    """Register a new custom provider class."""
    PROVIDER_REGISTRY[name.lower()] = provider_cls


def create_provider(config: ModelConfig) -> BaseLLMProvider:
    """Factory creating provider instance from ModelConfig."""
    prov_name = config.provider.lower()
    if prov_name not in PROVIDER_REGISTRY:
        raise LLMConfigurationError(
            f"Unknown provider '{prov_name}'. Registered providers: {list(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[prov_name](config)


__all__ = [
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "MockProvider",
    "PROVIDER_REGISTRY",
    "register_provider",
    "create_provider",
]
