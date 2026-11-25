"""
CineLearn Model-Agnostic LLM Layer.
Provides a unified interface across Google Gemini, local Ollama (Qwen), OpenAI, and mock providers,
with stage-to-model routing and centralized configuration.
"""

from .types import (
    Stage,
    Capability,
    ProviderType,
    LLMUsage,
    ModelConfig,
    LLMResponse,
)
from .exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMQuotaExhaustedError,
    LLMTimeoutError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMServiceUnavailableError,
    UnsupportedCapabilityError,
    LLMResponseFormatError,
)
from .base import BaseLLMProvider
from .config import LLMConfigManager
from .router import LLMRouter, get_router
from .interface import (
    generate_response,
    generate_response_token,
    get_llm,
    make_stage_caller,
)
from .providers import (
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    MockProvider,
    register_provider,
    create_provider,
)

__all__ = [
    # Types & Enums
    "Stage",
    "Capability",
    "ProviderType",
    "LLMUsage",
    "ModelConfig",
    "LLMResponse",
    # Exceptions
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMQuotaExhaustedError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMModelNotFoundError",
    "LLMServiceUnavailableError",
    "UnsupportedCapabilityError",
    "LLMResponseFormatError",
    # Core Base & Management
    "BaseLLMProvider",
    "LLMConfigManager",
    "LLMRouter",
    "get_router",
    # Public Functions
    "generate_response",
    "generate_response_token",
    "get_llm",
    "make_stage_caller",
    # Providers
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "MockProvider",
    "register_provider",
    "create_provider",
]

# Ensure singleton consistency between 'import llm' and 'import src.llm'
import sys
if __name__ == "llm":
    sys.modules.setdefault("src.llm", sys.modules[__name__])
elif __name__ == "src.llm":
    sys.modules.setdefault("llm", sys.modules[__name__])
