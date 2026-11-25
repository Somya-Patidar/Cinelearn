"""
Unified exception hierarchy for CineLearn LLM abstraction layer.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when configuration is invalid, missing, or malformed."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (invalid API key, billing issues)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when per-minute or concurrent rate limits are hit."""
    pass


class LLMQuotaExhaustedError(LLMError):
    """Raised when daily or persistent quota is completely exhausted."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM call exceeds the allowed timeout."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to the LLM service fails (e.g. Ollama not running)."""
    pass


class LLMModelNotFoundError(LLMError):
    """Raised when the requested model is not found on the provider/host."""
    pass


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM service returns 503 or is overloaded."""
    pass


class UnsupportedCapabilityError(LLMError):
    """Raised when an operation requires a capability not supported by the model/provider."""
    pass


class LLMResponseFormatError(LLMError):
    """Raised when the LLM response cannot be formatted or parsed into the required schema."""
    pass
