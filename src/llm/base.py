"""
Base abstract adapter class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from .types import Capability, LLMResponse, ModelConfig
from .exceptions import UnsupportedCapabilityError


class BaseLLMProvider(ABC):
    """Abstract base class that all provider adapters must implement."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider_name = config.provider
        self.model = config.model

    @property
    @abstractmethod
    def supported_capabilities(self) -> Set[Capability]:
        """Return set of capabilities supported by this provider/model."""
        pass

    def supports(self, capability: Capability) -> bool:
        """Check if this provider supports a specific capability."""
        return capability in self.supported_capabilities

    def check_capability(self, capability: Capability, operation_name: str = "operation") -> None:
        """Validate capability, raising UnsupportedCapabilityError if not supported."""
        if not self.supports(capability):
            raise UnsupportedCapabilityError(
                f"Provider '{self.provider_name}' (model: '{self.model}') does not support "
                f"'{capability.value}' for {operation_name}."
            )

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute request against LLM provider and return a normalized LLMResponse.
        """
        pass
