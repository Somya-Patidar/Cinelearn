"""
OpenAI & OpenAI-compatible provider adapter implementing BaseLLMProvider.
Supports OpenAI, vLLM, DeepSeek, Groq, LocalAI, etc.
"""

import os
from typing import Any, Dict, List, Optional, Set, Union
import base64

from ..base import BaseLLMProvider
from ..types import Capability, LLMResponse, LLMUsage, ModelConfig
from ..exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMError,
    UnsupportedCapabilityError,
)

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI and OpenAI-compatible API providers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not _OPENAI_AVAILABLE:
            raise LLMConfigurationError("The 'openai' library is required to use OpenAIProvider.")

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.strip() in ("", "...", "YOUR_OPENAI_KEY"):
            raise LLMAuthenticationError(
                "OPENAI_API_KEY is not configured in environment or .env!\n"
                "Please configure OPENAI_API_KEY to use the OpenAI provider."
            )
        self.api_key = api_key.strip()
        self.base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
        self.model = config.model or os.getenv("OPENAI_MODEL") or "gpt-4o"
        self._client: Optional[openai.OpenAI] = None

    @property
    def supported_capabilities(self) -> Set[Capability]:
        return {
            Capability.TEXT,
            Capability.SYSTEM_PROMPT,
            Capability.JSON_MODE,
            Capability.IMAGE_INPUT,
        }

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client

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
        if video_path:
            self.check_capability(Capability.VIDEO_INPUT, f"stage '{stage or 'video'}'")

        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        max_toks = max_tokens or self.config.max_tokens or 8000

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: Union[str, List[Dict[str, Any]]] = ""
        if image_path:
            self.check_capability(Capability.IMAGE_INPUT, f"stage '{stage or 'image'}'")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            with open(image_path, "rb") as imf:
                b64_img = base64.b64encode(imf.read()).decode("utf-8")
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            user_content = [
                {"type": "text", "text": prompt_str},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ]
            messages.append({"role": "user", "content": user_content})
        elif isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages.extend(prompt)

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_toks,
        }
        if timeout:
            call_kwargs["timeout"] = timeout

        if response_format == "json":
            call_kwargs["response_format"] = {"type": "json_object"}

        print(f"🤖 [OPENAI CALL] Model: {self.model} (stage: {stage or 'generic'})")

        try:
            completion = client.chat.completions.create(**call_kwargs)
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI Authentication error: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI Rate limit exceeded: {e}") from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI request timed out: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(f"OpenAI connection error: {e}") from e
        except openai.NotFoundError as e:
            raise LLMModelNotFoundError(f"OpenAI model '{self.model}' not found: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error calling OpenAI: {e}") from e

        text = completion.choices[0].message.content or ""
        usage_obj = completion.usage
        prompt_tokens = usage_obj.prompt_tokens if usage_obj else 0
        completion_tokens = usage_obj.completion_tokens if usage_obj else 0
        total_tokens = usage_obj.total_tokens if usage_obj else (prompt_tokens + completion_tokens)

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=text,
            usage=usage,
            model=self.model,
            provider=self.provider_name,
            stage=stage,
            raw_response=completion,
        )
