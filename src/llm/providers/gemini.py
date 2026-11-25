"""
Gemini provider adapter implementing BaseLLMProvider.
Integrates with QuotaManager, rate limiting, and multimodal capabilities.
"""

import os
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from ..base import BaseLLMProvider
from ..types import Capability, LLMResponse, LLMUsage, ModelConfig
from ..exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMQuotaExhaustedError,
    LLMServiceUnavailableError,
    LLMError,
)

# Import QuotaManager and helpers from gpt_request
import sys
_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not _GENAI_AVAILABLE:
            raise LLMConfigurationError("The 'google-genai' library is required to use the Gemini provider.")

        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.strip() in ("", "...", "YOUR_GEMINI_KEY", "your_gemini_api_key_here"):
            raise LLMAuthenticationError(
                "GEMINI_API_KEY is not configured in environment or .env!\n"
                "Please configure GEMINI_API_KEY to use the Gemini provider."
            )
        self.api_key = api_key.strip()
        self._client: Optional[genai.Client] = None

    @property
    def supported_capabilities(self) -> Set[Capability]:
        return {
            Capability.TEXT,
            Capability.SYSTEM_PROMPT,
            Capability.JSON_MODE,
            Capability.IMAGE_INPUT,
            Capability.VIDEO_INPUT,
        }

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
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
        import gpt_request

        max_toks = max_tokens or self.config.max_tokens or 8000
        client = self._get_client()
        qm = gpt_request.QuotaManager()
        model_name = self.model or gpt_request._get_gemini_model()

        # Build config
        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_toks}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        elif self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if response_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        cfg = types.GenerateContentConfig(**config_kwargs)

        # Handle multimodal contents
        contents = []
        extra_stat = ""
        if isinstance(prompt, str):
            contents.append(prompt)
        elif isinstance(prompt, list):
            # If prompt is a list of chat dicts or parts
            for item in prompt:
                if isinstance(item, dict):
                    contents.append(item.get("content", ""))
                else:
                    contents.append(str(item))

        if video_path:
            self.check_capability(Capability.VIDEO_INPUT, "video analysis")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            vstat = gpt_request._get_stable_file_stat(video_path)
            extra_stat = f"{extra_stat}:{vstat}" if extra_stat else vstat
            with open(video_path, "rb") as vf:
                vbytes = vf.read()
            contents.append(types.Part.from_bytes(data=vbytes, mime_type="video/mp4"))

        if image_path:
            self.check_capability(Capability.IMAGE_INPUT, "image analysis")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            istat = gpt_request._get_stable_file_stat(image_path)
            extra_stat = f"{extra_stat}:{istat}" if extra_stat else istat
            with open(image_path, "rb") as imf:
                ibytes = imf.read()
            contents.append(types.Part.from_bytes(data=ibytes, mime_type="image/png"))

        cache_prompt = prompt if isinstance(prompt, str) else str(prompt)
        cache_key = qm.get_cache_key(cache_prompt, model_name, extra=extra_stat if extra_stat else None)

        # 1. Check cache first
        cached = qm.get_cached(cache_key)
        if cached:
            text, usage_dict = cached
            print(f"[CACHE HIT] Returning cached Gemini response ({usage_dict.get('total_tokens', 0)} tokens)")
            return LLMResponse(
                text=text,
                usage=usage_dict,
                model=model_name,
                provider=self.provider_name,
                stage=stage,
            )

        # 2. Acquire Quota and call API
        retry_state = gpt_request.GeminiRetryState(max_rpm_retries=1, max_503_retries=4, max_other_retries=1)
        max_retries = kwargs.get("max_retries", 2)
        op_name = f"{stage or 'gemini'} request"

        while retry_state.can_continue():
            try:
                with qm.api_execution_lock:
                    cached = qm.get_cached(cache_key)
                    if cached:
                        text, usage_dict = cached
                        return LLMResponse(
                            text=text,
                            usage=usage_dict,
                            model=model_name,
                            provider=self.provider_name,
                            stage=stage,
                        )

                    qm.acquire()
                    qm.actual_calls_count += 1
                    daily_count = qm.increment_daily_requests_count()
                    rpd_info = f", daily: {daily_count}/{qm.effective_rpd}" if qm.effective_rpd else f", daily: {daily_count}"
                    print(f"📡 [GEMINI API CALL #{qm.actual_calls_count}{rpd_info}] Model: {model_name} (interval >= {qm.min_interval:.1f}s)")

                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=cfg,
                    )
                    wrapper, usage_dict = gpt_request._extract_text_and_usage(response)
                    qm.set_cached(cache_key, wrapper.text, usage_dict)
                    return LLMResponse(
                        text=wrapper.text,
                        usage=usage_dict,
                        model=model_name,
                        provider=self.provider_name,
                        stage=stage,
                        raw_response=response,
                    )

            except gpt_request.DailyQuotaExhaustedError as e:
                raise LLMQuotaExhaustedError(str(e)) from e
            except gpt_request.ServiceUnavailableError as e:
                raise LLMServiceUnavailableError(str(e)) from e
            except PermissionError as e:
                raise LLMAuthenticationError(str(e)) from e
            except Exception as e:
                try:
                    gpt_request._handle_gemini_error(e, qm, retry_state, max_retries, op_name=op_name)
                except Exception as final_e:
                    err_str = str(final_e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        raise LLMRateLimitError(str(final_e)) from final_e
                    raise LLMError(str(final_e)) from final_e

        raise LLMError(f"Gemini request failed after {max_retries} attempts.")
