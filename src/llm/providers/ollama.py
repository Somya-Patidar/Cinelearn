"""
Ollama local model provider adapter implementing BaseLLMProvider.
Supports open-weight models (Qwen, Llama, etc.) running locally via Ollama.
"""

import os
import json
import threading
from typing import Any, Dict, List, Optional, Set, Union
import requests

from ..base import BaseLLMProvider
from ..types import Capability, LLMResponse, LLMUsage, ModelConfig
from ..exceptions import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMResponseFormatError,
    LLMError,
    UnsupportedCapabilityError,
)


class OllamaProvider(BaseLLMProvider):
    """Adapter for local models running via Ollama."""
    _call_lock = threading.Lock()

    @staticmethod
    def _detect_wsl_host_ip() -> Optional[str]:
        """Detects the Windows host IP address from within WSL2."""
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" not in f.read().lower():
                    return None
            import subprocess
            res = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout:
                parts = res.stdout.strip().split()
                if "via" in parts:
                    return parts[parts.index("via") + 1]
        except Exception:
            pass
        return None

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        raw_url = (
            config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11435"
        ).rstrip("/")

        # Enforce isolated personal instance port 11435; never probe shared/default port 11434
        if ":11434" in raw_url:
            raw_url = raw_url.replace(":11434", ":11435")

        self.base_url = raw_url
        self.model = config.model or os.getenv("OLLAMA_MODEL") or "qwen2.5-coder:7b"

    @property
    def supported_capabilities(self) -> Set[Capability]:
        return {
            Capability.TEXT,
            Capability.SYSTEM_PROMPT,
            Capability.JSON_MODE,
        }

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
        # Multimodal capability check
        if video_path:
            self.check_capability(Capability.VIDEO_INPUT, f"stage '{stage or 'video'}'")
        if image_path:
            self.check_capability(Capability.IMAGE_INPUT, f"stage '{stage or 'image'}'")

        req_timeout = timeout or self.config.timeout or 120.0
        temp = temperature if temperature is not None else self.config.temperature
        max_toks = max_tokens or self.config.max_tokens or 8000
        num_ctx = max(int(os.getenv("OLLAMA_NUM_CTX", "16384")), max_toks + 2048)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, dict):
                    messages.append({
                        "role": item.get("role", "user"),
                        "content": str(item.get("content", "")),
                    })
                else:
                    messages.append({"role": "user", "content": str(item)})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_toks,
                "num_ctx": num_ctx,
            },
        }

        if response_format == "json":
            payload["format"] = "json"

        endpoint = f"{self.base_url}/api/chat"
        print(f"🦙 [OLLAMA CALL] Model: {self.model} -> {endpoint} (stage: {stage or 'generic'})")

        try:
            with self._call_lock:
                resp = requests.post(endpoint, json=payload, timeout=req_timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to local Ollama server at '{self.base_url}'.\n"
                f"Is Ollama running? To start it, run: 'ollama serve'\n"
                f"Original error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(
                f"Ollama request to '{self.model}' timed out after {req_timeout:.1f}s."
            ) from e
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Network error communicating with Ollama: {e}") from e

        if resp.status_code == 404:
            raise LLMModelNotFoundError(
                f"Model '{self.model}' not found on Ollama server at '{self.base_url}'.\n"
                f"Please download the model by running:\n"
                f"  ollama pull {self.model}"
            )
        elif resp.status_code != 200:
            raise LLMError(
                f"Ollama returned HTTP error {resp.status_code}: {resp.text}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise LLMResponseFormatError(
                f"Failed to parse JSON response from Ollama: {e}\nResponse body: {resp.text[:300]}"
            ) from e

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=content,
            usage=usage,
            model=self.model,
            provider=self.provider_name,
            stage=stage,
            raw_response=data,
        )
