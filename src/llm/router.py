"""
Centralized stage-to-model router for CineLearn.
Resolves and executes model providers per pipeline stage, handling fallbacks transparently.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from .types import Stage, ModelConfig, LLMResponse, Capability
from .config import LLMConfigManager
from .base import BaseLLMProvider
from .providers import create_provider
from .exceptions import (
    LLMError,
    LLMConfigurationError,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Centralized router that connects each generation stage to its configured model/provider.
    """

    def __init__(self, config_manager: Optional[LLMConfigManager] = None):
        self.config_manager = config_manager or LLMConfigManager()
        self._provider_cache: Dict[str, BaseLLMProvider] = {}
        self._override_providers: Dict[Stage, BaseLLMProvider] = {}

    def get_stage_config(self, stage: Union[str, Stage]) -> ModelConfig:
        """Resolve ModelConfig for the given stage."""
        return self.config_manager.get_stage_config(stage)

    def set_stage_provider(self, stage: Union[str, Stage], provider: BaseLLMProvider) -> None:
        """Explicitly override the provider instance for a stage (useful for testing & dependency injection)."""
        norm_stage = Stage.normalize(stage)
        self._override_providers[norm_stage] = provider

    def clear_overrides(self) -> None:
        """Clear all stage provider overrides."""
        self._override_providers.clear()
        self._provider_cache.clear()

    def get_provider_for_stage(
        self,
        stage: Optional[Union[str, Stage]] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> BaseLLMProvider:
        """Resolves or instantiates provider adapter for a specific stage."""
        norm_stage = Stage.normalize(stage)

        # Check explicit instance override first
        if not provider_override and not model_override and norm_stage in self._override_providers:
            return self._override_providers[norm_stage]

        stage_cfg = self.config_manager.get_stage_config(norm_stage)

        prov = (provider_override or stage_cfg.provider).lower()
        mod = model_override or stage_cfg.model
        base_url = stage_cfg.base_url if (not provider_override or provider_override.lower() == stage_cfg.provider.lower()) else None

        cache_key = f"{prov}::{mod}::{base_url}"
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        # Clone config with any overrides
        effective_cfg = ModelConfig(
            provider=prov,
            model=mod,
            temperature=stage_cfg.temperature,
            max_tokens=stage_cfg.max_tokens,
            timeout=stage_cfg.timeout,
            api_key=stage_cfg.api_key if (not provider_override or provider_override.lower() == stage_cfg.provider.lower()) else None,
            base_url=base_url,
            fallback_provider=stage_cfg.fallback_provider,
            fallback_model=stage_cfg.fallback_model,
        )

        provider_instance = create_provider(effective_cfg)
        self._provider_cache[cache_key] = provider_instance
        return provider_instance

    def execute_for_stage(
        self,
        stage: Optional[Union[str, Stage]],
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Executes an LLM call for a designated pipeline stage, using stage-specific
        routing and explicit configurable fallback if primary execution fails.
        """
        norm_stage = Stage.normalize(stage)
        stage_name = norm_stage.value

        primary_provider = self.get_provider_for_stage(
            stage=norm_stage,
            provider_override=provider_override,
            model_override=model_override,
        )

        try:
            return primary_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                video_path=video_path,
                image_path=image_path,
                stage=stage_name,
                **kwargs,
            )
        except UnsupportedCapabilityError:
            # Do not attempt fallback on architectural capability mismatches unless explicit fallback configured
            stage_cfg = self.config_manager.get_stage_config(norm_stage)
            if not stage_cfg.fallback_provider and not stage_cfg.fallback_model:
                raise
            return self._attempt_fallback(
                stage=norm_stage,
                primary_provider=primary_provider,
                original_error="Unsupported capability",
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                video_path=video_path,
                image_path=image_path,
                **kwargs,
            )
        except Exception as e:
            stage_cfg = self.config_manager.get_stage_config(norm_stage)
            fb_prov = stage_cfg.fallback_provider or getattr(primary_provider.config, "fallback_provider", None)
            fb_mod = stage_cfg.fallback_model or getattr(primary_provider.config, "fallback_model", None)
            if fb_prov or fb_mod:
                return self._attempt_fallback(
                    stage=norm_stage,
                    primary_provider=primary_provider,
                    original_error=str(e),
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    video_path=video_path,
                    image_path=image_path,
                    fallback_provider_override=fb_prov,
                    fallback_model_override=fb_mod,
                    **kwargs,
                )
            raise

    def _attempt_fallback(
        self,
        stage: Stage,
        primary_provider: BaseLLMProvider,
        original_error: str,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str],
        response_format: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        timeout: Optional[float],
        video_path: Optional[str],
        image_path: Optional[str],
        fallback_provider_override: Optional[str] = None,
        fallback_model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Executes explicit configurable fallback."""
        stage_cfg = self.config_manager.get_stage_config(stage)
        fb_provider_name = (
            fallback_provider_override
            or stage_cfg.fallback_provider
            or getattr(primary_provider.config, "fallback_provider", None)
            or primary_provider.provider_name
        )
        fb_model_name = (
            fallback_model_override
            or stage_cfg.fallback_model
            or getattr(primary_provider.config, "fallback_model", None)
            or self.config_manager.get_global_model(fb_provider_name)
        )

        print(
            f"\n🔄 [EXPLICIT STAGE FALLBACK] Stage '{stage.value}': primary '{primary_provider.provider_name}/{primary_provider.model}' "
            f"failed ({original_error}). Routing to fallback: '{fb_provider_name}/{fb_model_name}'"
        )

        fallback_provider = self.get_provider_for_stage(
            stage=stage,
            provider_override=fb_provider_name,
            model_override=fb_model_name,
        )

        return fallback_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            video_path=video_path,
            image_path=image_path,
            stage=stage.value,
            **kwargs,
        )


# Global default router instance
_DEFAULT_ROUTER = LLMRouter()


def get_router() -> LLMRouter:
    """Access the global LLMRouter instance."""
    return _DEFAULT_ROUTER
