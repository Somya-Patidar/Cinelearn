"""
Unified entry points and public functions for the CineLearn LLM abstraction layer.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from .types import Stage, ModelConfig, LLMResponse, Capability
from .router import get_router, LLMRouter
from .base import BaseLLMProvider


def generate_response(
    prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    stage: Optional[Union[str, Stage]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> LLMResponse:
    """
    Centralized unified interface handling all LLM calls across all CineLearn components.

    Args:
        prompt: User prompt string or list of message dictionaries.
        system_prompt: Optional system-level prompt/instructions.
        response_format: Optional format specification ('json' for structured JSON).
        stage: Pipeline generation stage ('outline', 'storyboard', 'code_generation',
               'verification', 'repair', 'evaluation'). Resolves stage-configured model.
        provider: Explicit provider override ('gemini', 'ollama', 'openai', 'mock').
        model: Explicit model name override.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        video_path: Path to video file for multimodal video understanding.
        image_path: Path to image file for visual reference.
        **kwargs: Additional model-specific parameters.

    Returns:
        LLMResponse: Normalized response with .text, .usage, .choices, .candidates.
    """
    router = get_router()
    return router.execute_for_stage(
        stage=stage,
        prompt=prompt,
        system_prompt=system_prompt,
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        video_path=video_path,
        image_path=image_path,
        provider_override=provider,
        model_override=model,
        **kwargs,
    )


def generate_response_token(
    prompt: Union[str, List[Dict[str, Any]]],
    stage: Optional[Union[str, Stage]] = None,
    max_tokens: Optional[int] = None,
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[LLMResponse, Dict[str, int]]:
    """
    Helper function returning (response_wrapper, usage_dict) for backwards compatibility
    with components expecting tuple return values.
    """
    resp = generate_response(
        prompt=prompt,
        stage=stage,
        max_tokens=max_tokens,
        video_path=video_path,
        image_path=image_path,
        **kwargs,
    )
    return resp, resp.usage_dict


def get_llm(
    stage: Optional[Union[str, Stage]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """Get the configured BaseLLMProvider for a given stage or override."""
    return get_router().get_provider_for_stage(
        stage=stage,
        provider_override=provider,
        model_override=model,
    )


def make_stage_caller(stage: Union[str, Stage]) -> Callable[..., Tuple[LLMResponse, Dict[str, int]]]:
    """
    Returns a callable bounded to a specific stage that returns (response, usage_dict),
    making it a 100% drop-in replacement for any legacy API function.
    """
    def _caller(prompt: str, max_tokens: int = 8000, **kwargs: Any) -> Tuple[LLMResponse, Dict[str, int]]:
        call_stage = kwargs.pop("stage", stage)
        return generate_response_token(
            prompt=prompt,
            stage=call_stage,
            max_tokens=max_tokens,
            **kwargs,
        )
    return _caller
