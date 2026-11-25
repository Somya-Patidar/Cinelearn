"""
Mock LLM provider adapter for testing without network calls or API keys.
"""

from typing import Any, Dict, List, Optional, Set, Union, Callable

from ..base import BaseLLMProvider
from ..types import Capability, LLMResponse, LLMUsage, ModelConfig, Stage


class MockProvider(BaseLLMProvider):
    """Mock provider for deterministic testing of model-agnostic architecture."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        default_response: str = "Mock LLM output",
        canned_responses: Optional[Dict[str, str]] = None,
        simulate_error: Optional[Exception] = None,
        capabilities: Optional[Set[Capability]] = None,
        response_generator: Optional[Callable[..., str]] = None,
    ):
        cfg = config or ModelConfig(provider="mock", model="mock-model")
        super().__init__(cfg)
        self.default_response = default_response
        self.canned_responses = canned_responses or {}
        self.simulate_error = simulate_error
        self._capabilities = capabilities or {
            Capability.TEXT,
            Capability.SYSTEM_PROMPT,
            Capability.JSON_MODE,
            Capability.IMAGE_INPUT,
            Capability.VIDEO_INPUT,
        }
        self.response_generator = response_generator
        self.calls: List[Dict[str, Any]] = []

    @property
    def supported_capabilities(self) -> Set[Capability]:
        return self._capabilities

    def set_error(self, error: Optional[Exception]):
        self.simulate_error = error

    def set_canned_response(self, stage_or_key: str, response: str):
        self.canned_responses[stage_or_key] = response

    def clear_calls(self):
        self.calls.clear()

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
        # Check capabilities
        if video_path:
            self.check_capability(Capability.VIDEO_INPUT, f"stage '{stage or 'video'}'")
        if image_path:
            self.check_capability(Capability.IMAGE_INPUT, f"stage '{stage or 'image'}'")

        call_record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response_format": response_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
            "video_path": video_path,
            "image_path": image_path,
            "stage": stage,
            "model": self.model,
            "provider": self.provider_name,
            "kwargs": kwargs,
        }
        self.calls.append(call_record)

        if self.simulate_error:
            raise self.simulate_error

        # Determine response
        if stage and stage in self.canned_responses:
            text = self.canned_responses[stage]
        elif self.response_generator:
            text = self.response_generator(prompt=prompt, stage=stage, **kwargs)
        elif self.default_response and self.default_response != "Mock LLM output":
            text = self.default_response
        elif stage == Stage.OUTLINE.value:
            text = (
                '{"topic": "Test Topic", "target_audience": "Students", '
                '"learning_level": "intermediate", "analogy_used": "Test Analogy", '
                '"sections": [{"id": "section_1", "title": "Intro"}]}'
            )
        elif stage == Stage.STORYBOARD.value:
            text = (
                '{"sections": [{"id": "section_1", "title": "Intro", '
                '"lecture_lines": ["Welcome to the lesson."], "animations": ["Show title"]}]}'
            )
        elif stage == Stage.CODE_GENERATION.value:
            text = (
                "```python\n"
                "from manim import *\n"
                "class Section1Scene(TeachingScene):\n"
                "    def construct(self):\n"
                "        self.play(FadeIn(Text('Hello')))\n"
                "```"
            )
        elif stage == Stage.VERIFICATION.value:
            text = '{"layout": {"has_issues": false, "improvements": []}}'
        else:
            text = self.default_response

        usage = LLMUsage(
            prompt_tokens=len(str(prompt)) // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(len(str(prompt)) + len(text)) // 4,
        )

        return LLMResponse(
            text=text,
            usage=usage,
            model=self.model,
            provider=self.provider_name,
            stage=stage,
            raw_response={"mock": True},
        )
