"""
Types, enums, dataclasses, and normalized response structures for CineLearn LLM layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple


class Stage(str, Enum):
    """Pipeline generation stages in CineLearn."""
    OUTLINE = "outline"
    STORYBOARD = "storyboard"
    CODE_GENERATION = "code_generation"
    VERIFICATION = "verification"
    REPAIR = "repair"
    EVALUATION = "evaluation"
    DEFAULT = "default"

    @classmethod
    def normalize(cls, name: Optional[Union[str, "Stage"]]) -> "Stage":
        """Normalize stage names and common aliases."""
        if isinstance(name, Stage):
            return name
        if not name:
            return cls.DEFAULT
        clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "outline": cls.OUTLINE,
            "stage1": cls.OUTLINE,
            "stage_1": cls.OUTLINE,
            "storyboard": cls.STORYBOARD,
            "stage2": cls.STORYBOARD,
            "stage_2": cls.STORYBOARD,
            "code": cls.CODE_GENERATION,
            "code_gen": cls.CODE_GENERATION,
            "code_generation": cls.CODE_GENERATION,
            "stage3": cls.CODE_GENERATION,
            "stage_3": cls.CODE_GENERATION,
            "critic": cls.VERIFICATION,
            "layout": cls.VERIFICATION,
            "layout_feedback": cls.VERIFICATION,
            "verification": cls.VERIFICATION,
            "stage4": cls.VERIFICATION,
            "stage_4": cls.VERIFICATION,
            "repair": cls.REPAIR,
            "fix": cls.REPAIR,
            "scope_refine": cls.REPAIR,
            "scoperefine": cls.REPAIR,
            "debug": cls.REPAIR,
            "evaluation": cls.EVALUATION,
            "eval": cls.EVALUATION,
            "eval_tq": cls.EVALUATION,
            "eval_aes": cls.EVALUATION,
            "default": cls.DEFAULT,
        }
        return aliases.get(clean, cls.DEFAULT)


class Capability(str, Enum):
    """Specific capabilities supported by LLM providers/models."""
    TEXT = "text"
    SYSTEM_PROMPT = "system_prompt"
    JSON_MODE = "json_mode"
    IMAGE_INPUT = "image_input"
    VIDEO_INPUT = "video_input"
    STREAMING = "streaming"


class ProviderType(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    MOCK = "mock"


@dataclass
class LLMUsage:
    """Normalized token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def get(self, key: str, default: Any = 0) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, item: str) -> int:
        return getattr(self, item)


@dataclass
class ModelConfig:
    """Configuration settings for a single model or stage."""
    provider: str = ProviderType.GEMINI.value
    model: str = "gemini-3.8-flash"
    temperature: float = 0.7
    max_tokens: int = 8000
    timeout: float = 120.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class LLMResponse:
    """
    Unified normalized response object that guarantees complete compatibility with:
    - response.text (Standard string content)
    - response.content (OpenAI-style content attribute)
    - response.choices[0].message.content (OpenAI format)
    - response.candidates[0].content.parts[0].text (Google GenAI format)
    - response.usage (Token counts)
    - str(response)
    - Tuple unpacking: response, usage = api(...)
    """

    def __init__(
        self,
        text: str,
        usage: Optional[Union[LLMUsage, Dict[str, int]]] = None,
        model: str = "",
        provider: str = "",
        stage: Optional[str] = None,
        raw_response: Any = None,
    ):
        self._text = text if text is not None else ""
        if isinstance(usage, LLMUsage):
            self._usage = usage
        elif isinstance(usage, dict):
            self._usage = LLMUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        else:
            self._usage = LLMUsage()

        self._model = model
        self._provider = provider
        self._stage = stage
        self._raw_response = raw_response

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> str:
        return self._text

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def stage(self) -> Optional[str]:
        return self._stage

    @property
    def raw_response(self) -> Any:
        return self._raw_response

    @property
    def usage(self) -> LLMUsage:
        return self._usage

    @property
    def usage_dict(self) -> Dict[str, int]:
        return self._usage.to_dict()

    @property
    def choices(self):
        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, msg):
                self.message = msg

        return [_Choice(_Msg(self._text))]

    @property
    def candidates(self):
        class _Part:
            def __init__(self, text):
                self.text = text

        class _Content:
            def __init__(self, parts):
                self.parts = parts

        class _Candidate:
            def __init__(self, content):
                self.content = content

        return [_Candidate(_Content([_Part(self._text)]))]

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"<LLMResponse provider={self._provider} model={self._model} text_length={len(self._text)} tokens={self._usage.total_tokens}>"

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking: response, usage = result"""
        yield self
        yield self.usage_dict
