"""
Configuration loader and stage-to-model configuration management for CineLearn.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

from .types import Stage, ProviderType, ModelConfig
from .exceptions import LLMConfigurationError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = Path(__file__).resolve().parent.parent

# Load environment files
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_SRC_DIR / ".env")
load_dotenv()

# Optional PyYAML support if available
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class LLMConfigManager:
    """Manages global and stage-specific LLM configuration."""

    DEFAULT_MODELS = {
        ProviderType.GEMINI.value: "gemini-3.8-flash",
        ProviderType.OLLAMA.value: "qwen2.5-coder:7b",
        ProviderType.OPENAI.value: "gpt-4o",
        ProviderType.MOCK.value: "mock-model",
    }

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = self._resolve_config_file(config_file)
        self.file_config = self._load_config_file(self.config_file)
        self._cached_stage_configs: Dict[Stage, ModelConfig] = {}

    def _resolve_config_file(self, path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Locates configuration file from argument, environment, or default locations."""
        if path:
            p = Path(path)
            if p.exists():
                return p
        env_path = os.getenv("LLM_CONFIG_FILE")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p
        candidate_paths = [
            _PROJECT_ROOT / "llm_config.yaml",
            _PROJECT_ROOT / "llm_config.json",
            _SRC_DIR / "llm_config.yaml",
            _SRC_DIR / "llm_config.json",
            _SRC_DIR / "api_config.json",
        ]
        for p in candidate_paths:
            if p.exists() and p.name != "api_config.json":
                return p
        return None

    def _load_config_file(self, path: Optional[Path]) -> Dict[str, Any]:
        """Loads YAML or JSON configuration file."""
        if not path or not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                if (path.suffix.lower() in (".yaml", ".yml")) and _YAML_AVAILABLE:
                    return yaml.safe_load(f) or {}
                return json.load(f) or {}
        except Exception as e:
            print(f"⚠️ Warning: Failed to parse LLM config file at {path}: {e}")
            return {}

    def reload(self):
        """Reload configuration from disk and environment."""
        self.file_config = self._load_config_file(self.config_file)
        self._cached_stage_configs.clear()

    def get_global_provider(self) -> str:
        """Determines default global provider from environment or config."""
        env_prov = os.getenv("LLM_PROVIDER")
        if env_prov:
            return env_prov.strip().lower()

        file_prov = self.file_config.get("default", {}).get("provider")
        if file_prov:
            return str(file_prov).strip().lower()

        # Fallback heuristic based on available keys
        if os.getenv("GEMINI_API_KEY"):
            return ProviderType.GEMINI.value
        if os.getenv("OLLAMA_BASE_URL"):
            return ProviderType.OLLAMA.value
        if os.getenv("OPENAI_API_KEY"):
            return ProviderType.OPENAI.value

        return ProviderType.GEMINI.value

    def get_global_model(self, provider: Optional[str] = None) -> str:
        """Determines default global model for the given provider."""
        prov = (provider or self.get_global_provider()).lower()
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            return env_model.strip()

        file_model = self.file_config.get("default", {}).get("model")
        if file_model:
            return str(file_model).strip()

        # Provider-specific environment variable legacy fallbacks
        if prov == ProviderType.GEMINI.value:
            gem_model = os.getenv("GEMINI_MODEL")
            if gem_model and gem_model.strip() not in ("", "...", "YOUR_GEMINI_MODEL"):
                return gem_model.strip()
        elif prov == ProviderType.OLLAMA.value:
            oll_model = os.getenv("OLLAMA_MODEL")
            if oll_model and oll_model.strip():
                return oll_model.strip()
        elif prov == ProviderType.OPENAI.value:
            oai_model = os.getenv("OPENAI_MODEL")
            if oai_model and oai_model.strip():
                return oai_model.strip()

        return self.DEFAULT_MODELS.get(prov, "gemini-3.8-flash")

    def get_stage_config(self, stage: Union[str, Stage]) -> ModelConfig:
        """
        Resolves model configuration for a specific stage using hierarchical resolution:
        1. Stage-specific environment variables (STAGE_<STAGE>_PROVIDER, STAGE_<STAGE>_MODEL)
        2. Config file stage entry (stages.<stage>...)
        3. Global environment variables (LLM_PROVIDER, LLM_MODEL)
        4. Config file default entry (default...)
        5. Built-in defaults
        """
        normalized_stage = Stage.normalize(stage)
        stage_key = normalized_stage.value
        stage_env_prefix = f"STAGE_{stage_key.upper()}_"

        # 1. Determine Provider
        stage_prov_env = os.getenv(f"{stage_env_prefix}PROVIDER")
        stage_file_cfg = self.file_config.get("stages", {}).get(stage_key, {})
        default_file_cfg = self.file_config.get("default", {})
        provider = (
            stage_prov_env
            or stage_file_cfg.get("provider")
            or (default_file_cfg.get("provider") if stage_key == "default" else None)
            or self.get_global_provider()
        ).strip().lower()

        # 2. Determine Model
        stage_model_env = os.getenv(f"{stage_env_prefix}MODEL")
        model = (
            stage_model_env
            or stage_file_cfg.get("model")
            or (default_file_cfg.get("model") if stage_key == "default" else None)
            or (self.get_global_model(provider) if (not stage_prov_env and not stage_file_cfg.get("provider")) else None)
        )
        if not model:
            # If provider was changed per-stage, resolve provider's default model
            model = self.get_global_model(provider)
        model = model.strip()

        # 3. Parameters (temperature, max_tokens, timeout)
        temp_env = os.getenv(f"{stage_env_prefix}TEMPERATURE") or os.getenv("LLM_TEMPERATURE")
        temperature = float(temp_env) if temp_env else float(stage_file_cfg.get("temperature") or default_file_cfg.get("temperature", 0.7))

        max_tokens_env = os.getenv(f"{stage_env_prefix}MAX_TOKENS") or os.getenv("LLM_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else int(stage_file_cfg.get("max_tokens") or default_file_cfg.get("max_tokens", 8000))

        timeout_env = os.getenv(f"{stage_env_prefix}TIMEOUT") or os.getenv("LLM_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else float(stage_file_cfg.get("timeout") or default_file_cfg.get("timeout", 120.0))

        # 4. Fallbacks
        fb_prov_env = os.getenv(f"{stage_env_prefix}FALLBACK_PROVIDER") or os.getenv("LLM_FALLBACK_PROVIDER")
        fallback_provider = fb_prov_env or stage_file_cfg.get("fallback_provider")
        if fallback_provider:
            fallback_provider = fallback_provider.strip().lower()

        fb_model_env = os.getenv(f"{stage_env_prefix}FALLBACK_MODEL") or os.getenv("LLM_FALLBACK_MODEL")
        fallback_model = fb_model_env or stage_file_cfg.get("fallback_model")
        if fallback_model:
            fallback_model = fallback_model.strip()

        # 5. Credentials & URLs
        api_key = None
        base_url = None
        if provider == ProviderType.GEMINI.value:
            api_key = os.getenv("GEMINI_API_KEY")
        elif provider == ProviderType.OLLAMA.value:
            base_url = (
                os.getenv(f"{stage_env_prefix}BASE_URL")
                or os.getenv("OLLAMA_BASE_URL")
                or stage_file_cfg.get("base_url")
                or self.file_config.get("default", {}).get("base_url")
                or "http://127.0.0.1:11435"
            )
        elif provider == ProviderType.OPENAI.value:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        return ModelConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

    def validate_stage_config(self, stage: Union[str, Stage]) -> None:
        """Validates configuration for a stage, raising LLMConfigurationError if invalid."""
        cfg = self.get_stage_config(stage)
        valid_providers = [p.value for p in ProviderType]
        if cfg.provider not in valid_providers:
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{cfg.provider}' configured for stage '{stage}'. "
                f"Supported providers: {', '.join(valid_providers)}"
            )

        if cfg.provider == ProviderType.GEMINI.value:
            if not cfg.api_key or cfg.api_key.strip() in ("", "...", "YOUR_GEMINI_KEY", "your_gemini_api_key_here"):
                raise LLMConfigurationError(
                    f"Gemini API key is missing for stage '{stage}'. Please configure GEMINI_API_KEY in .env."
                )

        if cfg.provider == ProviderType.OPENAI.value:
            if not cfg.api_key or cfg.api_key.strip() in ("", "...", "YOUR_OPENAI_KEY"):
                raise LLMConfigurationError(
                    f"OpenAI API key is missing for stage '{stage}'. Please configure OPENAI_API_KEY in .env."
                )
