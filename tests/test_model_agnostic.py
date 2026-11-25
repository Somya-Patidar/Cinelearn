"""
Tests for Model-Agnostic LLM layer in CineLearn.
Covers:
- Common LLM interface and response normalization
- Provider selection via configuration (.env and config files)
- Stage-specific model routing and fallback behavior
- Graceful error handling (Ollama down, model not found, timeout, unsupported capability)
- Pipeline integration (outline, storyboard, code generation, verification)
"""

import os
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.llm import (
    Stage,
    Capability,
    ProviderType,
    LLMResponse,
    LLMUsage,
    ModelConfig,
    LLMError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    UnsupportedCapabilityError,
    LLMRateLimitError,
    generate_response,
    generate_response_token,
    get_llm,
    get_router,
    make_stage_caller,
    LLMConfigManager,
    LLMRouter,
    MockProvider,
    OllamaProvider,
)
from src.agent import TeachingVideoAgent, RunConfig, Section, TeachingOutline, build_and_parse_args, get_api_and_output


class TestUnifiedLLMInterface(unittest.TestCase):
    def setUp(self):
        get_router().clear_overrides()

    def tearDown(self):
        get_router().clear_overrides()

    def test_response_normalization_properties(self):
        """Verify LLMResponse provides consistent OpenAI, Gemini, and string access."""
        resp = LLMResponse(
            text='{"result": "success"}',
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
            provider="mock",
            stage="outline",
        )
        self.assertEqual(resp.text, '{"result": "success"}')
        self.assertEqual(resp.content, '{"result": "success"}')
        self.assertEqual(str(resp), '{"result": "success"}')
        self.assertEqual(resp.model, "test-model")
        self.assertEqual(resp.provider, "mock")
        self.assertEqual(resp.stage, "outline")

        # OpenAI format compatibility
        self.assertEqual(resp.choices[0].message.content, '{"result": "success"}')

        # Gemini format compatibility
        self.assertEqual(resp.candidates[0].content.parts[0].text, '{"result": "success"}')

        # Token usage compatibility
        self.assertEqual(resp.usage.prompt_tokens, 10)
        self.assertEqual(resp.usage.completion_tokens, 20)
        self.assertEqual(resp.usage.total_tokens, 30)
        self.assertEqual(resp.usage_dict["total_tokens"], 30)

    def test_tuple_unpacking_compatibility(self):
        """Verify LLMResponse can be unpacked as (response, usage_dict) for legacy functions."""
        resp = LLMResponse(text="Hello", usage={"total_tokens": 15}, model="m", provider="p")
        unpacked_resp, unpacked_usage = resp
        self.assertEqual(unpacked_resp.text, "Hello")
        self.assertEqual(unpacked_usage["total_tokens"], 15)

    def test_generate_response_with_mock_provider(self):
        """Verify generate_response dispatches properly and captures parameters."""
        mock = MockProvider(default_response="Custom mock output")
        get_router().set_stage_provider("outline", mock)

        result = generate_response(
            prompt="Explain eigenvalues",
            system_prompt="You are a math tutor.",
            response_format="json",
            stage="outline",
            temperature=0.3,
        )

        self.assertEqual(result.text, "Custom mock output")
        self.assertEqual(len(mock.calls), 1)
        call = mock.calls[0]
        self.assertEqual(call["prompt"], "Explain eigenvalues")
        self.assertEqual(call["system_prompt"], "You are a math tutor.")
        self.assertEqual(call["response_format"], "json")
        self.assertEqual(call["temperature"], 0.3)
        self.assertEqual(call["stage"], "outline")


class TestProviderConfiguration(unittest.TestCase):
    def setUp(self):
        self.orig_env = dict(os.environ)
        get_router().clear_overrides()
        self._empty_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self._empty_cfg.write("{}")
        self._empty_cfg.close()
        os.environ["LLM_CONFIG_FILE"] = self._empty_cfg.name
        for k in list(os.environ.keys()):
            if k.startswith("STAGE_"):
                del os.environ[k]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        get_router().clear_overrides()
        if hasattr(self, "_empty_cfg") and os.path.exists(self._empty_cfg.name):
            try:
                os.remove(self._empty_cfg.name)
            except Exception:
                pass

    def test_global_provider_selection_via_env(self):
        """Verify global provider and model are determined by LLM_PROVIDER and LLM_MODEL."""
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_MODEL"] = "qwen2.5:7b"

        cfg_mgr = LLMConfigManager()
        self.assertEqual(cfg_mgr.get_global_provider(), "ollama")
        self.assertEqual(cfg_mgr.get_global_model(), "qwen2.5:7b")

        stage_cfg = cfg_mgr.get_stage_config("storyboard")
        self.assertEqual(stage_cfg.provider, "ollama")
        self.assertEqual(stage_cfg.model, "qwen2.5:7b")

    def test_yaml_config_file_stage_resolution(self):
        """Verify stage configurations are resolved from a YAML/JSON configuration file."""
        config_data = {
            "default": {"provider": "gemini", "model": "gemini-3.8-flash"},
            "stages": {
                "outline": {"provider": "gemini", "model": "gemini-3.8-flash"},
                "storyboard": {"provider": "ollama", "model": "qwen2.5:7b"},
                "code_generation": {"provider": "ollama", "model": "qwen2.5-coder:7b"},
                "verification": {"provider": "gemini", "model": "gemini-3.8-flash"},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            cfg_path = f.name

        try:
            cfg_mgr = LLMConfigManager(config_file=cfg_path)
            self.assertEqual(cfg_mgr.get_stage_config("outline").provider, "gemini")
            self.assertEqual(cfg_mgr.get_stage_config("outline").model, "gemini-3.8-flash")
            self.assertEqual(cfg_mgr.get_stage_config("storyboard").provider, "ollama")
            self.assertEqual(cfg_mgr.get_stage_config("storyboard").model, "qwen2.5:7b")
            self.assertEqual(cfg_mgr.get_stage_config("code_generation").provider, "ollama")
            self.assertEqual(cfg_mgr.get_stage_config("code_generation").model, "qwen2.5-coder:7b")
            self.assertEqual(cfg_mgr.get_stage_config("verification").provider, "gemini")
        finally:
            os.remove(cfg_path)

    def test_stage_env_var_overrides_file_and_global(self):
        """Verify stage environment variable has top priority."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_MODEL"] = "gemini-3.8-flash"
        os.environ["STAGE_CODE_GENERATION_PROVIDER"] = "ollama"
        os.environ["STAGE_CODE_GENERATION_MODEL"] = "qwen2.5-coder:7b"

        cfg_mgr = LLMConfigManager()
        outline_cfg = cfg_mgr.get_stage_config("outline")
        code_cfg = cfg_mgr.get_stage_config("code_generation")

        self.assertEqual(outline_cfg.provider, "gemini")
        self.assertEqual(outline_cfg.model, "gemini-3.8-flash")
        self.assertEqual(code_cfg.provider, "ollama")
        self.assertEqual(code_cfg.model, "qwen2.5-coder:7b")

    def test_invalid_provider_raises_configuration_error(self):
        """Verify invalid provider produces clear configuration error."""
        os.environ["LLM_PROVIDER"] = "non_existent_provider_xyz"
        cfg_mgr = LLMConfigManager()
        with self.assertRaises(LLMConfigurationError) as ctx:
            cfg_mgr.validate_stage_config("outline")
        self.assertIn("Unsupported LLM provider", str(ctx.exception))


class TestStageSpecificRoutingAndFallbacks(unittest.TestCase):
    def setUp(self):
        self.orig_env = dict(os.environ)
        get_router().clear_overrides()
        self._empty_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        self._empty_cfg.write("{}")
        self._empty_cfg.close()
        os.environ["LLM_CONFIG_FILE"] = self._empty_cfg.name
        for k in list(os.environ.keys()):
            if k.startswith("STAGE_"):
                del os.environ[k]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        get_router().clear_overrides()
        if hasattr(self, "_empty_cfg") and os.path.exists(self._empty_cfg.name):
            try:
                os.remove(self._empty_cfg.name)
            except Exception:
                pass

    def test_outline_and_code_generation_use_different_models(self):
        """Section 9.E Test 1: Verify outline and code generation can use different models."""
        mock_outline = MockProvider(ModelConfig(provider="mock", model="outline-model-v1"))
        mock_coder = MockProvider(ModelConfig(provider="mock", model="coder-model-v2"))

        get_router().set_stage_provider("outline", mock_outline)
        get_router().set_stage_provider("code_generation", mock_coder)

        resp1 = generate_response("Create curriculum outline", stage="outline")
        resp2 = generate_response("Generate Manim scene code", stage="code_generation")

        self.assertEqual(resp1.model, "outline-model-v1")
        self.assertEqual(resp2.model, "coder-model-v2")
        self.assertEqual(len(mock_outline.calls), 1)
        self.assertEqual(len(mock_coder.calls), 1)

    def test_storyboard_local_and_verification_cloud(self):
        """Section 9.E Test 2: Verify storyboard uses local model while verification uses cloud model."""
        os.environ["STAGE_STORYBOARD_PROVIDER"] = "ollama"
        os.environ["STAGE_STORYBOARD_MODEL"] = "qwen2.5:7b"
        os.environ["STAGE_VERIFICATION_PROVIDER"] = "gemini"
        os.environ["STAGE_VERIFICATION_MODEL"] = "gemini-3.8-flash"

        cfg_mgr = LLMConfigManager()
        sb_cfg = cfg_mgr.get_stage_config(Stage.STORYBOARD)
        verif_cfg = cfg_mgr.get_stage_config(Stage.VERIFICATION)

        self.assertEqual(sb_cfg.provider, "ollama")
        self.assertEqual(sb_cfg.model, "qwen2.5:7b")
        self.assertEqual(verif_cfg.provider, "gemini")
        self.assertEqual(verif_cfg.model, "gemini-3.8-flash")

    def test_changing_stage_config_changes_model_without_code_edits(self):
        """Section 9.E Test 3: Verify dynamic model selection via configuration alone."""
        cfg_mgr = LLMConfigManager()

        # Step A: default
        os.environ["STAGE_OUTLINE_MODEL"] = "model-alpha"
        cfg_a = cfg_mgr.get_stage_config("outline")
        self.assertEqual(cfg_a.model, "model-alpha")

        # Step B: switch configuration without any code change
        os.environ["STAGE_OUTLINE_MODEL"] = "model-beta"
        cfg_b = cfg_mgr.get_stage_config("outline")
        self.assertEqual(cfg_b.model, "model-beta")

    def test_stage_without_override_uses_configured_default(self):
        """Section 9.E Test 4: Verify stages without override inherit default provider and model."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_MODEL"] = "gemini-3.8-flash"
        # Explicit override only for storyboard
        os.environ["STAGE_STORYBOARD_PROVIDER"] = "ollama"
        os.environ["STAGE_STORYBOARD_MODEL"] = "qwen2.5:7b"

        cfg_mgr = LLMConfigManager()
        # Storyboard uses override
        self.assertEqual(cfg_mgr.get_stage_config("storyboard").provider, "ollama")
        self.assertEqual(cfg_mgr.get_stage_config("storyboard").model, "qwen2.5:7b")

        # Outline and code_generation use default
        self.assertEqual(cfg_mgr.get_stage_config("outline").provider, "gemini")
        self.assertEqual(cfg_mgr.get_stage_config("outline").model, "gemini-3.8-flash")
        self.assertEqual(cfg_mgr.get_stage_config("code_generation").provider, "gemini")
        self.assertEqual(cfg_mgr.get_stage_config("code_generation").model, "gemini-3.8-flash")

    def test_explicit_configurable_stage_fallback(self):
        """Section 9.C & 9.E: Verify explicit configurable fallback triggers without corrupting outputs."""
        failing_mock = MockProvider(
            ModelConfig(provider="mock", model="primary-failing", fallback_provider="mock", fallback_model="fallback-healthy"),
            simulate_error=LLMError("Simulated primary failure"),
        )
        healthy_mock = MockProvider(
            ModelConfig(provider="mock", model="fallback-healthy"),
            default_response="Fallback succeeded",
        )

        router = get_router()
        router.set_stage_provider("code_generation", failing_mock)
        router._provider_cache["mock::fallback-healthy::None"] = healthy_mock

        result = generate_response("Fix code", stage="code_generation")
        self.assertEqual(result.text, "Fallback succeeded")
        self.assertEqual(len(healthy_mock.calls), 1)


class TestErrorHandlingAndCapabilities(unittest.TestCase):
    def setUp(self):
        get_router().clear_overrides()

    def tearDown(self):
        get_router().clear_overrides()

    def test_ollama_connection_error_handling(self):
        """Verify Ollama connection error is handled gracefully with clear startup instructions."""
        import requests
        provider = OllamaProvider(ModelConfig(provider="ollama", model="qwen2.5:7b", base_url="http://127.0.0.1:99999"))
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with self.assertRaises(LLMConnectionError) as ctx:
                provider.generate("Test prompt")
            self.assertIn("Is Ollama running?", str(ctx.exception))

    def test_ollama_model_not_found_404_handling(self):
        """Verify HTTP 404 from Ollama raises LLMModelNotFoundError with pull instructions."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = '{"error": "model \'missing-model\' not found"}'

        provider = OllamaProvider(ModelConfig(provider="ollama", model="missing-model"))
        with patch("requests.post", return_value=mock_response):
            with self.assertRaises(LLMModelNotFoundError) as ctx:
                provider.generate("Test prompt")
            self.assertIn("ollama pull missing-model", str(ctx.exception))

    def test_timeout_error_handling(self):
        """Verify request timeout produces LLMTimeoutError."""
        import requests
        provider = OllamaProvider(ModelConfig(provider="ollama", model="qwen2.5:7b"))
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
            with self.assertRaises(LLMTimeoutError):
                provider.generate("Test prompt")

    def test_unsupported_capability_video_input_raises_clear_error(self):
        """Section 9.E Test 6: Verify unsupported video capability raises clear UnsupportedCapabilityError."""
        provider = OllamaProvider(ModelConfig(provider="ollama", model="qwen2.5:7b"))
        with self.assertRaises(UnsupportedCapabilityError) as ctx:
            provider.generate("Analyze video", video_path="/tmp/fake_video.mp4", stage="verification")
        self.assertIn("does not support 'video_input'", str(ctx.exception))

    def test_provider_failure_does_not_corrupt_outputs_from_other_stages(self):
        """Section 9.E Test 5: Verify failure in one stage leaves other stages uncorrupted."""
        failing_coder = MockProvider(ModelConfig(provider="mock", model="failing-coder"), simulate_error=LLMError("Code gen failed"))
        healthy_outline = MockProvider(ModelConfig(provider="mock", model="healthy-outline"), default_response='{"topic": "Physics"}')

        get_router().set_stage_provider("outline", healthy_outline)
        get_router().set_stage_provider("code_generation", failing_coder)

        # Outline succeeds
        outline_resp = generate_response("Create outline", stage="outline")
        self.assertEqual(outline_resp.text, '{"topic": "Physics"}')

        # Code gen fails cleanly without corrupting outline
        with self.assertRaises(LLMError):
            generate_response("Generate code", stage="code_generation")

        # Subsequent outline calls remain functional
        outline_resp2 = generate_response("Create outline 2", stage="outline")
        self.assertEqual(outline_resp2.text, '{"topic": "Physics"}')


class TestAgentPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        get_router().clear_overrides()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        get_router().clear_overrides()

    def test_agent_outline_generation_with_stage_routing(self):
        """Verify TeachingVideoAgent.generate_outline executes via unified stage routing."""
        mock_outline_json = json.dumps({
            "topic": "Graph Algorithms",
            "target_audience": "Students",
            "learning_level": "intermediate",
            "analogy_used": "Road Networks",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Introduction to Graphs",
                    "story_beat": "Traveler looks at road map.",
                    "content": "Vertices and edges.",
                    "example": "Town A to Town B.",
                }
            ]
        })
        mock_provider = MockProvider(default_response=mock_outline_json)
        get_router().set_stage_provider("outline", mock_provider)

        agent = TeachingVideoAgent(
            idx=101,
            knowledge_point="Graph Algorithms",
            folder=self.temp_dir,
            cfg=RunConfig(level="intermediate", analogy="Road Networks"),
        )

        outline = agent.generate_outline()

        self.assertIsNotNone(outline)
        self.assertEqual(outline.topic, "Graph Algorithms")
        self.assertEqual(len(outline.sections), 1)
        self.assertEqual(outline.sections[0]["id"], "section_1")
        # Verify file was written
        outline_file = agent.output_dir / "outline.json"
        self.assertTrue(outline_file.exists())

    def test_agent_storyboard_generation_with_stage_routing(self):
        """Verify TeachingVideoAgent.generate_storyboard executes via unified stage routing."""
        agent = TeachingVideoAgent(
            idx=102,
            knowledge_point="Calculus Limits",
            folder=self.temp_dir,
            cfg=RunConfig(level="intermediate"),
        )
        agent.outline = TeachingOutline(
            topic="Calculus Limits",
            target_audience="College Students",
            sections=[{"id": "section_1", "title": "Definition"}],
            learning_level="intermediate",
        )

        mock_sb_json = json.dumps({
            "sections": [
                {
                    "id": "section_1",
                    "title": "Definition of Limits",
                    "lecture_lines": ["A limit describes behavior near a point."],
                    "animations": ["Draw function graph approaching x = 2."],
                    "story_beat": "Approaching the threshold.",
                }
            ]
        })
        mock_provider = MockProvider(default_response=mock_sb_json)
        get_router().set_stage_provider("storyboard", mock_provider)

        sections = agent.generate_storyboard()
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].id, "section_1")
        self.assertEqual(sections[0].lecture_lines[0], "A limit describes behavior near a point.")

    def test_agent_code_generation_with_stage_routing(self):
        """Verify TeachingVideoAgent.generate_section_code executes via unified stage routing."""
        agent = TeachingVideoAgent(
            idx=103,
            knowledge_point="Linear Transformations",
            folder=self.temp_dir,
            cfg=RunConfig(level="intermediate"),
        )
        sec = Section(
            id="section_1",
            title="Introduction",
            lecture_lines=["Matrices transform vectors."],
            animations=["Rotate plane."],
        )
        mock_code = (
            "from manim import *\n"
            "class Section1Scene(TeachingScene):\n"
            "    def construct(self):\n"
            "        self.play(FadeIn(Text('Matrix')))\n"
        )
        mock_provider = MockProvider(default_response=mock_code)
        get_router().set_stage_provider("code_generation", mock_provider)

        code = agent.generate_section_code(sec, attempt=1)
        self.assertIn("Section1Scene", code)
        self.assertIn("TeachingScene", code)
        code_file = agent.output_dir / "section_1.py"
        self.assertTrue(code_file.exists())

    def test_cli_argument_parsing_provider_and_model(self):
        """Verify CLI parser parses --provider and --model flags correctly."""
        sys.argv = [
            "agent.py",
            "--topic", "Differential Equations",
            "--provider", "ollama",
            "--model", "qwen2.5:7b",
        ]
        args = build_and_parse_args()
        self.assertEqual(args.topic, "Differential Equations")
        self.assertEqual(args.provider, "ollama")
        self.assertEqual(args.model, "qwen2.5:7b")

    def test_get_api_and_output_ollama_and_custom(self):
        """Verify get_api_and_output returns proper callable and folder name for local and cloud models."""
        caller, folder = get_api_and_output("ollama")
        self.assertEqual(folder, "Ollama")
        self.assertTrue(callable(caller))

        caller_qwen, folder_qwen = get_api_and_output("qwen2.5:7b")
        self.assertIn("Ollama", folder_qwen)
        self.assertTrue(callable(caller_qwen))

        caller_prov, folder_prov = get_api_and_output(provider="ollama", model="qwen3:8b")
        self.assertEqual(folder_prov, "Ollama_qwen3:8b")
        self.assertTrue(callable(caller_prov))


if __name__ == "__main__":
    unittest.main()
