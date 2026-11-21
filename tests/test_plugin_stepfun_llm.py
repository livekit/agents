"""Unit tests for the StepFun LLM wrapper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.models import (
    DEFAULT_LLM_MODEL,
)

pytestmark = pytest.mark.plugin("stepfun")


@pytest.fixture(autouse=True)
def _stepfun_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the developer's provider credentials."""
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


_HAS_OPENAI = importlib.util.find_spec("livekit.plugins.openai") is not None
requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI, reason="needs the [llm] extra (livekit-plugins-openai)"
)


def test_llm_base_url_gets_version_suffix() -> None:
    # The OpenAI-compatible client expects the API version in the base URL
    # rather than in each request path.
    assert _utils.resolve_llm_base_url(None, "global") == "https://api.stepfun.ai/v1"
    assert _utils.resolve_llm_base_url(None, "china") == "https://api.stepfun.com/v1"


def test_llm_base_url_not_doubled() -> None:
    # Preserve an explicit /v1 suffix without duplicating it.
    assert (
        _utils.resolve_llm_base_url("https://api.stepfun.ai/v1", "global")
        == "https://api.stepfun.ai/v1"
    )


def test_llm_base_url_override_wins() -> None:
    assert (
        _utils.resolve_llm_base_url("https://proxy.internal", "global")
        == "https://proxy.internal/v1"
    )


def test_llm_base_url_strips_trailing_slash() -> None:
    assert (
        _utils.resolve_llm_base_url("https://api.stepfun.ai/", "global")
        == "https://api.stepfun.ai/v1"
    )


@requires_openai
def test_llm_is_bound_at_import_not_on_first_access() -> None:
    """LLM must already be a module attribute once the package is imported."""
    assert "LLM" in vars(stepfun)


def test_unknown_attribute_still_raises_attribute_error() -> None:
    # The lazy __getattr__ must not swallow genuine typos.
    with pytest.raises(AttributeError, match="no attribute 'Nonexistent'"):
        _ = stepfun.Nonexistent  # type: ignore[attr-defined]


def test_speech_imports_without_llm_extra() -> None:
    """Exercise a speech-only import even when the test environment has OpenAI."""
    code = """
import importlib.abc
import sys

class MissingOpenAIPlugin(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "livekit.plugins.openai":
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, MissingOpenAIPlugin())
from livekit.plugins import stepfun
assert stepfun.STT is not None
assert stepfun.TTS is not None
try:
    stepfun.LLM
except ImportError as exc:
    assert "livekit-plugins-stepfun[llm]" in str(exc)
else:
    raise AssertionError("LLM should require the optional OpenAI plugin")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


@requires_openai
def test_llm_defaults_to_global_cluster() -> None:
    llm = stepfun.LLM()
    assert llm.model == DEFAULT_LLM_MODEL
    assert llm.provider == "StepFun"
    assert llm.cluster == "global"


@requires_openai
def test_llm_honours_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "china")
    assert stepfun.LLM().cluster == "china"


@requires_openai
def test_llm_accepts_step_5_preview() -> None:
    llm = stepfun.LLM(model="step-5-preview")
    assert llm.model == "step-5-preview"


@requires_openai
def test_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        stepfun.LLM()


@requires_openai
def test_reasoning_effort_warns_on_undocumented_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Warn when the model has no documented reasoning_effort support.
    with caplog.at_level("WARNING", logger="livekit.plugins.stepfun"):
        stepfun.LLM(model="step-3.5-flash", reasoning_effort="low")
    assert "reasoning_effort" in caplog.text


@requires_openai
def test_reasoning_effort_silent_on_documented_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="livekit.plugins.stepfun"):
        stepfun.LLM(model="step-3.7-flash", reasoning_effort="low")
    assert "reasoning_effort" not in caplog.text


@requires_openai
@pytest.mark.parametrize("model", ["step-5-preview", "step-3.7-flash"])
def test_reasoning_effort_reaches_the_request(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both models document the parameter, so it must be on the wire."""
    from livekit.plugins.openai import LLM as OpenAILLM

    captured: dict = {}
    original = OpenAILLM.__init__

    def spy(self, **kwargs):
        captured.update(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(OpenAILLM, "__init__", spy)
    stepfun.LLM(model=model, reasoning_effort="medium")
    assert captured["reasoning_effort"] == "medium"


@requires_openai
def test_reasoning_effort_silent_on_step_5_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # step-5-preview documents reasoning_effort support.
    with caplog.at_level("WARNING", logger="livekit.plugins.stepfun"):
        stepfun.LLM(model="step-5-preview", reasoning_effort="high")
    assert "reasoning_effort" not in caplog.text
