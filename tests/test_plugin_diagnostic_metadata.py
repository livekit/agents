from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest

from livekit.agents.diagnostics import PluginCapability, PluginDiagnosticInfo
from livekit.agents.plugin import Plugin


@pytest.fixture(autouse=True)
def restore_plugins() -> Iterator[None]:
    previous = list(Plugin.registered_plugins)
    yield
    Plugin.registered_plugins[:] = previous


def _plugin_info(module_name: str, class_name: str) -> PluginDiagnosticInfo:
    module = importlib.import_module(module_name)
    plugin = getattr(module, class_name)()
    info = plugin.diagnostic_info()
    assert info is not None
    return info


def test_provider_plugin_diagnostic_metadata() -> None:
    cases = [
        (
            "livekit.plugins.openai",
            "OpenAIPlugin",
            {"OPENAI_API_KEY"},
            {
                PluginCapability.LLM,
                PluginCapability.STT,
                PluginCapability.TTS,
                PluginCapability.REALTIME,
            },
        ),
        (
            "livekit.plugins.deepgram",
            "DeepgramPlugin",
            {"DEEPGRAM_API_KEY"},
            {PluginCapability.STT, PluginCapability.TTS},
        ),
        (
            "livekit.plugins.cartesia",
            "CartesiaPlugin",
            {"CARTESIA_API_KEY"},
            {PluginCapability.STT, PluginCapability.TTS},
        ),
        (
            "livekit.plugins.elevenlabs",
            "ElevenLabsPlugin",
            {"ELEVEN_API_KEY"},
            {PluginCapability.STT, PluginCapability.TTS},
        ),
    ]

    for module_name, class_name, required_env_vars, capabilities in cases:
        info = _plugin_info(module_name, class_name)

        assert set(info.required_env_vars) == required_env_vars
        assert set(info.capabilities) == capabilities
        assert not info.downloadable_files
        assert info.docs_url is not None
        assert info.docs_url.startswith("https://docs.livekit.io/")

    openai_info = _plugin_info("livekit.plugins.openai", "OpenAIPlugin")
    assert "AZURE_OPENAI_API_KEY" in openai_info.optional_env_vars


def test_local_model_plugin_diagnostic_metadata() -> None:
    silero_info = _plugin_info("livekit.plugins.silero", "SileroPlugin")

    assert silero_info.required_env_vars == ()
    assert set(silero_info.capabilities) == {PluginCapability.VAD}
    assert silero_info.downloadable_files == ["Silero VAD model"]
    assert silero_info.docs_url == "https://docs.livekit.io/agents/build/turns/vad/"

    english = importlib.import_module("livekit.plugins.turn_detector.english")
    base = importlib.import_module("livekit.plugins.turn_detector.base")
    turn_detector_info = base.EOUPlugin(english._EUORunnerEn).diagnostic_info()

    assert turn_detector_info.required_env_vars == ()
    assert set(turn_detector_info.capabilities) == {PluginCapability.TURN_DETECTOR}
    assert turn_detector_info.downloadable_files == ["LiveKit turn detector model"]
    assert (
        turn_detector_info.docs_url == "https://docs.livekit.io/agents/build/turns/turn-detector/"
    )
