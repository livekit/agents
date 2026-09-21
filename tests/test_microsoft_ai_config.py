from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

from examples.other import microsoft_ai_smoke as smoke
from livekit.plugins import microsoft_ai
from livekit.plugins.microsoft_ai._http import HTTPClient

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

DUMMY_CONFIG = """\
MICROSOFT_AI_STT_URL=wss://stt.example.invalid/v1/realtime?intent=transcription
MICROSOFT_AI_STT_MODEL=file-transcriber
MICROSOFT_AI_STT_API_KEY="dummy-stt-key"
MICROSOFT_AI_STT_LANGUAGE=en
MICROSOFT_AI_TTS_URL=https://tts.example.invalid/cognitiveservices/v1
MICROSOFT_AI_TTS_MODEL=file-synthesizer
MICROSOFT_AI_TTS_API_KEY='dummy-tts-key'
MICROSOFT_AI_TTS_VOICE=en-US-Dummy:file-synthesizer
MICROSOFT_AI_TTS_SAMPLE_RATE=24000
"""


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("MICROSOFT_AI_"):
            monkeypatch.delenv(name)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Configuration tests must not create HTTP sessions")

    monkeypatch.setattr(HTTPClient, "session", forbidden)


def test_explicit_external_file_loads_without_mutating_environment(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    recognizer = microsoft_ai.STT(vad=None, env_file=path)
    synthesizer = microsoft_ai.TTS(env_file=path)
    assert recognizer.model == "file-transcriber"
    assert recognizer._language == "en"
    assert recognizer._client.headers["Authorization"] == "Bearer dummy-stt-key"
    assert synthesizer.model == "file-synthesizer"
    assert synthesizer.sample_rate == 24000
    assert synthesizer._opts.voice == "en-US-Dummy:file-synthesizer"
    assert synthesizer._client.headers["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in synthesizer._client.headers
    assert synthesizer._client.headers["Accept"] == "audio/wav"
    assert "MICROSOFT_AI_STT_API_KEY" not in os.environ
    assert "MICROSOFT_AI_TTS_API_KEY" not in os.environ


def test_explicit_arguments_then_environment_then_file_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_ENV_FILE", str(path))
    monkeypatch.setenv("MICROSOFT_AI_STT_MODEL", "environment-transcriber")
    monkeypatch.setenv("MICROSOFT_AI_TTS_MODEL", "environment-synthesizer")
    recognizer = microsoft_ai.STT(vad=None)
    synthesizer = microsoft_ai.TTS(
        model="argument-synthesizer", voice="en-US-Dummy:argument-synthesizer", sample_rate=48000
    )
    assert recognizer.model == "environment-transcriber"
    assert synthesizer.model == "argument-synthesizer"
    assert synthesizer.sample_rate == 48000
    other = microsoft_ai.STT(vad=None, language="fr", model="argument-transcriber")
    assert other._language == "fr" and other.model == "argument-transcriber"


def test_dotenv_values_are_literal_not_shell_sourced_or_interpolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace('"dummy-stt-key"', "'literal-${SHOULD_NOT_EXPAND}'"),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOULD_NOT_EXPAND", "not-a-credential")
    recognizer = microsoft_ai.STT(vad=None, env_file=path)
    assert recognizer._client.headers["Authorization"] == "Bearer literal-${SHOULD_NOT_EXPAND}"


def test_empty_template_fails_without_network_or_secret_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "empty.env"
    path.write_text("MICROSOFT_AI_STT_API_KEY=\nMICROSOFT_AI_TTS_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MICROSOFT_AI_STT_MODEL"):
        microsoft_ai.STT(vad=None, env_file=path)
    with pytest.raises(ValueError, match="MICROSOFT_AI_TTS_SAMPLE_RATE"):
        microsoft_ai.TTS(env_file=path)
    assert not caplog.records


def test_missing_selected_file_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read the selected") as caught:
        microsoft_ai.STT(vad=None, env_file=tmp_path / "private-location.env")
    assert "private-location" not in str(caught.value)


async def test_smoke_preflights_both_services_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "incomplete.env"
    path.write_text(
        DUMMY_CONFIG.replace(
            "MICROSOFT_AI_TTS_API_KEY='dummy-tts-key'", "MICROSOFT_AI_TTS_API_KEY="
        ),
        encoding="utf-8",
    )
    create_session = MagicMock()
    monkeypatch.setattr(HTTPClient, "session", create_session)
    with pytest.raises(ValueError, match="MICROSOFT_AI_TTS_API_KEY"):
        await smoke._run(pcm=b"\0\0", expected="test", check_tts=True, env_file=path)
    create_session.assert_not_called()


@pytest.mark.parametrize("voice", ["short-name", "en-US-Dummy:wrong-model", ":file-synthesizer"])
def test_voice_must_include_the_configured_model(tmp_path: Path, voice: str) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    with pytest.raises(ValueError, match="full voice ID"):
        microsoft_ai.TTS(env_file=path, voice=voice)


def test_model_validation_is_case_insensitive_but_does_not_invent_aliases(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    instance = microsoft_ai.TTS(
        env_file=path, model="mai-test-flash", voice="en-US-Dummy:MAI-Test-Flash"
    )
    assert instance.model == "mai-test-flash"
    with pytest.raises(ValueError, match="configured model"):
        microsoft_ai.TTS(
            env_file=path, model="mai-test-2.1-flash", voice="en-US-Dummy:MAI-Test-2-Flash"
        )


@pytest.mark.parametrize("region", ["eastus2", "EastUS2"])
def test_region_constructs_the_documented_public_cloud_endpoint(region: str) -> None:
    instance = microsoft_ai.TTS(
        region=region, api_key="dummy", model="test", voice="en-US-Dummy:test", sample_rate=24000
    )
    assert instance._client.url == "https://eastus2.tts.speech.microsoft.com/cognitiveservices/v1"


def test_region_from_file_and_constructor_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace(
            "MICROSOFT_AI_TTS_URL=https://tts.example.invalid/cognitiveservices/v1",
            "MICROSOFT_AI_TTS_URL=\nMICROSOFT_AI_TTS_REGION=eastus2",
        ),
        encoding="utf-8",
    )
    assert microsoft_ai.TTS(env_file=path)._client.url == (
        "https://eastus2.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    monkeypatch.setenv("MICROSOFT_AI_TTS_REGION", "westeurope")
    assert microsoft_ai.TTS(env_file=path)._client.url.startswith("https://westeurope.")
    assert microsoft_ai.TTS(env_file=path, region="eastus2")._client.url.startswith(
        "https://eastus2."
    )


def test_configured_url_wins_over_region_and_is_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    instance = microsoft_ai.TTS(env_file=path, region="eastus2")
    assert instance._client.url == "https://tts.example.invalid/cognitiveservices/v1"
    monkeypatch.setenv("MICROSOFT_AI_TTS_URL", "https://override.example.invalid/custom?q=dummy")
    instance = microsoft_ai.TTS(env_file=path, region="eastus2")
    assert instance._client.url == "https://override.example.invalid/custom?q=dummy"
    instance = microsoft_ai.TTS(
        env_file=path, region="eastus2", url="https://argument.example.invalid/exact"
    )
    assert instance._client.url == "https://argument.example.invalid/exact"


@pytest.mark.parametrize("region", ["east us 2", "eastus2.example.invalid/path", "../eastus2", ""])
def test_invalid_region_is_not_interpolated_into_a_host(region: str) -> None:
    with pytest.raises(ValueError):
        microsoft_ai.TTS(
            region=region,
            api_key="dummy",
            model="test",
            voice="en-US-Dummy:test",
            sample_rate=24000,
        )


@pytest.mark.parametrize(
    ("auth_header", "value"),
    [("Authorization", "Bearer dummy-stt-key"), ("api-key", "dummy-stt-key")],
)
def test_stt_auth_selector_from_file(
    tmp_path: Path, auth_header: Literal["Authorization", "api-key"], value: str
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG + f"MICROSOFT_AI_STT_AUTH_HEADER={auth_header}\n", encoding="utf-8"
    )
    instance = microsoft_ai.STT(vad=None, env_file=path)
    assert instance._client.headers == {auth_header: value, "User-Agent": "LiveKit Agents"}
    assert "dummy-stt-key" not in instance._client.url
    tts = microsoft_ai.TTS(env_file=path)
    assert tts._client.headers["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in tts._client.headers and "api-key" not in tts._client.headers


def test_stt_auth_selector_uses_argument_environment_file_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG + "MICROSOFT_AI_STT_AUTH_HEADER=api-key\n", encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", "Authorization")
    assert microsoft_ai.STT(vad=None, env_file=path)._client.headers["Authorization"] == (
        "Bearer dummy-stt-key"
    )
    instance = microsoft_ai.STT(
        vad=None, env_file=path, auth_header="api-key", api_key="argument-key"
    )
    assert instance._client.headers["api-key"] == "argument-key"
    assert "Authorization" not in instance._client.headers


@pytest.mark.parametrize(
    "selector",
    ["", " ", "Api-Key", "Bearer", "Ocp-Apim-Subscription-Key", "api-key\r\nx-secret: dummy"],
)
def test_invalid_auth_selector_fails_without_echoing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", selector)
    with pytest.raises(ValueError, match="must be Authorization or api-key") as caught:
        microsoft_ai.STT(vad=None, env_file=path)
    assert "dummy" not in str(caught.value)
    assert not caplog.records


def test_custom_headers_override_selector_environment_without_loading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        DUMMY_CONFIG.replace('MICROSOFT_AI_STT_API_KEY="dummy-stt-key"', ""), encoding="utf-8"
    )
    monkeypatch.setenv("MICROSOFT_AI_STT_AUTH_HEADER", "invalid-unused-value")
    instance = microsoft_ai.STT(vad=None, env_file=path, headers={"Authorization": "Bearer custom"})
    assert instance._client.headers == {
        "Authorization": "Bearer custom",
        "User-Agent": "LiveKit Agents",
    }
    assert microsoft_ai.STT(vad=None, env_file=path, headers={})._client.headers == {
        "User-Agent": "LiveKit Agents"
    }
    with pytest.raises(ValueError, match="either auth_header or headers"):
        microsoft_ai.STT(vad=None, env_file=path, headers={}, auth_header="api-key")


@pytest.mark.parametrize(
    "credential", ["dummy\r\nx-header:value", "dummy\n", "dummy\x00", "dummy\x7f"]
)
def test_stt_credentials_cannot_inject_header_values(
    tmp_path: Path, credential: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(DUMMY_CONFIG, encoding="utf-8")
    with pytest.raises(ValueError, match="control characters") as caught:
        microsoft_ai.STT(vad=None, env_file=path, api_key=credential, auth_header="api-key")
    assert "dummy" not in str(caught.value)
    assert not caplog.records


async def test_stt_only_smoke_reads_auth_selector_without_tts_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text(
        "\n".join(line for line in DUMMY_CONFIG.splitlines() if "MICROSOFT_AI_STT_" in line)
        + "\nMICROSOFT_AI_STT_AUTH_HEADER=api-key\n",
        encoding="utf-8",
    )
    checked = False

    async def check(provider: microsoft_ai.STT, pcm: bytes, expected: str) -> None:
        nonlocal checked
        assert provider._client.headers["api-key"] == "dummy-stt-key"
        assert "Authorization" not in provider._client.headers
        assert pcm == b"\0\0" and expected == "test"
        checked = True

    monkeypatch.setattr(smoke, "_check_stt", check)
    monkeypatch.setattr(
        microsoft_ai, "TTS", MagicMock(side_effect=AssertionError("TTS is not selected"))
    )
    await smoke._run(pcm=b"\0\0", expected="test", check_tts=False, env_file=path)
    assert checked
