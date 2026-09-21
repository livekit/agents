from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from xml.etree import ElementTree

import aiohttp
import pytest

from examples.other import microsoft_ai_smoke as smoke
from livekit.plugins import microsoft_ai

from .microsoft_ai_fakes import FakeResponse, FakeSocket, fake_session, wav_bytes

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Smoke-path unit tests must not make real network requests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.delenv("MICROSOFT_AI_ENV_FILE", raising=False)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--tts"],
        ["--run-live"],
        ["--run-live", "--stt-wav", "unused.wav"],
        ["--run-live", "--expected-text-file", "unused.txt"],
    ],
)
def test_smoke_requires_explicit_opt_in_and_complete_input_selection(
    args: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stt_factory, tts_factory = MagicMock(), MagicMock()
    monkeypatch.setattr(microsoft_ai, "STT", stt_factory)
    monkeypatch.setattr(microsoft_ai, "TTS", tts_factory)
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", *args])
    with pytest.raises(SystemExit) as caught:
        smoke.main()
    assert caught.value.code == 2
    stt_factory.assert_not_called()
    tts_factory.assert_not_called()


def test_smoke_refuses_audio_dumping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LK_DUMP_TTS", "1")
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", "--run-live", "--tts"])
    factory = MagicMock()
    monkeypatch.setattr(microsoft_ai, "TTS", factory)
    with pytest.raises(SystemExit) as caught:
        smoke.main()
    assert caught.value.code == 2
    factory.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_live_smoke_requires_owner_only_dotenv_permissions(tmp_path: Path) -> None:
    path = tmp_path / "endpoints.env"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions 0600"):
        smoke._check_env_file_permissions(path)
    path.chmod(0o600)
    smoke._check_env_file_permissions(path)


@pytest.mark.parametrize("samples", [1, 1307, 80000])
def test_approved_fixture_is_bounded_without_padding(tmp_path: Path, samples: int) -> None:
    path = tmp_path / "synthetic.wav"
    pcm = b"\x01\x00" * samples
    path.write_bytes(wav_bytes(pcm, sample_rate=16000))
    assert smoke._read_fixture(path) == pcm


@pytest.mark.parametrize(
    "data",
    [
        wav_bytes(b"", sample_rate=16000),
        wav_bytes(b"\x00\x00" * 80001, sample_rate=16000),
        wav_bytes(b"\x00\x00" * 100, sample_rate=24000),
        wav_bytes(b"\x00\x00" * 100, sample_rate=16000, channels=2),
        wav_bytes(b"\x00\x00" * 100, sample_rate=16000)[:-2],
        b"\x00" * (1024 * 1024 + 1),
    ],
)
def test_invalid_or_over_limit_fixtures_fail_before_network(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "synthetic.wav"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        smoke._read_fixture(path)


@pytest.mark.parametrize("text", ["", "  ", "...", "a" * 257, "b" * 4097])
def test_expected_text_is_bounded_and_contains_words(tmp_path: Path, text: str) -> None:
    path = tmp_path / "synthetic.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        smoke._read_expected(path)


@pytest.mark.parametrize("expected", ["Turn 1.", "Turn 1 missing tail"])
async def test_stt_smoke_checks_complete_words_without_printing_them(
    expected: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    socket = FakeSocket()
    session = fake_session()
    session.ws_connect = AsyncMock(return_value=socket)
    instance = microsoft_ai.STT(
        vad=None,
        url="wss://stt.example.invalid/realtime",
        model="test",
        headers={},
        http_session=session,
    )
    pcm = b"\x01\x00" * 337
    async with instance:
        if "missing" in expected:
            with pytest.raises(ValueError, match="including its tail"):
                await smoke._check_stt(instance, pcm, expected)
        else:
            await smoke._check_stt(instance, pcm, expected)
    assert socket.commits == [pcm]
    assert expected not in capsys.readouterr().out
    assert socket.closed
    session.ws_connect.assert_awaited_once()


async def test_tts_smoke_sends_only_the_fixed_approved_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = fake_session()
    response = FakeResponse(wav_bytes(b"\x01\x00" * 1001))
    session.post.return_value = response
    instance = microsoft_ai.TTS(
        url="https://tts.example.invalid/speech",
        model="test",
        voice="en-US-Dummy:test",
        sample_rate=24000,
        headers={},
        http_session=session,
    )
    async with instance:
        await smoke._check_tts(instance)
    session.post.assert_called_once()
    body = ElementTree.fromstring(session.post.call_args.kwargs["data"])
    assert next(iter(body)).text == smoke.TTS_TEXT
    assert "json" not in session.post.call_args.kwargs
    output = capsys.readouterr().out
    assert smoke.TTS_TEXT not in output
    assert "PCM16 mono frames at 24000 Hz" in output
    assert "audio_duration=" in output and "not model TTFA" in output
    assert response.closed
    assert smoke.CONNECT_OPTIONS.max_retry == 0
    assert smoke.MAX_DURATION == 5.0


def test_opted_in_cli_runs_only_selected_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LK_DUMP_TTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["microsoft_ai_smoke.py", "--run-live", "--tts"])
    run = AsyncMock()
    monkeypatch.setattr(smoke, "_run", run)
    smoke.main()
    run.assert_awaited_once_with(pcm=None, expected=None, check_tts=True, env_file=None)
