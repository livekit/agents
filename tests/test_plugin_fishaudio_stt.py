from __future__ import annotations

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.plugins import fishaudio


class _FakeFishResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.status = status
        self.headers = {"x-request-id": "fish-request"}
        self._payload = payload or {
            "text": "hello, world",
            "duration": 1.25,
            "segments": [
                {"text": "hello", "start": 0.0, "end": 0.6},
                {"text": "world", "start": 0.6, "end": 1.25},
            ],
        }

    async def __aenter__(self) -> _FakeFishResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self) -> dict[str, object]:
        return self._payload


class _FakeFishSession:
    def __init__(self, response: _FakeFishResponse | None = None) -> None:
        self.posts: list[dict[str, object]] = []
        self._response = response or _FakeFishResponse()

    def post(self, **kwargs: object) -> _FakeFishResponse:
        self.posts.append(kwargs)
        return self._response


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\0\0" * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


def _form_fields(form: aiohttp.FormData) -> dict[str, object]:
    return {field[0]["name"]: field[2] for field in form._fields}


async def test_fish_audio_stt_posts_wav_form_data_and_returns_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="livekit.plugins.fishaudio"):
        fake_session = _FakeFishSession()
        fish_stt = fishaudio.STT(
            api_key="fish-key",
            language="zh-CN",
            ignore_timestamps=False,
            http_session=fake_session,  # type: ignore[arg-type]
        )

        event = await fish_stt.recognize(
            _audio_frame(),
            conn_options=APIConnectOptions(max_retry=0),
        )

    assert "Fish Audio STT transcript: hello, world" in caplog.records[-1].getMessage()
    assert caplog.records[-1].stt_provider == "FishAudio"
    assert caplog.records[-1].stt_model == "transcribe-1"
    assert caplog.records[-1].stt_language == "zh-CN"
    assert caplog.records[-1].stt_duration == 1.25

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == "fish-request"
    assert event.alternatives[0].text == "hello, world"
    assert event.alternatives[0].language == "zh-CN"
    assert event.alternatives[0].start_time == 0.0
    assert event.alternatives[0].end_time == 1.25
    assert event.alternatives[0].metadata == {
        "segments": [
            {"text": "hello", "start": 0.0, "end": 0.6},
            {"text": "world", "start": 0.6, "end": 1.25},
        ]
    }

    request = fake_session.posts[0]
    assert request["url"] == "https://api.fish.audio/v1/asr"
    assert request["headers"] == {
        "Authorization": "Bearer fish-key",
        "User-Agent": f"livekit-plugins-fishaudio/{fishaudio.__version__}",
    }

    form = request["data"]
    assert isinstance(form, aiohttp.FormData)
    fields = _form_fields(form)
    assert fields["language"] == "zh"
    assert fields["ignore_timestamps"] == "false"
    assert fields["audio"][:4] == b"RIFF"


def test_fish_audio_stt_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="FISH_API_KEY"):
        fishaudio.STT()


async def test_fish_audio_stt_omits_auto_language() -> None:
    fake_session = _FakeFishSession()
    fish_stt = fishaudio.STT(
        api_key="fish-key",
        language="auto",
        http_session=fake_session,  # type: ignore[arg-type]
    )

    await fish_stt.recognize(
        _audio_frame(),
        conn_options=APIConnectOptions(max_retry=0),
    )

    form = fake_session.posts[0]["data"]
    assert isinstance(form, aiohttp.FormData)
    fields = _form_fields(form)
    assert "language" not in fields


async def test_fish_audio_stt_logs_http_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_session = _FakeFishSession(
        _FakeFishResponse(
            status=401,
            payload={"message": "bad fish key", "code": 401},
        )
    )
    fish_stt = fishaudio.STT(
        api_key="fish-key",
        http_session=fake_session,  # type: ignore[arg-type]
    )

    with (
        caplog.at_level("ERROR", logger="livekit.plugins.fishaudio"),
        pytest.raises(Exception, match="Fish Audio ASR request failed"),
    ):
        await fish_stt.recognize(
            _audio_frame(),
            conn_options=APIConnectOptions(max_retry=0),
        )

    assert "Fish Audio STT request failed" in caplog.records[-1].getMessage()
    assert caplog.records[-1].request_id == "fish-request"
    assert caplog.records[-1].stt_provider == "FishAudio"
    assert caplog.records[-1].stt_model == "transcribe-1"
    assert caplog.records[-1].stt_status_code == 401
    assert caplog.records[-1].stt_error_body == {
        "message": "bad fish key",
        "code": 401,
    }
