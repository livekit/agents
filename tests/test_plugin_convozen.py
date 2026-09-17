"""Tests for the ConvoZen Akshara STT / Ragini TTS plugin.

Hermetic: every HTTP call is served by a fake aiohttp session, so no API key and
no network are needed.
"""

from __future__ import annotations

import io
import wave
from typing import Any
from unittest.mock import patch

import pytest

from livekit import rtc

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# fake aiohttp
# --------------------------------------------------------------------------- #


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunks(self):  # noqa: ANN202
        for chunk in self._chunks:
            yield chunk, True


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        json_body: dict | None = None,
        chunks: list[bytes] | None = None,
        text_body: str = "",
    ) -> None:
        self.status = status
        self._json = json_body or {}
        self._text = text_body
        self.content = _FakeContent(chunks or [])

    async def json(self) -> dict:
        return self._json

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeSession:
    """Stands in for aiohttp.ClientSession, recording every request."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str | None = None, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self._response


def _form_fields(form: Any) -> dict[str, Any]:
    """Flatten an aiohttp.FormData into {field name: value}."""
    return {opts["name"]: value for opts, _headers, value in form._fields}


def _audio_buffer(duration: float = 1.0, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = int(duration * sample_rate)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def _wav_bytes(duration: float = 0.2, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * int(duration * sample_rate))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# STT — construction
# --------------------------------------------------------------------------- #


def test_stt_requires_api_key() -> None:
    from livekit.plugins.convozen import STT

    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="API key"):
        STT()


def test_stt_reads_api_key_from_env() -> None:
    from livekit.plugins.convozen import STT

    with patch.dict("os.environ", {"CONVOZEN_API_KEY": "env-key"}):
        assert STT()._opts.api_key == "env-key"


def test_base_url_defaults_to_public_api() -> None:
    """CONVOZEN_BASE_URL is optional — an API key is the only required config."""
    from livekit.plugins.convozen import STT, TTS
    from livekit.plugins.convozen.models import DEFAULT_BASE_URL

    with patch.dict("os.environ", {"CONVOZEN_API_KEY": "k"}, clear=True):
        assert STT()._opts.base_url == DEFAULT_BASE_URL
        assert TTS()._opts.base_url == DEFAULT_BASE_URL


def test_base_url_precedence() -> None:
    """Explicit argument wins over the env var, which wins over the default."""
    from livekit.plugins.convozen import STT, TTS

    env = {"CONVOZEN_API_KEY": "k", "CONVOZEN_BASE_URL": "https://staging.example/api"}
    with patch.dict("os.environ", env, clear=True):
        assert STT()._opts.base_url == "https://staging.example/api"
        assert TTS()._opts.base_url == "https://staging.example/api"
        assert STT(base_url="https://local.test/api")._opts.base_url == "https://local.test/api"
        assert TTS(base_url="https://local.test/api")._opts.base_url == "https://local.test/api"


def test_base_url_trailing_slash_is_tolerated() -> None:
    from livekit.plugins.convozen import STT, TTS

    stt = STT(api_key="k", base_url="https://local.test/api/")
    tts = TTS(api_key="k", base_url="https://local.test/api/")
    assert stt._opts.transcribe_url() == "https://local.test/api/v2/akshara/transcribe"
    assert tts._opts.tts_url() == "https://local.test/api/v1/ragini/tts"


def test_stt_is_not_streaming() -> None:
    """Akshara is batch-only; the framework relies on this to insert a StreamAdapter."""
    from livekit.plugins.convozen import STT

    caps = STT(api_key="k").capabilities
    assert caps.streaming is False
    assert caps.interim_results is False
    assert caps.keyterms is True


def test_stt_model_and_provider() -> None:
    from livekit.plugins.convozen import STT

    stt = STT(api_key="k", model="akshara")
    assert stt.model == "akshara"
    assert stt.provider == "ConvoZen"


def test_stt_lang_tags_derived_from_language() -> None:
    from livekit.plugins.convozen import STT

    assert STT(api_key="k", language="hi")._opts.lang_tags == ["hi"]


def test_stt_lang_tags_omitted_for_unknown_language() -> None:
    """An unrecognized language sends no hint rather than one the server rejects."""
    from livekit.plugins.convozen import STT

    assert STT(api_key="k", language="fr")._opts.lang_tags is None


def test_stt_explicit_lang_tags_are_validated() -> None:
    from livekit.plugins.convozen import STT

    assert STT(api_key="k", lang_tags=["hi", "en"])._opts.lang_tags == ["hi", "en"]

    with pytest.raises(ValueError, match="invalid lang_tags"):
        STT(api_key="k", lang_tags=["hi", "xx"])


def test_stt_update_options_moves_lang_tags_with_language() -> None:
    from livekit.plugins.convozen import STT

    stt = STT(api_key="k", language="en")
    stt.update_options(language="ta")
    assert stt._opts.language == "ta"
    assert stt._opts.lang_tags == ["ta"]


def test_stt_session_keyterms_merge_with_user_keywords() -> None:
    from livekit.plugins.convozen import STT

    stt = STT(api_key="k", keywords=["ConvoZen"])
    stt._update_session_keyterms(["Akshara", "ConvoZen"])
    assert stt._opts.all_keywords() == ["ConvoZen", "Akshara"]


def test_stt_keywords_are_passed_through_verbatim() -> None:
    """Terms go to the server exactly as given — no case or spacing changes."""
    from livekit.plugins.convozen import STT

    stt = STT(api_key="k", keywords=["LiveKit", "Convo Zen"])
    assert stt._opts.all_keywords() == ["LiveKit", "Convo Zen"]


def test_stt_keywords_dedupe() -> None:
    from livekit.plugins.convozen import STT

    stt = STT(api_key="k", keywords=["LiveKit", "LiveKit"])
    stt._update_session_keyterms(["LiveKit", "Akshara"])
    assert stt._opts.all_keywords() == ["LiveKit", "Akshara"]


# --------------------------------------------------------------------------- #
# STT — recognition
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stt_sends_expected_form_and_auth() -> None:
    from livekit.plugins.convozen import STT

    session = _FakeSession(_FakeResponse(json_body={"text": "namaste", "score": -1.2}))
    stt = STT(
        api_key="test-key",
        language="hi",
        model="akshara-pro",
        keywords=["ConvoZen"],
        blank_penalty=1.5,
        word_timestamps=True,
        http_session=session,  # type: ignore[arg-type]
    )

    await stt.recognize(_audio_buffer())

    call = session.calls[0]
    assert call["url"].endswith("/v2/akshara/transcribe")
    assert call["headers"] == {"x-api-key": "test-key"}

    fields = _form_fields(call["data"])
    assert fields["model"] == "akshara-pro"
    assert fields["audio_channels"] == "mono"
    assert fields["lang_tags"] == '["hi"]'
    # JSON list — comma/space-separated forms make the server 500
    assert fields["keywords"] == '["ConvoZen"]'
    assert fields["blank_penalty"] == "1.5"
    assert fields["word_timestamps"] == "true"
    # a real RIFF header, since the server takes a file upload
    assert fields["file"].startswith(b"RIFF")
    # not exposed: diarization returns a different response shape
    assert "speaker_labels" not in fields
    assert "num_speakers" not in fields


@pytest.mark.asyncio
async def test_stt_defaults_omit_optional_fields() -> None:
    from livekit.plugins.convozen import STT

    session = _FakeSession(_FakeResponse(json_body={"text": "hello"}))
    stt = STT(api_key="k", http_session=session)  # type: ignore[arg-type]

    await stt.recognize(_audio_buffer())

    fields = _form_fields(session.calls[0]["data"])
    for absent in ("keywords", "blank_penalty", "word_timestamps"):
        assert absent not in fields
    # denoise is deliberately not exposed: it only applies on the diarization
    # pipeline, and this plugin always uses the plain-transcription path
    assert "denoise" not in fields


@pytest.mark.asyncio
async def test_stt_returns_single_final_transcript() -> None:
    from livekit.agents import stt as agents_stt
    from livekit.plugins.convozen import STT

    session = _FakeSession(_FakeResponse(json_body={"text": "namaste duniya", "score": -2.45}))
    stt = STT(api_key="k", language="hi", http_session=session)  # type: ignore[arg-type]

    event = await stt.recognize(_audio_buffer())

    assert event.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == "namaste duniya"
    assert event.alternatives[0].language == "hi"


@pytest.mark.asyncio
async def test_stt_score_is_metadata_not_confidence() -> None:
    """`score` is a log-probability; putting it in confidence would misread as ~0."""
    from livekit.plugins.convozen import STT

    session = _FakeSession(_FakeResponse(json_body={"text": "hi", "score": -2.45}))
    stt = STT(api_key="k", http_session=session)  # type: ignore[arg-type]

    alt = (await stt.recognize(_audio_buffer())).alternatives[0]
    assert alt.metadata == {"score": -2.45}
    assert alt.confidence == 0.0


@pytest.mark.asyncio
async def test_stt_word_timestamps_become_timed_strings() -> None:
    from livekit.plugins.convozen import STT

    session = _FakeSession(
        _FakeResponse(
            json_body={
                "text": "hello world",
                "word_timestamps": [
                    {"word": "hello", "start_s": 0.1, "end_s": 0.4},
                    {"word": "world", "start_s": 0.5, "end_s": 0.9},
                ],
            }
        )
    )
    stt = STT(api_key="k", word_timestamps=True, http_session=session)  # type: ignore[arg-type]

    alt = (await stt.recognize(_audio_buffer())).alternatives[0]
    assert alt.words is not None
    assert [str(w) for w in alt.words] == ["hello", "world"]
    assert alt.words[0].start_time == 0.1
    assert alt.start_time == 0.1
    assert alt.end_time == 0.9


@pytest.mark.asyncio
async def test_stt_maps_http_error_to_api_status_error() -> None:
    from livekit.agents import APIConnectOptions, APIStatusError
    from livekit.plugins.convozen import STT

    session = _FakeSession(_FakeResponse(status=401, text_body="invalid api key"))
    stt = STT(api_key="k", http_session=session)  # type: ignore[arg-type]

    # max_retry=0 is what stt.StreamAdapter passes, i.e. the path an AgentSession
    # actually takes. Note that STT.recognize() with the default conn_options
    # retries even a 401: unlike RecognizeStream/TTS/LLM, its retry loop does not
    # consult APIError.retryable, so a bad key costs four round-trips there.
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(
            _audio_buffer(), conn_options=APIConnectOptions(max_retry=0, timeout=10)
        )

    assert exc.value.status_code == 401
    assert exc.value.retryable is False
    assert len(session.calls) == 1


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #


def test_tts_requires_api_key() -> None:
    from livekit.plugins.convozen import TTS

    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="API key"):
        TTS()


def test_tts_is_not_streaming() -> None:
    from livekit.plugins.convozen import TTS

    assert TTS(api_key="k").capabilities.streaming is False


def test_tts_sample_rate_defaults_per_model() -> None:
    from livekit.plugins.convozen import TTS

    assert TTS(api_key="k").sample_rate == 24000
    assert TTS(api_key="k", model="ragini-lite").sample_rate == 22050
    assert TTS(api_key="k", sample_rate=8000).sample_rate == 8000


def test_tts_rejects_unknown_language() -> None:
    from livekit.plugins.convozen import TTS

    with pytest.raises(ValueError, match="unsupported language"):
        TTS(api_key="k", language="fr")


def test_tts_rejects_empty_voice() -> None:
    from livekit.plugins.convozen import TTS

    with pytest.raises(ValueError, match="voice cannot be empty"):
        TTS(api_key="k", voice="  ")


def test_tts_accepts_unknown_voice() -> None:
    """Voices are added server-side; the plugin must not gate on its own list."""
    from livekit.plugins.convozen import TTS

    assert TTS(api_key="k", voice="brand-new-voice")._opts.voice == "brand-new-voice"


def test_tts_form_uses_speaker_and_omits_format() -> None:
    from livekit.plugins.convozen import TTS

    form = TTS(api_key="k", voice="roohi", language="hi", speed=1.2)._opts.to_form("hello")

    assert form["speaker"] == "roohi"  # wire name is `speaker`, not `voice`
    assert "voice" not in form
    assert "format" not in form  # server has no format field; output is always WAV
    assert form == {
        "text": "hello",
        "language": "hi",
        "speaker": "roohi",
        "model": "ragini-v1",
        "sample_rate": "24000",
        "speed": "1.2",
        "stream": "true",
    }


@pytest.mark.asyncio
async def test_tts_synthesizes_audio_from_wav() -> None:
    from livekit.plugins.convozen import TTS

    wav = _wav_bytes(duration=0.2, sample_rate=24000)
    # split across chunks, the way a chunked response arrives
    session = _FakeSession(_FakeResponse(chunks=[wav[:100], wav[100:]]))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    frames = []
    stream = tts.synthesize("hello")
    async for ev in stream:
        frames.append(ev.frame)
    await stream.aclose()

    assert frames, "expected at least one synthesized frame"
    assert session.calls[0]["headers"] == {"x-api-key": "test-key"}
    assert session.calls[0]["url"].endswith("/v1/ragini/tts")

    total_samples = sum(f.samples_per_channel for f in frames)
    assert total_samples / 24000 == pytest.approx(0.2, abs=0.05)


@pytest.mark.asyncio
async def test_tts_maps_http_error_to_api_status_error() -> None:
    from livekit.agents import APIStatusError
    from livekit.plugins.convozen import TTS

    session = _FakeSession(_FakeResponse(status=422, text_body="unsupported speaker"))
    tts = TTS(api_key="k", http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError) as exc:
        stream = tts.synthesize("hello")
        async for _ in stream:
            pass

    assert exc.value.status_code == 422
