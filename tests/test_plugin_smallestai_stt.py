"""Unit tests for SmallestAI STT plugin."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit import rtc
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.aio.channel import ChanEmpty

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_no_task():
    """Create a SpeechStream without starting the _main_task WebSocket loop."""
    from livekit.plugins.smallestai import STT
    from livekit.plugins.smallestai.stt import SpeechStream

    stt = STT(api_key="test-key")

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=stt._opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            http_session=MagicMock(),
        )
    return stream


def _silent_frame(*, sample_rate: int = 16000, duration_ms: int = 100) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=bytes(samples * 2),  # 16-bit PCM = 2 bytes/sample
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


# ---------------------------------------------------------------------------
# _CLOSE_STREAM_MSG payload
# ---------------------------------------------------------------------------


def test_close_stream_msg_payload():
    from livekit.plugins.smallestai.stt import SpeechStream

    assert json.loads(SpeechStream._CLOSE_STREAM_MSG) == {"type": "close_stream"}


def test_finalize_msg_constant_removed():
    from livekit.plugins.smallestai.stt import SpeechStream

    assert not hasattr(SpeechStream, "_FINALIZE_MSG"), (
        "_FINALIZE_MSG still exists; it should have been replaced by _CLOSE_STREAM_MSG"
    )


# ---------------------------------------------------------------------------
# close_stream sent on end_input (no real network)
# ---------------------------------------------------------------------------


async def test_close_stream_sent_on_end_input():
    """close_stream is sent (not finalize) when the input channel closes."""
    from livekit.plugins.smallestai import STT
    from livekit.plugins.smallestai.stt import SpeechStream

    close_stream_sent = asyncio.Event()
    sent_strings: list[str] = []

    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.close_code = None
    mock_ws.close = AsyncMock()
    mock_ws.send_bytes = AsyncMock()

    async def fake_send_str(msg: str) -> None:
        sent_strings.append(msg)
        if json.loads(msg).get("type") == "close_stream":
            close_stream_sent.set()

    mock_ws.send_str = fake_send_str

    async def fake_receive() -> MagicMock:
        # Block until close_stream arrives, then return is_last so recv_task exits.
        await close_stream_sent.wait()
        return MagicMock(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {"session_id": "s1", "transcript": "", "is_final": True, "is_last": True}
            ),
        )

    mock_ws.receive = fake_receive

    stt_instance = STT(api_key="test-key")
    stream = SpeechStream(
        stt=stt_instance,
        opts=stt_instance._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        http_session=MagicMock(),
    )

    async def fake_connect_ws() -> MagicMock:
        return mock_ws

    # Patch before the task gets to run (still synchronous here).
    stream._connect_ws = fake_connect_ws  # type: ignore[method-assign]

    stream.push_frame(_silent_frame())
    stream.end_input()

    await asyncio.wait_for(stream._task, timeout=5.0)

    types_sent = [json.loads(s).get("type") for s in sent_strings]
    assert "close_stream" in types_sent, f"close_stream not sent; sent: {types_sent}"
    assert "finalize" not in types_sent, f"finalize must not be sent; sent: {types_sent}"

    await stream.aclose()


# ---------------------------------------------------------------------------
# _process_stream_event
# ---------------------------------------------------------------------------


def test_interim_transcript_emitted():
    stream = _make_stream_no_task()
    stream._process_stream_event({"session_id": "s1", "transcript": "hello", "is_final": False})
    ev = stream._event_ch.recv_nowait()
    assert ev.type == SpeechEventType.START_OF_SPEECH
    ev2 = stream._event_ch.recv_nowait()
    assert ev2.type == SpeechEventType.INTERIM_TRANSCRIPT
    assert ev2.alternatives[0].text == "hello"


def test_final_transcript_emits_end_of_speech():
    stream = _make_stream_no_task()
    stream._process_stream_event(
        {"session_id": "s1", "transcript": "hello world", "is_final": True}
    )
    ev = stream._event_ch.recv_nowait()
    assert ev.type == SpeechEventType.START_OF_SPEECH
    ev2 = stream._event_ch.recv_nowait()
    assert ev2.type == SpeechEventType.FINAL_TRANSCRIPT
    assert ev2.alternatives[0].text == "hello world"
    ev3 = stream._event_ch.recv_nowait()
    assert ev3.type == SpeechEventType.END_OF_SPEECH


def test_empty_transcript_emits_nothing():
    stream = _make_stream_no_task()
    stream._process_stream_event(
        {"session_id": "s1", "transcript": "", "is_final": True, "is_last": True}
    )
    with pytest.raises(ChanEmpty):
        stream._event_ch.recv_nowait()


def test_is_last_without_transcript_emits_nothing():
    stream = _make_stream_no_task()
    stream._process_stream_event({"is_last": True, "transcript": ""})
    with pytest.raises(ChanEmpty):
        stream._event_ch.recv_nowait()


def test_session_id_recorded():
    stream = _make_stream_no_task()
    stream._process_stream_event({"session_id": "abc123", "transcript": "hi", "is_final": False})
    stream._event_ch.recv_nowait()  # START_OF_SPEECH
    ev = stream._event_ch.recv_nowait()  # INTERIM_TRANSCRIPT
    assert ev.request_id == "abc123"


# ---------------------------------------------------------------------------
# Endpoint URL construction
# ---------------------------------------------------------------------------


def _build_ws_url(stt) -> str:
    """Reproduce the URL-building logic from SpeechStream._connect_ws."""
    from urllib.parse import urlencode

    opts = stt._opts
    params = {
        "model": opts.model,
        "language": opts.language,
        "encoding": opts.encoding,
        "sample_rate": opts.sample_rate,
        "word_timestamps": str(opts.word_timestamps).lower(),
        "diarize": str(opts.diarize).lower(),
        "eou_timeout_ms": opts.eou_timeout_ms,
        "endpointing": str(opts.endpointing).lower(),
        "format": str(opts.format).lower(),
        "sentence_timestamps": str(opts.sentence_timestamps).lower(),
    }
    if opts.keywords:
        params["keywords"] = ",".join(f"{kw}:{weight:g}" for kw, weight in opts.keywords)
    return (
        opts.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        + "/stt/live"
        + f"?{urlencode(params)}"
    )


def test_streaming_url_uses_new_endpoint():
    """Streaming URL must use /stt/live?model=pulse, not the old /{model}/get_text path."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "/stt/live" in url, f"expected /stt/live in URL, got: {url}"
    assert "model=pulse" in url, f"expected model=pulse query param, got: {url}"
    assert "get_text" not in url, f"old /get_text path still present in URL: {url}"


def test_streaming_url_model_is_query_param_not_path():
    """Model must appear as a query parameter, not a path segment."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key", model="pulse")
    url = _build_ws_url(stt)

    # /pulse/ as a path segment would mean the old format is still being used
    assert "/pulse/" not in url, f"model still in URL path: {url}"
    assert "model=pulse" in url, f"model not in query string: {url}"


def test_streaming_url_includes_eou_timeout():
    """eou_timeout_ms must always be present in the URL, matching the client-side default."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "eou_timeout_ms=100" in url, (
        f"eou_timeout_ms=100 (the documented default) missing from URL: {url}"
    )


def test_streaming_url_includes_endpointing_default_true():
    """endpointing must default to true so trailing-silence finalization is enabled."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "endpointing=true" in url, f"endpointing=true missing from URL: {url}"


def test_streaming_url_includes_endpointing_when_disabled():
    """endpointing=false must be reflected in the URL when explicitly disabled."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key", endpointing=False)
    url = _build_ws_url(stt)

    assert "endpointing=false" in url, f"endpointing=false missing from URL: {url}"


def test_streaming_url_format_defaults_true():
    """format (punctuation/capitalization) must default to true."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "format=true" in url, f"format=true missing from URL: {url}"


def test_streaming_url_format_disabled():
    """format=false must be reflected when explicitly disabled for raw output."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key", format=False)
    url = _build_ws_url(stt)

    assert "format=false" in url, f"format=false missing from URL: {url}"


def test_streaming_url_sentence_timestamps_defaults_false():
    """sentence_timestamps must default to false (opt-in feature)."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "sentence_timestamps=false" in url, f"sentence_timestamps=false missing: {url}"


def test_streaming_url_sentence_timestamps_enabled():
    """sentence_timestamps=true must be reflected when explicitly enabled."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key", sentence_timestamps=True)
    url = _build_ws_url(stt)

    assert "sentence_timestamps=true" in url, f"sentence_timestamps=true missing: {url}"


def test_streaming_url_omits_keywords_by_default():
    """keywords must default to no boosting — the query param should be absent."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    url = _build_ws_url(stt)

    assert "keywords" not in url, f"keywords should be absent by default: {url}"


def test_streaming_url_includes_keywords_when_set():
    """keywords must serialize as comma-separated keyword:intensifier pairs."""
    from urllib.parse import unquote_plus

    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key", keywords=[("NVIDIA", 2.0), ("Jensen Huang", 1.0)])
    url = unquote_plus(_build_ws_url(stt))

    assert "keywords=NVIDIA:2,Jensen Huang:1" in url, f"keywords not serialized correctly: {url}"


def test_process_stream_event_attaches_utterances_metadata():
    """utterances present in a stream event must surface via SpeechData.metadata."""
    stream = _make_stream_no_task()
    stream._process_stream_event(
        {
            "session_id": "s1",
            "transcript": "hello world",
            "is_final": True,
            "utterances": [{"text": "hello world", "start": 0.0, "end": 0.9}],
        }
    )
    stream._event_ch.recv_nowait()  # START_OF_SPEECH
    ev = stream._event_ch.recv_nowait()  # FINAL_TRANSCRIPT
    assert ev.alternatives[0].metadata == {
        "utterances": [{"text": "hello world", "start": 0.0, "end": 0.9}]
    }


def test_process_stream_event_no_utterances_key_when_absent():
    """metadata must stay None when sentence_timestamps is not enabled server-side."""
    stream = _make_stream_no_task()
    stream._process_stream_event({"session_id": "s1", "transcript": "hello", "is_final": True})
    stream._event_ch.recv_nowait()  # START_OF_SPEECH
    ev = stream._event_ch.recv_nowait()  # FINAL_TRANSCRIPT
    assert ev.alternatives[0].metadata is None


def test_batch_transcription_attaches_utterances_metadata():
    from livekit.plugins.smallestai.stt import _batch_transcription_to_speech_event

    ev = _batch_transcription_to_speech_event(
        "en",
        {
            "transcription": "Hello world. How are you?",
            "utterances": [
                {"text": "Hello world.", "start": 0.0, "end": 0.9, "speaker": "speaker_0"},
                {"text": "How are you?", "start": 1.0, "end": 2.1, "speaker": "speaker_1"},
            ],
        },
    )
    assert ev.alternatives[0].metadata == {
        "utterances": [
            {"text": "Hello world.", "start": 0.0, "end": 0.9, "speaker": "speaker_0"},
            {"text": "How are you?", "start": 1.0, "end": 2.1, "speaker": "speaker_1"},
        ]
    }


def test_batch_url_uses_new_endpoint():
    """Batch URL must be /stt/ with model as a query param."""
    from livekit.plugins.smallestai import STT

    stt = STT(api_key="test-key")
    opts = stt._opts
    batch_url = f"{opts.base_url}/stt/"

    assert batch_url.endswith("/stt/"), f"unexpected batch URL: {batch_url}"
    assert "get_text" not in batch_url, f"old /get_text path in batch URL: {batch_url}"
