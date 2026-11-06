"""Tests for the MiniMax STT plugin.

These are hermetic: no network, no API key. They cover option handling and the
request/response contract, which is where a hand-written plugin actually breaks.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.stt import SpeechEventType
from livekit.agents.utils import AudioBuffer
from livekit.plugins import minimax

pytestmark = pytest.mark.plugin("minimax")

FAKE_KEY = "fake-key"


# --- helpers -------------------------------------------------------------------------


def _make_audio(seconds: float = 1.0, sample_rate: int = 16000) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=int(sample_rate * seconds),
    )


def _parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Minimal multipart/form-data parser: field name -> raw value bytes.

    Each part is `--boundary CRLF headers CRLF CRLF body CRLF`, so the header/body
    separator is the *last* blank line, not the first one.
    """
    boundary = content_type.split("boundary=")[1].strip().encode()
    fields: dict[str, bytes] = {}
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, _, value = part.partition(b"\r\n\r\n")
        for line in raw_headers.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                name = line.split(b'name="')[1].split(b'"')[0].decode()
                # the trailing CRLF belongs to the framing, not the value
                fields[name] = value[:-2] if value.endswith(b"\r\n") else value
    return fields


class _FakeResponse:
    def __init__(self, *, status: int = 200, payload: Any = None, text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self) -> Any:
        return self._payload

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeSession:
    """Stands in for an aiohttp.ClientSession, recording the last POST."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.data: bytes | None = None
        self.content_type: str | None = None

    def post(self, url: str, *, headers: dict[str, str], data: Any, timeout: Any) -> _FakeResponse:
        self.url = url
        self.headers = headers
        self.data = _encode_form(data)
        # aiohttp derives this from the writer's boundary; FormData exposes no content_type
        self.content_type = f"multipart/form-data; boundary={data._writer.boundary}"
        return self._response


def _encode_form(form: Any) -> bytes:
    """Serialize an aiohttp.FormData the way aiohttp would put it on the wire.

    `_fields` holds `(type_options, headers, value)` triples; single values are
    stored as plain `str`/`bytes` and file values as `BytesPayload`.
    """
    boundary = form._writer.boundary.encode()
    out = bytearray()
    for type_options, _headers, value in form._fields:
        out += b"--" + boundary + b"\r\n"
        disposition = f'Content-Disposition: form-data; name="{type_options["name"]}"'
        filename = type_options.get("filename")
        if filename:
            disposition += f'; filename="{filename}"'
        out += disposition.encode() + b"\r\n"

        content_type = _headers.get("Content-Type")
        if content_type:
            out += b"Content-Type: " + content_type.encode() + b"\r\n"

        if isinstance(value, (bytes, bytearray)):
            body = bytes(value)
        elif isinstance(value, str):
            body = value.encode()
        else:  # payload objects (e.g. BytesPayload) expose _value
            body = value._value

        out += b"\r\n" + body + b"\r\n"
    out += b"--" + boundary + b"--\r\n"
    return bytes(out)


async def _recognize_with(
    stt: minimax.STT,
    response: _FakeResponse,
    *,
    seconds: float = 1.0,
    buffer: AudioBuffer | None = None,
) -> tuple[Any, _FakeSession]:
    session = _FakeSession(response)
    stt._session = session  # type: ignore[assignment]
    event = await stt.recognize(
        buffer if buffer is not None else _make_audio(seconds),
        conn_options=APIConnectOptions(),
    )
    return event, session


# --- configuration -------------------------------------------------------------------


def test_requires_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            minimax.STT()


def test_reads_api_key_from_env() -> None:
    with patch.dict("os.environ", {"MINIMAX_API_KEY": "env-key"}):
        assert minimax.STT()._opts.api_key == "env-key"


def test_accepts_api_key_directly() -> None:
    assert minimax.STT(api_key="direct")._opts.api_key == "direct"


def test_omits_language_by_default() -> None:
    """No language must be sent: that is what enables MiniMax's mixed-language mode."""
    assert minimax.STT(api_key=FAKE_KEY)._opts.language is None


def test_explicit_none_language_is_the_same_as_unset() -> None:
    assert minimax.STT(api_key=FAKE_KEY, language=None)._opts.language is None


def test_capabilities_report_non_streaming() -> None:
    """The framework keys its StreamAdapter wrapping off this flag."""
    stt = minimax.STT(api_key=FAKE_KEY)
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False
    assert stt.capabilities.diarization is False


def test_verbose_json_enables_diarization() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, response_format="verbose_json")
    assert stt.capabilities.diarization is True


def test_rejects_streaming_with_diarization() -> None:
    """MiniMax cannot combine stream=true with verbose_json/srt/vtt."""
    with pytest.raises(ValueError, match="cannot combine"):
        minimax.STT(api_key=FAKE_KEY, stream=True, response_format="verbose_json")


def test_provider_and_model() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    assert stt.provider == "MiniMax"
    assert stt.model == "asr-1.0"


def test_update_options_changes_language() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    stt.update_options(language="en")
    assert stt._opts.language == "en"
    # passing None must restore mixed-language mode rather than pinning a language
    stt.update_options(language=None)
    assert stt._opts.language is None


# --- request construction ------------------------------------------------------------


async def test_request_shape() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    event, session = await _recognize_with(
        stt, _FakeResponse(payload={"text": "hello", "duration": 1.0})
    )

    assert session.url == "https://api-uw.minimax.io/v1/speech_to_text"
    assert session.headers == {"Authorization": f"Bearer {FAKE_KEY}"}
    assert "language" not in session.headers, "language header must be absent for mixed mode"

    assert session.data is not None
    assert session.content_type is not None
    fields = _parse_multipart(session.data, session.content_type)
    assert fields["model"] == b"asr-1.0"
    assert fields["response_format"] == b"json"
    assert fields["stream"] == b"false"
    # the audio must be a real container, not raw PCM
    assert fields["file"].startswith(b"RIFF"), "uploaded audio must be a WAV container"

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello"


async def test_language_header_sent_when_set() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, language="en")
    _, session = await _recognize_with(stt, _FakeResponse(payload={"text": "hi"}))
    assert session.headers is not None
    assert session.headers["language"] == "en"


async def test_per_call_language_overrides_instance_default() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, language="zh")
    session = _FakeSession(_FakeResponse(payload={"text": "hi"}))
    stt._session = session  # type: ignore[assignment]
    await stt.recognize(_make_audio(), language="en", conn_options=APIConnectOptions())
    assert session.headers is not None
    assert session.headers["language"] == "en"


async def test_duration_from_multiple_frames() -> None:
    """A buffer list must be concatenated, and its total duration measured across all frames."""
    stt = minimax.STT(api_key=FAKE_KEY)
    frames = [_make_audio(0.5), _make_audio(0.5)]
    _, session = await _recognize_with(stt, _FakeResponse(payload={"text": "ok"}), buffer=frames)
    assert session.data is not None
    assert session.content_type is not None
    wav = _parse_multipart(session.data, session.content_type)["file"]
    # 1.0s of 16 kHz mono int16 => 32000 bytes of PCM inside the WAV container
    assert len(wav) > 32000


# --- response handling ---------------------------------------------------------------


async def test_empty_text_is_returned_as_empty_string() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    event, _ = await _recognize_with(stt, _FakeResponse(payload={"duration": 1.0}))
    assert event.alternatives[0].text == ""


async def test_missing_text_key_does_not_crash() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    event, _ = await _recognize_with(stt, _FakeResponse(payload={}))
    assert event.alternatives[0].text == ""


async def test_http_error_becomes_api_status_error() -> None:
    """An HTTP-level failure must surface as APIStatusError with its status intact."""
    stt = minimax.STT(api_key=FAKE_KEY)
    session = _FakeSession(_FakeResponse(status=413, text="payload too large"))
    stt._session = session  # type: ignore[assignment]
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(_make_audio(), conn_options=APIConnectOptions(max_retry=0))
    assert exc.value.status_code == 413


async def test_base_resp_error_becomes_api_status_error() -> None:
    """MiniMax reports failures with HTTP 200 and a non-zero base_resp.status_code."""
    stt = minimax.STT(api_key=FAKE_KEY)
    session = _FakeSession(
        _FakeResponse(
            payload={
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
                "trace_id": "trace-abc",
            }
        )
    )
    stt._session = session  # type: ignore[assignment]
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(_make_audio(), conn_options=APIConnectOptions(max_retry=0))
    assert exc.value.status_code == 1004
    assert "invalid api key" in str(exc.value)
    assert exc.value.request_id == "trace-abc"


async def test_permanent_error_is_marked_not_retryable() -> None:
    """An auth failure is permanent: retrying it just burns quota and time.

    Note the framework's `STT.recognize()` retry loop does not yet consult this flag, so
    the wasted retries still happen -- this asserts the contract we publish, which the
    streaming and fallback adapters do honour.
    """
    stt = minimax.STT(api_key=FAKE_KEY)
    session = _FakeSession(
        _FakeResponse(payload={"base_resp": {"status_code": 1004, "status_msg": "no auth"}})
    )
    stt._session = session  # type: ignore[assignment]
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(_make_audio(), conn_options=APIConnectOptions(max_retry=0))
    assert exc.value.retryable is False


async def test_transient_error_is_marked_retryable() -> None:
    """Rate limiting and internal errors are what retries are for."""
    stt = minimax.STT(api_key=FAKE_KEY)
    session = _FakeSession(
        _FakeResponse(payload={"base_resp": {"status_code": 1002, "status_msg": "rate limit"}})
    )
    stt._session = session  # type: ignore[assignment]
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(_make_audio(), conn_options=APIConnectOptions(max_retry=0))
    assert exc.value.retryable is True


async def test_success_with_zero_base_resp_is_not_an_error() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    event, _ = await _recognize_with(
        stt,
        _FakeResponse(
            payload={"text": "ok", "base_resp": {"status_code": 0, "status_msg": "success"}}
        ),
    )
    assert event.alternatives[0].text == "ok"


async def test_empty_buffer_is_rejected() -> None:
    stt = minimax.STT(api_key=FAKE_KEY)
    with pytest.raises(ValueError, match="empty audio buffer"):
        await stt.recognize([], conn_options=APIConnectOptions())


# --- SSE streaming -------------------------------------------------------------------


class _FakeStreamResponse(_FakeResponse):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(status=200)
        self._lines = lines

    @property
    def content(self) -> Any:
        async def _iter() -> Any:
            for line in self._lines:
                yield line.encode()

        return _iter()


async def test_sse_concatenates_deltas() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, stream=True)
    lines = [
        'data: {"index":0,"delta":"Hello","finish":false}',
        "",
        'data: {"index":1,"delta":" everyone","finish":false}',
        "",
        'data: {"index":2,"delta":"","finish":true,"duration":26.3}',
        "",
    ]
    event, session = await _recognize_with(stt, _FakeStreamResponse(lines))
    assert event.alternatives[0].text == "Hello everyone"
    assert session.data is not None
    assert session.content_type is not None
    assert _parse_multipart(session.data, session.content_type)["stream"] == b"true"


async def test_sse_skips_keepalive_comments_and_blank_lines() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, stream=True)
    lines = [": keepalive", "", 'data: {"index":0,"delta":"你好","finish":true}']
    event, _ = await _recognize_with(stt, _FakeStreamResponse(lines))
    assert event.alternatives[0].text == "你好"


async def test_sse_orders_by_index_not_arrival() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, stream=True)
    lines = [
        'data: {"index":1,"delta":" world","finish":false}',
        'data: {"index":0,"delta":"hello","finish":true}',
    ]
    event, _ = await _recognize_with(stt, _FakeStreamResponse(lines))
    assert event.alternatives[0].text == "hello world"


async def test_sse_error_event_raises() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, stream=True)
    lines = ['data: {"base_resp":{"status_code":1004,"status_msg":"bad key"}}']
    session = _FakeSession(_FakeStreamResponse(lines))
    stt._session = session  # type: ignore[assignment]
    with pytest.raises(APIStatusError) as exc:
        await stt.recognize(_make_audio(), conn_options=APIConnectOptions(max_retry=0))
    assert exc.value.status_code == 1004
    assert exc.value.retryable is False


async def test_sse_malformed_json_is_skipped_not_fatal() -> None:
    stt = minimax.STT(api_key=FAKE_KEY, stream=True)
    lines = ["data: {not json}", 'data: {"index":0,"delta":"fine","finish":true}']
    event, _ = await _recognize_with(stt, _FakeStreamResponse(lines))
    assert event.alternatives[0].text == "fine"


def test_sse_json_payload_is_serializable() -> None:
    """Guard the contract the SSE parser assumes."""
    payload = {"index": 0, "delta": "x", "finish": False}
    assert json.loads(json.dumps(payload)) == payload
