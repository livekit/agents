from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    tts,
    utils,
)
from livekit.plugins.sarvam import tts as sarvam_tts_module
from livekit.plugins.sarvam.tts import (
    _ALAW_TABLE,
    _CODEC_TO_MIME,
    _MULAW_TABLE,
    _TELEPHONY_CODECS,
    ALLOWED_OUTPUT_AUDIO_CODECS,
    TTS,
    ConnectionState,
    SynthesizeStream,
    _codec_to_mime_type,
    _decode_telephony,
)


@pytest.fixture(autouse=True)
async def _cleanup_leaked_keepalive_tasks() -> AsyncIterator[None]:
    """Cancel any orphan ``sarvam-tts-ws-keepalive`` tasks at test teardown.

    Tests that exercise the success path of ``_run_ws`` leave a freshly
    spawned keepalive task running on the TTS instance. Real users would
    clean it up via ``TTS.aclose()`` (which closes the pool and so
    ``_close_ws``), but tests use ``_FakePool`` and never close the TTS,
    so we need this safety net to avoid pytest's "leaked tasks" warnings.
    """
    yield
    for task in list(asyncio.all_tasks()):
        if task.get_name() == "sarvam-tts-ws-keepalive" and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


# ---------------------------------------------------------------------------
# Bug 1: _codec_to_mime_type mapping
# ---------------------------------------------------------------------------


class TestCodecToMimeType:
    def test_mp3(self) -> None:
        assert _codec_to_mime_type("mp3") == "audio/mp3"

    def test_wav(self) -> None:
        assert _codec_to_mime_type("wav") == "audio/wav"

    def test_opus(self) -> None:
        assert _codec_to_mime_type("opus") == "audio/opus"

    def test_flac(self) -> None:
        assert _codec_to_mime_type("flac") == "audio/flac"

    def test_aac(self) -> None:
        assert _codec_to_mime_type("aac") == "audio/aac"

    def test_linear16_maps_to_pcm(self) -> None:
        assert _codec_to_mime_type("linear16") == "audio/pcm"

    def test_mulaw_maps_to_audio_pcm(self) -> None:
        # mulaw is decoded to linear PCM in the plugin before being pushed
        assert _codec_to_mime_type("mulaw") == "audio/pcm"

    def test_alaw_maps_to_audio_pcm(self) -> None:
        # alaw is decoded to linear PCM in the plugin before being pushed
        assert _codec_to_mime_type("alaw") == "audio/pcm"

    def test_unsupported_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported output_audio_codec"):
            _codec_to_mime_type("unknown_codec")

    def test_all_allowed_codecs_have_mime_mapping(self) -> None:
        for codec in ALLOWED_OUTPUT_AUDIO_CODECS:
            mime = _codec_to_mime_type(codec)
            assert mime.startswith("audio/"), f"{codec} -> {mime} is not a valid MIME"

    def test_codec_to_mime_dict_covers_allowed_codecs(self) -> None:
        assert set(_CODEC_TO_MIME.keys()) == ALLOWED_OUTPUT_AUDIO_CODECS


# ---------------------------------------------------------------------------
# Bug 1: ALLOWED_OUTPUT_AUDIO_CODECS includes telephony/PCM codecs
# ---------------------------------------------------------------------------


class TestAllowedCodecs:
    def test_includes_original_codecs(self) -> None:
        for codec in ("mp3", "opus", "flac", "aac", "wav"):
            assert codec in ALLOWED_OUTPUT_AUDIO_CODECS

    def test_includes_telephony_and_pcm_codecs(self) -> None:
        for codec in ("linear16", "mulaw", "alaw"):
            assert codec in ALLOWED_OUTPUT_AUDIO_CODECS


# ---------------------------------------------------------------------------
# Bug 1: TTS constructor validation accepts new codecs
# ---------------------------------------------------------------------------


class TestTTSConstructorCodecValidation:
    def test_rejects_unknown_codec(self) -> None:
        with pytest.raises(ValueError, match="output_audio_codec must be one of"):
            TTS(
                target_language_code="hi-IN",
                api_key="sk_test",
                output_audio_codec="ogg",
            )

    def test_accepts_linear16(self) -> None:
        t = TTS(
            target_language_code="hi-IN",
            api_key="sk_test",
            output_audio_codec="linear16",
        )
        assert t._opts.output_audio_codec == "linear16"

    def test_accepts_mulaw(self) -> None:
        t = TTS(
            target_language_code="hi-IN",
            api_key="sk_test",
            output_audio_codec="mulaw",
        )
        assert t._opts.output_audio_codec == "mulaw"

    def test_accepts_alaw(self) -> None:
        t = TTS(
            target_language_code="hi-IN",
            api_key="sk_test",
            output_audio_codec="alaw",
        )
        assert t._opts.output_audio_codec == "alaw"

    def test_default_codec_is_mp3(self) -> None:
        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        assert t._opts.output_audio_codec == "mp3"


# ---------------------------------------------------------------------------
# Bug 1: update_options validates new codecs
# ---------------------------------------------------------------------------


class TestUpdateOptionsCodecValidation:
    def test_update_rejects_unknown_codec(self) -> None:
        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        with pytest.raises(ValueError, match="output_audio_codec must be one of"):
            t.update_options(output_audio_codec="ogg")

    def test_update_accepts_linear16(self) -> None:
        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t.update_options(output_audio_codec="linear16")
        assert t._opts.output_audio_codec == "linear16"


# ---------------------------------------------------------------------------
# Bug 2: WebSocket config always includes output_audio_codec
# ---------------------------------------------------------------------------


class _FakeWS:
    """Minimal fake WebSocket that records sent messages and then closes."""

    def __init__(
        self,
        recv_messages: list[SimpleNamespace] | None = None,
        *,
        close_code: int | None = 1000,
        send_error: BaseException | None = None,
    ) -> None:
        self.sent: list[str] = []
        self._recv_messages = iter(recv_messages or [])
        self.close_code = close_code
        self._send_error = send_error
        self.closed = False

    async def send_str(self, data: str) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(data)

    async def receive(self, timeout: float = 30.0) -> SimpleNamespace:
        try:
            return next(self._recv_messages)
        except StopIteration:
            return SimpleNamespace(
                type=aiohttp.WSMsgType.CLOSED,
                data=self.close_code,
                extra=None,
            )

    async def close(self) -> None:
        self.closed = True


def _build_audio_response(audio_b64: str = "AAAA") -> SimpleNamespace:
    return SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps({"type": "audio", "data": {"audio": audio_b64}}),
        extra=None,
    )


def _build_final_event() -> SimpleNamespace:
    return SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps({"type": "event", "data": {"event_type": "final"}}),
        extra=None,
    )


def _make_stream(
    model: str = "bulbul:v3",
    codec: str = "mp3",
) -> SynthesizeStream:
    """Build a SynthesizeStream with minimal setup via object.__new__."""
    t = TTS(
        target_language_code="hi-IN",
        api_key="sk_test",
        model=model,
        output_audio_codec=codec,
    )
    stream = object.__new__(SynthesizeStream)
    stream._tts = t
    from dataclasses import replace

    stream._opts = replace(t._opts)
    stream._segments_ch = utils.aio.Chan()
    stream._connection_state = ConnectionState.DISCONNECTED
    stream._session_id = id(stream)
    stream._client_request_id = None
    stream._server_request_id = None
    stream._send_task = None
    stream._recv_task = None
    stream._ws_conn = None
    stream._conn_options = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
    return stream


@pytest.mark.asyncio
async def test_ws_config_always_includes_output_audio_codec_v3() -> None:
    """output_audio_codec must be in the config for bulbul:v3."""
    stream = _make_stream(model="bulbul:v3", codec="opus")

    audio_msg = _build_audio_response()
    final_msg = _build_final_event()
    ws = _FakeWS(recv_messages=[audio_msg, final_msg])

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    await stream._run_ws(_FakeWordStream(), emitter)

    assert len(ws.sent) >= 1
    config_raw = json.loads(ws.sent[0])
    assert config_raw["type"] == "config"
    config_data = config_raw["data"]
    assert "output_audio_codec" in config_data
    assert config_data["output_audio_codec"] == "opus"
    assert "output_audio_bitrate" in config_data
    assert "min_buffer_size" in config_data
    assert "max_chunk_length" in config_data


@pytest.mark.asyncio
async def test_ws_config_v3_includes_temperature_but_not_pitch() -> None:
    stream = _make_stream(model="bulbul:v3", codec="mp3")

    ws = _FakeWS(recv_messages=[_build_audio_response(), _build_final_event()])

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    await stream._run_ws(_FakeWordStream(), emitter)

    config_data = json.loads(ws.sent[0])["data"]
    assert "temperature" in config_data
    assert "pitch" not in config_data
    assert "loudness" not in config_data


# ---------------------------------------------------------------------------
# Bug 3: Interruption handling – send_task transport errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_task_raises_api_connection_error_on_closing_transport() -> None:
    """``ConnectionResetError("...closing transport")`` must raise as
    ``APIConnectionError`` so the pool evicts the dead connection.

    Returning silently here would let the pool's ``__aexit__`` see a clean
    exit and put the dead connection back, causing the next request to
    fail again on the same socket.
    """
    stream = _make_stream()

    err = ConnectionResetError("Cannot write to closing transport")
    ws = _FakeWS(
        recv_messages=[_build_final_event()],
        send_error=err,
    )

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    with pytest.raises(APIConnectionError, match="transport closed before send"):
        await stream._run_ws(_FakeWordStream(), emitter)


@pytest.mark.asyncio
async def test_send_task_raises_on_non_closing_connection_reset() -> None:
    """ConnectionResetError without 'closing' should propagate as APIConnectionError."""
    stream = _make_stream()

    err = ConnectionResetError("peer reset connection unexpectedly")
    ws = _FakeWS(
        recv_messages=[_build_final_event()],
        send_error=err,
    )

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    with pytest.raises(APIConnectionError, match="Send task failed"):
        await stream._run_ws(_FakeWordStream(), emitter)


@pytest.mark.asyncio
async def test_send_task_raises_api_connection_error_on_runtime_closed() -> None:
    """``RuntimeError("...closed")`` from aiohttp must surface as
    ``APIConnectionError`` so the dead connection gets evicted from the pool.
    """
    stream = _make_stream()

    err = RuntimeError("WebSocket connection is closed")
    ws = _FakeWS(
        recv_messages=[_build_final_event()],
        send_error=err,
    )

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    with pytest.raises(APIConnectionError, match="transport closed before send"):
        await stream._run_ws(_FakeWordStream(), emitter)


# ---------------------------------------------------------------------------
# Bug 3: Interruption handling – recv_task transport errors
# ---------------------------------------------------------------------------


class _FakeWSRecvError:
    """WS where receive() raises a specific error."""

    def __init__(self, error: BaseException) -> None:
        self.sent: list[str] = []
        self._error = error
        self.close_code = None
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float = 30.0) -> SimpleNamespace:
        raise self._error

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_recv_task_raises_api_connection_error_on_closing_transport() -> None:
    """``ClientConnectionResetError`` from auto-PONG must raise as
    ``APIConnectionError`` so the pool evicts the dead connection.
    """
    stream = _make_stream()

    err = aiohttp.ClientConnectionResetError("Cannot write to closing transport")
    ws = _FakeWSRecvError(error=err)

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    with pytest.raises(APIConnectionError, match="transport closed during receive"):
        await stream._run_ws(_FakeWordStream(), emitter)


@pytest.mark.asyncio
async def test_recv_task_raises_on_non_closing_connection_reset() -> None:
    """ConnectionResetError without transport-closing message should propagate."""
    stream = _make_stream()

    err = ConnectionResetError("peer reset connection unexpectedly")
    ws = _FakeWSRecvError(error=err)

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    with pytest.raises((ConnectionResetError, APIStatusError)):
        await stream._run_ws(_FakeWordStream(), emitter)


# ---------------------------------------------------------------------------
# Bug 3: input_sent_event coordination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_sent_event_unblocks_recv_on_send_failure() -> None:
    """recv_task must not deadlock if send_task fails before setting event."""
    stream = _make_stream()

    err = ValueError("simulated config error")
    ws = _FakeWS(
        recv_messages=[_build_final_event()],
        send_error=err,
    )

    emitter = MagicMock(spec=tts.AudioEmitter)
    emitter.start_segment = MagicMock()
    emitter.push = MagicMock()
    emitter.end_input = MagicMock()

    class _FakeWordStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream._mark_started = MagicMock()

    class _FakePool:
        last_acquire_time = 0.0
        last_connection_reused = False

        class _ctx:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *args):
                pass

        def connection(self, *, timeout):
            return self._ctx(ws)

    stream._tts._pool = _FakePool()  # type: ignore[assignment]

    # Should complete (not deadlock) within timeout.
    # The send_task fails with ValueError -> wrapped as APIConnectionError,
    # but the finally block sets input_sent_event so recv_task unblocks.
    with pytest.raises((APIConnectionError, APIStatusError)):
        await asyncio.wait_for(
            stream._run_ws(_FakeWordStream(), emitter),
            timeout=5.0,
        )


# ---------------------------------------------------------------------------
# Bug 3: Simplified aclose()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_segments_channel() -> None:
    """aclose() should close the segments channel and delegate to super."""
    t = TTS(target_language_code="hi-IN", api_key="sk_test")
    stream = t.stream()

    assert not stream._segments_ch.closed
    stream.end_input()
    await stream.aclose()
    assert stream._segments_ch.closed


# ---------------------------------------------------------------------------
# TTS _handle_websocket_message tests
# ---------------------------------------------------------------------------


class TestHandleWebsocketMessage:
    def _make_stream(self) -> SynthesizeStream:
        return _make_stream()

    @pytest.mark.asyncio
    async def test_audio_message_pushes_to_emitter(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        audio_b64 = base64.b64encode(b"\x00\x01\x02").decode()
        msg = json.dumps({"type": "audio", "data": {"audio": audio_b64}})

        result = await stream._handle_websocket_message(msg, emitter)
        assert result is True
        emitter.push.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_message_non_recoverable_raises_status_error(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)

        msg = json.dumps(
            {
                "type": "error",
                "data": {"message": "invalid speaker", "code": "400"},
            }
        )

        with pytest.raises(APIStatusError, match="TTS API error from Sarvam"):
            await stream._handle_websocket_message(msg, emitter)

    @pytest.mark.asyncio
    async def test_error_message_recoverable_raises_connection_error(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)

        msg = json.dumps(
            {
                "type": "error",
                "data": {"message": "rate_limit exceeded", "code": "429"},
            }
        )

        with pytest.raises(APIConnectionError, match="Recoverable"):
            await stream._handle_websocket_message(msg, emitter)

    @pytest.mark.asyncio
    async def test_final_event_returns_false(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.end_input = MagicMock()

        msg = json.dumps({"type": "event", "data": {"event_type": "final"}})

        result = await stream._handle_websocket_message(msg, emitter)
        assert result is False
        emitter.end_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_true(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)

        result = await stream._handle_websocket_message("not json{", emitter)
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_message_type_returns_true(self) -> None:
        stream = self._make_stream()
        emitter = MagicMock(spec=tts.AudioEmitter)

        msg = json.dumps({"type": "ping", "data": {}})

        result = await stream._handle_websocket_message(msg, emitter)
        assert result is True


# ---------------------------------------------------------------------------
# G.711 telephony codec decoding (mulaw / alaw)
# ---------------------------------------------------------------------------


class TestTelephonyCodecsConstants:
    def test_telephony_codecs_set(self) -> None:
        assert _TELEPHONY_CODECS == frozenset({"mulaw", "alaw"})

    def test_mulaw_table_shape_and_dtype(self) -> None:
        import numpy as np

        assert _MULAW_TABLE.shape == (256,)
        assert _MULAW_TABLE.dtype == np.int16

    def test_alaw_table_shape_and_dtype(self) -> None:
        import numpy as np

        assert _ALAW_TABLE.shape == (256,)
        assert _ALAW_TABLE.dtype == np.int16


class TestMulawDecoding:
    """Reference values come from ITU-T G.711 / standard mu-law decode tables."""

    def test_silence_byte_decodes_to_zero(self) -> None:
        # 0xFF is the canonical mu-law silence byte
        out = _decode_telephony("mulaw", b"\xff")
        assert out == b"\x00\x00"

    def test_max_positive_byte(self) -> None:
        # 0x80 -> largest positive sample
        out = _decode_telephony("mulaw", b"\x80")
        # int16 little-endian
        sample = int.from_bytes(out, byteorder="little", signed=True)
        assert sample == 32124

    def test_max_negative_byte(self) -> None:
        # 0x00 -> largest negative sample
        out = _decode_telephony("mulaw", b"\x00")
        sample = int.from_bytes(out, byteorder="little", signed=True)
        assert sample == -32124

    def test_output_length_doubles(self) -> None:
        # 8-bit input -> 16-bit output
        out = _decode_telephony("mulaw", b"\xff\x80\x00\x7f")
        assert len(out) == 8

    def test_sequence_decodes_correctly(self) -> None:
        out = _decode_telephony("mulaw", b"\xff\x80\x00")
        samples = [
            int.from_bytes(out[i : i + 2], byteorder="little", signed=True)
            for i in range(0, len(out), 2)
        ]
        assert samples == [0, 32124, -32124]

    def test_empty_input(self) -> None:
        assert _decode_telephony("mulaw", b"") == b""

    def test_table_is_symmetric_around_zero(self) -> None:
        # mu-law sign bit flips → samples should be exact negatives across pairs
        # (e.g. byte 0x80 ↔ 0x00 give ±32124)
        assert _MULAW_TABLE[0x80] == -_MULAW_TABLE[0x00]
        assert _MULAW_TABLE[0xFF] == 0  # silence
        assert _MULAW_TABLE[0x7F] == 0  # negative-side silence


class TestAlawDecoding:
    """Reference values come from ITU-T G.711 / standard A-law decode tables."""

    def test_silence_byte_decodes_to_eight(self) -> None:
        # A-law silence byte is 0xD5; decoded to a small positive value (8)
        out = _decode_telephony("alaw", b"\xd5")
        sample = int.from_bytes(out, byteorder="little", signed=True)
        assert sample == 8

    def test_max_positive_byte(self) -> None:
        # 0xAA -> max positive A-law sample (~32256)
        out = _decode_telephony("alaw", b"\xaa")
        sample = int.from_bytes(out, byteorder="little", signed=True)
        assert sample == 32256

    def test_max_negative_byte(self) -> None:
        # 0x2A -> max negative A-law sample
        out = _decode_telephony("alaw", b"\x2a")
        sample = int.from_bytes(out, byteorder="little", signed=True)
        assert sample == -32256

    def test_output_length_doubles(self) -> None:
        out = _decode_telephony("alaw", b"\x00\x55\xaa\xff")
        assert len(out) == 8

    def test_empty_input(self) -> None:
        assert _decode_telephony("alaw", b"") == b""

    def test_all_byte_values_in_int16_range(self) -> None:
        for i in range(256):
            v = int(_ALAW_TABLE[i])
            assert -32768 <= v <= 32767


class TestDecodeTelephonyContract:
    def test_unsupported_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support codec"):
            _decode_telephony("mp3", b"\x00")

    def test_returns_bytes(self) -> None:
        assert isinstance(_decode_telephony("mulaw", b"\xff"), bytes)
        assert isinstance(_decode_telephony("alaw", b"\xd5"), bytes)

    def test_large_buffer_works(self) -> None:
        # Realistic chunk size (e.g. 20ms of 8kHz mulaw = 160 bytes)
        data = bytes(range(256)) * 4
        out = _decode_telephony("mulaw", data)
        assert len(out) == len(data) * 2


class TestHandleAudioMessageTelephonyDecoding:
    """Verify _handle_audio_message decodes mulaw/alaw before pushing."""

    def _make_stream(self, codec: str) -> SynthesizeStream:
        t = TTS(
            target_language_code="hi-IN",
            api_key="sk_test",
            output_audio_codec=codec,
        )
        stream = object.__new__(SynthesizeStream)
        from dataclasses import replace

        stream._tts = t
        stream._opts = replace(t._opts)
        stream._segments_ch = utils.aio.Chan()
        stream._connection_state = ConnectionState.DISCONNECTED
        stream._session_id = id(stream)
        stream._client_request_id = None
        stream._server_request_id = None
        stream._send_task = None
        stream._recv_task = None
        stream._ws_conn = None
        stream._conn_options = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
        return stream

    @pytest.mark.asyncio
    async def test_mulaw_audio_is_decoded_before_push(self) -> None:
        stream = self._make_stream("mulaw")
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        # 4 bytes of mulaw -> should become 8 bytes of pcm
        raw = b"\xff\x80\x00\x7f"
        b64 = base64.b64encode(raw).decode()
        resp = {"type": "audio", "data": {"audio": b64}}

        result = await stream._handle_audio_message(resp, emitter)
        assert result is True
        emitter.push.assert_called_once()
        pushed = emitter.push.call_args[0][0]
        assert len(pushed) == 8  # 4 mulaw samples → 4 int16 = 8 bytes
        # First sample (0xFF) must decode to silence
        assert pushed[:2] == b"\x00\x00"

    @pytest.mark.asyncio
    async def test_alaw_audio_is_decoded_before_push(self) -> None:
        stream = self._make_stream("alaw")
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        raw = b"\xd5\xaa\x2a\x00"  # silence, max+, max-, neg-zero
        b64 = base64.b64encode(raw).decode()
        resp = {"type": "audio", "data": {"audio": b64}}

        result = await stream._handle_audio_message(resp, emitter)
        assert result is True
        pushed = emitter.push.call_args[0][0]
        assert len(pushed) == 8

    @pytest.mark.asyncio
    async def test_mp3_audio_is_passed_through_unchanged(self) -> None:
        stream = self._make_stream("mp3")
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        raw = b"\xff\xfb\x90\x00fakemp3frame"
        b64 = base64.b64encode(raw).decode()
        resp = {"type": "audio", "data": {"audio": b64}}

        result = await stream._handle_audio_message(resp, emitter)
        assert result is True
        pushed = emitter.push.call_args[0][0]
        # MP3 must NOT be decoded by the telephony helper
        assert pushed == raw

    @pytest.mark.asyncio
    async def test_linear16_audio_is_passed_through_unchanged(self) -> None:
        stream = self._make_stream("linear16")
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        raw = b"\x00\x10\xff\x7f\x00\x80"
        b64 = base64.b64encode(raw).decode()
        resp = {"type": "audio", "data": {"audio": b64}}

        await stream._handle_audio_message(resp, emitter)
        pushed = emitter.push.call_args[0][0]
        assert pushed == raw

    @pytest.mark.asyncio
    async def test_empty_audio_skipped(self) -> None:
        stream = self._make_stream("mulaw")
        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.push = MagicMock()

        resp = {"type": "audio", "data": {"audio": ""}}
        result = await stream._handle_audio_message(resp, emitter)
        assert result is True
        emitter.push.assert_not_called()


# ---------------------------------------------------------------------------
# WebSocket keepalive (ping every 30s to defeat Sarvam's 60s idle timeout)
# ---------------------------------------------------------------------------


class _FakeKeepaliveWS:
    """A tiny stand-in for aiohttp.ClientWebSocketResponse.

    The new active-drain keepalive loop calls ``ws.receive()`` and only
    sends an app-level ping when ``receive()`` times out. This fake mirrors
    that contract:

    - ``receive()`` pops queued messages first (used to inject CLOSE,
      PONG-like, or arbitrary frames). When the queue is empty it blocks
      until the transport is closed, at which point it returns a synthetic
      CLOSED message. ``asyncio.wait_for`` is what produces the
      ``TimeoutError`` that drives the ping cadence.
    - ``send_str`` records calls; can be configured to raise.
    """

    def __init__(
        self,
        *,
        send_error: BaseException | None = None,
        recv_messages: list[SimpleNamespace] | None = None,
    ) -> None:
        self.sent: list[str] = []
        self.closed: bool = False
        self.close_code: int | None = None
        self._send_error = send_error
        self._recv_queue: list[SimpleNamespace] = list(recv_messages or [])

    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError("Cannot write to closing transport")
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(data)

    async def receive(self) -> SimpleNamespace:
        if self._recv_queue:
            return self._recv_queue.pop(0)
        # Block until the test marks ``ws.closed`` -- the keepalive loop
        # uses ``asyncio.wait_for(receive(), timeout=...)`` so the loop's
        # ping cadence is driven by the timeout firing here.
        while not self.closed:
            await asyncio.sleep(0.001)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

    async def close(self) -> None:
        self.closed = True

    def queue_recv(self, msg: SimpleNamespace) -> None:
        """Push a message that the next ``receive()`` call will return."""
        self._recv_queue.append(msg)


def _make_tts_with_short_keepalive(monkeypatch: pytest.MonkeyPatch, interval: float) -> TTS:
    """Build a TTS instance with the keepalive interval shrunk for tests."""
    monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", interval)
    return TTS(target_language_code="hi-IN", api_key="sk_test")


class TestKeepaliveLifecycle:
    @pytest.mark.asyncio
    async def test_start_keepalive_registers_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.05)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        assert id(ws) in t._ws_keepalive_tasks
        task = t._ws_keepalive_tasks[id(ws)]
        assert not task.done()

        await t._stop_keepalive(ws)  # type: ignore[arg-type]
        assert id(ws) not in t._ws_keepalive_tasks
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_keepalive_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.05)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        await t._stop_keepalive(ws)  # type: ignore[arg-type]
        # Second stop must not raise even though no task is registered.
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_disabled_keepalive_when_interval_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.0)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]
        assert id(ws) not in t._ws_keepalive_tasks


class TestKeepaliveBehaviour:
    @pytest.mark.asyncio
    async def test_sends_periodic_ping_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        # Wait long enough for at least 3 receive() timeouts -> pings
        await asyncio.sleep(0.1)
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

        assert len(ws.sent) >= 3
        for raw in ws.sent:
            payload = json.loads(raw)
            assert payload == {"type": "ping"}

    @pytest.mark.asyncio
    async def test_loop_exits_when_ws_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        # Marking ws closed makes the fake's blocking receive() return CLOSED;
        # the keepalive sees CLOSE -> evicts -> exits cleanly.
        ws.closed = True
        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done() and task.exception() is None

    @pytest.mark.asyncio
    async def test_loop_exits_silently_on_send_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)
        # Simulate a transport that has already been torn down by the server.
        ws = _FakeKeepaliveWS(send_error=ConnectionResetError("Cannot write to closing transport"))
        t._start_keepalive(ws)  # type: ignore[arg-type]

        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done() and task.exception() is None
        # No successful sends should have been recorded.
        assert ws.sent == []


class TestKeepaliveActiveDrain:
    """The new active-drain keepalive must call ``ws.receive()`` so that
    aiohttp's protocol-level heartbeat can process incoming PONG frames.
    Without that, a heartbeat-armed but unread WS gets torn down by aiohttp
    locally even though the server is healthy.
    """

    @pytest.mark.asyncio
    async def test_receive_drives_loop_no_ping_when_messages_flow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If the server keeps sending things (PONGs, etc.), receive() never
        # times out, so the loop never sends an app-level ping.
        monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", 0.5)

        # Pre-load a big stream of harmless TEXT messages so receive() always
        # returns immediately and never times out within the test window.
        msgs = [
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data="{}",
                extra=None,
            )
            for _ in range(100)
        ]
        ws = _FakeKeepaliveWS(recv_messages=msgs)

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._start_keepalive(ws)  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

        assert ws.sent == [], "no ping should fire while server messages flow"

    @pytest.mark.asyncio
    async def test_receive_close_message_evicts_and_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", 1.0)

        removed: list[object] = []

        class _FakePool:
            def remove(self, conn: object) -> None:
                removed.append(conn)

        ws = _FakeKeepaliveWS(
            recv_messages=[SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=1006, extra=None)]
        )

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._pool = _FakePool()  # type: ignore[assignment]
        t._start_keepalive(ws)  # type: ignore[arg-type]

        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)

        assert removed == [ws]

    @pytest.mark.asyncio
    async def test_receive_error_message_evicts_and_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", 1.0)

        removed: list[object] = []

        class _FakePool:
            def remove(self, conn: object) -> None:
                removed.append(conn)

        ws = _FakeKeepaliveWS(
            recv_messages=[SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None, extra=None)]
        )

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._pool = _FakePool()  # type: ignore[assignment]
        t._start_keepalive(ws)  # type: ignore[arg-type]

        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)

        assert removed == [ws]

    @pytest.mark.asyncio
    async def test_receive_pong_like_messages_are_drained_and_loop_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A burst of incoming messages followed by silence should drain
        the burst and then start sending pings on receive() timeouts."""
        monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", 0.02)

        # Three drained messages, then receive() blocks until closed.
        ws = _FakeKeepaliveWS(
            recv_messages=[
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{}", extra=None),
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{}", extra=None),
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{}", extra=None),
            ]
        )

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._start_keepalive(ws)  # type: ignore[arg-type]
        await asyncio.sleep(0.1)  # enough time for several timeouts after drain
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

        # After draining the burst, receive() blocks; timeouts fire and pings
        # are sent.
        assert len(ws.sent) >= 2
        for raw in ws.sent:
            assert json.loads(raw) == {"type": "ping"}


class TestKeepalivePoolEviction:
    """When a keepalive ping fails, the dead connection must be evicted
    from the pool so the next checkout yields a fresh socket instead of
    handing out the dead one and forcing a wasted retry round-trip.
    """

    @pytest.mark.asyncio
    async def test_failed_ping_calls_pool_remove(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)

        removed: list[object] = []

        class _FakePool:
            def remove(self, conn: object) -> None:
                removed.append(conn)

        t._pool = _FakePool()  # type: ignore[assignment]

        ws = _FakeKeepaliveWS(send_error=ConnectionResetError("Cannot write to closing transport"))
        t._start_keepalive(ws)  # type: ignore[arg-type]

        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)

        assert removed == [ws], "dead connection must be evicted from the pool"

    @pytest.mark.asyncio
    async def test_successful_ping_does_not_call_pool_remove(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)

        removed: list[object] = []

        class _FakePool:
            def remove(self, conn: object) -> None:
                removed.append(conn)

        t._pool = _FakePool()  # type: ignore[assignment]

        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        # Allow several successful pings to fire, then stop.
        await asyncio.sleep(0.08)
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

        assert removed == [], "healthy connections must not be evicted"
        assert len(ws.sent) >= 2

    @pytest.mark.asyncio
    async def test_pool_remove_exception_does_not_break_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if ``pool.remove`` raises (e.g. connection already gone),
        the keepalive task must exit cleanly without propagating."""
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)

        class _FakePool:
            def remove(self, conn: object) -> None:
                raise RuntimeError("pool already shut down")

        t._pool = _FakePool()  # type: ignore[assignment]

        ws = _FakeKeepaliveWS(send_error=ConnectionResetError("Cannot write to closing transport"))
        t._start_keepalive(ws)  # type: ignore[arg-type]

        task = t._ws_keepalive_tasks[id(ws)]
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done() and task.exception() is None


class TestKeepaliveWiring:
    @pytest.mark.asyncio
    async def test_close_ws_cancels_keepalive_and_closes_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)
        ws = _FakeKeepaliveWS()
        t._start_keepalive(ws)  # type: ignore[arg-type]

        await t._close_ws(ws)  # type: ignore[arg-type]
        assert ws.closed is True
        assert id(ws) not in t._ws_keepalive_tasks

    @pytest.mark.asyncio
    async def test_close_ws_without_prior_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even if _connect_ws was bypassed (e.g. test paths), _close_ws must
        # still close the socket without raising.
        t = _make_tts_with_short_keepalive(monkeypatch, 0.02)
        ws = _FakeKeepaliveWS()
        await t._close_ws(ws)  # type: ignore[arg-type]
        assert ws.closed is True


class TestModuleConstants:
    def test_default_keepalive_interval_is_under_sarvam_idle_timeout(self) -> None:
        # Sarvam closes idle connections after 60 s; pings must fire well below
        # that to actually keep the connection warm.
        assert 0 < sarvam_tts_module._KEEPALIVE_INTERVAL < 60

    def test_default_ws_heartbeat_interval_is_under_sarvam_idle_timeout(self) -> None:
        # The aiohttp protocol-level heartbeat must fire well within Sarvam's
        # 60 s server-side timeout window so the server keeps PONGing back and
        # never gives up on the connection.
        assert 0 < sarvam_tts_module._WS_HEARTBEAT_INTERVAL < 60

    def test_ws_heartbeat_is_at_most_keepalive_interval(self) -> None:
        # The protocol-level heartbeat is the primary defense; the
        # application-level ping is secondary. Heartbeat at <= keepalive
        # interval ensures aiohttp catches a dead transport before our
        # app-level ping does.
        assert sarvam_tts_module._WS_HEARTBEAT_INTERVAL <= sarvam_tts_module._KEEPALIVE_INTERVAL


# ---------------------------------------------------------------------------
# WebSocket heartbeat plumbing in _connect_ws
# ---------------------------------------------------------------------------


class TestWsConnectHeartbeat:
    """Verify the aiohttp protocol-level ``heartbeat`` parameter is wired
    into ``ws_connect`` so the connection survives Sarvam's idle timeout
    even while sitting unread in the pool.
    """

    @pytest.mark.asyncio
    async def test_connect_ws_passes_heartbeat_kwarg(self) -> None:
        captured: dict[str, object] = {}

        class _FakeSession:
            async def ws_connect(self, url, **kwargs):  # type: ignore[no-untyped-def]
                captured["url"] = url
                captured["kwargs"] = kwargs
                return _FakeKeepaliveWS()

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._session = _FakeSession()  # type: ignore[assignment]

        ws = await t._connect_ws(timeout=5.0)

        assert "heartbeat" in captured["kwargs"], (
            "heartbeat must be passed so aiohttp keeps the WS warm while idle"
        )
        assert captured["kwargs"]["heartbeat"] == sarvam_tts_module._WS_HEARTBEAT_INTERVAL

        # Cleanup: stop the keepalive task spawned for the fake ws.
        await t._stop_keepalive(ws)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_connect_ws_uses_short_heartbeat_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Module-level constant overrides should propagate through.
        monkeypatch.setattr(sarvam_tts_module, "_WS_HEARTBEAT_INTERVAL", 5.0)

        captured: dict[str, object] = {}

        class _FakeSession:
            async def ws_connect(self, url, **kwargs):  # type: ignore[no-untyped-def]
                captured["kwargs"] = kwargs
                return _FakeKeepaliveWS()

        t = TTS(target_language_code="hi-IN", api_key="sk_test")
        t._session = _FakeSession()  # type: ignore[assignment]

        ws = await t._connect_ws(timeout=5.0)

        assert captured["kwargs"]["heartbeat"] == 5.0

        await t._stop_keepalive(ws)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Keepalive pause/resume around active _run_ws sessions
# ---------------------------------------------------------------------------


class TestKeepalivePauseResumeIntegration:
    """Verify the keepalive task is paused while a session is active and
    resumed when the connection is returned to the pool, but NOT resumed
    when the connection is discarded due to an error.
    """

    @pytest.mark.asyncio
    async def test_keepalive_paused_during_session_and_resumed_on_success(
        self,
    ) -> None:
        stream = _make_stream(model="bulbul:v3", codec="mp3")
        stream._mark_started = MagicMock()  # type: ignore[method-assign]

        ws = _FakeWS(recv_messages=[_build_audio_response(), _build_final_event()])

        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.start_segment = MagicMock()
        emitter.push = MagicMock()
        emitter.end_input = MagicMock()

        class _FakeWordStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class _FakePool:
            last_acquire_time = 0.0
            last_connection_reused = False

            class _ctx:
                def __init__(self, ws):
                    self.ws = ws

                async def __aenter__(self):
                    return self.ws

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            def connection(self, *, timeout):
                return self._ctx(ws)

        stream._tts._pool = _FakePool()  # type: ignore[assignment]

        call_log: list[tuple[str, int]] = []
        original_start = stream._tts._start_keepalive
        original_stop = stream._tts._stop_keepalive

        def _logging_start(socket):  # type: ignore[no-untyped-def]
            call_log.append(("start", id(socket)))
            return original_start(socket)

        async def _logging_stop(socket):  # type: ignore[no-untyped-def]
            call_log.append(("stop", id(socket)))
            return await original_stop(socket)

        stream._tts._start_keepalive = _logging_start  # type: ignore[assignment]
        stream._tts._stop_keepalive = _logging_stop  # type: ignore[assignment]

        await stream._run_ws(_FakeWordStream(), emitter)

        # Expected: stop happens before any work, start happens at the end.
        assert call_log[0] == ("stop", id(ws))
        assert call_log[-1] == ("start", id(ws))

        # The keepalive task must now exist and be running for ws.
        assert id(ws) in stream._tts._ws_keepalive_tasks
        await stream._tts._stop_keepalive(ws)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_keepalive_not_resumed_when_session_fails(
        self,
    ) -> None:
        stream = _make_stream(model="bulbul:v3", codec="mp3")
        stream._mark_started = MagicMock()  # type: ignore[method-assign]

        # send_str raises a non-transport-closing error, which propagates as
        # APIConnectionError. The pool will discard this connection.
        ws = _FakeWS(
            recv_messages=[],
            send_error=ValueError("simulated catastrophic config error"),
        )

        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.start_segment = MagicMock()
        emitter.push = MagicMock()
        emitter.end_input = MagicMock()

        class _FakeWordStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class _FakePool:
            last_acquire_time = 0.0
            last_connection_reused = False

            class _ctx:
                def __init__(self, ws):
                    self.ws = ws

                async def __aenter__(self):
                    return self.ws

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            def connection(self, *, timeout):
                return self._ctx(ws)

        stream._tts._pool = _FakePool()  # type: ignore[assignment]

        call_log: list[str] = []
        original_start = stream._tts._start_keepalive
        original_stop = stream._tts._stop_keepalive

        def _logging_start(socket):  # type: ignore[no-untyped-def]
            call_log.append("start")
            return original_start(socket)

        async def _logging_stop(socket):  # type: ignore[no-untyped-def]
            call_log.append("stop")
            return await original_stop(socket)

        stream._tts._start_keepalive = _logging_start  # type: ignore[assignment]
        stream._tts._stop_keepalive = _logging_stop  # type: ignore[assignment]

        with pytest.raises((APIConnectionError, APIStatusError)):
            await stream._run_ws(_FakeWordStream(), emitter)

        # We must have paused the keepalive at the start, and we must NOT
        # have restarted it after the failure. The pool will discard the
        # connection, so restarting would orphan a task pointing at a dying
        # socket.
        assert "stop" in call_log
        assert "start" not in call_log
        assert id(ws) not in stream._tts._ws_keepalive_tasks

    @pytest.mark.asyncio
    async def test_no_pings_sent_during_active_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """During ``_run_ws`` the keepalive must be paused so its
        ``ws.receive()`` never competes with ``recv_task`` and no ping
        frames are interleaved with config/text/flush.
        """
        # Shrink the interval so any unintended pings would surface quickly.
        monkeypatch.setattr(sarvam_tts_module, "_KEEPALIVE_INTERVAL", 0.01)

        stream = _make_stream(model="bulbul:v3", codec="mp3")
        stream._mark_started = MagicMock()  # type: ignore[method-assign]

        ws = _FakeWS(recv_messages=[_build_audio_response(), _build_final_event()])

        emitter = MagicMock(spec=tts.AudioEmitter)
        emitter.start_segment = MagicMock()
        emitter.push = MagicMock()
        emitter.end_input = MagicMock()

        class _FakeWordStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class _FakePool:
            last_acquire_time = 0.0
            last_connection_reused = False

            class _ctx:
                def __init__(self, ws):
                    self.ws = ws

                async def __aenter__(self):
                    return self.ws

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            def connection(self, *, timeout):
                return self._ctx(ws)

        stream._tts._pool = _FakePool()  # type: ignore[assignment]

        await stream._run_ws(_FakeWordStream(), emitter)

        # Every payload sent during the session must be a known protocol
        # frame -- never a ping.
        for raw in ws.sent:
            payload = json.loads(raw)
            assert payload.get("type") in {"config", "text", "flush"}, (
                f"unexpected message during active session: {payload}"
            )

        # Cleanup: stop the resumed keepalive task.
        await stream._tts._stop_keepalive(ws)  # type: ignore[arg-type]
