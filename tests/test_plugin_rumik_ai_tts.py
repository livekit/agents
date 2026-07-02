"""Tests for Rumik AI TTS plugin configuration and websocket behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import utils
from livekit.plugins.rumik_ai import tts as rumik_tts

pytestmark = pytest.mark.plugin("rumik-ai")


def test_tts_requires_api_key() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("RUMIK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            rumik_tts.TTS(api_key=None)


def test_muga_without_fallback_tone_requires_tagged_text() -> None:
    tts = rumik_tts.TTS(api_key="test-key")

    with pytest.raises(ValueError, match="global tone tag"):
        rumik_tts._prepare_text("Arre yaar", tts._opts)


def test_payload_omits_unset_sampling_params() -> None:
    tts = rumik_tts.TTS(api_key="test-key")
    # Only the text is sent; Rumik AI applies its own defaults for everything else.
    assert rumik_tts._synthesis_payload("[happy] hello", tts._opts) == {"text": "[happy] hello"}


def test_payload_includes_explicitly_set_sampling_params() -> None:
    tts = rumik_tts.TTS(api_key="test-key", temperature=0.8, top_p=0.9)
    assert rumik_tts._synthesis_payload("[happy] hello", tts._opts) == {
        "text": "[happy] hello",
        "temperature": 0.8,
        "top_p": 0.9,
    }


def test_mulberry_rejects_tone_option() -> None:
    with pytest.raises(ValueError, match="tone"):
        rumik_tts.TTS(api_key="test-key", model="mulberry", tone="happy")


def test_mulberry_accepts_plain_text() -> None:
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    assert rumik_tts._prepare_text("Arre yaar", tts._opts) == "Arre yaar"


def test_mulberry_accepts_devanagari_hindi() -> None:
    # Mulberry expects Hindi in Devanagari (unlike muga, which rejects it); English words
    # inside a Hindi sentence stay in Latin script.
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    text = "कुछ heavy लग रहा है क्या?"
    assert rumik_tts._prepare_text(text, tts._opts) == text


def test_mulberry_accepts_pure_english() -> None:
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    text = "Hey, how was your day today?"
    assert rumik_tts._prepare_text(text, tts._opts) == text


@pytest.mark.parametrize(
    "text",
    [
        "[happy] Arre yaar",
        "[neutral] Status update ready hai",
    ],
)
def test_mulberry_rejects_muga_tone_tags(text: str) -> None:
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    with pytest.raises(ValueError, match="tone tags"):
        rumik_tts._prepare_text(text, tts._opts)


@pytest.mark.parametrize(
    "text",
    [
        "Arre <laugh> yaar",
        "Theek hai <sigh>",
    ],
)
def test_mulberry_rejects_muga_event_tags(text: str) -> None:
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    with pytest.raises(ValueError, match="event tags"):
        rumik_tts._prepare_text(text, tts._opts)


def test_muga_fallback_tone_prepares_untagged_text() -> None:
    tts = rumik_tts.TTS(api_key="test-key", tone="happy")

    assert rumik_tts._prepare_text("Arre yaar", tts._opts) == "[happy] Arre yaar"


def test_muga_accepts_llm_selected_tone_tag() -> None:
    tts = rumik_tts.TTS(api_key="test-key")

    assert rumik_tts._prepare_text("[sad] <sigh> Theek hai", tts._opts) == (
        "[sad] <sigh> Theek hai"
    )


def test_muga_accepts_matching_existing_tone_tag() -> None:
    tts = rumik_tts.TTS(api_key="test-key", tone="sad")

    assert rumik_tts._prepare_text("[sad] <sigh> Theek hai", tts._opts) == (
        "[sad] <sigh> Theek hai"
    )


def test_muga_normalizes_newline_after_tone_tag() -> None:
    # A newline (not a space) after the tag would otherwise fail the "one space" rule.
    tts = rumik_tts.TTS(api_key="test-key")

    assert rumik_tts._prepare_text("[happy]\nArre yaar", tts._opts) == "[happy] Arre yaar"


def test_prepare_text_collapses_internal_whitespace() -> None:
    # Buffered LLM output can carry newlines and double spaces; Rumik gets clean text.
    tts = rumik_tts.TTS(api_key="test-key", model="mulberry")

    assert rumik_tts._prepare_text("Arre\n\nyaar   kaise  ho", tts._opts) == "Arre yaar kaise ho"


@pytest.mark.parametrize(
    ("text", "tone", "match"),
    [
        ("[sad] Arre yaar", "happy", "must match"),
        ("[Happy] Arre yaar", "happy", "Unsupported"),
        ("[happy]Arre yaar", "happy", "one space"),
        ("[happy] Arre [sad] yaar", "happy", "exactly one"),
        ("[happy] <sneeze> Arre", "happy", "Unsupported"),
        ("[sad] <laugh> sab khatam", "sad", "not compatible"),
        ("[happy] <laugh> <laugh> <laugh> Arre", "happy", "at most two"),
        ("मैं ठीक हूँ", "happy", "Roman script"),
    ],
)
def test_muga_rejects_invalid_prompts(text: str, tone: str, match: str) -> None:
    tts = rumik_tts.TTS(api_key="test-key", tone=tone)

    with pytest.raises(ValueError, match=match):
        rumik_tts._prepare_text(text, tts._opts)


def test_mulberry_rejects_invalid_speaker() -> None:
    with pytest.raises(ValueError, match="speaker"):
        rumik_tts.TTS(api_key="test-key", model="mulberry", speaker="speaker_5")


def test_mulberry_payload_includes_voice_controls() -> None:
    tts = rumik_tts.TTS(
        api_key="test-key",
        model="mulberry",
        description="warm narrator",
        speaker="speaker_2",
        f0_up_key=1,
    )

    assert rumik_tts._synthesis_payload("hello", tts._opts) == {
        "text": "hello",
        "description": "warm narrator",
        "speaker": "speaker_2",
        "f0_up_key": 1,
    }


class _FakePostContext:
    def __init__(self, session: _FakeSession, response: _FakeResponse) -> None:
        self._session = session
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeResponse:
    def __init__(self, status: int, body: object) -> None:
        self.status = status
        self._body = body

    async def json(self) -> object:
        return self._body

    async def text(self) -> str:
        return json.dumps(self._body)


class _FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.sent: list[str] = []
        self.closed = False
        self.close_code: int | None = None

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, *, timeout: float | None = None) -> object:
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)

    def exception(self) -> BaseException | None:
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, response: _FakeResponse, ws: _FakeWebSocket) -> None:
        self._response = response
        self._ws = ws
        self.post_calls: list[dict[str, object]] = []
        self.ws_connect_urls: list[str] = []

    def post(self, url: str, **kwargs: object) -> _FakePostContext:
        self.post_calls.append({"url": url, **kwargs})
        return _FakePostContext(self, self._response)

    async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
        self.ws_connect_urls.append(url)
        return self._ws


class _FakeEmitter:
    def __init__(self) -> None:
        self.initialized: dict[str, object] | None = None
        self.segments: list[str] = []
        self.audio: list[bytes] = []
        self.ended_count = 0

    @property
    def ended(self) -> bool:
        return self.ended_count > 0

    def initialize(self, **kwargs: object) -> None:
        self.initialized = kwargs

    def start_segment(self, *, segment_id: str) -> None:
        self.segments.append(segment_id)

    def push(self, data: bytes) -> None:
        self.audio.append(data)

    def end_segment(self) -> None:
        self.ended_count += 1


def _make_stream(tts: rumik_tts.TTS) -> rumik_tts.SynthesizeStream:
    stream = object.__new__(rumik_tts.SynthesizeStream)
    stream._tts = tts  # pyright: ignore[reportPrivateUsage]
    stream._opts = tts._opts  # pyright: ignore[reportPrivateUsage]
    stream._conn_options = rumik_tts.DEFAULT_API_CONNECT_OPTIONS  # pyright: ignore[reportPrivateUsage]
    stream._started_time = 0  # pyright: ignore[reportPrivateUsage]
    return stream


async def _run_with_inputs(
    stream: rumik_tts.SynthesizeStream,
    emitter: _FakeEmitter,
    inputs: list[str | object],
) -> None:
    stream._input_ch = utils.aio.Chan()  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(stream._run(emitter))  # pyright: ignore[reportPrivateUsage]
    for item in inputs:
        if item is _FLUSH:
            stream._input_ch.send_nowait(stream._FlushSentinel())  # pyright: ignore[reportPrivateUsage]
        else:
            stream._input_ch.send_nowait(item)  # pyright: ignore[reportPrivateUsage]
    stream._input_ch.close()  # pyright: ignore[reportPrivateUsage]
    await task


_FLUSH = object()
_CONN = rumik_tts.DEFAULT_API_CONNECT_OPTIONS


def _ws_text(payload: dict[str, object]) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def _ws_binary(data: bytes) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data)


def _ws_close(code: int, reason: str) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=code, extra=reason)


def test_default_capability_is_model_aware() -> None:
    # muga buffers the full response (its leading [tone] tag must condition the whole
    # utterance), so streaming defaults True; mulberry has no tone tag and defaults to
    # sentence streaming for lower latency, so streaming defaults False.
    assert rumik_tts.TTS(api_key="test-key", model="muga").capabilities.streaming is True
    assert rumik_tts.TTS(api_key="test-key", model="mulberry").capabilities.streaming is False


def test_full_response_aggregation_false_disables_streaming() -> None:
    # Opt out -> the framework's StreamAdapter sentence-splits and calls synthesize()
    # per sentence for lower latency.
    for model in ("muga", "mulberry"):
        tts = rumik_tts.TTS(api_key="test-key", model=model, full_response_aggregation=False)
        assert tts.capabilities.streaming is False


@pytest.mark.asyncio
async def test_muga_stream_uses_full_response() -> None:
    # Muga synthesizes the entire response in one request, ignoring mid-stream
    # flushes so the leading [tone] tag conditions the whole utterance.
    tts = rumik_tts.TTS(api_key="test-key")
    stream = _make_stream(tts)
    emitter = _FakeEmitter()
    texts: list[str] = []

    async def fake_stream_synthesis(text, _opts, _conn, _emitter, *, on_started=None) -> None:  # noqa: ANN001
        texts.append(text)

    tts._stream_synthesis = fake_stream_synthesis  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]

    await _run_with_inputs(
        stream,
        emitter,
        ["[happy] Hey, ", "main Mira hoon.", _FLUSH, " Aur tum?"],
    )

    assert texts == ["[happy] Hey, main Mira hoon. Aur tum?"]


@pytest.mark.asyncio
async def test_stream_mints_reusable_ws_session_and_pushes_binary_audio() -> None:
    response = _FakeResponse(200, {"ws_url": "wss://rumik.test/tts", "token": "tok en"})
    ws = _FakeWebSocket([_ws_binary(b"pcm-1"), _ws_binary(b"pcm-2"), _ws_text({"type": "done"})])
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )
    emitter = _FakeEmitter()

    await tts._stream_synthesis(  # pyright: ignore[reportPrivateUsage]
        "[happy] Arre yaar", tts._opts, _CONN, emitter
    )

    assert session.post_calls[0]["url"] == "https://rumik.test/v1/tts/ws-connect"
    assert session.post_calls[0]["headers"] == {"Authorization": "Bearer test-key"}
    # The mint uses a throwaway "init" text; the real text is sent over the socket.
    assert session.post_calls[0]["json"] == {"model": "muga", "text": "init"}
    assert session.ws_connect_urls == ["wss://rumik.test/tts?token=tok%20en"]
    sent_payload = json.loads(ws.sent[0])
    assert sent_payload["text"] == "[happy] Arre yaar"
    assert "model" not in sent_payload
    assert emitter.audio == [b"pcm-1", b"pcm-2"]


@pytest.mark.asyncio
async def test_pooled_ws_session_is_reused_across_requests() -> None:
    response = _FakeResponse(200, {"ws_url": "wss://rumik.test/tts", "token": "tok"})
    ws = _FakeWebSocket(
        [
            _ws_binary(b"a"),
            _ws_text({"type": "done"}),
            _ws_binary(b"b"),
            _ws_text({"type": "done"}),
        ]
    )
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )
    emitter = _FakeEmitter()

    await tts._stream_synthesis("[happy] one", tts._opts, _CONN, emitter)  # pyright: ignore[reportPrivateUsage]
    await tts._stream_synthesis("[happy] two", tts._opts, _CONN, emitter)  # pyright: ignore[reportPrivateUsage]

    # The session is minted and connected only once, then reused for both requests.
    assert len(session.post_calls) == 1
    assert len(session.ws_connect_urls) == 1
    assert len(ws.sent) == 2


@pytest.mark.asyncio
async def test_ws_connect_error_raises_status_error() -> None:
    response = _FakeResponse(429, {"error": "rate limited", "code": "rate_limit"})
    ws = _FakeWebSocket([])
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        tone="happy",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )

    with pytest.raises(rumik_tts.APIStatusError) as exc:
        await tts._mint_ws_session(10.0)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 429
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_provider_error_event_raises_status_error() -> None:
    response = _FakeResponse(200, {"ws_url": "wss://rumik.test/tts", "token": "tok"})
    ws = _FakeWebSocket(
        [
            _ws_text(
                {
                    "error": "validation failed",
                    "code": "bad_prompt",
                    "status": "422",
                }
            )
        ]
    )
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        tone="happy",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )
    emitter = _FakeEmitter()

    with pytest.raises(rumik_tts.APIStatusError) as exc:
        await tts._stream_synthesis("Arre yaar", tts._opts, _CONN, emitter)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 422
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_invalid_websocket_json_raises_connection_error() -> None:
    response = _FakeResponse(200, {"ws_url": "wss://rumik.test/tts", "token": "tok"})
    ws = _FakeWebSocket([SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{")])
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        tone="happy",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )
    emitter = _FakeEmitter()

    with pytest.raises(rumik_tts.APIConnectionError, match="invalid JSON"):
        await tts._stream_synthesis("Arre yaar", tts._opts, _CONN, emitter)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_websocket_close_raises_connection_error_with_context() -> None:
    response = _FakeResponse(200, {"ws_url": "wss://rumik.test/tts", "token": "tok"})
    ws = _FakeWebSocket([_ws_close(1011, "internal error")])
    session = _FakeSession(response, ws)
    tts = rumik_tts.TTS(
        api_key="test-key",
        tone="happy",
        base_url="https://rumik.test",
        http_session=session,  # type: ignore[arg-type]
    )
    emitter = _FakeEmitter()

    with pytest.raises(rumik_tts.APIConnectionError, match="code=1011"):
        await tts._stream_synthesis("Arre yaar", tts._opts, _CONN, emitter)  # pyright: ignore[reportPrivateUsage]


class _CancelWebSocket:
    """Yields ``before`` frames, raises CancelledError at ``receive()`` (simulating an
    interruption), then yields ``after`` frames for the cancel-drain phase."""

    def __init__(self, before: list[object], after: list[object]) -> None:
        self._before = list(before)
        self._after = list(after)
        self.sent: list[str] = []
        self.closed = False
        self.close_code: int | None = None
        self._interrupted = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, *, timeout: float | None = None) -> object:
        if not self._interrupted:
            if self._before:
                return self._before.pop(0)
            self._interrupted = True
            raise asyncio.CancelledError
        if self._after:
            return self._after.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)

    def exception(self) -> BaseException | None:
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_barge_in_sends_cancel_and_reuses_clean_socket() -> None:
    # On interruption the plugin sends {"type": "cancel"}, drains the server's ack, drops
    # any late audio, and keeps the socket warm so the next utterance avoids a re-mint.
    tts = rumik_tts.TTS(api_key="test-key", model="muga", tone="happy")
    ws = _CancelWebSocket(
        before=[_ws_binary(b"old-1"), _ws_binary(b"old-2")],
        after=[_ws_binary(b"late-stale"), _ws_text({"type": "cancelled", "reason": "cancel"})],
    )
    emitter = _FakeEmitter()

    result = await tts._stream_on_ws(ws, {"text": "hi"}, _CONN, emitter, None)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]

    assert result is True  # interrupted but clean -> caller re-raises, socket reused
    assert json.loads(ws.sent[-1]) == {"type": "cancel"}
    assert emitter.audio == [b"old-1", b"old-2"]  # audio after the cancel is discarded
    assert ws.closed is False  # kept warm
    await tts.aclose()


@pytest.mark.asyncio
async def test_barge_in_drops_socket_when_drain_unclean() -> None:
    # If the socket closes during the drain it cannot be reused -> re-raise so the pool
    # drops it (the cancel is still sent first).
    tts = rumik_tts.TTS(api_key="test-key", model="muga", tone="happy")
    ws = _CancelWebSocket(before=[_ws_binary(b"old")], after=[_ws_close(1000, "bye")])
    emitter = _FakeEmitter()

    with pytest.raises(asyncio.CancelledError):
        await tts._stream_on_ws(ws, {"text": "hi"}, _CONN, emitter, None)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]

    assert json.loads(ws.sent[-1]) == {"type": "cancel"}
    await tts.aclose()


@pytest.mark.asyncio
async def test_server_cancelled_frame_is_clean_terminal() -> None:
    # A server-sent {"type": "cancelled"} (e.g. a barge-in replaced the request) ends the
    # generation cleanly without raising; the socket stays reusable.
    tts = rumik_tts.TTS(api_key="test-key", model="muga", tone="happy")
    ws = _FakeWebSocket([_ws_binary(b"a"), _ws_text({"type": "cancelled", "reason": "interrupt"})])
    emitter = _FakeEmitter()

    result = await tts._stream_on_ws(ws, {"text": "hi"}, _CONN, emitter, None)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]

    assert result is False
    assert emitter.audio == [b"a"]
    await tts.aclose()
