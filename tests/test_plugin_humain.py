"""HUMAIN Voice TTS plugin tests: offline, with a fake of the ``humain-voice`` SDK client.

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)

pytestmark = pytest.mark.unit

_AUDIO = b"\x10\x00" * 1200  # 50 ms at 24 kHz
_VOICES = [
    {"id": "11111111-1111-4111-8111-111111111111", "label": "mul_nourah"},
    {"id": "22222222-2222-4222-8222-222222222222", "label": "mul_sara"},
]


@dataclass
class _Chunk:
    audio: bytes
    is_last: bool = False


@dataclass
class _Error:
    code: str
    retryable: bool | None = None
    message: str = "the provider message may echo the text"


class _FakeClient:
    """Stands in for ``humain_voice.tts.TTSClient``; ``mode`` picks the reply per request."""

    instances: list[_FakeClient] = []
    mode = "audio"

    def __init__(self, *, api_url: str, api_key: str) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self.voice_lists = 0
        self.closed = False
        _FakeClient.instances.append(self)

    async def connect(self) -> None:
        if _FakeClient.mode == "connect-hangs":
            await asyncio.sleep(10)
        elif _FakeClient.mode.startswith("connect-status:"):
            # What python-socketio raises for a refused handshake: a generic
            # ConnectionError whose context holds aiohttp's handshake error.
            status = int(_FakeClient.mode.split(":", 1)[1])
            try:
                raise aiohttp.WSServerHandshakeError(
                    request_info=None,  # type: ignore[arg-type]
                    history=(),
                    status=status,
                    message="Invalid response status",
                )
            except aiohttp.WSServerHandshakeError:
                raise ConnectionError("Connection error")  # noqa: B904

    async def close(self) -> None:
        self.closed = True

    async def list_voices(self, *, timeout_seconds: float | None = None) -> list[dict[str, str]]:
        self.voice_lists += 1
        return _VOICES

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        *,
        model: str | None = None,
        on_error: Callable[[Any], None] | None = None,
        request_id: Any = None,
        timeout_seconds: float | None = None,
        voice_references: Any = None,
    ) -> AsyncIterator[_Chunk]:
        self.requests.append({"text": text, "voice_id": voice_id, "model": model})
        mode = _FakeClient.mode
        if mode == "audio":
            yield _Chunk(_AUDIO)
            yield _Chunk(_AUDIO, is_last=True)
        elif mode.startswith("error:"):
            code = mode.split(":", 1)[1]
            assert on_error is not None
            on_error(_Error(code=code, retryable=False if code == "PERMANENT" else None))
            raise RuntimeError("the provider message may echo the text")
        elif mode == "timeout":
            raise asyncio.TimeoutError("Timeout waiting for TTS response")
        elif mode == "disconnect":
            raise ConnectionError("socket closed")
        elif mode == "hang":
            await asyncio.sleep(10)
        elif mode == "invalid":
            raise ValueError("text must be a non-empty string")


@pytest.fixture(autouse=True)
def fake_client() -> Any:
    _FakeClient.instances = []
    _FakeClient.mode = "audio"
    with patch("livekit.plugins.humain.tts.TTSClient", _FakeClient):
        yield _FakeClient


_OPTS = APIConnectOptions(max_retry=0, timeout=5.0)


# --- configuration ---


def test_tts_requires_api_key():
    from livekit.plugins.humain import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_reads_key_and_url_from_env():
    from livekit.plugins.humain import TTS

    with patch.dict(
        "os.environ", {"HUMAIN_API_KEY": "env-key", "HUMAIN_API_URL": "https://x.test"}
    ):
        t = TTS()
    assert t._api_key == "env-key"
    assert t._api_url == "https://x.test"


def test_tts_properties():
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    assert t.model == "nebula"
    assert t.provider == "HUMAIN"
    assert t.capabilities.streaming is True
    assert t.sample_rate == 24000
    assert t.num_channels == 1
    assert t._api_url == "https://api.voice.humain.com"


def test_update_options_changes_the_voice():
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    t.update_options(voice="mul_sara")
    assert t._opts.voice == "mul_sara"
    with pytest.raises(ValueError):
        t.update_options(voice="")
    assert t._opts.voice == "mul_sara"


# --- synthesize() ---


async def test_synthesize_resolves_the_voice_label_and_pushes_audio(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="sk-test", voice="mul_sara")
    frame = await t.synthesize("مرحبا", conn_options=_OPTS).collect()
    await t.aclose()

    (client,) = fake_client.instances
    assert client.api_key == "sk-test"
    assert client.requests == [{"text": "مرحبا", "voice_id": _VOICES[1]["id"], "model": "nebula"}]
    assert frame.sample_rate == 24000
    assert frame.samples_per_channel >= 2 * len(_AUDIO) // 2
    assert client.closed


async def test_a_voice_uuid_skips_the_voice_list(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k", voice=_VOICES[0]["id"])
    await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert fake_client.instances[0].voice_lists == 0


async def test_one_connection_and_one_voice_lookup_serve_many_requests(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    for text in ("أهلا", "وسهلا", "مرحبا"):
        await t.synthesize(text, conn_options=_OPTS).collect()
    await t.aclose()
    (client,) = fake_client.instances
    assert len(client.requests) == 3
    assert client.voice_lists == 1


async def test_an_unknown_voice_is_a_404():
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k", voice="mul_nobody")
    with pytest.raises(APIStatusError) as exc:
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert exc.value.status_code == 404
    assert exc.value.retryable is False


@pytest.mark.parametrize(
    ("code", "status", "retryable"),
    [
        ("TTS_VOICE_NOT_FOUND", 404, False),
        ("TTS_INPUT_NOT_ALLOWED", 400, False),
        ("CREDITS_EXHAUSTED", 402, False),
        ("CONCURRENCY_LIMIT_EXCEEDED", 429, True),
    ],
)
async def test_humain_error_codes_map_to_status_errors(
    fake_client: Any, code: str, status: int, retryable: bool
):
    from livekit.plugins.humain import TTS

    fake_client.mode = f"error:{code}"
    t = TTS(api_key="k")
    with pytest.raises(APIStatusError) as exc:
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert exc.value.status_code == status
    assert exc.value.retryable is retryable
    assert "echo the text" not in str(exc.value)


async def test_an_unlisted_error_code_follows_humains_retryable_flag(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "error:PERMANENT"
    t = TTS(api_key="k")
    with pytest.raises(APIError) as exc:
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert exc.value.retryable is False
    assert "PERMANENT" in str(exc.value)


async def test_a_stalled_request_times_out(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "timeout"
    t = TTS(api_key="k")
    with pytest.raises(APITimeoutError):
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()


@pytest.mark.virtual_time
async def test_a_hanging_connect_times_out_and_closes_the_client(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "connect-hangs"
    t = TTS(api_key="k")
    with pytest.raises(APITimeoutError):
        await t.synthesize(
            "أهلا", conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
        ).collect()
    await t.aclose()
    assert fake_client.instances[0].closed


async def test_a_dropped_connection_reconnects_on_the_next_request(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "disconnect"
    t = TTS(api_key="k")
    with pytest.raises(APIConnectionError):
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    fake_client.mode = "audio"
    await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert len(fake_client.instances) == 2
    assert fake_client.instances[0].closed


@pytest.mark.parametrize(("status", "retryable"), [(401, False), (403, False), (429, True)])
async def test_a_refused_connection_is_a_status_error(
    fake_client: Any, status: int, retryable: bool
):
    from livekit.plugins.humain import TTS

    fake_client.mode = f"connect-status:{status}"
    t = TTS(api_key="bad")
    with pytest.raises(APIStatusError) as exc_info:
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable
    assert exc_info.value.__cause__ is None
    assert fake_client.instances[0].closed


async def test_a_server_error_on_connect_stays_a_connection_error(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "connect-status:503"
    t = TTS(api_key="k")
    with pytest.raises(APIConnectionError):
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()


# --- stream() ---


async def _synthesize_stream(pushes: list[str | None]) -> list[Any]:
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    stream = t.stream(conn_options=_OPTS)
    for p in pushes:
        stream.flush() if p is None else stream.push_text(p)
    stream.end_input()
    try:
        return [ev async for ev in stream]
    finally:
        await stream.aclose()
        await t.aclose()


async def test_llm_token_deltas_become_sentence_requests(fake_client: Any):
    deltas = [
        "هلا وغلا فيك في مركز",
        " الرياض الصحي، معك ليلى. وش أقدر ",
        "أخدمك فيه اليوم؟ العيادات تشتغل ",
        "من السبت للخميس.",
    ]
    events = await _synthesize_stream(deltas)
    (client,) = fake_client.instances
    texts = [r["text"] for r in client.requests]
    assert 1 < len(texts) < len(deltas)
    assert " ".join(texts) == "".join(deltas).strip()
    assert sum(len(ev.frame.data) for ev in events) > 0
    assert len({ev.segment_id for ev in events}) == 1


async def test_a_stream_with_no_text_makes_no_request(fake_client: Any):
    events = await _synthesize_stream([None])
    assert events == []
    assert fake_client.instances == []


async def test_prewarm_connects_and_resolves_the_voice(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    t.prewarm()
    assert t._prewarm_task is not None
    await t._prewarm_task
    (client,) = fake_client.instances
    assert client.voice_lists == 1
    await t.synthesize("أهلا", conn_options=_OPTS).collect()
    await t.aclose()
    assert len(fake_client.instances) == 1
    assert client.voice_lists == 1


# --- review follow-ups ---


@pytest.mark.virtual_time
async def test_a_reconnect_after_a_dropped_socket_is_bounded_by_the_timeout(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    await t.synthesize("أهلا", conn_options=_OPTS).collect()
    fake_client.mode = "connect-hangs"  # the socket dropped and reconnecting hangs
    with pytest.raises(APITimeoutError):
        await t.synthesize(
            "أهلا", conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
        ).collect()
    assert fake_client.instances[0].closed
    fake_client.mode = "audio"
    await t.synthesize("أهلا", conn_options=_OPTS).collect()  # a fresh client takes over
    await t.aclose()
    assert len(fake_client.instances) == 2


async def test_aclose_stops_an_in_flight_synthesize(fake_client: Any):
    from livekit.plugins.humain import TTS

    fake_client.mode = "hang"
    t = TTS(api_key="k")
    stream = t.synthesize("أهلا", conn_options=APIConnectOptions(max_retry=3, timeout=30.0))
    task = asyncio.create_task(stream.collect())
    for _ in range(20):
        if fake_client.instances and fake_client.instances[0].requests:
            break
        await asyncio.sleep(0)
    await t.aclose()
    await asyncio.gather(task, return_exceptions=True)
    assert task.done()
    assert len(fake_client.instances) == 1  # no reconnect after shutdown
    assert fake_client.instances[0].closed


async def test_empty_text_makes_no_request(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    events = [ev async for ev in t.synthesize("   ", conn_options=_OPTS)]
    await t.aclose()
    assert events == []
    assert fake_client.instances == []


async def test_a_request_the_sdk_rejects_keeps_the_connection(fake_client: Any):
    from livekit.plugins.humain import TTS

    t = TTS(api_key="k")
    await t.synthesize("أهلا", conn_options=_OPTS).collect()
    fake_client.mode = "invalid"
    with pytest.raises(APIStatusError) as exc:
        await t.synthesize("أهلا", conn_options=_OPTS).collect()
    assert exc.value.status_code == 400
    assert exc.value.retryable is False
    assert not fake_client.instances[0].closed  # concurrent requests are unaffected
    await t.aclose()
