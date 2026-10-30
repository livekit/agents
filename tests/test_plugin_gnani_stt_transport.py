"""Gnani STT streaming/REST behaviour against a local fake of the Vachana API.

The fake reproduces behaviour recorded from the live API (2026-10-30):

* ``connected`` first, then ``speech_start`` when audio starts and one final
  ``transcript`` per utterance. There is no ``speech_end`` message, and no interim results.
* A ``commit`` control frame closes the open utterance immediately.
* Idle expiry is announced as an ``error`` carrying ``timeout_reason``, then ``session_end``,
  then a normal 1001 close.
* Rejected handshakes send an ``error`` message and close with 1008.
* Utterance-level failures ("Transcription failed or audio unclear") are ``error`` messages
  that leave the connection open.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt

pytestmark = pytest.mark.plugin("gnani")

OPTS = APIConnectOptions(max_retry=0, timeout=5, retry_interval=0.05)
FRAME = rtc.AudioFrame(bytes(3200), 16000, 1, 1600)  # 100 ms of silence


class FakeVachanaSTT:
    def __init__(self) -> None:
        self.sessions: list[SimpleNamespace] = []
        self.text = "नमस्ते, आप कैसे हैं?"
        self.detected: str | None = "hi-IN"
        # per-session behaviour by index: "idle" | "reject" | "quick_close"
        self.behaviours: dict[int, str] = {}
        self.utterance_errors_left = 0
        # seconds the server takes before it answers the audio it received (speech_start
        # and the transcript both come late), and a switch for a server that never answers
        self.reply_delay = 0.0
        self.silent = False
        self.rest_calls: list[dict[str, Any]] = []
        self.rest_status = 200

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        index = len(self.sessions)
        s = SimpleNamespace(headers=dict(request.headers), audio=0, controls=[], speaking=False)
        self.sessions.append(s)
        mode = self.behaviours.get(index)

        if mode == "reject":
            await ws.send_json({"type": "error", "message": "Invalid x-format. Supported values"})
            await ws.close(code=1008, message=b"policy violation")
            return ws

        await ws.send_json({"type": "connected", "config": {}})
        if mode == "quick_close":
            await ws.send_json({"type": "session_end", "reason": "idle_timeout"})
            await ws.close(code=1001, message=b"idle_timeout")
            return ws

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                s.audio += len(msg.data)
                if mode == "idle":
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "Session closed: no speech segment received for 5 seconds.",
                            "timeout_reason": "idle",
                        }
                    )
                    await ws.send_json({"type": "session_end", "reason": "idle_timeout"})
                    await ws.close(code=1001, message=b"idle_timeout")
                    break
                if not s.speaking and not self.silent:
                    s.speaking = True
                    if self.reply_delay:
                        await asyncio.sleep(self.reply_delay)
                    await ws.send_json({"type": "speech_start", "utterance_id": "u1"})
            elif msg.type == aiohttp.WSMsgType.TEXT:
                body = json.loads(msg.data)
                s.controls.append(body)
                if body["type"] == "commit" and s.speaking:
                    s.speaking = False
                    if self.utterance_errors_left > 0:
                        self.utterance_errors_left -= 1
                        await ws.send_json(
                            {"type": "error", "message": "Transcription failed or audio unclear"}
                        )
                        continue
                    payload = {
                        "type": "transcript",
                        "text": self.text,
                        "is_final": True,
                        "utterance_id": "u1",
                        "segment_id": f"seg-{index}",
                    }
                    if self.detected:
                        payload["detected_language_code"] = self.detected
                    await ws.send_json(payload)
                elif body["type"] == "update_settings":
                    await ws.send_json({"type": "settings_updated", "config": {}})
        return ws

    async def rest(self, request: web.Request) -> web.Response:
        form = await request.post()
        self.rest_calls.append(
            {k: v for k, v in form.items() if k != "audio_file"}
            | {"_headers": dict(request.headers)}
        )
        if self.rest_status != 200:
            return web.json_response(
                {"success": False, "error": {"type": "INVALID_LANGUAGE_CODE", "message": "nope"}},
                status=self.rest_status,
            )
        body: dict[str, Any] = {
            "success": True,
            "request_id": "req-1",
            "transcript": self.text,
            "model": form.get("model", "prisma-v2.5"),
        }
        if self.detected:
            body["detected_language_code"] = self.detected
        return web.json_response(body)


@pytest.fixture
async def server():
    fake = FakeVachanaSTT()
    app = web.Application()
    app.router.add_get("/stt/v3/stream", fake.ws)
    app.router.add_post("/stt/v3", fake.rest)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    fake.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"  # type: ignore[attr-defined]
    fake.session = aiohttp.ClientSession()  # type: ignore[attr-defined]
    try:
        yield fake
    finally:
        await fake.session.close()  # type: ignore[attr-defined]
        await runner.cleanup()


def _make_stt(server: Any, **kwargs: Any):
    from livekit.plugins.gnani import STT

    stt_ = STT(api_key="test-key", base_url=server.base_url, **kwargs)
    stt_._session = server.session
    return stt_


async def _utterance(stream, *, flush: bool = True) -> None:
    stream.push_frame(FRAME)
    if flush:
        stream.flush()


async def _collect(stream) -> list[stt.SpeechEvent]:
    events = []
    async for ev in stream:
        events.append(ev)
    return events


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [e.type for e in events]


SOS = stt.SpeechEventType.START_OF_SPEECH
EOS = stt.SpeechEventType.END_OF_SPEECH
FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


async def test_stream_sends_the_expected_handshake(server):
    stt_ = _make_stt(
        server,
        model="prisma-v3.0",
        language=["hi-IN", "en-IN"],
        format="codemix",
        keyterms=["Gnani"],
        topic="recharge",
        speaker_names=["Asha"],
        idle_timeout_sec=120,
        min_silence_ms=300,
    )
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    await _collect(stream)

    h = server.sessions[0].headers
    assert h["x-api-key-id"] == "test-key"
    assert h["model"] == "prisma-v3.0"
    assert h["language_code"] == "indic-auto"
    assert h["x-language-candidates"] == "hi-IN,en-IN"
    assert h["format"] == "codemix"
    assert json.loads(h["keyterms"]) == ["Gnani"]
    assert h["topic"] == "recharge"
    assert json.loads(h["speaker_names"]) == ["Asha"]
    assert h["x-idle-timeout-sec"] == "120"
    assert h["x-min-silence-ms"] == "300"
    assert h["x-sample-rate"] == "16000"


async def test_utterance_produces_start_final_end_in_order(server):
    """The server sends no end-of-speech message. STT-based turn detection needs an
    END_OF_SPEECH, so it follows the final transcript."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()

    events = await _collect(stream)

    assert _types(events) == [SOS, FINAL, EOS]
    final = events[1]
    assert final.alternatives[0].text == server.text
    assert final.alternatives[0].language == "hi-IN"
    assert final.request_id == "seg-0"


async def test_detected_language_is_reported_not_the_configured_one(server):
    server.detected = "en-IN"
    stt_ = _make_stt(server, model="prisma-v3.0", language="indic-auto")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()

    final = next(e for e in await _collect(stream) if e.type == FINAL)

    assert final.alternatives[0].language == "en-IN"


async def test_language_falls_back_to_the_configured_one_when_not_detected(server):
    server.detected = None  # prisma-v2.5 and format="transliterate" do not return it
    stt_ = _make_stt(server, language="ta-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    final = next(e for e in await _collect(stream) if e.type == FINAL)
    assert final.alternatives[0].language == "ta-IN"

    stt_ = _make_stt(server, model="prisma-v3.0", language="indic-auto", format="transliterate")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream)
    stream.end_input()
    final = next(e for e in await _collect(stream) if e.type == FINAL)
    assert final.alternatives[0].language == "multi"  # unknown, not the literal "indic-auto"


async def test_flush_sends_a_commit_frame(server):
    """manual_flush: LiveKit's flush() closes the utterance without waiting for the
    server's trailing-silence window."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=True)

    await _wait_for(lambda: server.sessions and {"type": "commit"} in server.sessions[0].controls)
    async for ev in stream:
        if ev.type == FINAL:
            break
    stream.end_input()
    await _collect(stream)

    assert {"type": "commit"} in server.sessions[0].controls


async def test_end_input_commits_so_the_last_utterance_is_not_lost(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    stream.end_input()  # no explicit flush from the caller

    events = await _collect(stream)

    assert FINAL in _types(events)


async def test_end_input_waits_for_a_late_transcript_of_the_final_utterance(server):
    """The server may take far longer than a fixed grace period to answer a short clip:
    speech_start and the transcript both arrive after end_input(). They must not be lost."""
    server.reply_delay = 1.2
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    stream.end_input()

    events = await _collect(stream)

    assert _types(events) == [SOS, FINAL, EOS]
    assert events[1].alternatives[0].text == server.text


async def test_end_input_returns_once_the_final_utterance_failed(server):
    """An utterance error answers the commit too, so there is nothing left to wait for."""
    server.reply_delay = 0.3
    server.utterance_errors_left = 1
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.push_frame(FRAME)
    stream.end_input()

    started = asyncio.get_running_loop().time()
    events = await _collect(stream)

    assert FINAL not in _types(events)
    assert asyncio.get_running_loop().time() - started < 5  # well below the 30 s timeout


async def test_end_input_gives_up_at_the_connection_timeout_when_never_answered(server):
    """A final utterance without speech is never answered: wait for the configured
    timeout (not forever) and finish quietly."""
    server.silent = True
    errors: list[stt.STTError] = []
    stt_ = _make_stt(server, language="hi-IN")
    stt_.on("error", errors.append)
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1))
    stream.push_frame(FRAME)
    stream.end_input()

    started = asyncio.get_running_loop().time()
    events = await _collect(stream)
    elapsed = asyncio.get_running_loop().time() - started

    assert events == []
    assert not errors
    assert 0.9 < elapsed < 4


async def test_end_input_does_not_wait_when_nothing_was_sent(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
    stream.end_input()

    started = asyncio.get_running_loop().time()
    assert await _collect(stream) == []
    assert asyncio.get_running_loop().time() - started < 5


async def test_partial_frames_are_chunked_to_1024_bytes(server):
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    for _ in range(5):
        stream.push_frame(FRAME)  # 5 x 3200 B = 16000 B -> 15 full chunks + a remainder
    stream.end_input()
    await _collect(stream)

    assert server.sessions[0].audio == 16000


async def test_idle_timeout_reconnects_transparently(server):
    """The server ends idle sessions (60 s by default) with an `error`, `session_end` and a
    1001 close. That is routine: it must not surface as an error, and the stream carries on
    with a new session (an error here counts toward LiveKit's unrecoverable-error limit)."""
    server.behaviours[0] = "idle"
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    errors: list[Any] = []
    stt_.on("error", errors.append)

    stream.push_frame(FRAME)  # session 0 answers this with the idle timeout
    await _wait_for(lambda: len(server.sessions) == 2)
    await _utterance(stream)
    stream.end_input()
    events = await _collect(stream)

    assert FINAL in _types(events)
    assert errors == []
    assert server.sessions[1].headers["model"] == "prisma-v3.0"
    assert server.sessions[1].audio > 0


async def test_reconnect_uses_options_updated_in_the_meantime(server):
    server.behaviours[0] = "idle"
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", format="verbatim")
    stream = stt_.stream(conn_options=OPTS)
    stt_.update_options(format="transcribe")

    stream.push_frame(FRAME)
    await _wait_for(lambda: len(server.sessions) == 2)
    stream.end_input()
    await _collect(stream)

    assert server.sessions[1].headers["format"] == "transcribe"


async def test_session_that_keeps_closing_immediately_gives_up(server):
    for i in range(10):
        server.behaviours[i] = "quick_close"
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    with pytest.raises(APIConnectionError, match="keeps closing"):
        await _collect(stream)

    assert len(server.sessions) == 3


async def test_utterance_error_does_not_kill_the_stream(server):
    """ "Transcription failed or audio unclear" leaves the connection open; a noisy moment
    must not end the stream. The utterance it belonged to is closed."""
    server.utterance_errors_left = 1
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    await _utterance(stream)  # fails
    first = []
    async for ev in stream:
        first.append(ev)
        if ev.type == EOS:
            break
    assert _types(first) == [SOS, EOS]

    await _utterance(stream)  # next utterance on the same connection works
    stream.end_input()
    rest = await _collect(stream)

    assert FINAL in _types(rest)
    assert len(server.sessions) == 1


async def test_rejected_handshake_is_a_clear_non_retryable_error(server):
    server.behaviours[0] = "reject"
    stt_ = _make_stt(server, language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)

    with pytest.raises(APIStatusError) as exc:
        await _collect(stream)

    assert exc.value.status_code == 400
    assert "Invalid x-format" in str(exc.value)
    assert exc.value.retryable is False
    assert len(server.sessions) == 1


async def test_update_options_applies_live_via_update_settings(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", format="verbatim")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language="en-IN", format="transcribe", keyterms=["Gnani", "Vachana"])

    await _wait_for(
        lambda: any(c["type"] == "update_settings" for c in server.sessions[0].controls)
    )
    stream.end_input()
    await _collect(stream)

    update = next(c for c in server.sessions[0].controls if c["type"] == "update_settings")
    assert update == {
        "type": "update_settings",
        "language_code": "en-IN",
        "format": "transcribe",
        "bias": {"list": ["Gnani", "Vachana"]},
    }


def _language_updates(session: Any) -> list[dict[str, Any]]:
    return [c for c in session.controls if c["type"] == "update_settings" and "language_code" in c]


@pytest.mark.parametrize(
    ("start", "update", "headers"),
    [
        # single -> list: the new session needs the candidate restriction
        (
            "hi-IN",
            ["hi-IN", "en-IN"],
            {"language_code": "indic-auto", "x-language-candidates": "hi-IN,en-IN"},
        ),
        # list -> a different list: the old restriction would stay in force otherwise
        (
            ["hi-IN", "en-IN"],
            ["ta-IN", "te-IN"],
            {"language_code": "indic-auto", "x-language-candidates": "ta-IN,te-IN"},
        ),
        # list -> single: the candidate header must go away
        (["hi-IN", "en-IN"], "ta-IN", {"language_code": "ta-IN"}),
    ],
)
async def test_language_list_changes_reconnect_with_the_new_handshake(
    server, start, update, headers
):
    """update_settings can only change `language_code`; the candidate list lives in the
    handshake. Sending `indic-auto` alone would enable unrestricted detection (or keep the
    old restriction), so a change involving a list opens a new session instead."""
    stt_ = _make_stt(server, model="prisma-v3.0", language=start)
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language=update)

    await _wait_for(lambda: len(server.sessions) >= 2)
    stream.end_input()
    await _collect(stream)

    assert _language_updates(server.sessions[0]) == []  # never a partial live update
    new = server.sessions[1].headers
    for key, value in headers.items():
        assert new[key] == value
    if "x-language-candidates" not in headers:
        assert "x-language-candidates" not in new


async def test_single_language_change_still_updates_live(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_.update_options(language="en-IN")

    await _wait_for(lambda: _language_updates(server.sessions[0]))
    stream.end_input()
    await _collect(stream)

    assert _language_updates(server.sessions[0]) == [
        {"type": "update_settings", "language_code": "en-IN"}
    ]
    assert len(server.sessions) == 1  # no reconnect


async def test_language_list_change_keeps_the_utterance_in_progress(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)  # the server answers with speech_start
    first = await stream.__anext__()
    assert first.type == SOS

    stt_.update_options(language=["hi-IN", "en-IN"], format="transcribe")

    await _wait_for(lambda: len(server.sessions) >= 2)
    stream.end_input()
    events = [first, *await _collect(stream)]

    assert {"type": "commit"} in server.sessions[0].controls  # committed before closing
    assert FINAL in _types(events)
    assert server.sessions[1].headers["format"] == "transcribe"  # other options carried over


async def test_repeated_language_changes_are_not_mistaken_for_a_broken_session(server):
    """Reconnecting on request must not trip the guard against sessions that keep closing
    immediately."""
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = stt_.stream(conn_options=OPTS)
    stream.push_frame(FRAME)
    for n, language in enumerate(
        [["hi-IN", "en-IN"], ["ta-IN", "te-IN"], "hi-IN", ["en-IN", "hi-IN"]]
    ):
        await _wait_for(lambda n=n: len(server.sessions) == n + 1 and server.sessions[n].audio > 0)
        stt_.update_options(language=language)
        await _wait_for(lambda n=n: len(server.sessions) == n + 2)
        stream.push_frame(FRAME)

    stream.end_input()
    await _collect(stream)  # would raise APIConnectionError if throttled as quick closes
    assert len(server.sessions) == 5


async def test_session_keyterms_reach_a_running_stream(server):
    stt_ = _make_stt(server, model="prisma-v3.0", language="hi-IN", keyterms=["user"])
    stream = stt_.stream(conn_options=OPTS)
    await _utterance(stream, flush=False)
    await _wait_for(lambda: server.sessions and server.sessions[0].audio > 0)

    stt_._update_session_keyterms(["auto"])

    await _wait_for(
        lambda: any(c["type"] == "update_settings" for c in server.sessions[0].controls)
    )
    stream.end_input()
    await _collect(stream)

    update = next(c for c in server.sessions[0].controls if c["type"] == "update_settings")
    assert update["bias"] == {"list": ["user", "auto"]}


async def test_stream_language_override_is_validated(server):
    stt_ = _make_stt(server, language="hi-IN")  # default model: one language only
    with pytest.raises(ValueError, match="prisma-v3.0"):
        stt_.stream(language=["hi-IN", "en-IN"], conn_options=OPTS)

    v3 = _make_stt(server, model="prisma-v3.0", language="hi-IN")
    stream = v3.stream(language=["hi-IN", "en-IN"], conn_options=OPTS)
    stream.end_input()
    await _collect(stream)
    assert server.sessions[-1].headers["x-language-candidates"] == "hi-IN,en-IN"


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


async def test_recognize_sends_model_aware_form_and_reports_detected_language(server):
    server.detected = "en-IN"
    stt_ = _make_stt(
        server,
        model="prisma-v3.0",
        language=["hi-IN", "en-IN"],
        format="codemix",
        keyterms=["Gnani"],
        topic="recharge",
    )

    event = await stt_.recognize(FRAME, conn_options=OPTS)

    call = server.rest_calls[0]
    assert call["language_code"] == "hi-IN,en-IN"
    assert call["model"] == "prisma-v3.0"
    assert call["format"] == "codemix"
    assert json.loads(call["keyterms"]) == ["Gnani"]
    assert call["topic"] == "recharge"
    assert call["_headers"]["X-API-Key-ID"] == "test-key"
    assert event.alternatives[0].text == server.text
    assert event.alternatives[0].language == "en-IN"
    assert event.request_id == "req-1"


async def test_recognize_default_model_sends_no_model(server):
    server.detected = None
    stt_ = _make_stt(server, language="hi-IN")

    event = await stt_.recognize(FRAME, conn_options=OPTS)

    assert "model" not in server.rest_calls[0]
    assert server.rest_calls[0]["language_code"] == "hi-IN"
    assert event.alternatives[0].language == "hi-IN"


async def test_recognize_language_override_is_validated(server):
    stt_ = _make_stt(server, language="hi-IN")
    with pytest.raises(ValueError, match="prisma-v3.0"):
        await stt_.recognize(FRAME, language="indic-auto", conn_options=OPTS)
    assert server.rest_calls == []


async def test_recognize_api_error_is_surfaced(server):
    server.rest_status = 400
    stt_ = _make_stt(server, language="hi-IN")

    with pytest.raises(APIStatusError) as exc:
        await stt_.recognize(FRAME, conn_options=OPTS)

    assert exc.value.status_code == 400
    assert "INVALID_LANGUAGE_CODE" in str(exc.value)
