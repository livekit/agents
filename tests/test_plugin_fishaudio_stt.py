from __future__ import annotations

from unittest.mock import MagicMock, patch

import aiohttp
import msgpack
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents.llm import ChatMessage
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins.fishaudio import STT
from livekit.plugins.fishaudio.stt import SpeechStream, _close_code_to_status

from .stt_reconnect import FakeWebSocket, wait_until

pytestmark = pytest.mark.unit

_INPUT_STARTED_AT = 1_700_000_000.0


def _make_stream(stt: STT | None = None) -> SpeechStream:
    """Build a stream without starting its connection loop."""
    if stt is None:
        stt = STT(api_key="test-key", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream()
    stream._input_ch.input_started_at = _INPUT_STARTED_AT
    return stream


def _drain(stream: SpeechStream) -> list[SpeechEvent]:
    events = []
    while True:
        try:
            events.append(stream._event_ch.recv_nowait())
        except ChanEmpty:
            return events


def _committed(**overrides):
    msg = {
        "type": "committed",
        "turn_index": 0,
        "text": "Hi, I'd like to book a table.",
        "words": [
            {"text": "Hi", "start_ms": 500, "end_ms": 900},
            {"text": "table", "start_ms": 2100, "end_ms": 2500},
        ],
        "language_code": "en",
        "audio_start_ms": 256,
        "audio_end_ms": 3600,
        "reason": "eot",
        "degraded": False,
        "user_turn": True,
        "turn_kind": "user",
    }
    msg.update(overrides)
    return msg


def _user_turn(stream: SpeechStream, turn_index: int = 0) -> None:
    stream._process_message({"type": "speech_start", "turn_index": turn_index, "audio_ms": 256})
    stream._process_message({"type": "user_turn_taken", "turn_index": turn_index, "audio_ms": 900})


# ---------------------------------------------------------------------------
# turn events
# ---------------------------------------------------------------------------


async def test_auto_mode_waits_for_user_turn_taken():
    stream = _make_stream()

    stream._process_message({"type": "speech_start", "turn_index": 0, "audio_ms": 256})
    stream._process_message({"type": "draft", "turn_index": 0, "text": "Hi,", "audio_ms": 880})
    assert _drain(stream) == []

    stream._process_message({"type": "user_turn_taken", "turn_index": 0, "audio_ms": 1280})
    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
    ]
    # the onset of speech, not the moment the turn was taken
    assert events[0].speech_start_time == pytest.approx(_INPUT_STARTED_AT + 0.256)
    assert events[1].alternatives[0].text == "Hi,"


async def test_auto_mode_turn_events():
    stream = _make_stream()
    _user_turn(stream)
    stream._process_message(
        {"type": "draft", "turn_index": 0, "text": "Hi, I'd like", "audio_ms": 1600}
    )
    stream._process_message(
        {
            "type": "eager_end_of_turn",
            "turn_index": 0,
            "text": "Hi, I'd like to book a table.",
            "language_code": "en",
            "audio_ms": 3000,
        }
    )
    stream._process_message(_committed())

    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.PREFLIGHT_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[2].alternatives[0].text == "Hi, I'd like to book a table."

    final = events[3].alternatives[0]
    assert final.text == "Hi, I'd like to book a table."
    assert final.language == "en"
    assert final.start_time == pytest.approx(0.5)
    assert final.end_time == pytest.approx(2.5)
    assert final.words is not None
    assert [str(word) for word in final.words] == ["Hi", "table"]

    # end of speech carries the last word's end, not the commit time
    end_of_speech = events[4]
    assert end_of_speech.alternatives[0].end_time == pytest.approx(2.5)


async def test_timestamps_follow_the_stream_offset():
    stream = _make_stream()
    stream._input_ch._start_time_offset = 2.0
    _user_turn(stream)
    stream._process_message(_committed())

    events = _drain(stream)
    assert events[0].speech_start_time == pytest.approx(_INPUT_STARTED_AT + 2.0 + 0.256)
    final = events[1].alternatives[0]
    assert final.start_time == pytest.approx(2.5)
    assert final.end_time == pytest.approx(4.5)
    assert final.words is not None
    assert final.words[0].start_time == pytest.approx(2.5)
    assert events[2].alternatives[0].end_time == pytest.approx(4.5)


async def test_auto_mode_ignores_turns_that_are_not_the_users():
    stream = _make_stream()
    stream._process_message({"type": "speech_start", "turn_index": 0, "audio_ms": 256})
    stream._process_message({"type": "draft", "turn_index": 0, "text": "[noise]", "audio_ms": 500})
    stream._process_message(_committed(user_turn=False, turn_kind="other", text="mm"))

    assert _drain(stream) == []
    assert list(stream._history) == []


async def test_non_user_commit_after_start_closes_the_turn():
    stream = _make_stream()
    _user_turn(stream)
    stream._process_message(_committed(user_turn=False, turn_kind="other"))

    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.END_OF_SPEECH,
    ]


async def test_user_commit_without_user_turn_taken_still_starts_the_turn():
    # the protocol sends user_turn_taken first; don't depend on it
    stream = _make_stream()
    stream._process_message({"type": "speech_start", "turn_index": 0, "audio_ms": 256})
    stream._process_message(_committed())

    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[0].speech_start_time == pytest.approx(_INPUT_STARTED_AT + 0.256)


async def test_empty_user_turn_still_sends_a_final_transcript():
    # after a flush, the session waits for a final transcript before committing the turn
    stt = STT(api_key="test-key", http_session=MagicMock(), turn_detection_mode="manual")
    stream = _make_stream(stt)
    stream._process_message({"type": "speech_start", "turn_index": 0, "audio_ms": 256})
    stream._process_message(_committed(text="", words=[], reason="finalize"))

    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == ""
    assert list(stream._history) == []


async def test_vad_mode_starts_turn_on_speech_start():
    stt = STT(api_key="test-key", http_session=MagicMock(), turn_detection_mode="vad")
    stream = _make_stream(stt)

    stream._process_message({"type": "speech_start", "turn_index": 0, "audio_ms": 384})
    stream._process_message({"type": "draft", "turn_index": 0, "text": "Hi", "audio_ms": 900})

    events = _drain(stream)
    assert [ev.type for ev in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
    ]


async def test_messages_for_other_turns_are_ignored():
    stream = _make_stream()
    _user_turn(stream, turn_index=1)
    _drain(stream)

    stream._process_message({"type": "draft", "turn_index": 0, "text": "stale", "audio_ms": 10})
    stream._process_message({"type": "eager_end_of_turn", "turn_index": 0, "text": "stale"})
    assert _drain(stream) == []

    # a late commit for an earlier turn leaves the open turn alone
    stream._process_message(_committed(turn_index=0, user_turn=False, turn_kind="other"))
    stream._process_message({"type": "draft", "turn_index": 1, "text": "still here"})
    events = _drain(stream)
    assert events[-1].type == SpeechEventType.INTERIM_TRANSCRIPT
    assert events[-1].alternatives[0].text == "still here"


async def test_degraded_commit_uses_turn_bounds():
    stream = _make_stream()
    _user_turn(stream)
    msg = _committed(degraded=True)
    del msg["words"]
    stream._process_message(msg)

    events = _drain(stream)
    final = events[1].alternatives[0]
    assert final.words is None
    assert final.start_time == pytest.approx(0.256)
    assert final.end_time == pytest.approx(3.6)
    # no word timings, so no speech end time
    assert events[2].alternatives == []


async def test_committed_user_turn_is_kept_as_context():
    stream = _make_stream()
    stream._process_message(_committed())

    assert list(stream._history) == [{"role": "user", "text": "Hi, I'd like to book a table."}]


async def test_session_started_sets_ids():
    stream = _make_stream()
    assert stream._session_id is None

    stream._process_message({"type": "session_started", "session_id": "s_1", "request_id": "r_1"})
    _user_turn(stream)

    assert stream._session_id == "s_1"
    assert _drain(stream)[0].request_id == "r_1"


async def test_errors():
    stream = _make_stream()

    # recoverable errors and server-ended sessions don't fail the stream
    stream._process_message(
        {"type": "error", "code": 503, "error_type": "engine_error", "recoverable": True}
    )
    stream._process_message(
        {"type": "error", "code": 503, "error_type": "server_restarting", "recoverable": True}
    )
    stream._process_message(
        {
            "type": "error",
            "code": 413,
            "error_type": "max_duration_exceeded",
            "recoverable": False,
        }
    )

    with pytest.raises(APIStatusError) as exc_info:
        stream._process_message(
            {
                "type": "error",
                "code": 400,
                "error_type": "invalid_config",
                "message": "bad config",
                "recoverable": False,
            }
        )
    assert exc_info.value.status_code == 400
    assert not exc_info.value.retryable

    with pytest.raises(APIStatusError) as exc_info:
        stream._process_message(
            {"type": "error", "code": 500, "error_type": "internal", "recoverable": False}
        )
    assert exc_info.value.retryable


def test_close_code_to_status():
    assert _close_code_to_status(4400) == 400
    assert _close_code_to_status(4408) == 408
    assert _close_code_to_status(1006) == -1


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


async def test_default_config():
    stream = _make_stream()
    config = stream._config()

    assert config["type"] == "config"
    assert config["audio"] == {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    assert config["turn_detection"] == {"mode": "auto", "eager": True}
    assert config["results"] == {
        "drafts": True,
        "ignore_timestamps": False,
        "diarize": False,
        "tag_audio_events": False,
    }
    assert "context" not in config


async def test_vad_config():
    stt = STT(
        api_key="test-key",
        http_session=MagicMock(),
        turn_detection_mode="vad",
        silence_timeout_ms=500,
        interim_results=False,
    )
    config = _make_stream(stt)._config()

    # eager end of turn only exists in "auto" mode
    assert config["turn_detection"] == {"mode": "vad", "silence_timeout_ms": 500}
    assert config["results"]["drafts"] is False


def test_sample_rate_validated():
    with pytest.raises(ValueError):
        STT(api_key="test-key", sample_rate=11025)
    STT(api_key="test-key", sample_rate=8000)


def test_silence_timeout_validated_per_mode():
    with pytest.raises(ValueError):
        STT(api_key="test-key", silence_timeout_ms=700)

    STT(api_key="test-key", turn_detection_mode="vad", silence_timeout_ms=700)

    stt = STT(api_key="test-key")
    with pytest.raises(ValueError):
        stt.update_options(silence_timeout_ms=100)


async def test_keyterms_are_sanitized():
    stt = STT(
        api_key="test-key",
        http_session=MagicMock(),
        keyterms=["LiveKit", " LiveKit ", "", "x" * 33, "Fish Audio"],
    )
    config = _make_stream(stt)._config()

    assert config["context"]["terms"] == ["LiveKit", "Fish Audio"]


async def test_keyterms_are_capped():
    terms = [f"term {index}" for index in range(120)]
    stt = STT(api_key="test-key", http_session=MagicMock(), keyterms=terms)

    assert stt._opts.keyterms == terms[:100]


async def test_keyterm_updates_are_sent_to_active_streams():
    stt = STT(api_key="test-key", http_session=MagicMock(), keyterms=["LiveKit"])
    stream = _make_stream(stt)
    stream._ws = MagicMock()

    stt._update_session_keyterms(["Fish Audio"])
    msg = stream._control_queue.get_nowait()
    assert msg == {"type": "configure", "context": {"terms": ["LiveKit", "Fish Audio"]}}

    stt.update_options(keyterms=["Agents"], silence_timeout_ms=4000)
    msg = stream._control_queue.get_nowait()
    assert msg == {
        "type": "configure",
        "turn_detection": {"silence_timeout_ms": 4000},
        "context": {"terms": ["Agents", "Fish Audio"]},
    }


async def test_eager_update_is_dropped_outside_auto_mode():
    stt = STT(api_key="test-key", http_session=MagicMock(), turn_detection_mode="vad")
    stream = _make_stream(stt)
    stream._ws = MagicMock()

    stt.update_options(eager_eot=True)
    assert stream._control_queue.empty()


async def test_agent_replies_are_sent_as_context():
    stt = STT(api_key="test-key", http_session=MagicMock())
    stream = _make_stream(stt)
    stream._ws = MagicMock()

    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="user", content=["hello"]))
    )
    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["What time?"]))
    )

    msg = stream._control_queue.get_nowait()
    assert msg == {
        "type": "context_add",
        "items": [{"role": "assistant", "text": "What time?"}],
    }
    assert stream._control_queue.empty()

    # a new stream starts with the agent's replies so far
    config = _make_stream(stt)._config()
    assert config["context"]["history"] == [{"role": "assistant", "text": "What time?"}]


async def test_user_turns_stay_with_their_stream():
    stt = STT(api_key="test-key", http_session=MagicMock())
    first = _make_stream(stt)
    first._process_message(_committed())

    second = _make_stream(stt)
    assert "context" not in second._config()


async def test_context_is_kept_while_disconnected():
    stt = STT(api_key="test-key", http_session=MagicMock())
    stream = _make_stream(stt)

    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["What time?"]))
    )

    # not queued: the next connection's config carries it
    assert stream._control_queue.empty()
    assert stream._config()["context"]["history"] == [{"role": "assistant", "text": "What time?"}]


async def test_history_is_capped():
    stt = STT(api_key="test-key", http_session=MagicMock())
    for index in range(40):
        stt._push_conversation_item(
            ConversationItemAddedEvent(
                item=ChatMessage(role="assistant", content=[f"{index} " + "x" * 2500])
            )
        )

    history = _make_stream(stt)._config()["context"]["history"]
    assert len(history) == 32
    assert all(len(item["text"]) == 2000 for item in history)


# ---------------------------------------------------------------------------
# wire protocol
# ---------------------------------------------------------------------------


class FishSocket(FakeWebSocket):
    def respond(self, data: dict) -> None:
        payload = msgpack.packb(data)
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, payload, ""))

    def server_close(self, code: int) -> None:
        self.close_code = code
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, ""))

    async def ping(self) -> None:
        pass

    def messages(self) -> list[dict]:
        return [msgpack.unpackb(data) for data in self.sent]

    def types(self) -> list[str]:
        return [msg["type"] for msg in self.messages()]


class BrokenSocket(FishSocket):
    async def send_bytes(self, data: bytes) -> None:
        raise ConnectionResetError("Cannot write to closing transport")


class FishServer:
    def __init__(self, socket_type: type[FishSocket] = FishSocket) -> None:
        self.sockets: list[FishSocket] = []
        self.closed = False
        self._socket_type = socket_type

    def ws_connect(self, *args, **kwargs) -> FishSocket:
        ws = self._socket_type()
        self.sockets.append(ws)
        return ws


def _silence(ms: int, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(b"\x00\x00" * samples, sample_rate, 1, samples)


async def _collect(stream: SpeechStream) -> list[SpeechEvent]:
    events = []
    async for event in stream:
        events.append(event)
    return events


async def test_wire_protocol_in_manual_mode():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server, turn_detection_mode="manual")
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)
    ws = server.sockets[0]

    stream.push_frame(_silence(30))
    stream.flush()
    await wait_until(lambda: "finalize" in ws.types())

    messages = ws.messages()
    assert messages[0]["type"] == "config"
    assert messages[0]["turn_detection"] == {"mode": "manual"}
    # 20 ms chunks as msgpack bin; the flush sends the rest, then finalizes the turn
    assert [msg["type"] for msg in messages[1:]] == ["audio", "audio", "finalize"]
    assert isinstance(messages[1]["data"], bytes)
    assert len(messages[1]["data"]) == 640
    assert len(messages[2]["data"]) == 320

    stream.end_input()
    await wait_until(lambda: "audio_done" in ws.types())
    ws.respond({"type": "session_done", "stats": {"audio_ms": 30, "turns": 0}})
    ws.server_close(1000)

    events = await _collect(stream)
    usage = [event for event in events if event.type == SpeechEventType.RECOGNITION_USAGE]
    assert sum(event.recognition_usage.audio_duration for event in usage) == pytest.approx(0.03)


async def test_flush_finalizes_the_turn_in_auto_mode():
    # the session flushes when its own VAD or turn detector ends the turn
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)
    ws = server.sockets[0]

    stream.push_frame(_silence(20))
    stream.flush()
    await wait_until(lambda: "finalize" in ws.types())

    messages = ws.messages()
    assert messages[0]["turn_detection"]["mode"] == "auto"
    await stream.aclose()


async def test_flush_with_an_open_turn_waits_for_the_commit():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)
    ws = server.sockets[0]

    ws.respond({"type": "speech_start", "turn_index": 0, "audio_ms": 100})
    await wait_until(lambda: stream._turn is not None)
    stream.push_frame(_silence(20))
    stream.flush()
    await wait_until(lambda: "finalize" in ws.types())
    ws.respond(_committed(reason="finalize"))
    stream.end_input()
    await wait_until(lambda: "audio_done" in ws.types())
    ws.server_close(1000)

    events = await _collect(stream)
    transcripts = [event for event in events if event.type != SpeechEventType.RECOGNITION_USAGE]
    assert [event.type for event in transcripts] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert transcripts[1].alternatives[0].text == "Hi, I'd like to book a table."


async def test_server_restart_starts_a_new_session_with_the_conversation():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stt._push_conversation_item(
        ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["What time?"]))
    )
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)
    first = server.sockets[0]

    first.respond({"type": "session_started", "session_id": "s_1", "request_id": "r_1"})
    first.respond({"type": "speech_start", "turn_index": 0, "audio_ms": 100})
    first.respond({"type": "user_turn_taken", "turn_index": 0, "audio_ms": 300})
    first.respond(_committed(text="Seven thirty."))
    first.respond({"type": "error", "code": 503, "error_type": "server_restarting"})
    first.server_close(1012)

    await wait_until(lambda: len(server.sockets) == 2)
    second = server.sockets[1]
    await wait_until(lambda: len(second.sent) >= 1)
    config = second.messages()[0]
    assert config["type"] == "config"
    assert config["context"]["history"] == [
        {"role": "assistant", "text": "What time?"},
        {"role": "user", "text": "Seven thirty."},
    ]
    await stream.aclose()


async def test_session_ending_before_any_speech_raises_a_retryable_error():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
    await wait_until(lambda: len(server.sockets) == 1)
    server.sockets[0].server_close(1012)

    with pytest.raises(APIConnectionError) as exc_info:
        await _collect(stream)
    assert exc_info.value.retryable


async def test_failed_config_send_raises_a_connection_error():
    server = FishServer(socket_type=BrokenSocket)
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

    with pytest.raises(APIConnectionError):
        await _collect(stream)


async def test_idle_session_at_maximum_duration_starts_a_new_one():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
    await wait_until(lambda: len(server.sockets) == 1)
    server.sockets[0].server_close(4413)

    await wait_until(lambda: len(server.sockets) == 2)
    assert server.sockets[1].messages()[0]["type"] == "config"
    await stream.aclose()


async def test_closing_the_stt_closes_its_streams():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)

    await stt.aclose()
    assert stream._task.done()
    assert server.sockets[0].closed


async def test_invalid_config_close_is_not_retried():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server)
    stream = stt.stream()
    await wait_until(lambda: len(server.sockets) == 1)
    server.sockets[0].server_close(4400)

    with pytest.raises(APIStatusError) as exc_info:
        await _collect(stream)
    assert exc_info.value.status_code == 400
    assert len(server.sockets) == 1


async def test_dropped_connection_ends_the_open_turn():
    server = FishServer()
    stt = STT(api_key="test-key", http_session=server, turn_detection_mode="vad")
    stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
    await wait_until(lambda: len(server.sockets) == 1)
    ws = server.sockets[0]

    ws.respond({"type": "speech_start", "turn_index": 0, "audio_ms": 100})
    ws.respond({"type": "draft", "turn_index": 0, "text": "Book a table for"})
    ws.server_close(1006)

    events = []
    with pytest.raises(APIStatusError):
        async for event in stream:
            events.append(event)

    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[2].alternatives[0].text == "Book a table for"
