"""Agent state through the real GPT-Live SDK over a loopback WebSocket.

Adapted from zek's reproduction in PR #8379; no provider or SDK internals are patched.
"""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.agents.llm._realtime import gpt_live_types as wire
from livekit.agents.llm._realtime.gpt_live import SAMPLE_RATE, GPTLiveModel, GPTLiveSession
from livekit.agents.voice import io
from livekit.agents.voice.events import ToolCallEnded
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_realtime import FakeRealtimeModel

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

DEADLINE_SECONDS = TIMEOUT = 6
ANSWER = "The fixture result is verified."
RESPONSE_CREATED = "response.created"
RESPONSE_COMPLETED = "response.completed"
RESPONSE_ITEM_DONE = "response.output_item.done"
OUTPUT_TRANSCRIPT_DELTA = "session.output_transcript.delta"
AUDIBLE_SAMPLES = SAMPLE_RATE // 10


class MemorySpeaker(io.AudioOutput):
    """Complete playout when the adapter flushes, without an audio device."""

    def __init__(self):
        super().__init__(
            label="memory",
            next_in_chain=None,
            sample_rate=SAMPLE_RATE,
            capabilities=io.AudioOutputCapabilities(pause=False),
        )
        self.duration = 0.0
        self.captured = asyncio.Event()

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        if not self.duration:
            self.on_playback_started(created_at=time.time())
        self.duration += frame.duration
        self.captured.set()

    def flush(self):
        super().flush()
        duration, self.duration = self.duration, 0.0
        if duration:
            self.on_playback_finished(playback_position=duration, interrupted=False)

    def clear_buffer(self):
        self.duration = 0.0


class ProviderPeer:
    """Exchange real SDK wire events over a local WebSocket."""

    def __init__(self):
        self.ws = None
        self.commands = asyncio.Queue()
        self.processed = asyncio.Queue()
        self.runner = None
        self.close_received = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()

    async def handler(self, request):
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        async for message in self.ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            self.commands.put_nowait(event)
            if event["type"] == wire.SessionStartEvent.model_fields["type"].default:
                await self.send(
                    wire.SessionStartedEvent(session=wire.SessionResource(id="session"))
                )
            elif event["type"] == wire.SessionCloseEvent.model_fields["type"].default:
                self.close_received.set()
                await self.allow_close.wait()
                await self.send(wire.SessionClosedEvent(reason="close_requested"))
        return self.ws

    async def start(self):
        app = web.Application()
        app.router.add_get("/v1/live/sessions", self.handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def send(self, event):
        await self.ws.send_str(event.model_dump_json(exclude_none=True))

    async def receive(self, event_type):
        async with asyncio.timeout(DEADLINE_SECONDS):
            while True:
                event = await self.commands.get()
                if event["type"] == event_type:
                    return event

    async def user_turn(self, realtime):
        """Supply a real input turn so an earlier startup away timer cannot taint assertions."""
        event = wire.TranscriptDeltaEvent(
            type="session.input_transcript.delta", delta="Please check.", start_ms=0, end_ms=100
        )
        await self.send(event)
        async with asyncio.timeout(TIMEOUT):
            while (await self.processed.get()).get("type") != event.type:
                pass
        realtime.push_audio(
            rtc.AudioFrame(
                data=b"\x00\x00" * SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLE_RATE,
            )
        )
        await self.receive(wire.InputAudioAppendEvent.model_fields["type"].default)

    async def backend(self, event_type, *, item=None, delegation_id="delegation", **fields):
        event = wire.ResponseEventEnvelope(
            delegation_id=delegation_id,
            event=wire.ResponsesEvent(type=event_type, item=item, **fields),
        )
        await self.send(event)
        async with asyncio.timeout(DEADLINE_SECONDS):
            while True:
                received = await self.processed.get()
                if (
                    received.get("type") == event.type
                    and received.get("event", {}).get("type") == event_type
                ):
                    return

    async def late_transcript(self, speeches):
        """Deliver old text and wait until its text-only generation has finished."""
        event = wire.TranscriptDeltaEvent(
            type=OUTPUT_TRANSCRIPT_DELTA, delta="Late old words.", start_ms=50, end_ms=100
        )
        await self.send(event)
        async with asyncio.timeout(TIMEOUT):
            while (await self.processed.get()).get("type") != event.type:
                pass
        await speeches[-1]

    async def tool_call(self, call_id="later", *, delegation_id="delegation"):
        await self.backend(
            RESPONSE_ITEM_DONE,
            delegation_id=delegation_id,
            item=wire.OutputItem(
                id=call_id,
                type="function_call",
                status="completed",
                call_id=call_id,
                name="lookup",
                arguments="{}",
            ),
        )

    async def continuation(self, status=RESPONSE_COMPLETED, *, delegation_id="delegation"):
        await self.backend(RESPONSE_CREATED, delegation_id=delegation_id)
        await self.backend(status, delegation_id=delegation_id)

    async def audio(self, *, speech=True):
        pcm = b"\x00\x10" * AUDIBLE_SAMPLES if speech else b"\x00\x00" * SAMPLE_RATE
        await self.send(wire.OutputAudioDeltaEvent(delta=base64.b64encode(pcm).decode()))

    async def speech(self, session, speeches, text, *, speech_handle=None, **timestamps):
        index = len(speeches)
        created = asyncio.Event()
        if speech_handle is not None:
            created.set()

        def observe(event):
            if len(speeches) > index:
                created.set()

        session.on("speech_created", observe)
        try:
            await self.send(
                wire.TranscriptDeltaEvent(type=OUTPUT_TRANSCRIPT_DELTA, delta=text, **timestamps)
            )
            await self.audio()
            await self.audio(speech=False)
            await asyncio.wait_for(created.wait(), DEADLINE_SECONDS)
            handle = speech_handle if speech_handle is not None else speeches[index]
            await asyncio.wait_for(handle.wait_for_playout(), DEADLINE_SECONDS)
        finally:
            session.off("speech_created", observe)

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
        await self.runner.cleanup()


async def _wait_for_agent_state(session, state):
    """Wait for a state edge without polling or depending on scheduler delays."""
    changed = asyncio.Event()

    def observe(event):
        if event.new_state == state:
            changed.set()

    session.on("agent_state_changed", observe)
    try:
        if session.agent_state != state:
            await asyncio.wait_for(changed.wait(), TIMEOUT)
    finally:
        session.off("agent_state_changed", observe)


@dataclass
class _Conversation:
    """Shared wire/session fixture; tests below describe each lifecycle explicitly."""

    peer: ProviderPeer
    session: AgentSession
    realtime: llm.RealtimeSession
    live: GPTLiveSession
    speeches: list[SpeechHandle]
    release: asyncio.Event
    release_enter: asyncio.Event
    started: asyncio.Queue[None]
    away: asyncio.Event
    run: RunResult[None] | None
    count: int = 1

    @property
    def activity(self):
        return self.session._activity

    @property
    def pending_calls(self):
        return {call for call, reply in self.live._tool_replies.items() if reply.submitted}

    async def open_tools(self, *, count=1, complete=True):
        self.count = count
        await self.peer.backend(RESPONSE_CREATED)
        for index in range(count):
            await self.peer.tool_call(f"call-{index}")
        if complete:
            await self.peer.backend(RESPONSE_COMPLETED)
        for _ in range(count):
            await asyncio.wait_for(self.started.get(), TIMEOUT)
        await self.peer.speech(self.session, self.speeches, "Checking. ", start_ms=0, end_ms=100)
        await self.peer.user_turn(self.realtime)
        self.away.clear()
        assert self.session.agent_state == "thinking"

    async def submit_results(self):
        self.release.set()
        for _ in range(self.count):
            await self.peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
        await self.peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)

    async def ready_reply(self, *, count=1):
        await self.open_tools(count=count)
        await self.submit_results()
        await self.peer.continuation()
        await asyncio.gather(*self.speeches)
        self.assert_pending()
        if self.run is not None:
            assert self.activity._pending_auto_tool_reply_fut is not None
            assert not self.run.done()

    def assert_pending(self):
        assert self.live.pending_tool_reply
        assert self.session.agent_state == "thinking"
        assert self.session._user_away_timer is None
        assert not self.away.is_set()

    async def release_hook(self):
        self.release_enter.set()
        if self.activity._on_enter_task is not None:
            await self.activity._on_enter_task

    async def finish(self):
        await self.release_hook()
        await self.peer.speech(self.session, self.speeches, ANSWER)
        await self.assert_idle()
        assert any(
            getattr(item, "text_content", None) == ANSWER for item in self.session.history.items
        )
        await asyncio.wait_for(self.away.wait(), TIMEOUT)

    async def assert_idle(self):
        await _wait_for_agent_state(self.session, "listening")
        assert not self.live.pending_tool_reply
        if self.run is not None:
            await asyncio.wait_for(self.run, TIMEOUT)


@asynccontextmanager
async def _conversation(with_run, *, no_reply=False, fallback=None):
    peer = ProviderPeer()
    url = await peer.start()
    release, release_enter, away = asyncio.Event(), asyncio.Event(), asyncio.Event()
    started = asyncio.Queue()
    speeches = []

    class ToolAgent(Agent):
        async def on_enter(self):
            if with_run:
                await release_enter.wait()

    @function_tool
    async def lookup():
        """Return a local fixture after the test releases execution."""
        started.put_nowait(None)
        await release.wait()
        return llm.ToolResult(ANSWER, reply_required=False) if no_reply else ANSWER

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as http:

        def make_model():
            return GPTLiveModel(
                model="gpt-live-1",
                delegation="responses",
                api_key="local-peer-not-a-key",
                base_url=url,
                http_session=http,
            )

        model = make_model()
        session_model = model
        if fallback:
            primary = llm.DuplexRealtimeAdapter(model)
            backup = (
                FakeRealtimeModel(capabilities=primary.capabilities)
                if fallback == "legacy"
                else llm.DuplexRealtimeAdapter(make_model())
            )
            session_model = llm.RealtimeModelFallbackAdapter(
                [primary, backup], regenerate_on_swap=fallback == "regenerate"
            )
        session = AgentSession(llm=session_model, user_away_timeout=0.1)
        session.output.audio = MemorySpeaker()
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        session.on(
            "user_state_changed", lambda event: away.set() if event.new_state == "away" else None
        )
        start_task = asyncio.create_task(
            session.start(
                ToolAgent(instructions="Answer briefly.", tools=[lookup]), capture_run=with_run
            )
        )
        try:
            await peer.receive(wire.SessionStartEvent.model_fields["type"].default)
            if not with_run:
                await start_task
            realtime = session._activity.realtime_llm_session
            live = (realtime._active if fallback else realtime).duplex_session
            live.on("openai_server_event_received", peer.processed.put_nowait)
            yield _Conversation(
                peer,
                session,
                realtime,
                live,
                speeches,
                release,
                release_enter,
                started,
                away,
                session._global_run_state if with_run else None,
            )
        finally:
            release.set()
            release_enter.set()
            await asyncio.wait_for(session.aclose(), TIMEOUT)
            await asyncio.wait_for(start_task, TIMEOUT)
            await model.aclose()
            await peer.close()


@pytest.fixture(params=[False, True], ids=["no_run", "active_run"])
def with_run(request):
    return request.param


@pytest.mark.parametrize("complete_before_playout", [False, True], ids=["running", "completed"])
async def test_inline_tool_preamble_stays_busy_through_result_submission(
    with_run, complete_before_playout
):
    """A call in the audible burst owns work until its result reaches the provider."""
    async with _conversation(with_run) as c:
        ended, executed = asyncio.Event(), asyncio.Event()
        states = []
        c.session.on("agent_state_changed", lambda event: states.append(event.new_state))
        c.session.on(
            "tool_execution_updated",
            lambda event: ended.set() if isinstance(event.update, ToolCallEnded) else None,
        )
        c.session.on("function_tools_executed", lambda _: executed.set())
        await c.peer.user_turn(c.realtime)
        await c.peer.send(
            wire.TranscriptDeltaEvent(
                type=OUTPUT_TRANSCRIPT_DELTA, delta="I'll check that now.", start_ms=0, end_ms=100
            )
        )
        await c.peer.audio()
        await _wait_for_agent_state(c.session, "speaking")
        await c.peer.backend(RESPONSE_CREATED)
        await c.peer.tool_call("call-0")
        await asyncio.wait_for(c.started.get(), TIMEOUT)
        await c.release_hook()
        states.clear()

        # Hold actual result submission after execution has finished. Neither an
        # unfinished hook nor provider pending state can hide this local interval.
        async with c.activity._rt_chat_ctx_lock:
            if complete_before_playout:
                c.release.set()
                await asyncio.wait_for(ended.wait(), TIMEOUT)
            await c.peer.backend(RESPONSE_COMPLETED)
            await c.peer.audio(speech=False)
            await _wait_for_agent_state(c.session, "thinking")
            assert "listening" not in states
            assert c.session._user_away_timer is None
            c.release.set()
            await asyncio.wait_for(executed.wait(), TIMEOUT)
            assert not c.live.pending_tool_reply
            assert c.session.agent_state == "thinking"
            assert c.session._user_away_timer is None

        await c.peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
        await c.peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
        await asyncio.gather(*c.speeches)
        c.assert_pending()
        assert "listening" not in states
        await c.peer.continuation()
        await c.finish()
        assert states.count("listening") == 1


@pytest.mark.parametrize(
    "case",
    [
        "control",
        "batch",
        "interrupt",
        "late_transcript",
        "no_reply",
        "unrelated_failure",
        "tool_only",
    ],
)
async def test_tool_results_remain_busy_until_audio(with_run, case):
    async with _conversation(with_run, no_reply=case == "no_reply") as c:
        await c.ready_reply(count=2 if case == "batch" else 1)
        await c.release_hook()
        if case == "interrupt":
            await asyncio.wait_for(c.session.interrupt(), TIMEOUT)
        elif case == "late_transcript":
            await c.peer.late_transcript(c.speeches)
        elif case == "unrelated_failure":
            await c.peer.backend("response.failed", delegation_id="other")
        elif case == "tool_only":
            await c.peer.backend(RESPONSE_CREATED)
            await c.peer.tool_call("extra")
            await c.peer.backend(RESPONSE_COMPLETED)
            await c.peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
            await c.peer.continuation()
        c.assert_pending()
        if case == "control":
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(c.away.wait(), 0.2)
        await c.finish()


@pytest.mark.parametrize(
    "outcome", ["response.failed", "response.incomplete", "orphan", "early_audio"]
)
async def test_failed_continuations_release_the_wait(with_run, outcome):
    async with _conversation(with_run) as c:
        await c.open_tools(complete=outcome != "orphan")
        if outcome == "orphan":
            await c.peer.backend("response.failed")
            c.release.set()
            await c.peer.receive(wire.ThinkingAppendEvent.model_fields["type"].default)
        else:
            await c.submit_results()
            await c.peer.backend(RESPONSE_CREATED)
            if outcome == "early_audio":
                await c.peer.speech(c.session, c.speeches, "Still checking.")
                c.assert_pending()
            await c.peer.backend(outcome if outcome != "early_audio" else "response.failed")
        await asyncio.gather(*c.speeches)
        await c.release_hook()
        await c.assert_idle()


@pytest.mark.parametrize("phase", ["before_continuation", "during_continuation"])
async def test_unrelated_audio_cannot_answer_unprocessed_results(with_run, phase):
    async with _conversation(with_run) as c:
        await c.open_tools()
        await c.submit_results()
        if phase == "during_continuation":
            await c.peer.backend(RESPONSE_CREATED)
        await c.peer.speech(c.session, c.speeches, "Still checking.")
        c.assert_pending()
        if phase == "before_continuation":
            await c.peer.backend(RESPONSE_CREATED)
        await c.peer.backend(RESPONSE_COMPLETED)
        await c.finish()


async def test_backend_completion_during_audio_hands_off_on_the_next_frame(with_run):
    async with _conversation(with_run) as c:
        await c.open_tools()
        await c.submit_results()
        await c.peer.backend(RESPONSE_CREATED)
        speaker = c.session.output.audio
        await c.peer.send(wire.TranscriptDeltaEvent(type=OUTPUT_TRANSCRIPT_DELTA, delta=ANSWER))
        speaker.captured.clear()
        await c.peer.audio()
        await asyncio.wait_for(speaker.captured.wait(), TIMEOUT)
        assert c.live.pending_tool_reply
        await c.peer.backend(RESPONSE_COMPLETED)
        speaker.captured.clear()
        await c.peer.audio()
        await asyncio.wait_for(speaker.captured.wait(), TIMEOUT)
        assert not c.live.pending_tool_reply
        await c.peer.audio(speech=False)
        await c.speeches[-1].wait_for_playout()
        await c.release_hook()
        await c.assert_idle()
        assert any(
            getattr(item, "text_content", None) == ANSWER for item in c.session.history.items
        )


async def test_audio_handoff_preserves_a_newer_unprocessed_batch(with_run):
    async with _conversation(with_run) as c:
        await c.ready_reply()
        await c.peer.backend(RESPONSE_CREATED, delegation_id="other")
        await c.peer.tool_call(delegation_id="other")
        await c.peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
        await c.peer.speech(c.session, c.speeches, "First result.")
        c.assert_pending()
        assert c.pending_calls == {"later"}
        await c.peer.backend(RESPONSE_COMPLETED, delegation_id="other")
        await c.peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
        await c.peer.continuation(delegation_id="other")
        await c.finish()


@pytest.mark.parametrize("status", ["response.failed", "response.incomplete"])
async def test_later_failure_preserves_a_completed_answer_in_the_same_delegation(with_run, status):
    async with _conversation(with_run) as c:
        await c.ready_reply()
        await c.peer.backend(RESPONSE_CREATED)
        await c.peer.tool_call()
        await c.peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
        await c.peer.backend(status)
        c.assert_pending()
        assert c.pending_calls == {"call-0"}
        await c.finish()


@pytest.mark.parametrize(
    "output", ["text_before_call", "text_after_call", "message_after_call", "empty_text"]
)
async def test_mixed_responses_hand_off_only_answered_results(with_run, output):
    async with _conversation(with_run) as c:
        await c.open_tools()
        await c.submit_results()
        await c.peer.backend(RESPONSE_CREATED)
        if output == "text_before_call":
            await c.peer.backend("response.output_text.delta", delta="First result.")
        await c.peer.tool_call()
        if output == "text_after_call" or output == "empty_text":
            await c.peer.backend(
                "response.output_text.delta",
                delta="  " if output == "empty_text" else "First result.",
            )
        elif output == "message_after_call":
            await c.peer.backend(
                RESPONSE_ITEM_DONE,
                item=wire.OutputItem.model_validate(
                    {
                        "id": "answer",
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "First result."}],
                    }
                ),
            )
        await c.peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
        await c.peer.backend(RESPONSE_COMPLETED)
        await c.peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
        await c.peer.speech(
            c.session, c.speeches, "Checking next." if output == "empty_text" else "First result."
        )
        assert c.pending_calls == ({"call-0", "later"} if output == "empty_text" else {"later"})
        await c.peer.continuation()
        await c.finish()


@pytest.mark.parametrize("fallback", ["managed", "legacy", "regenerate"])
async def test_fallback_abandons_old_waits_before_observing_new_work(with_run, fallback):
    async with _conversation(with_run, fallback=fallback) as c:
        await c.ready_reply()
        await c.release_hook()
        states = []
        c.session.on("agent_state_changed", lambda event: states.append(event.new_state))
        await c.realtime.restart(switch_model=True)
        assert c.activity._pending_auto_tool_reply_fut is None
        if fallback == "regenerate":
            assert "listening" not in states
            if c.run is not None:
                assert not c.run.done()
            await c.peer.receive(wire.CommentaryAppendEvent.model_fields["type"].default)
            await c.peer.speech(c.session, c.speeches, ANSWER, speech_handle=c.speeches[-1])
        await c.assert_idle()


async def test_session_close_abandons_the_provider_wait(with_run):
    async with _conversation(with_run) as c:
        await c.ready_reply()
        activity = c.activity
        c.release_enter.set()
        await asyncio.wait_for(c.session.aclose(), TIMEOUT)
        assert not c.live.pending_tool_reply
        assert activity._pending_auto_tool_reply_fut is None


@pytest.mark.parametrize("case", ["late_transcript", "no_reply"])
async def test_provider_wait_outlives_run_placeholder(case):
    async with _conversation(True, no_reply=case == "no_reply") as c:
        await c.ready_reply()
        waiters = [
            handle
            for handle in c.run._handles
            if isinstance(handle, asyncio.Task)
            and handle.get_coro().__name__ == "_wait_for_auto_tool_reply"
        ]
        assert len(waiters) == 1
        await asyncio.wait_for(asyncio.shield(waiters[0]), TIMEOUT)
        assert c.activity._pending_auto_tool_reply_fut is None
        await c.release_hook()
        await asyncio.wait_for(c.run, TIMEOUT)
        if case == "late_transcript":
            await c.peer.late_transcript(c.speeches)
        c.assert_pending()
        await c.finish()


async def test_fallback_keeps_busy_while_closing_the_old_provider(with_run):
    """A hook completing mid-swap must not observe a false idle interval."""
    async with _conversation(with_run, fallback="regenerate") as c:
        await c.ready_reply()
        c.peer.allow_close.clear()
        swap = asyncio.create_task(c.realtime.restart(switch_model=True))
        await asyncio.wait_for(c.peer.close_received.wait(), TIMEOUT)
        try:
            await c.release_hook()
            assert c.realtime.pending_tool_reply is True
            assert c.session.agent_state == "thinking"
            assert c.session._user_away_timer is None
        finally:
            c.peer.allow_close.set()
            await asyncio.wait_for(swap, TIMEOUT)
            await c.peer.receive(wire.CommentaryAppendEvent.model_fields["type"].default)
            await c.peer.speech(c.session, c.speeches, ANSWER, speech_handle=c.speeches[-1])
            await c.assert_idle()
