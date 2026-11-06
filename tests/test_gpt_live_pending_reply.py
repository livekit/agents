"""Agent state through the real GPT-Live SDK over a loopback WebSocket.

Adapted from zek\'s reproduction in PR #8379; no provider or SDK internals are patched.
"""

import asyncio
import base64
import json

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import Agent, AgentSession, function_tool, llm
from livekit.agents.llm._realtime import gpt_live_types as wire
from livekit.agents.llm._realtime.gpt_live import SAMPLE_RATE, GPTLiveModel
from livekit.agents.voice import io

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
            for pcm in (b"\x00\x10" * AUDIBLE_SAMPLES, b"\x00\x00" * SAMPLE_RATE):
                await self.send(wire.OutputAudioDeltaEvent(delta=base64.b64encode(pcm).decode()))
            await asyncio.wait_for(created.wait(), DEADLINE_SECONDS)
            handle = speech_handle if speech_handle is not None else speeches[index]
            await asyncio.wait_for(handle.wait_for_playout(), DEADLINE_SECONDS)
        finally:
            session.off("speech_created", observe)

    async def close(self):
        if self.ws is not None:
            await self.ws.close()
        await self.runner.cleanup()


@pytest.mark.parametrize(
    "case",
    [
        "control",
        "batch",
        "interrupt",
        "failed",
        "late_transcript",
        "orphan",
        "no_reply",
        "incomplete",
        "unrelated_failure",
        "tool_only",
        "close",
        "before_continuation",
        "during_continuation",
        "before_failed_continuation",
        "partial_delivery",
        "fallback_managed",
        "fallback_legacy",
        "fallback_regenerate",
        "streaming_completion",
        "later_response_failed",
        "later_response_incomplete",
        "mixed_text_before_call",
        "mixed_text_after_call",
        "mixed_message_after_call",
        "mixed_empty_text",
    ],
)
@pytest.mark.parametrize("with_run", [False, True])
async def test_gpt_live_pending_reply(case, with_run, expire_wait=False):
    """Only audio, failure, or abandonment releases an accepted tool continuation."""
    peer = ProviderPeer()
    url = await peer.start()
    started = asyncio.Event()
    release = asyncio.Event()
    speeches = []
    count = 2 if case == "batch" else 1
    running = 0
    release_enter = asyncio.Event()

    class ToolAgent(Agent):
        async def on_enter(self):
            if with_run:
                await release_enter.wait()

    @function_tool
    async def lookup():
        """Read a deterministic local fixture."""
        nonlocal running
        running += 1
        if running == count:
            started.set()
        await release.wait()
        return llm.ToolResult(ANSWER, reply_required=False) if case == "no_reply" else ANSWER

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as http:
        model = GPTLiveModel(
            model="gpt-live-1",
            delegation="responses",
            api_key="local-peer-not-a-key",
            base_url=url,
            http_session=http,
        )
        session_model = model
        if case.startswith("fallback_"):
            primary = llm.DuplexRealtimeAdapter(model)
            backup = (
                llm.DuplexRealtimeAdapter(
                    GPTLiveModel(
                        model="gpt-live-1",
                        delegation="responses",
                        api_key="local-peer-not-a-key",
                        base_url=url,
                        http_session=http,
                    )
                )
                if case in ("fallback_managed", "fallback_regenerate")
                else FakeRealtimeModel(capabilities=primary.capabilities)
            )
            session_model = llm.RealtimeModelFallbackAdapter(
                [primary, backup], regenerate_on_swap=case == "fallback_regenerate"
            )
        session = AgentSession(llm=session_model, user_away_timeout=0.1)
        user_states = []
        session.on("user_state_changed", lambda event: user_states.append(event.new_state))
        session.output.audio = MemorySpeaker()
        session.on("speech_created", lambda event: speeches.append(event.speech_handle))
        try:
            start_task = asyncio.create_task(
                session.start(
                    ToolAgent(instructions="Answer briefly.", llm=session_model, tools=[lookup]),
                    capture_run=with_run,
                )
            )
            await peer.receive(wire.SessionStartEvent.model_fields["type"].default)
            if not with_run:
                await start_task
            run = session._global_run_state if with_run else None
            activity = session._activity
            realtime = activity.realtime_llm_session
            live = (realtime._active if case.startswith("fallback_") else realtime).duplex_session
            live.on("openai_server_event_received", peer.processed.put_nowait)
            await peer.backend(RESPONSE_CREATED)
            for index in range(count):
                await peer.backend(
                    RESPONSE_ITEM_DONE,
                    item=wire.OutputItem(
                        id=f"item-{index}",
                        type="function_call",
                        status="completed",
                        call_id=f"call-{index}",
                        name="lookup",
                        arguments="{}",
                    ),
                )
            if case != "orphan":
                await peer.backend(RESPONSE_COMPLETED)
            await asyncio.wait_for(started.wait(), TIMEOUT)
            await peer.speech(session, speeches, "Checking. ", start_ms=0, end_ms=100)
            await asyncio.wait_for(started.wait(), TIMEOUT)
            assert session.agent_state == "thinking"
            await peer.user_turn(realtime)
            assert session.user_state == "listening"
            if case == "orphan":
                await peer.backend("response.failed")
            release.set()
            if case == "orphan":
                await peer.receive(wire.ThinkingAppendEvent.model_fields["type"].default)
            else:
                for _ in range(count):
                    await peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
                await peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
                if case == "before_continuation":
                    await peer.speech(session, speeches, "Still checking.")
                    assert live.pending_tool_reply
                    assert session.agent_state == "thinking"
                await peer.backend(RESPONSE_CREATED)
                if case in ("during_continuation", "before_failed_continuation"):
                    await peer.speech(session, speeches, "Still checking.")
                    assert live.pending_tool_reply
                    assert session.agent_state == "thinking"
                if case.startswith("mixed_"):
                    if case == "mixed_text_before_call":
                        await peer.backend("response.output_text.delta", delta="First result.")
                    await peer.backend(
                        RESPONSE_ITEM_DONE,
                        item=wire.OutputItem(
                            id="later",
                            type="function_call",
                            status="completed",
                            call_id="later",
                            name="lookup",
                            arguments="{}",
                        ),
                    )
                    if case == "mixed_text_after_call":
                        await peer.backend("response.output_text.delta", delta="First result.")
                    elif case == "mixed_message_after_call":
                        # Construct from the wire dict so this also runs on older schemas
                        # that silently discarded message content.
                        item = wire.OutputItem.model_validate(
                            {
                                "id": "answer",
                                "type": "message",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "First result."}],
                            }
                        )
                        await peer.backend(RESPONSE_ITEM_DONE, item=item)
                    elif case == "mixed_empty_text":
                        await peer.backend("response.output_text.delta", delta="  ")
                    await peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
                    await peer.backend(RESPONSE_COMPLETED)
                    await peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
                    await peer.speech(
                        session,
                        speeches,
                        "Checking next." if case == "mixed_empty_text" else "First result.",
                    )
                    expected = {"call-0", "later"} if case == "mixed_empty_text" else {"later"}
                    assert set(live._awaiting_tool_audio) == expected
                    await peer.backend(RESPONSE_CREATED)
                if case == "streaming_completion":
                    speaker = session.output.audio
                    speaker.captured.clear()
                    await peer.send(
                        wire.TranscriptDeltaEvent(type=OUTPUT_TRANSCRIPT_DELTA, delta=ANSWER)
                    )
                    await peer.send(
                        wire.OutputAudioDeltaEvent(
                            delta=base64.b64encode(b"\x00\x10" * AUDIBLE_SAMPLES).decode()
                        )
                    )
                    await asyncio.wait_for(speaker.captured.wait(), TIMEOUT)
                    assert live.pending_tool_reply
                    await peer.backend(RESPONSE_COMPLETED)
                    speaker.captured.clear()
                    await peer.send(
                        wire.OutputAudioDeltaEvent(
                            delta=base64.b64encode(b"\x00\x10" * AUDIBLE_SAMPLES).decode()
                        )
                    )
                    await asyncio.wait_for(speaker.captured.wait(), TIMEOUT)
                    assert not live.pending_tool_reply
                    await peer.send(
                        wire.OutputAudioDeltaEvent(
                            delta=base64.b64encode(b"\x00\x00" * SAMPLE_RATE).decode()
                        )
                    )
                    await asyncio.wait_for(speeches[-1].wait_for_playout(), TIMEOUT)
                    release_enter.set()
                    if activity._on_enter_task is not None:
                        await activity._on_enter_task
                    await asyncio.sleep(0)
                    assert session.agent_state == "listening"
                    assert any(
                        getattr(item, "text_content", None) == ANSWER
                        for item in session.history.items
                    )
                    if run is not None:
                        await asyncio.wait_for(run, TIMEOUT)
                    return
                await peer.backend(
                    "response.failed"
                    if case in ("failed", "before_failed_continuation")
                    else "response.incomplete"
                    if case == "incomplete"
                    else RESPONSE_COMPLETED
                )
            await asyncio.gather(*(asyncio.wait_for(speech, TIMEOUT) for speech in speeches))
            before = session.agent_state
            if case in ("failed", "incomplete", "orphan", "before_failed_continuation"):
                print(
                    json.dumps(
                        {
                            "case": case,
                            "state": before,
                            "pending": len(getattr(activity, "_pending_auto_tool_replies", ())),
                        }
                    ),
                    flush=True,
                )
                release_enter.set()
                if activity._on_enter_task is not None:
                    await activity._on_enter_task
                await asyncio.sleep(0)
                assert session.agent_state == "listening"
                assert not live.pending_tool_reply
                if run is not None:
                    await asyncio.wait_for(run, TIMEOUT)
                return
            assert live.pending_tool_reply
            if case == "close":
                release_enter.set()
                await asyncio.wait_for(session.aclose(), TIMEOUT)
                assert not live.pending_tool_reply
                assert activity._pending_auto_tool_reply_fut is None
                return
            if run is not None:
                assert activity._pending_auto_tool_reply_fut is not None
                if expire_wait:
                    await asyncio.sleep(5.1)
                    assert activity._pending_auto_tool_reply_fut is None
                    assert live.pending_tool_reply
                release_enter.set()
                await activity._on_enter_task
                if expire_wait:
                    await asyncio.wait_for(run, TIMEOUT)
                else:
                    assert not run.done()
            if case.startswith("fallback_"):
                states = []
                session.on("agent_state_changed", lambda event: states.append(event.new_state))
                await realtime.restart(switch_model=True)
                assert activity._pending_auto_tool_reply_fut is None
                if case == "fallback_regenerate":
                    assert "listening" not in states
                    if run is not None:
                        assert not run.done()
                    await peer.receive(wire.CommentaryAppendEvent.model_fields["type"].default)
                    await peer.speech(session, speeches, ANSWER, speech_handle=speeches[-1])
                assert session.agent_state == "listening"
                if run is not None:
                    await asyncio.wait_for(run, 1)
                return
            if case in ("later_response_failed", "later_response_incomplete"):
                await peer.backend(RESPONSE_CREATED)
                await peer.backend(
                    RESPONSE_ITEM_DONE,
                    item=wire.OutputItem(
                        id="later",
                        type="function_call",
                        status="completed",
                        call_id="later",
                        name="lookup",
                        arguments="{}",
                    ),
                )
                await peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
                await peer.backend(
                    "response.failed" if case == "later_response_failed" else "response.incomplete"
                )
                assert live.pending_tool_reply
                assert set(live._awaiting_tool_audio) == {"call-0"}
            if case == "partial_delivery":
                await peer.backend(RESPONSE_CREATED, delegation_id="other")
                await peer.backend(
                    RESPONSE_ITEM_DONE,
                    delegation_id="other",
                    item=wire.OutputItem(
                        id="later",
                        type="function_call",
                        status="completed",
                        call_id="later",
                        name="lookup",
                        arguments="{}",
                    ),
                )
                await peer.receive(wire.ResponseItemCreateEvent.model_fields["type"].default)
                await peer.speech(session, speeches, "First result.")
                assert live.pending_tool_reply
                assert session.agent_state == "thinking"
                assert set(live._awaiting_tool_audio) == {"later"}
                await peer.backend(RESPONSE_COMPLETED, delegation_id="other")
                await peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
                await peer.backend(RESPONSE_CREATED, delegation_id="other")
                await peer.backend(RESPONSE_COMPLETED, delegation_id="other")
            if case == "unrelated_failure":
                await peer.backend("response.failed", delegation_id="other")
                assert live.pending_tool_reply
            elif case == "tool_only":
                await peer.backend(RESPONSE_CREATED)
                await peer.backend(
                    RESPONSE_ITEM_DONE,
                    item=wire.OutputItem(
                        id="extra",
                        type="function_call",
                        status="completed",
                        call_id="extra",
                        name="lookup",
                        arguments="{}",
                    ),
                )
                await peer.backend(RESPONSE_COMPLETED)
                await peer.receive(wire.ResponseCreateEvent.model_fields["type"].default)
                assert live.pending_tool_reply
                await peer.backend(RESPONSE_CREATED)
                await peer.backend(RESPONSE_COMPLETED)
            if case == "interrupt":
                await asyncio.wait_for(session.interrupt(), TIMEOUT)
            elif case == "late_transcript":
                event = wire.TranscriptDeltaEvent(
                    type=OUTPUT_TRANSCRIPT_DELTA,
                    delta="Late old words.",
                    start_ms=50,
                    end_ms=100,
                )
                await peer.send(event)
                async with asyncio.timeout(TIMEOUT):
                    while (await peer.processed.get()).get("type") != event.type:
                        pass
                await asyncio.wait_for(speeches[-1], TIMEOUT)
            user_states.clear()
            await asyncio.sleep(0.2)
            assert "away" not in user_states
            between = session.agent_state
            pending_before_audio = len(getattr(activity, "_pending_auto_tool_replies", ()))
            await peer.speech(session, speeches, ANSWER)
            after = session.agent_state
            print(
                json.dumps(
                    {
                        "case": case,
                        "before": before,
                        "between": between,
                        "pending_before_audio": pending_before_audio,
                        "after": after,
                        "pending_after_audio": len(
                            getattr(activity, "_pending_auto_tool_replies", ())
                        ),
                        "answer_delivered": any(
                            getattr(item, "text_content", None) == ANSWER
                            for item in session.history.items
                        ),
                    }
                ),
                flush=True,
            )
            assert any(
                getattr(item, "text_content", None) == ANSWER for item in session.history.items
            )
            assert not live.pending_tool_reply
            assert before == "thinking"
            assert between == "thinking"
            assert after == "listening"
            await asyncio.sleep(0.2)
            assert "away" in user_states
            if run is not None:
                await asyncio.wait_for(run, TIMEOUT)
        finally:
            release.set()
            release_enter.set()
            await asyncio.wait_for(session.aclose(), TIMEOUT)
            await asyncio.wait_for(start_task, TIMEOUT)
            await model.aclose()
            await peer.close()


@pytest.mark.parametrize("case", ["late_transcript", "no_reply"])
async def test_provider_wait_outlives_run_placeholder(case):
    """A bounded RunResult wait expiring must not claim the audible reply was dropped."""
    await test_gpt_live_pending_reply(case, with_run=True, expire_wait=True)
