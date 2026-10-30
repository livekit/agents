"""Customer workflows through the real SDK, local protocol peer, and AgentSession."""

from __future__ import annotations

import asyncio
import base64

import pytest
from rimelabs_sdk import RimeTimeoutError, realtime as r

from livekit import rtc
from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool, llm
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
from livekit.plugins.rime.realtime import PrismHistoryUpdateError

from .fake_tts import FakeTTS
from .test_plugin_rime import messages, peer_model, wait_until

pytestmark = pytest.mark.unit


class ControlledOutput(AudioOutput):
    def __init__(self, *, sample_rate=None, synchronous=False):
        super().__init__(
            label="controlled",
            capabilities=AudioOutputCapabilities(pause=True),
            sample_rate=sample_rate,
        )
        self.frames = []
        self.segments = []
        self.duration = 0
        self.synchronous = synchronous
        self.paused = False

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        self.frames.append(frame)
        self.duration += frame.duration

    def flush(self):
        super().flush()
        if self.duration:
            self.segments.append(self.duration)
            self.duration = 0
            if self.synchronous:
                self.finish()

    def finish(self, *, interrupted=False):
        duration = self.segments.pop(0)
        self.on_playback_finished(playback_position=duration, interrupted=interrupted)

    def clear_buffer(self):
        if self.duration:
            self.flush()
        if self.segments:
            self.finish(interrupted=True)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class CountingTTS(FakeTTS):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        return super().stream(**kwargs)

    def synthesize(self, text, **kwargs):
        self.calls += 1
        return super().synthesize(text, **kwargs)


async def emit_message(peer, response, name, *, seconds=0.02):
    await peer.emit(
        "response.output_item.added",
        response_id=response,
        output_index=0,
        item={"id": name, "type": "message"},
    )
    fields = {"response_id": response, "output_index": 0, "item_id": name, "content_index": 0}
    await peer.emit("response.text.delta", **fields, delta=name)
    await peer.emit("response.text.done", **fields, text=name)
    await peer.emit(
        "response.audio.delta",
        **fields,
        delta=base64.b64encode(b"\0\0" * int(24000 * seconds)).decode(),
    )
    await peer.emit("response.audio.done", **fields)


@pytest.mark.parametrize("sink,tts", [(False, False), (True, False), (True, True)])
async def test_discard_keeps_text_without_audio_or_automatic_tts(sink, tts):
    async with peer_model(playback="discard") as (peer, model, _):
        speech_tts = CountingTTS()
        async with AgentSession(llm=model, tts=speech_tts if tts else None) as session:
            output = ControlledOutput(synchronous=True)
            if sink:
                session.output.audio = output
            await session.start(Agent(instructions="test"))
            speech = session.generate_reply(user_input="hello")
            response = await peer.start(await peer.next())
            await peer.message(response, text="Visible answer", seconds=3)
            assert peer.receipts.empty()
            await peer.end(response)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response
            assert receipt["played_ms"] == 0
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert not output.frames
            assert speech_tts.calls == 0
            assert any(
                getattr(i, "text_content", None) == "Visible answer" for i in session.history.items
            )
            await peer.end(response)
            await asyncio.sleep(0.01)
            assert peer.receipts.empty()


@pytest.mark.parametrize("sample_rate", [None, 48000])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_measured_playback_through_real_pipeline(sample_rate, synchronous):
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            sink = ControlledOutput(sample_rate=sample_rate, synchronous=synchronous)
            wrapper = model.wrap_audio_output(sink)
            forwarded = []
            wrapper.on("playback_finished", forwarded.append)
            session.output.audio = wrapper
            assert wrapper.sample_rate is None
            wrapper.pause()
            assert sink.paused
            wrapper.resume()
            await session.start(Agent(instructions="test"))
            speech = session.generate_reply()
            response = await peer.start(await peer.next())
            await emit_message(peer, response, "first")
            await emit_message(peer, response, "second")
            await peer.end(response)
            if not synchronous:
                await wait_until(lambda: bool(sink.segments))
                assert peer.receipts.empty()
                sink.finish()
                await wait_until(lambda: bool(sink.segments))
                assert peer.receipts.empty()
                sink.finish()
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response
            assert receipt["played_ms"] == pytest.approx(40)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert len(forwarded) == 2
            if sample_rate:
                assert {frame.sample_rate for frame in sink.frames} == {sample_rate}
            assert peer.receipts.empty()


@pytest.mark.parametrize("mode", ["estimated", "auto"])
async def test_unwrapped_output_keeps_estimate(mode):
    async with peer_model(playback=mode) as (peer, model, _):
        async with AgentSession(llm=model) as session:
            session.output.audio = ControlledOutput(synchronous=True)
            await session.start(Agent(instructions="test"))
            speech = session.generate_reply()
            response = await peer.start(await peer.next())
            await peer.message(response)
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert peer.receipts.empty()


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
async def test_exact_edit_failure_repairs_local_projection_and_keeps_connection(failure_at):
    notices = []
    async with peer_model(on_history_update=notices.append) as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            await model.replace_chat_ctx(agent, messages("old-a", "old-b"))
            peer.fail_history_at = len(peer.history_operations) + failure_at
            with pytest.raises(PrismHistoryUpdateError) as caught:
                await model.replace_chat_ctx(agent, messages("new-a", "new-b"))
            error = caught.value
            assert error.status == ("rejected" if failure_at == 1 else "partial")
            assert error.recoverable and not error.needs_reconciliation
            assert [
                i.text_content for i in agent.chat_ctx.items if isinstance(i, llm.ChatMessage)
            ] == [i["text"] for i in peer.server_items]
            assert len(notices) == 1 and notices[0].needs_reconciliation
            assert "old-a" not in repr(notices[0])
            assert "old-a" not in str(error)
            assert peer.connections == 1
            speech = session.generate_reply()
            response = await peer.start(await peer.next())
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            await model.clear_chat_ctx(agent)
            assert not peer.server_items


async def test_ordinary_partial_blocks_reply_until_public_reconciliation():
    notices = []
    async with peer_model(on_history_update=notices.append) as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            await model.replace_chat_ctx(agent, messages("old"))
            peer.fail_history_at = len(peer.history_operations) + 2
            with pytest.raises(PrismHistoryUpdateError) as caught:
                await agent.update_chat_ctx(messages("new"))
            assert caught.value.status == "partial"
            assert caught.value.needs_reconciliation
            with pytest.raises(PrismHistoryUpdateError):
                await agent.update_chat_ctx(messages("another stale edit"))
            assert peer.server_items == []
            await model.reconcile_chat_ctx(agent)
            assert not agent.chat_ctx.copy(exclude_config_update=True, exclude_handoff=True).items
            await model.replace_chat_ctx(agent, messages("recovered"))
            assert peer.server_items[0]["text"] == "recovered"


async def test_shared_views_preserve_omissions_and_explicit_clear_is_exact():
    async with peer_model(context_mode="shared") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            original = messages("one", "two")
            await agent.update_chat_ctx(original)
            count = len(peer.history_operations)
            await agent.update_chat_ctx(llm.ChatContext.empty())
            await agent.update_chat_ctx(llm.ChatContext([original.items[-1]]))
            assert len(peer.history_operations) == count
            assert [i["text"] for i in peer.server_items] == ["one", "two"]
            target = original.copy()
            target.items[0].content = ["one!"]
            with pytest.raises(PrismHistoryUpdateError):
                await agent.update_chat_ctx(target)
            await model.reconcile_chat_ctx(agent)
            await model.replace_chat_ctx(agent, target)
            assert [i["text"] for i in peer.server_items] == ["one!", "two"]
            await model.clear_chat_ctx(agent)
            assert not peer.server_items


@pytest.mark.parametrize("inside_task", [False, True])
async def test_task_inside_live_tool_preserves_original_result(inside_task, caplog):
    ran = []

    class Task(AgentTask[str]):
        def __init__(self):
            super().__init__(instructions="Task", chat_ctx=llm.ChatContext.empty())

        async def on_enter(self):
            if inside_task:
                result = await Child()
                self.complete(result)
            else:
                self.complete("finished")

    class Child(AgentTask[str]):
        def __init__(self):
            super().__init__(instructions="Child", chat_ctx=llm.ChatContext.empty())

        async def on_enter(self):
            self.complete("finished")

    @function_tool
    async def task_tool(context: RunContext) -> str:
        """Run the task."""
        ran.append("called")
        return await Task()

    async with peer_model(context_mode="shared", playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Parent", tools=[task_tool]))
            speech = session.generate_reply()
            parent = await peer.start(await peer.next())
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="parent-item",
                call_id="parent-call",
                name="task_tool",
                arguments="{}",
            )
            await peer.end(parent)
            continuation = await peer.next()
            assert continuation["response"]["metadata"]["prism_parent_response_id"] == parent
            response = await peer.start(continuation)
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert ran == ["called"]
            assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
            assert "active tool call" not in caplog.text
            assert peer.connections == 1


@pytest.mark.parametrize("operation", ["replace", "clear", "reconcile"])
async def test_inactive_agent_cannot_claim_helper_success(operation):
    async with peer_model() as (_, model, _):
        agent = Agent(instructions="inactive", llm=model)
        with pytest.raises(PrismHistoryUpdateError, match="active"):
            if operation == "replace":
                await model.replace_chat_ctx(agent, messages("new"))
            elif operation == "clear":
                await model.clear_chat_ctx(agent)
            else:
                await model.reconcile_chat_ctx(agent)


@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_failing_task_inside_tool_submits_one_result(failure, caplog):
    class Task(AgentTask[str]):
        def __init__(self):
            super().__init__(instructions="Task")

        async def on_enter(self):
            if failure == "cancel":
                self.cancel()
            else:
                self.complete(ValueError("task failed"))

    @function_tool
    async def task_tool() -> str:
        """Run a task that cannot finish."""
        return await Task()

    async with peer_model(context_mode="shared", playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Parent", tools=[task_tool]))
            speech = session.generate_reply()
            parent = await peer.start(await peer.next())
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="parent-item",
                call_id="parent-call",
                name="task_tool",
                arguments="{}",
            )
            await peer.end(parent)
            request = await peer.next()
            response = await peer.start(request)
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
            assert "active tool call" not in caplog.text


async def test_long_task_receives_two_turns_and_child_tool_before_parent_result(caplog):
    ran = []
    entered = asyncio.Event()

    class Task(AgentTask[str]):
        def __init__(self):
            super().__init__(instructions="Task", chat_ctx=llm.ChatContext.empty())

        async def on_enter(self):
            entered.set()

        @function_tool
        async def child(self) -> str:
            """Read a detail."""
            ran.append("child")
            return "detail"

        @function_tool
        async def finish(self) -> None:
            """Finish the task."""
            self.complete("finished")

    @function_tool
    async def parent_tool() -> str:
        """Run the task."""
        ran.append("parent")
        return await Task()

    async with peer_model(context_mode="shared", playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            await session.start(Agent(instructions="Parent", tools=[parent_tool]))
            speech = session.generate_reply()
            parent = await peer.start(await peer.next())
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=parent,
                item_id="parent-item",
                call_id="parent-call",
                name="parent_tool",
                arguments="{}",
            )
            await peer.end(parent)
            await asyncio.wait_for(entered.wait(), 3)
            for number, tool in enumerate(("child", "finish")):
                item = f"user-{number}"
                await peer.emit("input_audio_buffer.speech_started", item_id=item, audio_start_ms=0)
                await peer.emit("input_audio_buffer.speech_stopped", item_id=item, audio_end_ms=20)
                await peer.emit("input_audio_buffer.committed", item_id=item)
                await peer.emit(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item,
                    transcript=f"turn {number}",
                )
                response = await peer.start()
                await peer.emit(
                    "response.function_call_arguments.done",
                    response_id=response,
                    item_id=f"{tool}-item",
                    call_id=f"{tool}-call",
                    name=tool,
                    arguments="{}",
                )
                await peer.end(response)
                if tool == "child":
                    request = await peer.next()
                    reply = await peer.start(request)
                    await peer.message(reply)
                    await peer.end(reply)
                    await wait_until(
                        lambda: any(
                            getattr(i, "text_content", None) == "Hello"
                            for i in session.history.items
                        )
                    )
            await wait_until(lambda: any(i.get("call_id") == "parent-call" for i in peer.items))
            request = await peer.next()
            assert request["response"]["prism_call_id"] == "parent-call"
            reply = await peer.start(request)
            await peer.end(reply)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            assert ran == ["parent", "child"]
            assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1
            assert "active tool call" not in caplog.text
            assert "does not support resuming function calls" in caplog.text


async def test_observed_interruption_variant_appends_without_rebuild():
    async with peer_model(context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        response = await peer.start()
        item_id = await peer.message(response, text="Full answer.")
        await peer.end(response)
        await wait_until(lambda: provider.chat_ctx.get_by_id(item_id) is not None)
        provider.truncate(
            message_id=item_id,
            modalities=["text", "audio"],
            audio_end_ms=10,
            audio_transcript="Full",
        )
        await asyncio.wait_for(peer.truncations.get(), 3)
        await peer.emit(
            "conversation.item.truncated",
            item_id=item_id,
            content_index=0,
            audio_end_ms=10,
            prism_kept_text="Full ",
        )
        await wait_until(lambda: provider.chat_ctx.get_by_id(item_id).text_content == "Full ")
        observed = provider.chat_ctx
        observed.get_by_id(item_id).content = ["Full"]
        observed.add_message(role="user", content="Next")
        before = len(peer.history_operations)
        await provider.update_chat_ctx(observed)
        assert [op["type"] for op in peer.history_operations[before:]] == [
            "conversation.item.create"
        ]
        assert provider.chat_ctx.get_by_id(item_id).text_content == "Full "


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unconsumed", "unrelated", "detach"])
async def test_unproven_playback_never_reports_finished(mutation):
    async with peer_model(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (peer, model, _):
        provider = model.session()
        sink = ControlledOutput(synchronous=True)
        wrapper = model.wrap_audio_output(sink)
        await provider.update_instructions("test")
        pending = provider.generate_reply()
        response = await peer.start(await peer.next())
        generation = await pending
        await peer.message(response)
        await peer.end(response)
        message = await anext(generation.message_stream)
        frames = [frame async for frame in message.audio_stream]
        if mutation == "missing":
            for frame in frames:
                frame.userdata.clear()
        if mutation == "duplicate":
            frames *= 2
        if mutation == "unrelated":
            frames = [rtc.AudioFrame(b"\0\0" * 480, 24000, 1, 480)]
        if mutation != "unconsumed":
            for frame in frames:
                await wrapper.capture_frame(frame)
            if mutation == "detach":
                wrapper.on_detached()
            wrapper.flush()
            await wrapper.wait_for_playout()
        await asyncio.sleep(0.08)
        assert peer.receipts.empty()
        assert peer.connections == 1


async def test_cancelled_explicit_edit_settles_before_cancellation():
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            peer.history_release.clear()
            edit = asyncio.create_task(model.replace_chat_ctx(agent, messages("accepted")))
            await asyncio.wait_for(peer.history_started.wait(), 3)
            edit.cancel()
            await asyncio.sleep(0)
            assert not edit.done()
            peer.history_release.set()
            with pytest.raises(asyncio.CancelledError):
                await edit
            assert peer.server_items[0]["text"] == "accepted"
            await model.reconcile_chat_ctx(agent)
            assert agent.chat_ctx.items[0].text_content == "accepted"


async def test_reconciliation_rejects_revision_race():
    class DelayedAgent(Agent):
        delayed = False
        entered = asyncio.Event()
        release = asyncio.Event()

        async def update_chat_ctx(self, chat_ctx, **kwargs):
            if self.delayed:
                self.entered.set()
                await self.release.wait()
            await super().update_chat_ctx(chat_ctx, **kwargs)

    async with peer_model(context_mode="shared") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = DelayedAgent(instructions="test")
            await session.start(agent)
            await model.replace_chat_ctx(agent, messages("one"))
            agent.delayed = True
            edit = asyncio.create_task(model.reconcile_chat_ctx(agent))
            await asyncio.wait_for(agent.entered.wait(), 3)
            await peer.emit(
                "conversation.item.input_audio_transcription.completed",
                item_id="new-input",
                transcript="new",
            )
            await wait_until(
                lambda: any(
                    getattr(i, "text_content", None) == "new" for i in session.history.items
                )
            )
            agent.release.set()
            with pytest.raises(PrismHistoryUpdateError, match="changed"):
                await edit
            agent.delayed = False
            await model.reconcile_chat_ctx(agent)
            assert any(getattr(i, "text_content", None) == "new" for i in agent.chat_ctx.items)


async def test_pending_parent_result_survives_installed_sdk_reference_lifetime():
    # alpha26 supports only its existing cache window; the companion SDK must pass 140.
    count = 140 if hasattr(r.RealtimeSession, "hold_response") else 64
    async with peer_model(playback="discard", context_mode="shared") as (peer, model, _):
        provider = model.session()
        await provider.update_instructions("test")
        parent = await peer.start()
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=parent,
            item_id="parent",
            call_id="parent-call",
            name="lookup",
            arguments="{}",
        )
        await peer.end(parent)
        await asyncio.wait_for(peer.receipts.get(), 3)
        for _ in range(count):
            response = await peer.start()
            await peer.end(response)
            await asyncio.wait_for(peer.receipts.get(), 3)
        target = llm.ChatContext(
            [llm.FunctionCallOutput(call_id="parent-call", output="done", is_error=False)]
        )
        await provider.update_chat_ctx(target)
        reply = provider.generate_reply()
        request = await peer.next()
        assert request["response"]["prism_call_id"] == "parent-call"
        response = await peer.start(request)
        await peer.end(response)
        await reply
        assert len([i for i in peer.items if i.get("call_id") == "parent-call"]) == 1


async def test_missing_acknowledgment_is_terminal_without_replay():
    async with peer_model(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            peer.history_release.clear()
            try:
                with pytest.raises(PrismHistoryUpdateError) as caught:
                    await model.replace_chat_ctx(agent, messages("unknown outcome"))
                assert caught.value.status == "uncertain"
                assert not caught.value.recoverable
                assert caught.value.confirmed_chat_ctx is None
                assert len(peer.history_operations) == 1
                with pytest.raises((llm.RealtimeError, RimeTimeoutError)):
                    await model.reconcile_chat_ctx(agent)
            finally:
                peer.history_release.set()


async def test_headless_empty_response_and_wrong_model_helper():
    async with peer_model(playback="discard") as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            speech = session.generate_reply()
            response = await peer.start(await peer.next())
            await peer.end(response)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["played_ms"] == 0
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            async with peer_model() as (_, other, _):
                with pytest.raises(PrismHistoryUpdateError, match="directly"):
                    await other.clear_chat_ctx(agent)


async def test_roomio_wrapper_installs_before_opening_reply():
    """Real RoomIO and RTC audio source; only room publication/subscription is local."""
    from types import SimpleNamespace

    from livekit.agents.voice.room_io import RoomIO, RoomOptions

    class Room(rtc.EventEmitter):
        name = "local-validation"

        def __init__(self):
            super().__init__()
            self.local_participant = SimpleNamespace(
                identity="agent",
                publish_track=self.publish_track,
                set_attributes=self.set_attributes,
            )
            self.remote_participants = {
                "user": SimpleNamespace(
                    identity="user",
                    sid="PA_user",
                    name="user",
                    metadata="",
                    attributes={},
                    kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
                )
            }

        def isconnected(self):
            return True

        async def publish_track(self, track, options):
            return SimpleNamespace(sid="TR_audio", wait_for_subscription=self.subscribed)

        async def subscribed(self):
            pass

        async def set_attributes(self, attributes):
            pass

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            room_io = RoomIO(
                session,
                Room(),
                options=RoomOptions(audio_input=False, text_input=False, text_output=False),
            )
            await room_io.start()
            try:

                class OpeningAgent(Agent):
                    async def on_enter(self):
                        self.session.output.audio = model.wrap_audio_output(
                            self.session.output.audio
                        )
                        self.session.generate_reply(instructions="Say hello")

                await session.start(OpeningAgent(instructions="test"))
                response = await peer.start(await peer.next())
                await peer.message(response, seconds=0.02)
                await peer.end(response)
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["played_ms"] == pytest.approx(20)
            finally:
                await room_io.aclose()


@pytest.mark.parametrize("acknowledgment", [1, 2, 3, 4])
async def test_new_reply_at_each_rebuild_acknowledgment_is_nonfatal(acknowledgment):
    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            await model.replace_chat_ctx(agent, messages("old-a", "old-b"))
            before = len(peer.history_operations)
            peer.block_history_at = before + acknowledgment
            peer.history_release.clear()
            edit = asyncio.create_task(model.replace_chat_ctx(agent, messages("new-a", "new-b")))
            try:
                await wait_until(lambda: len(peer.history_operations) == peer.block_history_at)
                response = await peer.start()
            finally:
                peer.history_release.set()
            with pytest.raises(PrismHistoryUpdateError) as caught:
                await edit
            assert caught.value.status == "partial" and caught.value.recoverable
            assert not caught.value.needs_reconciliation
            assert len(peer.history_operations) == before + acknowledgment
            assert [
                i.text_content for i in agent.chat_ctx.items if isinstance(i, llm.ChatMessage)
            ] == [i["text"] for i in peer.server_items]
            await peer.end(response)
            assert peer.connections == 1


async def test_refused_typed_input_cannot_start_reply_and_callback_cannot_break_recovery(caplog):
    observed = []

    def notice(value):
        observed.append(value)
        raise RuntimeError("callback-transcript-marker")

    async with peer_model(on_history_update=notice) as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = Agent(instructions="test")
            await session.start(agent)
            peer.fail_history_at = 1
            session.generate_reply(user_input="refused input")
            await wait_until(lambda: bool(observed))
            await asyncio.sleep(0.02)
            assert peer.requests.empty() and not peer.server_items
            assert observed[0].status == "rejected"
            assert "callback-transcript-marker" not in caplog.text
            await model.reconcile_chat_ctx(agent)
            assert not agent.chat_ctx.copy(exclude_config_update=True, exclude_handoff=True).items


async def test_late_playback_cannot_release_fresh_provider_session():
    async with peer_model() as (peer, model, _):
        sink = ControlledOutput()
        wrapper = model.wrap_audio_output(sink)
        provider = model.session()
        await provider.update_instructions("test")
        pending = provider.generate_reply()
        response = await peer.start(await peer.next())
        generation = await pending
        await peer.message(response)
        await peer.end(response)
        message = await anext(generation.message_stream)
        async for frame in message.audio_stream:
            await wrapper.capture_frame(frame)
        wrapper.flush()
        await provider.aclose()
        fresh = model.session()
        await fresh.update_instructions("fresh")
        sink.finish()
        await asyncio.sleep(0.01)
        assert peer.receipts.empty()
        assert peer.connections == 2


async def test_detached_task_cannot_consume_explicit_edit_token():
    class DetachedAgent(Agent):
        async def update_chat_ctx(self, chat_ctx, **kwargs):
            await asyncio.create_task(super().update_chat_ctx(chat_ctx, **kwargs))

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            agent = DetachedAgent(instructions="test")
            await session.start(agent)
            with pytest.raises(PrismHistoryUpdateError, match="mismatched"):
                await model.replace_chat_ctx(agent, messages("not accepted"))
            assert not peer.history_operations


@pytest.mark.parametrize("outcome", ["accepted", "refused", "active-omission"])
async def test_acknowledged_append_merges_concurrent_speech_in_server_order(outcome):
    async with peer_model() as (peer, model, _):
        provider = model.session()
        await provider.update_chat_ctx(messages("old"))
        target = provider.chat_ctx
        target.add_message(role="user", content="A")
        target.add_message(role="user", content="B")
        peer.history_started.clear()
        peer.history_release.clear()
        if outcome == "refused":
            peer.fail_history_at = len(peer.history_operations) + 2
        update = asyncio.create_task(provider.update_chat_ctx(target))
        try:
            await asyncio.wait_for(peer.history_started.wait(), 3)
            peer.server_items.append(
                {"id": "speech", "type": "message", "role": "user", "text": "incoming"}
            )
            await peer.emit(
                "conversation.item.input_audio_transcription.completed",
                item_id="speech",
                transcript="incoming",
            )
            if outcome == "active-omission":
                await peer.emit(
                    "response.created",
                    response={"id": "active", "prism_input_item_ids": ["speech"]},
                )
            await wait_until(lambda: provider.chat_ctx.get_by_id("speech") is not None)
        finally:
            peer.history_release.set()
        if outcome == "accepted":
            await update
        else:
            with pytest.raises(PrismHistoryUpdateError) as caught:
                await update
            assert caught.value.status == ("partial" if outcome == "refused" else "uncertain")
            if outcome == "active-omission":
                assert caught.value.confirmed_chat_ctx is None
                assert not caught.value.recoverable
                assert peer.history_reads == 1
                return
        assert peer.history_reads == 1
        assert [i.text_content for i in provider.chat_ctx.items] == [
            i["text"] for i in peer.server_items
        ]
        assert [i.text_content for i in provider.chat_ctx.items] == ["old", "incoming", "A"] + (
            ["B"] if outcome == "accepted" else []
        )
