from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents import AMD, NOT_GIVEN, Agent, AgentSession, LanguageCode, llm, stt, utils, vad
from livekit.agents.voice.amd import AMDCategory, AMDLifecycle
from livekit.agents.voice.amd.detector import (
    _DEFAULT_HUMAN_INSTRUCTIONS,
    _DEFAULT_IVR_INSTRUCTIONS,
    _DEFAULT_SCREENING_INSTRUCTIONS,
    _DEFAULT_VOICEMAIL_INSTRUCTIONS,
)
from livekit.agents.voice.speech_handle import SpeechHandle

from .amd_test_utils import detector_clock  # noqa: F401
from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, _audio_frame, fake_capabilities
from .fake_stt import DrainingSTT, FakeSTT
from .fake_vad import FakeVAD
from .test_amd_detector import (
    ClassifierLLM,
    CustomerAgent,
    end_of_turn,
    eventually,
    speech_ended,
    speech_started,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@asynccontextmanager
async def running(
    *,
    machine_silence_threshold: float = 0,
    agent: Agent | None = None,
    session_stt: bool = True,
    amd_stt: stt.STT | None = None,
    user_transcription: bool = True,
    turn_detection: Literal["manual", "vad"] = "manual",
) -> AsyncIterator[tuple[AMD, AgentSession, ClassifierLLM, FakeRealtimeSession]]:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            can_disable_turn_detection=True,
            auto_tool_reply_generation=False,
            user_transcription=user_transcription,
        )
    )
    async with AgentSession(
        llm=model,
        stt=FakeSTT() if session_stt else None,
        vad=FakeVAD() if turn_detection == "vad" else None,
        turn_handling={"turn_detection": turn_detection, "interruption": {"mode": "vad"}},
        aec_warmup_duration=None,
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(agent or Agent(instructions="Call about an appointment."))
        classifier = ClassifierLLM()
        async with AMD(
            session,
            llm=classifier,
            stt=amd_stt,
            machine_silence_threshold=machine_silence_threshold,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            yield detector, session, classifier, model.active_session


async def reply(
    session: AgentSession,
    classifier: ClassifierLLM,
    rt: FakeRealtimeSession,
    category: AMDCategory,
) -> SpeechHandle:
    handle = asyncio.get_running_loop().create_future()
    session.once("speech_created", lambda ev: handle.set_result(ev.speech_handle))
    count = rt.generate_reply_calls
    assert session._activity.on_end_of_turn(end_of_turn())
    request = await classifier.request()
    classifier.prediction(request.current_turn.extra["turn_id"], category)
    await eventually(lambda: rt.generate_reply_calls == count + 1)
    return await handle


def respond(
    rt: FakeRealtimeSession, *, tool: llm.FunctionCall | None = None, duration: float = 0.01
) -> None:
    messages = utils.aio.Chan[llm.MessageGeneration]()
    functions = utils.aio.Chan[llm.FunctionCall]()
    if tool is not None:
        functions.send_nowait(tool)
    else:
        text = utils.aio.Chan[str]()
        audio = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])
        text.send_nowait("Please call me back.")
        text.close()
        audio.send_nowait(_audio_frame(duration))
        audio.close()
        messages.send_nowait(
            llm.MessageGeneration(
                message_id=utils.shortuuid("message_"),
                text_stream=text,
                audio_stream=audio,
                modalities=modalities,
            )
        )
    messages.close()
    functions.close()
    rt._reply_futs[-1].set_result(
        llm.GenerationCreatedEvent(
            message_stream=messages, function_stream=functions, user_initiated=True
        )
    )


@pytest.mark.parametrize(
    ("configuration", "error"),
    [
        ("server_turn_detection", "client-side turn detection"),
        ("auto_tool_reply", "client-controlled tool replies"),
        ("session_tools_only", "per-response tools"),
        ("missing_stt", "session STT"),
        ("missing_classifier", "LLM for classification"),
    ],
)
async def test_unsupported_configuration_does_not_install_amd(
    configuration: str, error: str
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            can_disable_turn_detection=configuration != "server_turn_detection",
            auto_tool_reply_generation=configuration == "auto_tool_reply",
            per_response_tool_choice=configuration != "session_tools_only",
        )
    )
    async with AgentSession(
        llm=model,
        stt=None if configuration == "missing_stt" else FakeSTT(),
        vad=None,
        turn_handling={"turn_detection": "manual"},
    ) as session:
        await session.start(Agent(instructions="Call about an appointment."))
        detector = AMD(
            session,
            llm=None if configuration == "missing_classifier" else ClassifierLLM(),
            stt=None,
        )
        with pytest.raises(ValueError, match=error):
            await detector.__aenter__()
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
        assert detector.lifecycle is AMDLifecycle.INITIALIZED


async def test_reply_waits_for_classification_and_silence() -> None:
    async with running(machine_silence_threshold=1.5) as (detector, session, classifier, rt):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        speech_started(detector)
        speech_ended(detector, 0.5)
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        await asyncio.sleep(0.1)
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.sleep(0.5)
        assert rt.generate_reply_calls == 0
        await eventually(lambda: rt.generate_reply_calls == 1)
        assert rt.committed
        assert rt.reply_instructions == [_DEFAULT_SCREENING_INSTRUCTIONS]
        respond(rt)
        await asyncio.wait_for(handles[0], 2)


async def test_untranscribed_speech_does_not_revoke_reply_authorization() -> None:
    async with running(machine_silence_threshold=0.5, turn_detection="vad") as (
        detector,
        session,
        classifier,
        rt,
    ):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        recognition = activity._audio_recognition
        output = session.output.audio
        await asyncio.wait_for(recognition._vad_atask, 2)
        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0,
                speech_duration=0,
                silence_duration=0,
            )
        )
        respond(rt)
        await asyncio.sleep(0.1)
        assert not handle.done()
        assert output.captured_playout_segments == 0

        await recognition._on_vad_event(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=0,
                speech_duration=0.1,
                silence_duration=0,
            )
        )
        await asyncio.wait_for(handle, 0.2)
        assert not handle.interrupted
        assert output.captured_playout_segments == 1
        assert detector.lifecycle is AMDLifecycle.ACTIVE
        assert detector._turn_id == 1
        assert classifier.requests.empty()

        assert activity.on_end_of_turn(end_of_turn("Please hold."))
        await classifier.request()
        assert not activity._authorization_allowed.is_set()
        assert rt.generate_reply_calls == 1
        classifier.prediction(2, AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        assert rt.generate_reply_calls == 1
        assert output.captured_playout_segments == 1


async def test_amd_interrupts_a_pending_reply_despite_disabled_interruptions() -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    async with running(agent=agent) as (_, session, classifier, rt):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        assert activity.on_end_of_turn(end_of_turn("Please hold."))
        await classifier.request()
        classifier.prediction(2, AMDCategory.WAIT)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        await asyncio.wait_for(handle, 0.1)
        assert handle.interrupted
        assert rt.generate_reply_calls == 1
        assert session.output.audio.captured_playout_segments == 0


async def test_amd_preserves_false_interruption_resume() -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    async with running(agent=agent, turn_detection="vad") as (detector, session, classifier, rt):
        output = FakeAudioOutput(can_pause=True)
        session.output.audio = output
        session.options.interruption["min_words"] = 0
        session.options.interruption["false_interruption_timeout"] = 0.1
        resumed = []
        session.on("agent_false_interruption", lambda event: resumed.append(event.resumed))
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_SCREENING)
        activity = session._activity
        await activity._audio_recognition._vad_atask
        respond(rt, duration=1)
        await eventually(lambda: session.agent_state == "speaking")
        activity.on_start_of_speech(None, time.time())
        activity._interrupt_by_audio_activity()
        assert output._paused_at is not None
        assert handle.allow_interruptions
        activity.on_end_of_speech(None, speech_end_time=time.time())
        await asyncio.wait_for(handle, 2)
        assert resumed == [True]
        assert not handle.interrupted
        assert output.captured_playout_segments == 1
        assert detector._turn_id == 1


async def test_stage_instructions_expire_after_first_human_reply() -> None:
    async with running() as (_, session, classifier, rt):
        for category, instructions in (
            (AMDCategory.MACHINE_SCREENING, _DEFAULT_SCREENING_INSTRUCTIONS),
            (AMDCategory.HUMAN, _DEFAULT_HUMAN_INSTRUCTIONS),
        ):
            handle = await reply(session, classifier, rt, category)
            assert rt.reply_instructions[-1] == instructions
            respond(rt)
            await asyncio.wait_for(handle, 2)

        assert session.amd is None
        assert session._turn_hooks is None
        handle = session.generate_reply()
        await eventually(lambda: rt.generate_reply_calls == 3)
        assert rt.reply_instructions[-1] is NOT_GIVEN
        respond(rt)
        await asyncio.wait_for(handle, 2)
        assert rt.updated_instructions == "Call about an appointment."
        for context in (session.history, session.current_agent.chat_ctx, rt.chat_ctx):
            assert not any(
                "Call state:" in (message.text_content or "") for message in context.messages()
            )


@pytest.mark.parametrize("category", [AMDCategory.WAIT, AMDCategory.MACHINE_UNAVAILABLE])
@pytest.mark.parametrize("transcription_before_verdict", [False, True])
async def test_suppressed_reply_keeps_only_the_realtime_transcript(
    category: AMDCategory, transcription_before_verdict: bool
) -> None:
    async with running() as (_, session, classifier, rt):
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        transcript = llm.InputTranscriptionCompleted(
            item_id="user-audio", transcript="hello", is_final=True
        )
        if transcription_before_verdict:
            rt.emit("input_audio_transcription_completed", transcript)
        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        assert rt.committed
        assert rt.generate_reply_calls == 0
        if not transcription_before_verdict:
            rt.emit("input_audio_transcription_completed", transcript)
        for context in (session.current_agent.chat_ctx, session.history):
            messages = [m for m in context.messages() if m.role == "user"]
            assert [(m.id, m.text_content) for m in messages] == [("user-audio", "hello")]


@pytest.mark.parametrize("source", ["session", "amd", "both"])
@pytest.mark.parametrize(
    "category", [AMDCategory.WAIT, AMDCategory.MACHINE_UNAVAILABLE, AMDCategory.HUMAN]
)
async def test_external_stt_preserves_history_without_native_transcription(
    source: str, category: AMDCategory
) -> None:
    amd_stt = DrainingSTT() if source != "session" else None
    async with running(session_stt=source != "amd", amd_stt=amd_stt, user_transcription=False) as (
        detector,
        session,
        classifier,
        rt,
    ):
        handles: list[SpeechHandle] = []
        added: list[llm.ChatMessage] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        session.on("conversation_item_added", lambda ev: added.append(ev.item))
        activity = session._activity
        activity.push_audio(_audio_frame(0.1))
        if source != "amd":
            await activity._audio_recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text="Session transcript.", language=LanguageCode("en"))
                    ],
                )
            )
        if amd_stt is not None and source == "amd":
            amd_stt.streams[0].send_fake_transcript("AMD transcript.")
            await eventually(
                lambda: (
                    detector._resources.stt._current.snapshot("amd").transcript == "AMD transcript."
                )
            )

        await session.commit_user_turn()
        request = await classifier.request()
        assert request.current_turn.text_content == (
            "AMD transcript." if source == "amd" else "Session transcript."
        )
        if source == "both":
            assert not detector._resources.stt.amd_stt_active
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, category)
        await asyncio.wait_for(activity._user_turn_completed_atask, 2)
        if category is AMDCategory.HUMAN:
            await eventually(lambda: rt.generate_reply_calls == 1)
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        else:
            assert rt.generate_reply_calls == 0

        expected = "AMD transcript." if source == "amd" else "Session transcript."
        for context in (session.current_agent.chat_ctx, session.history):
            assert [m.text_content for m in context.messages() if m.role == "user"] == [expected]
        assert [m.text_content for m in added if m.role == "user"] == [expected]
        assert not [m for m in rt.chat_ctx.messages() if m.role == "user"]
        assert rt.committed


@pytest.mark.parametrize("category", [AMDCategory.WAIT, AMDCategory.HUMAN])
async def test_session_stt_classifies_before_realtime_transcription(category: AMDCategory) -> None:
    async with running() as (_, session, classifier, rt):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        recognition = session._activity._audio_recognition
        await recognition._on_stt_event(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="Please hold.", language=LanguageCode("en"))],
            )
        )
        assert await session.commit_user_turn() == "Please hold."
        request = await classifier.request()
        assert request.current_turn.text_content == "Please hold."
        assert not session.current_agent.chat_ctx.messages()
        assert not handles
        classifier.prediction(1, category)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if category is AMDCategory.HUMAN:
            await eventually(lambda: rt.generate_reply_calls == 1)
            assert rt.reply_instructions[-1] is NOT_GIVEN
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        else:
            assert rt.generate_reply_calls == 0


@pytest.mark.parametrize("session_stt", [False, True])
@pytest.mark.parametrize("skip_reply", [False, True])
async def test_manual_realtime_commit_without_amd(session_stt: bool, skip_reply: bool) -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    async with AgentSession(
        llm=model,
        stt=FakeSTT() if session_stt else None,
        turn_handling={"turn_detection": "manual"},
    ) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Call about an appointment."))
        rt = model.active_session
        rt.commit_audio = Mock(wraps=rt.commit_audio)
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        session._activity.push_audio(_audio_frame(0.1))
        if session_stt:
            await session._activity._audio_recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="Hello.", language=LanguageCode("en"))],
                )
            )

        committed = session.commit_user_turn(skip_reply=skip_reply)
        rt.commit_audio.assert_called_once_with()
        assert await committed == ("Hello." if session_stt else "")
        recognition = session._activity._audio_recognition
        if recognition._end_of_turn_task is not None:
            await asyncio.wait_for(recognition._end_of_turn_task, 2)
        if session._activity._user_turn_completed_atask is not None:
            await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if not skip_reply:
            await eventually(lambda: rt.generate_reply_calls == 1)
            respond(rt)
            await asyncio.wait_for(handles[0], 2)
        assert rt.generate_reply_calls == (0 if skip_reply else 1)
        rt.commit_audio.assert_called_once_with()
    if not session_stt:
        assert not [m for m in session.history.messages() if m.role == "user"]


@pytest.mark.parametrize("session_stt", [False, True])
@pytest.mark.parametrize("turn_detection", ["manual", "vad"])
async def test_amd_stt_supplies_transcripts_at_eot(
    session_stt: bool, turn_detection: Literal["manual", "vad"]
) -> None:
    amd_stt = DrainingSTT()
    async with running(session_stt=session_stt, amd_stt=amd_stt, turn_detection=turn_detection) as (
        detector,
        session,
        classifier,
        rt,
    ):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        activity = session._activity
        recognition = activity._audio_recognition
        activity.push_audio(_audio_frame(0.1))
        stream = amd_stt.streams[0]
        if not session_stt:
            rt.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id="previous-audio", transcript="Late realtime transcript.", is_final=True
                ),
            )
        assert detector._resources.stt.amd_stt_active

        if turn_detection == "vad":
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=0,
                    timestamp=0,
                    speech_duration=0,
                    silence_duration=0,
                )
            )
        if session_stt:
            await recognition._on_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text="Session transcript.", language=LanguageCode("en"))
                    ],
                )
            )
        stream.send_fake_transcript("Please state your name.")
        await eventually(
            lambda: (
                detector._resources.stt._current.snapshot("amd").transcript
                == "Please state your name."
            )
        )
        assert rt.generate_reply_calls == 0
        assert classifier.requests.empty()

        if turn_detection == "manual":
            transcript = await session.commit_user_turn()
            assert transcript == ("Session transcript." if session_stt else "")
        else:
            await recognition._on_vad_event(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=0,
                    timestamp=0.5,
                    speech_duration=0.5,
                    silence_duration=0.5,
                )
            )
        request = await classifier.request()
        assert request.current_turn.text_content == "Please state your name."
        assert request.current_turn.extra["transcript_source"] == "amd"
        assert rt.generate_reply_calls == 0
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await eventually(lambda: rt.generate_reply_calls == 1)
        assert rt.reply_instructions[-1] == _DEFAULT_SCREENING_INSTRUCTIONS
        assert rt.pushed_audio
        assert rt.committed
        respond(rt)
        await asyncio.wait_for(handles[0], 2)


async def test_ivr_tool_executes_and_retains_instructions_for_its_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_dtmf = AsyncMock()
    monkeypatch.setattr(
        "livekit.agents.beta.tools.send_dtmf.get_job_context",
        lambda: SimpleNamespace(
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish_dtmf))
        ),
    )
    async with running() as (_, session, classifier, rt):
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_IVR)
        assert [tool.info.name for tool in rt.reply_tools[0]] == ["send_dtmf_events"]
        assert session.current_agent.tools == []
        respond(
            rt,
            tool=llm.FunctionCall(
                call_id="dtmf-1", name="send_dtmf_events", arguments='{"events": ["1"]}'
            ),
        )
        await eventually(lambda: rt.generate_reply_calls == 2)
        publish_dtmf.assert_awaited_once_with(code=1, digit="1")
        assert rt.reply_instructions == [_DEFAULT_IVR_INSTRUCTIONS, _DEFAULT_IVR_INSTRUCTIONS]
        assert rt.reply_tools[1] == rt.reply_tools[0]
        outputs = [i for i in rt.chat_ctx.items if i.type == "function_call_output"]
        assert len(outputs) == 1
        assert not outputs[0].is_error
        respond(rt)
        await asyncio.wait_for(handle, 2)

        human = await reply(session, classifier, rt, AMDCategory.HUMAN)
        assert rt.reply_tools[-1] is NOT_GIVEN
        respond(rt)
        await asyncio.wait_for(human, 2)


@pytest.mark.parametrize("interrupted", [False, True])
async def test_voicemail_is_retried_only_after_interrupted_playout(interrupted: bool) -> None:
    async with running() as (detector, session, classifier, rt):
        handles: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: handles.append(ev.speech_handle))
        handle = await reply(session, classifier, rt, AMDCategory.MACHINE_VM)
        assert rt.reply_instructions[-1] == _DEFAULT_VOICEMAIL_INSTRUCTIONS
        respond(rt, duration=10 if interrupted else 0.01)
        if interrupted:
            await eventually(lambda: session.output.audio._pushed_duration > 0)
            session.interrupt()
        await asyncio.wait_for(handle, 2)
        assert detector._voicemail_message_played is not interrupted

        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        classifier.prediction(2, AMDCategory.MACHINE_VM)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        if interrupted:
            await eventually(lambda: rt.generate_reply_calls == 2)
            respond(rt)
            await asyncio.wait_for(handles[-1], 2)
        else:
            assert rt.generate_reply_calls == 1


async def test_customer_stop_response_prevents_realtime_generation() -> None:
    async with running(agent=CustomerAgent(stop=True)) as (_, session, classifier, rt):
        assert session._activity.on_end_of_turn(end_of_turn())
        await classifier.request()
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await asyncio.wait_for(session._activity._user_turn_completed_atask, 2)
        assert rt.generate_reply_calls == 0
