"""AMD model selection and resource ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from livekit import rtc
from livekit.agents import AMD, NOT_GIVEN, Agent, AgentSession, inference, llm, utils
from livekit.agents.types import NotGivenOr
from livekit.agents.voice.amd import AMDCategory, AMDLifecycle
from livekit.agents.voice.events import SpeechCreatedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

from .amd_test_utils import detector_clock  # noqa: F401
from .fake_llm import FakeLLM
from .fake_stt import DrainingSTT, FakeSTT
from .test_amd_detector import ClassifierLLM, commit, eventually, push_audio, running

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.virtual_time]


@pytest.fixture
def cloud_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")


@pytest.fixture
def model_factories(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = ClassifierLLM()
    model.aclose = AsyncMock()
    stt = DrainingSTT()
    stt.aclose = AsyncMock()
    llm_factory = Mock(return_value=model)
    stt_factory = Mock(return_value=stt)
    monkeypatch.setattr(inference.LLM, "from_model_string", llm_factory)
    monkeypatch.setattr(inference.STT, "from_model_string", stt_factory)
    return llm_factory, stt_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", ["project", "inference", "mixed"])
@pytest.mark.parametrize("finish", [False, True])
async def test_default_models_are_auto_selected_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
    credentials: str,
    finish: bool,
) -> None:
    if credentials != "project":
        monkeypatch.delenv("LIVEKIT_API_KEY")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "test-key")
    if credentials == "inference":
        monkeypatch.delenv("LIVEKIT_API_SECRET")
        monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "test-secret")
    llm_factory, stt_factory = model_factories
    model, stt = llm_factory.return_value, stt_factory.return_value
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            llm_factory.assert_called_once_with("google/gemini-3.1-flash-lite")
            stt_factory.assert_called_once_with("cartesia/ink-whisper")
            assert detector._llm is model
            assert detector._stt is stt
            stream = push_audio(detector, stt)
            if finish:
                stream.send_fake_transcript("hello")
                await eventually(
                    lambda: detector._resources.stt._current.snapshot("amd").transcript == "hello"
                )
                await commit(detector, session, model)
                model.prediction(1, AMDCategory.HUMAN)
                assert (await detector.execute()).category == AMDCategory.HUMAN
        assert stream._task.done()
        model.aclose.assert_awaited_once()
        stt.aclose.assert_awaited_once()
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.parametrize("missing", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"])
def test_auto_selection_requires_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
    missing: str,
) -> None:
    monkeypatch.delenv(missing)
    detector = AMD(AgentSession())
    assert detector._llm is None
    assert detector._stt is None
    for factory in model_factories:
        factory.assert_not_called()


def test_auto_selection_does_not_use_cloud_models_for_a_local_server(
    monkeypatch: pytest.MonkeyPatch,
    cloud_credentials: None,
    model_factories: tuple[Mock, Mock],
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    detector = AMD(AgentSession())
    assert detector._llm is None
    assert detector._stt is None
    for factory in model_factories:
        factory.assert_not_called()


@pytest.mark.asyncio
async def test_none_always_inherits_the_active_agent_models(
    cloud_credentials: None, model_factories: tuple[Mock, Mock]
) -> None:
    model = ClassifierLLM()
    model.aclose = AsyncMock()
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment.", llm=model))
    try:
        async with AMD(session, llm=None, stt=None) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            assert detector._resources.llm is model
            assert detector._stt is None
            assert detector._resources.stt._model is None
            await commit(detector, session, model)
            model.prediction(1, AMDCategory.HUMAN)
            assert (await detector.execute()).category == AMDCategory.HUMAN
        model.aclose.assert_not_awaited()
        for factory in model_factories:
            factory.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.parametrize("inherited", ["llm", "stt"])
def test_model_selection_is_independent_for_each_model(
    cloud_credentials: None, model_factories: tuple[Mock, Mock], inherited: str
) -> None:
    detector = AMD(AgentSession(), **{inherited: None})
    llm_factory, stt_factory = model_factories
    if inherited == "llm":
        llm_factory.assert_not_called()
        stt_factory.assert_called_once_with("cartesia/ink-whisper")
        assert detector._llm is None
    else:
        llm_factory.assert_called_once_with("google/gemini-3.1-flash-lite")
        stt_factory.assert_not_called()
        assert detector._stt is None


@pytest.mark.parametrize("as_strings", [False, True])
def test_explicit_models_override_auto_selection(
    cloud_credentials: None, model_factories: tuple[Mock, Mock], as_strings: bool
) -> None:
    llm_factory, stt_factory = model_factories
    model, stt = llm_factory.return_value, stt_factory.return_value
    detector = AMD(
        AgentSession(),
        llm="google/gemma-4-31b-it" if as_strings else model,
        stt="cartesia/ink-2" if as_strings else stt,
    )
    assert detector._llm is model
    assert detector._stt is stt
    assert detector._owns_llm == as_strings
    assert detector._owns_stt == as_strings
    if as_strings:
        llm_factory.assert_called_once_with("google/gemma-4-31b-it")
        stt_factory.assert_called_once_with("cartesia/ink-2")
    else:
        llm_factory.assert_not_called()
        stt_factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_stt", [None, NOT_GIVEN])
async def test_default_falls_back_to_session_model_without_amd_credentials(
    monkeypatch: pytest.MonkeyPatch, extra_stt: Any
) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    model = ClassifierLLM()
    session = AgentSession(llm=model, turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(session, stt=extra_stt) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            assert detector._resources.llm is model
            await commit(detector, session, model)
            model.prediction(1, AMDCategory.HUMAN)
            assert (await detector.execute()).category == AMDCategory.HUMAN
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_enabled", [False, True])
@pytest.mark.parametrize("agent_enabled", [False, True, NOT_GIVEN])
async def test_amd_restores_interruption_settings(
    session_enabled: bool, agent_enabled: NotGivenOr[bool]
) -> None:
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": agent_enabled}}
        if utils.is_given(agent_enabled)
        else {},
    )
    async with running(
        agent=agent,
        session_options={
            "turn_handling": {
                "turn_detection": "manual",
                "interruption": {"enabled": session_enabled},
            }
        },
    ) as (detector, session, classifier, _):
        assert session.options.interruption["enabled"] is True
        assert agent.allow_interruptions is True
        handles = []
        session.on("speech_created", lambda event: handles.append(event.speech_handle))
        await commit(detector, session, classifier, reply=True)
        classifier.prediction(1, AMDCategory.MACHINE_SCREENING)
        await session._activity._user_turn_completed_atask
        assert handles[0].allow_interruptions
        await handles[0]

        await commit(detector, session, classifier)
        classifier.prediction(2, AMDCategory.HUMAN)
        await detector.execute()
        assert session.options.interruption["enabled"] is session_enabled
        assert agent.allow_interruptions is agent_enabled
        following = session.generate_reply(user_input="hello")
        expected = agent_enabled if utils.is_given(agent_enabled) else session_enabled
        assert following.allow_interruptions is expected
        await following


@pytest.mark.asyncio
async def test_missing_llm_does_not_install_turn_hooks() -> None:
    session = AgentSession(turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        with pytest.raises(ValueError, match="requires an LLM"):
            await AMD(session, llm=None, stt=None).__aenter__()
        assert session.amd is None
        assert session._turn_hooks is None
        assert session._activity._authorization_allowed.is_set()
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pause", "participant", "session_on", "room_on", "listening"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_setup_failure_cleans_up_amd(
    monkeypatch: pytest.MonkeyPatch,
    model_factories: tuple[Mock, Mock],
    failure: str,
    error_type: type[BaseException],
) -> None:
    session = AgentSession(
        llm=FakeLLM(),
        turn_handling={"turn_detection": "manual", "interruption": {"enabled": False}},
    )
    agent = Agent(
        instructions="Call about an appointment.",
        turn_handling={"interruption": {"enabled": False}},
    )
    await session.start(agent)
    activity = session._activity
    assert activity is not None
    room = utils.EventEmitter()
    room_io = SimpleNamespace(room=room, set_participant=Mock())
    session._room_io = room_io
    detector = AMD(
        session,
        llm="google/gemma-4-31b-it",
        stt="cartesia/ink-2",
        participant_identity="callee",
    )
    target, method = {
        "pause": (activity, "_pause_authorization"),
        "participant": (room_io, "set_participant"),
        "session_on": (session, "on"),
        "room_on": (room, "on"),
        "listening": (asyncio.get_running_loop(), "create_task"),
    }[failure]
    original = getattr(target, method)
    error = error_type("setup failed")

    def fail_setup(*args: Any, **kwargs: Any) -> Any:
        if failure == "listening":
            if args[0].cr_code is AMD._setup_listening.__code__:
                raise error
            return original(*args, **kwargs)
        result = original(*args, **kwargs)
        if failure == "session_on" and args[0] != "speech_created":
            return result
        raise error

    monkeypatch.setattr(target, method, fail_setup)
    try:
        with pytest.raises(error_type, match="setup failed") as exc_info:
            await detector.__aenter__()
        assert exc_info.value is error
        assert session.amd is None
        assert session._turn_hooks is None
        assert activity._authorization_allowed.is_set()
        assert session.options.interruption["enabled"] is False
        assert agent.allow_interruptions is False
        assert detector.lifecycle is AMDLifecycle.FINISHED
        assert not detector._tasks
        for emitter, event, handler in detector._subscriptions:
            assert handler not in emitter._events.get(event, ())
        for factory in model_factories:
            factory.return_value.aclose.assert_awaited_once()
    finally:
        await detector.aclose()
        session._room_io = None
        await session.aclose()


@pytest.mark.asyncio
async def test_stt_model_string_uses_the_standard_factory_and_closes_owned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    model = DrainingSTT()
    model.aclose = AsyncMock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(inference.STT, "from_model_string", factory)
    async with running(stt="cartesia/ink-2") as (detector, _, _, _):
        stream = push_audio(detector, model)
        factory.assert_called_once_with("cartesia/ink-2")
        await detector.aclose()
        assert stream._task.done()
    model.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_closes_the_run_stream_and_does_not_close_supplied_stt() -> None:
    stt = DrainingSTT()
    stt.aclose = AsyncMock()
    async with running(stt=stt) as (detector, session, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("hello")
        await eventually(
            lambda: detector._resources.stt._current.snapshot("amd").transcript == "hello"
        )
        await commit(detector, session, classifier)
        assert push_audio(detector, stt) is stream
        await asyncio.sleep(0)
        await detector.aclose()
        assert stream._task.done()
        assert len(stt.streams) == 1
        assert not detector._tasks
        assert session.amd is None
        stt.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_owned_model_cleanup_failure_still_detaches_and_releases_turn_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import inference

    stt = DrainingSTT()
    stt.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    monkeypatch.setattr(inference.STT, "from_model_string", Mock(return_value=stt))
    async with running(stt="cartesia/ink-2") as (detector, session, classifier, _):
        stream = push_audio(detector, stt)
        stream.send_fake_transcript("hello")
        await eventually(
            lambda: detector._resources.stt._current.snapshot("amd").transcript == "hello"
        )
        await commit(detector, session, classifier)
        await asyncio.wait_for(detector.aclose(), 2)
        assert (await detector.execute()).reason == "cancelled"
        assert session.amd is None
        assert session._activity._authorization_allowed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_finish_detaches_before_task_cleanup_and_preserves_a_new_run(
    monkeypatch: pytest.MonkeyPatch, category: AMDCategory
) -> None:
    async with running() as (detector, session, classifier, _):
        activity = session._activity
        assert activity is not None
        started = asyncio.Event()
        cancelling = asyncio.Event()
        release = asyncio.Event()

        async def slow_cancellation() -> None:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()

        detector._spawn(slow_cancellation())
        await started.wait()
        terminal_session_state = []
        detector.on(
            "amd_prediction",
            lambda _: terminal_session_state.append(
                (session.amd, session._turn_hooks, activity._authorization_allowed.is_set())
            ),
        )
        try:
            await commit(detector, session, classifier)
            classifier.prediction(1, category)
            await asyncio.wait_for(cancelling.wait(), 2)
            assert terminal_session_state == [(None, None, True)]
            assert session.current_agent.allow_interruptions is NOT_GIVEN
            assert not detector._resources.completion.done()

            session_audio = Mock()
            amd_audio = Mock(wraps=detector.push_audio)
            monkeypatch.setattr(activity._audio_recognition, "_push_audio", session_audio)
            monkeypatch.setattr(detector, "push_audio", amd_audio)
            frame = rtc.AudioFrame.create(16000, 1, 320)
            activity.push_audio(frame)
            session_audio.assert_called_once_with(frame, stt_frame=None)
            amd_audio.assert_not_called()

            following = AMD(session, llm=ClassifierLLM(), stt=None)

            async def wait_to_listen() -> None:
                await asyncio.Future()

            monkeypatch.setattr(following, "_setup_listening", wait_to_listen)
            async with following:
                assert following.lifecycle is AMDLifecycle.PENDING
                cancel_pending = Mock(wraps=activity._cancel_pending_speeches)
                cancel_preemptive = Mock(wraps=activity._cancel_preemptive_generation)
                monkeypatch.setattr(activity, "_cancel_pending_speeches", cancel_pending)
                monkeypatch.setattr(activity, "_cancel_preemptive_generation", cancel_preemptive)
                release.set()
                assert (await asyncio.wait_for(detector.execute(), 2)).category == category
                assert session.amd is following
                assert session._turn_hooks is following._turn_hooks
                assert not activity._authorization_allowed.is_set()
                assert session.current_agent.allow_interruptions is True
                cancel_pending.assert_not_called()
                cancel_preemptive.assert_not_called()
            assert session.current_agent.allow_interruptions is NOT_GIVEN
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_target", ["run_stt", "owned_stt", "owned_llm"])
@pytest.mark.parametrize("category", [AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE])
async def test_slow_resource_close_does_not_keep_amd_attached(
    monkeypatch: pytest.MonkeyPatch,
    model_factories: tuple[Mock, Mock],
    close_target: str,
    category: AMDCategory,
) -> None:
    llm_factory, stt_factory = model_factories
    classifier, stt_model = llm_factory.return_value, stt_factory.return_value
    session = AgentSession(llm=FakeLLM(), turn_handling={"turn_detection": "manual"})
    await session.start(Agent(instructions="Call about an appointment."))
    try:
        async with AMD(
            session,
            llm="google/gemma-4-31b-it",
            stt="cartesia/ink-2",
            machine_silence_threshold=0,
        ) as detector:
            await eventually(lambda: detector.lifecycle is AMDLifecycle.ACTIVE)
            stream = push_audio(detector, stt_model)
            target = {
                "run_stt": detector._resources.stt,
                "owned_stt": stt_model,
                "owned_llm": classifier,
            }[close_target]
            close_started = asyncio.Event()
            release_close = asyncio.Event()
            original_close = target.aclose

            async def slow_close() -> None:
                close_started.set()
                await release_close.wait()
                await original_close()

            stream.send_fake_transcript("hello")
            await eventually(
                lambda: detector._resources.stt._current.snapshot("amd").transcript == "hello"
            )
            hooks = await commit(detector, session, classifier)
            classifier.prediction(1, AMDCategory.MACHINE_VM)
            assert await hooks.should_reply(llm.ChatContext())
            speech = SpeechHandle.create()
            session.emit(
                "speech_created",
                SpeechCreatedEvent(
                    speech_handle=speech, user_initiated=False, source="generate_reply"
                ),
            )
            hooks.on_agent_turn_committed(speech)
            remove_callback = Mock(wraps=speech.remove_done_callback)
            monkeypatch.setattr(speech, "remove_done_callback", remove_callback)
            activity = session._activity
            assert activity is not None
            cancel_pending = Mock(wraps=activity._cancel_pending_speeches)
            cancel_preemptive = Mock(wraps=activity._cancel_preemptive_generation)
            monkeypatch.setattr(activity, "_cancel_pending_speeches", cancel_pending)
            monkeypatch.setattr(activity, "_cancel_preemptive_generation", cancel_preemptive)
            completed_events = []
            detector.on("amd_completed", completed_events.append)
            completion = asyncio.create_task(detector.execute())
            try:
                monkeypatch.setattr(target, "aclose", AsyncMock(side_effect=slow_close))
                stream.send_fake_transcript("hello")
                await eventually(
                    lambda: detector._resources.stt._current.snapshot("amd").transcript == "hello"
                )
                await commit(detector, session, classifier)
                classifier.prediction(2, category)
                await asyncio.wait_for(close_started.wait(), 2)

                assert session.amd is None
                assert session._turn_hooks is None
                assert activity._authorization_allowed.is_set()
                for emitter, event, handler in detector._subscriptions:
                    assert handler not in emitter._events.get(event, ())
                remove_callback.assert_any_call(detector._on_speech_done)
                remove_callback.assert_any_call(detector._on_voicemail_done)
                assert cancel_pending.called == (category == AMDCategory.MACHINE_UNAVAILABLE)
                cancel_preemptive.assert_called()
                assert not completion.done()
                assert completed_events == []
            finally:
                release_close.set()
                result = await asyncio.wait_for(completion, 2)
                speech._mark_done()
            assert result.category == category
            assert result.reason == "finished"
            assert completed_events == [result]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_nonstreaming_stt_is_rejected() -> None:
    stt = FakeSTT()
    stt._capabilities.streaming = False
    with pytest.raises(ValueError, match="streaming STT"):
        AMD(AgentSession(), llm=None, stt=stt)
