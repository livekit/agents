"""Public Python journeys through the built extension, including process boundaries."""

import os
from array import array

import pytest

from livekit import rtc
from livekit.agents import _core
from livekit.agents.llm import RealtimeCapabilities, RealtimeModel
from livekit.agents.storage.backends import _MemoryStorage
from livekit.agents.utils.audio import AudioByteStream
from livekit.agents.voice.avatar import DataStreamAudioOutput

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["bytes", "bytearray", "byte_view", "pcm_view"])
def test_pcm_inputs_preserve_bytes_after_the_caller_mutates_its_buffer(kind: str) -> None:
    samples = array("h", [1, -2, 300, -400])
    expected = samples.tobytes()
    mutable = bytearray(expected)
    payload = {
        "bytes": expected,
        "bytearray": mutable,
        "byte_view": memoryview(mutable),
        "pcm_view": memoryview(samples),
    }[kind]
    stream = AudioByteStream(sample_rate=16000, num_channels=1, samples_per_channel=4)
    # Leave a partial frame queued in Rust, then change both potential mutable exporters.
    chunk = payload[:2]
    assert stream.push(chunk) == []
    if isinstance(chunk, bytearray):
        chunk[:] = b"\0" * len(chunk)
    mutable[:] = b"\0" * len(mutable)
    samples[0] = samples[1] = 0
    # A typed PCM view slices in samples; byte-oriented inputs slice in bytes.
    retained = expected[:4] if kind == "pcm_view" else expected[:2]
    remaining = expected[len(retained) :]
    frames = stream.push(remaining)
    frames.extend(stream.flush())
    assert b"".join(frame.data.tobytes() for frame in frames) == expected


@pytest.mark.parametrize("invalid", [memoryview(bytearray(8))[::2], memoryview(array("f", [1.0]))])
def test_invalid_pcm_layouts_fail_without_corrupting_queued_audio(invalid: memoryview) -> None:
    stream = AudioByteStream(sample_rate=16000, num_channels=1, samples_per_channel=2)
    assert stream.push(b"\x01") == []
    with pytest.raises(ValueError, match="one-dimensional contiguous byte or i16le buffer"):
        stream.push(invalid)
    frames = stream.push(b"\x02\x03\x04")
    assert b"".join(frame.data.tobytes() for frame in frames) == b"\x01\x02\x03\x04"


def test_initialization_preserves_the_public_role_contract_and_is_repeatable() -> None:
    assert _core.Role.__module__ == "livekit.agents._core"
    assert _core.Role.Worker == 0
    assert _core.Role.Job == 1
    assert _core.Role.Worker != _core.Role.Job
    assert _core.runtime_busy_seconds() == []
    for role in (_core.Role.Worker, _core.Role.Job, _core.Role.Job):
        assert _core.init(role) is None
        assert _core.runtime_busy_seconds() == []
    for invalid in (0, "Job", None):
        with pytest.raises(TypeError):
            _core.init(invalid)
    stream = _core.AudioByteStream(16000, 1, samples_per_channel=2)
    frames = stream.push(b"\x01\x02\x03\x04")
    assert frames[0].data.tobytes() == b"\x01\x02\x03\x04"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_requires_reinitialization_and_parent_remains_usable() -> None:
    import traceback

    _core.init(_core.Role.Job)
    pid = os.fork()
    if pid == 0:
        try:
            with pytest.raises(RuntimeError, match="inherited across a fork"):
                _core.ipc_pair()
            _core.init(_core.Role.Job)
            for fd in _core.ipc_pair():
                os.close(fd)
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    for fd in _core.ipc_pair():
        os.close(fd)


async def test_run_handle_keeps_loaded_snapshot_and_commits_state() -> None:
    storage = _MemoryStorage()
    try:
        thread_id = (await storage.threads.create_thread()).thread_id
        thread = await storage._open(thread_id)
        try:
            run = await thread.open_run("desk", "first")
            assert isinstance(run, _core.Run)
            assert run.state is None and run.history == [] and run.context_scopes == []
            version = await run.commit([], {"visits": 1})
            assert version > run.version
            assert run.state is None, "the handle exposes the snapshot loaded at open"

            next_run = await thread.open_run("desk", "second")
            assert next_run.state == {"visits": 1}
            await next_run.commit([], None)
            last_run = await thread.open_run("desk", "third")
            assert last_run.state == {"visits": 1}, "None preserves stored state"
            with pytest.raises(ValueError, match="unknown outcome"):
                await last_run.abandon("invalid", "reason")
            await last_run.abandon("failed", "finished testing")
            assert await thread.task("missing") is None
            assert await thread.lane_info("missing") is None
            with pytest.raises(ValueError, match="unknown task order"):
                await thread.list_tasks(order="invalid")
        finally:
            thread.close()
    finally:
        await storage.aclose()


async def test_avatar_output_keeps_sdk_room_and_validation_contract() -> None:
    room = rtc.Room()
    with pytest.raises(ValueError, match="clear_buffer_timeout"):
        DataStreamAudioOutput(
            room, destination_identity="avatar", clear_buffer_timeout=float("nan")
        )
    output = DataStreamAudioOutput(room, destination_identity="avatar")
    assert isinstance(output, _core.DataStreamAudioOutput)
    await output.aclose()
    await output.aclose()


async def test_realtime_chain_accepts_host_and_native_members() -> None:
    class Provider(RealtimeModel):
        def session(self, *, turn_detection_disabled=False):
            raise AssertionError("construction must not open a provider session")

    capabilities = RealtimeCapabilities(
        message_truncation=False,
        turn_detection=False,
        user_transcription=False,
        auto_tool_reply_generation=False,
        audio_output=False,
        manual_function_calls=False,
    )
    provider = Provider(capabilities=capabilities)
    first = _core.RealtimeFallbackModel([provider])
    chain = _core.RealtimeFallbackModel([first, provider])
    assert chain.active_index == 0
    assert chain.capabilities() == capabilities
    await chain.restart(switch_model=True)


async def test_http_route_kind_is_validated_before_the_host_runs() -> None:
    from livekit.agents._generated.worker_options import ServerOptions
    from livekit.agents._generated.worker_server import EndpointRoute

    runtime = _core.ServerRuntime(_core.ServerConfig(ServerOptions()), object())
    with pytest.raises(TypeError, match="EndpointRoute.kind") as failure:
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=False,
            routes=[EndpointRoute(path="/", methods=["GET"], kind="grpc", public=False)],
            has_load_fnc=False,
            endpoints=object(),
        )
    assert isinstance(failure.value.__cause__, ValueError)
    assert "not one of 'http', 'text'" in str(failure.value.__cause__)


@pytest.mark.parametrize("host_failure", [False, True])
async def test_drain_distinguishes_deadlines_from_identical_host_error_text(
    host_failure: bool,
) -> None:
    import asyncio

    from livekit.agents._generated.worker_options import ServerOptions

    class Host:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def _core_start(self, options: object) -> None:
            pass

        async def _core_started(self) -> None:
            self.started.set()

        async def _core_worker_load(self, load: float) -> None:
            pass

        async def _core_join_jobs(self) -> None:
            pass

        async def _core_close(self) -> None:
            pass

    class Endpoints:
        async def _core_drain(self) -> None:
            if host_failure:
                raise RuntimeError("drain timed out")
            await asyncio.Future()

    host = Host()
    config = _core.ServerConfig(ServerOptions())
    config.simulation = True
    runtime = _core.ServerRuntime(config, host)
    running = asyncio.ensure_future(
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=True,
            routes=[],
            has_load_fnc=False,
            endpoints=Endpoints(),
        )
    )
    try:
        await asyncio.wait_for(host.started.wait(), 5)
        expected = RuntimeError if host_failure else TimeoutError
        with pytest.raises(expected, match="drain timed out") as failure:
            await runtime.drain(timeout=0.1)
        assert type(failure.value) is expected
    finally:
        await asyncio.wait_for(runtime.aclose(), 5)
        await asyncio.wait_for(running, 5)


@pytest.mark.parametrize("name", ["FlushSentinel", "AudioSegmentEnd"])
def test_generated_markers_keep_public_identity_and_distinct_instances(name: str) -> None:
    from livekit.agents import types
    from livekit.agents._generated import avatar as generated_avatar, types as generated_types
    from livekit.agents.voice import avatar

    public, generated = {
        "FlushSentinel": (types.FlushSentinel, generated_types.FlushSentinel),
        "AudioSegmentEnd": (avatar.AudioSegmentEnd, generated_avatar.AudioSegmentEnd),
    }[name]
    assert public is generated
    assert public.__module__ == (types.__name__ if name == "FlushSentinel" else avatar.__name__)
    assert public() != public()


def test_generated_session_connect_options_keep_positional_fields_and_replace() -> None:
    from dataclasses import replace

    from livekit.agents._generated.connect import SessionConnectOptions as GeneratedOptions
    from livekit.agents._generated.session_wire import DEFAULT_MAX_UNRECOVERABLE_ERRORS
    from livekit.agents.types import APIConnectOptions
    from livekit.agents.voice.agent_session import SessionConnectOptions

    assert SessionConnectOptions is GeneratedOptions
    assert SessionConnectOptions().max_unrecoverable_errors == DEFAULT_MAX_UNRECOVERABLE_ERRORS
    policy = APIConnectOptions(max_retry=1)
    options = SessionConnectOptions(policy, None, None, 7)
    updated = replace(options, max_unrecoverable_errors=9)
    assert updated.stt_conn_options is policy
    assert updated.max_unrecoverable_errors == 9
    assert options.max_unrecoverable_errors == 7


def test_generated_connect_options_keep_frozen_replace_and_validation() -> None:
    from dataclasses import FrozenInstanceError, replace

    from livekit.agents._generated.connect import APIConnectOptions as GeneratedOptions
    from livekit.agents.types import APIConnectOptions

    assert APIConnectOptions is GeneratedOptions
    original = APIConnectOptions(2, 0.5)
    changed = replace(original, read_timeout=None)
    assert changed.max_retry == 2 and changed.retry_interval == 0.5
    assert changed.read_timeout is None and original.read_timeout == 10.0
    with pytest.raises(FrozenInstanceError):
        changed.max_retry = 9
    with pytest.raises(ValueError, match="max_retry"):
        APIConnectOptions(max_retry=True)
    assert original.interval_for_retry(0) == 0.1


async def test_generated_connect_options_total_deadline_keeps_timeout_classification() -> None:
    import asyncio

    from livekit.agents._exceptions import APITimeoutError
    from livekit.agents.types import APIConnectOptions

    with pytest.raises(APITimeoutError) as failure:
        async with APIConnectOptions(total_timeout=0).total_deadline():
            await asyncio.Future()
    assert failure.value.phase == "total"
    with pytest.raises(TimeoutError, match="provider timeout"):
        async with APIConnectOptions(total_timeout=None).total_deadline():
            raise TimeoutError("provider timeout")


def test_generated_overlap_event_retains_arrays_but_serializes_them_as_null() -> None:
    import numpy as np

    from livekit.agents._generated.overlap import OverlappingSpeechEvent as GeneratedEvent
    from livekit.agents.inference.interruption import OverlappingSpeechEvent

    assert OverlappingSpeechEvent is GeneratedEvent
    audio = np.array([1, -2], dtype=np.int16)
    probabilities = np.array([0.3, 0.8], dtype=np.float32)
    event = OverlappingSpeechEvent(speech_input=audio, probabilities=probabilities)
    assert event.speech_input is audio and event.probabilities is probabilities
    dumped = event.model_dump(mode="json")
    assert dumped["speech_input"] is None and dumped["probabilities"] is None
    assert dumped["type"] == "overlapping_speech"


def test_generated_missing_value_keeps_pydantic_omission_contract() -> None:
    from pydantic import BaseModel, ValidationError

    from livekit.agents._generated.types import NOT_GIVEN as generated_missing
    from livekit.agents.types import NOT_GIVEN, NotGiven

    class Input(BaseModel):
        value: int | NotGiven = NOT_GIVEN

    assert NOT_GIVEN is generated_missing
    assert not NOT_GIVEN and repr(NOT_GIVEN) == "NOT_GIVEN"
    assert Input().value is NOT_GIVEN
    assert Input(value=3).value == 3
    with pytest.raises(ValidationError):
        Input(value=None)


async def test_transcript_record_roundtrip_keeps_metadata_without_timestamps() -> None:
    from dataclasses import asdict

    from livekit.agents import tts, utils
    from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TranscriptChunk
    from livekit.agents.voice._session_callbacks import _transcript_chunk

    chunk = TranscriptChunk("hello", confidence=0.7, speaker_id="speaker-1")
    assert not isinstance(chunk, str)
    assert chunk != TranscriptChunk("hello", confidence=0.7, speaker_id="speaker-2")
    assert asdict(chunk)["text"] == "hello"
    assert _transcript_chunk(chunk) is chunk
    assert _transcript_chunk("plain") == TranscriptChunk("plain")

    output = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="transcript-contract", dst_ch=output)
    emitter.initialize(
        request_id="request", sample_rate=16000, num_channels=1, mime_type="audio/pcm"
    )
    with pytest.raises(TypeError):
        emitter.push_timed_transcript("untyped")
    with pytest.raises(TypeError):
        emitter.push_timed_transcript(TranscriptChunk("invalid", start_time="bad"))
    emitter.push_timed_transcript(chunk)
    emitter.push_frame(rtc.AudioFrame.create(16000, 1, 320))
    emitter.end_input()
    try:
        events = []
        async for event in output:
            events.append(event)
            if event.is_final:
                break
        await emitter.join()
        words = [
            word
            for event in events
            for word in event.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
        ]
        assert words == [chunk]
    finally:
        await emitter.aclose()
        output.close()


def test_generated_room_options_preserve_sdk_defaults_and_resolution() -> None:
    from livekit.agents.voice.room_io import (
        AudioInputOptions,
        AudioOutputOptions,
        RoomOptions,
        TextInputOptions,
    )

    first, second = AudioOutputOptions(), AudioOutputOptions()
    first.track_publish_options.source = rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO
    assert second.track_publish_options.source == rtc.TrackSource.SOURCE_MICROPHONE
    assert TextInputOptions().text_input_cb is TextInputOptions().text_input_cb
    options = RoomOptions()
    assert options.get_video_input_options() is None
    assert options._to_core()["audio_input"]["auto_gain_control"] is True
    nc = rtc.NoiseCancellationOptions(module_id="test", options={})
    options.audio_input = AudioInputOptions(noise_cancellation=nc)
    assert options._to_core()["audio_input"]["auto_gain_control"] is False
    options.audio_input.auto_gain_control = True
    assert options._to_core()["audio_input"]["auto_gain_control"] is True
    options.audio_output = False
    assert options._to_core()["audio_output"] is None


def test_generated_session_events_validate_and_preserve_control_and_serialization() -> None:
    from dataclasses import FrozenInstanceError

    from pydantic import ValidationError

    from livekit.agents.llm import FunctionCall, FunctionCallOutput
    from livekit.agents.voice.events import (
        ErrorEvent,
        FunctionToolsExecutedEvent,
        SpeechCreatedEvent,
        event_dict,
    )
    from livekit.agents.voice.speech_handle import SpeechHandle

    call = FunctionCall(call_id="call", name="example", arguments="{}")
    output = FunctionCallOutput(call_id="call", output="done", is_error=False)
    with pytest.raises(ValidationError, match="number of function_calls"):
        FunctionToolsExecutedEvent(function_calls=[call], function_call_outputs=[])
    event = FunctionToolsExecutedEvent(function_calls=[call], function_call_outputs=[output])
    assert event.zipped() == [(call, output)]
    assert event.has_tool_reply
    event.cancel_tool_reply()
    assert not output.reply_required
    event._handoff_required = True
    event.cancel_agent_handoff()
    assert not event.has_agent_handoff
    assert "_handoff_required" not in event.model_dump()

    # The excluded handle must never be inspected by the event serializer.
    handle = object.__new__(SpeechHandle)
    speech = SpeechCreatedEvent(user_initiated=True, source="say", speech_handle=handle)
    assert speech.speech_handle is handle
    assert "speech_handle" not in event_dict(speech, json=True)
    with pytest.raises(FrozenInstanceError):
        speech.source = "generate_reply"
    error = ErrorEvent(error=ValueError("bad input"), source="adapter")
    assert error.model_dump()["error"] == "ValueError('bad input')"
    assert error.model_dump()["source"] == "'adapter'"


def test_generated_tool_records_keep_signatures_forward_types_and_exceptions() -> None:
    from dataclasses import FrozenInstanceError
    from typing import get_type_hints

    from livekit.agents.llm import FunctionTool, InputRequired, ModelRetry, ToolError, ToolFlag
    from livekit.agents.llm.realtime import RealtimeError
    from livekit.agents.llm.tool_context import FunctionToolInfo, StopResponse, ToolResult

    result = ToolResult("hello", reply_required=False)
    assert result.output == "hello" and not result.reply_required
    with pytest.raises(TypeError):
        ToolResult("hello", False)
    assert get_type_hints(FunctionToolInfo)["flags"] is ToolFlag
    assert FunctionTool in get_type_hints(InputRequired)["tool"].__args__
    with pytest.raises(FrozenInstanceError):
        InputRequired("Question?").question = "Changed"
    assert (ToolFlag.NONE | ToolFlag.IGNORE_ON_ENTER) is ToolFlag.IGNORE_ON_ENTER
    for cls in (ToolError, ModelRetry):
        error = cls("retry reason")
        assert error.args == ("retry reason",) and error.message == "retry reason"
        with pytest.raises(AttributeError):
            error.message = "Changed"
    assert StopResponse().args == ()
    with pytest.raises(TypeError):
        StopResponse("unexpected")
    error = RealtimeError("busy", code="response_active")
    assert error.args == ("busy",) and error.code == "response_active"


def test_generated_evaluation_results_share_the_native_judgment_and_score() -> None:
    from livekit.agents._generated.evals import JudgmentResult as NativeJudgment
    from livekit.agents.evals import EvaluationResult, JudgmentResult

    judgment = JudgmentResult("pass", "criteria met")
    assert type(judgment) is NativeJudgment and judgment.passed
    assert not judgment.failed and not judgment.uncertain
    result = EvaluationResult({"first": judgment, "second": JudgmentResult("maybe", "unclear")})
    assert result.score == 0.75
    assert not result.all_passed and result.any_passed and result.none_failed
    assert EvaluationResult().judgments is not EvaluationResult().judgments
