"""Public Python journeys through the built extension, including process boundaries."""

import os
from array import array
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal, Never, NoReturn

import pytest

from livekit import rtc
from livekit.agents import _core, telemetry
from livekit.agents.beta.workflows import format_dtmf
from livekit.agents.llm import RealtimeCapabilities, RealtimeModel
from livekit.agents.storage import SQLiteStorage
from livekit.agents.utils.media import AudioByteStream
from livekit.agents.voice.avatar import DataStreamAudioOutput

pytestmark = pytest.mark.unit


def test_inference_headers_keep_the_public_list_of_pairs_interface() -> None:
    """Outside any job a request names only the SDK, and the scheduling class it asks for."""
    headers = _core.inference_identity_headers()
    assert isinstance(headers, list)
    [(name, agent)] = headers
    assert name == "User-Agent" and agent.startswith("LiveKit Agents/")
    assert _core.inference_identity_headers(inference_class="low")[1] == (
        "X-LiveKit-Inference-Priority",
        "low",
    )


def test_provider_updates_bind_only_the_changed_legs_and_their_nodes() -> None:
    from types import SimpleNamespace

    from livekit.agents.types import NOT_GIVEN
    from livekit.agents.voice._session_bindings import SessionHost

    stt_node, tts_node = object(), object()
    adapter = SimpleNamespace(_nodes=lambda _: {"stt_node": stt_node, "tts_node": tts_node})
    host = SessionHost.__new__(SessionHost)
    host.session = SimpleNamespace(current_agent=object(), _adapter=adapter)

    first = host.bind_providers({"stt": None})
    assert first.stt is None and first.tts is NOT_GIVEN and first.vad is NOT_GIVEN
    assert first.stt_node is stt_node and first.tts_node is None

    speaker, detector = object(), object()
    second = host.bind_providers({"tts": speaker, "vad": detector})
    assert second.stt is NOT_GIVEN and second.tts is speaker and second.vad is detector
    assert second.stt_node is None and second.tts_node is tts_node


def _native_facades() -> list[tuple[type, tuple[str, ...], str]]:
    """Every native provider facade, the hooks it runs itself and where their behaviour belongs."""
    from livekit.agents import inference, llm, stt, tts
    from livekit.agents.llm import _realtime, duplex, realtime

    tts_hooks, stt_hooks = ("_synthesize", "_stream"), ("stream", "recognize", "_recognize_impl")
    return [
        (inference.TTS, tts_hooks, "session.tts_node"),
        (tts.StreamAdapter, tts_hooks, "session.tts_node"),
        (tts.FallbackAdapter, tts_hooks, "session.tts_node"),
        (inference.STT, stt_hooks, "session.stt_node"),
        (stt.StreamAdapter, stt_hooks, "session.stt_node"),
        (stt.FallbackAdapter, stt_hooks, "session.stt_node"),
        (inference.LLM, ("chat",), "session.llm_node"),
        (llm.FallbackAdapter, ("chat",), "session.llm_node"),
        (inference.VAD, ("stream",), "session.stt_node"),
        (_realtime.RealtimeModel, ("session",), "session.llm_node"),
        (realtime.RealtimeModelFallbackAdapter, ("session",), "session.llm_node"),
        (duplex.DuplexRealtimeAdapter, ("session",), "session.llm_node"),
    ]


def test_a_hook_override_on_a_native_facade_is_refused_where_the_class_is_defined() -> None:
    """The core runs a native facade itself: a subclass replacing a hook it would bypass is
    refused as it is defined, with the same words for every kind; one that keeps them is not."""
    for facade, hooks, node in _native_facades():
        type("Inherited", (facade,), {"__module__": "app"})
        for hook in hooks:
            with pytest.raises(TypeError) as refused:
                type("Replacing", (facade,), {"__module__": "app", hook: lambda *_a, **_k: None})
            assert str(refused.value) == (
                f"Replacing overrides {hook}, which {facade.__qualname__} runs natively;"
                f" customize its output in {node} instead"
            )


def test_a_native_facade_whose_init_never_ran_is_refused_where_it_reaches_the_core() -> None:
    from livekit.agents import APIConnectOptions, inference

    class Unbuilt(inference.TTS):
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError, match=r"InferenceTts was never initialized"):
        _core.TtsStream(tts=Unbuilt(), conn_options=APIConnectOptions(max_retry=0))


def test_conversion_runs_no_host_code_of_the_facade() -> None:
    """Taking a facade as the core's provider reads only its native state: a class whose every
    host hook raises is still taken, and natively."""
    from livekit.agents import APIConnectOptions, tts

    def never(*_a: object, **_k: object) -> NoReturn:
        raise AssertionError("conversion called the host")

    class Guarded(tts.StreamAdapter):
        pass

    for name in dir(tts.TTS):
        if name.startswith("_core_") and callable(getattr(tts.TTS, name)):
            setattr(Guarded, name, never)
    from tests.fakes import FakeTTS

    adapter = Guarded(tts=FakeTTS(streaming=False))
    stream = _core.TtsStream(tts=adapter, conn_options=APIConnectOptions(max_retry=0))
    assert stream.native


def test_a_facade_passing_itself_to_the_core_while_it_is_built_does_not_recurse() -> None:
    from livekit.agents import APIConnectOptions, tts
    from tests.fakes import FakeTTS

    taken: list[bool] = []

    class Eager(tts.StreamAdapter):
        def __init__(self) -> None:
            super().__init__(tts=FakeTTS(streaming=False))
            taken.append(_core.TtsStream(tts=self, conn_options=APIConnectOptions()).native)

    Eager()
    assert taken == [True]


@pytest.mark.parametrize("mutable", [False, True])
def test_duplex_adapter_exposes_the_engine_capabilities_without_opening_a_session(
    mutable: bool,
) -> None:
    from typing import Never

    from livekit.agents.llm.duplex import DuplexCapabilities, DuplexModel, DuplexRealtimeAdapter

    class Model(DuplexModel):
        def session(self) -> Never:
            raise AssertionError("reading capabilities opened a provider session")

        async def aclose(self) -> None:
            pass

    model = Model(
        capabilities=DuplexCapabilities(
            user_transcription=mutable,
            auto_tool_reply_generation=mutable,
            mutable_chat_context=mutable,
            mutable_instructions=mutable,
            mutable_tools=mutable,
        )
    )
    caps = DuplexRealtimeAdapter(model).capabilities
    assert caps.auto_turn_reply_generation and caps.turn_detection and caps.audio_output
    assert caps.supports_overlapping_speech
    assert caps.user_transcription is mutable and caps.auto_tool_reply_generation is mutable
    assert caps.mutable_chat_context is mutable and caps.mutable_instructions is mutable
    assert caps.mutable_tools is mutable
    assert not caps.message_truncation and not caps.manual_function_calls and not caps.supports_say


@pytest.mark.parametrize("kind", ["llm", "stt", "tts", "realtime_model"])
def test_generated_provider_error_schema_retains_exception_but_excludes_it_from_json(
    kind: str,
) -> None:
    from livekit.agents import llm, stt, tts

    models = {
        "llm": llm.LLMError,
        "stt": stt.STTError,
        "tts": tts.TTSError,
        "realtime_model": llm.RealtimeModelError,
    }
    exception = ValueError("provider failed")
    event = models[kind](timestamp=1.5, label="provider", error=exception, recoverable=True)
    assert event.error is exception
    assert event.model_dump() == {
        "type": f"{kind}_error",
        "timestamp": 1.5,
        "label": "provider",
        "recoverable": True,
    }
    assert "provider failed" not in event.model_dump_json()


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


def test_native_scalars_refuse_a_value_of_another_kind() -> None:
    """A bool is not taken as a native integer nor an int as a native bool, and a negative count
    is refused instead of wrapping."""
    with pytest.raises(TypeError, match="window_size"):
        _core.MovingAverage(True)
    with pytest.raises(OverflowError):
        _core.MovingAverage(-1)
    with pytest.raises(TypeError, match="bool"):
        telemetry.set_capture_content(1)  # type: ignore[arg-type]  # the refused input


def test_a_native_list_argument_is_checked_element_by_element() -> None:
    """Every element of a list argument is converted, and one of the wrong type is refused."""
    assert format_dtmf(["1", "2", "#"]) == "1 2 #"
    with pytest.raises(TypeError, match="str"):
        format_dtmf(["1", 2])  # type: ignore[list-item]  # the refused input


@pytest.mark.parametrize(
    ("update", "options", "error", "message"),
    [
        (False, {"extra_kwargs": []}, ValueError, "expected a dict of extra kwargs, got []"),
        (True, {"extra_kwargs": []}, ValueError, "expected a dict of extra kwargs, got []"),
        (False, {"fallback": 7}, TypeError, "expected FallbackModels, got int"),
    ],
)
def test_inference_tts_refuses_misshapen_options_with_a_python_error(
    update: bool, options: dict[str, object], error: type[Exception], message: str
) -> None:
    """Options of the wrong shape, at construction or in an update, raise the Python error
    naming what was expected."""
    from livekit.agents import inference

    def tts(**kwargs: object) -> inference.TTS:
        return inference.TTS(
            "cartesia/sonic-2",
            base_url="http://unused.invalid",
            api_key="key",
            api_secret="secret",
            **kwargs,  # type: ignore[arg-type]  # the refused inputs
        )

    with pytest.raises(error) as raised:
        if update:
            tts().update_options(**options)  # type: ignore[arg-type]  # the refused inputs
        else:
            tts(**options)

    assert str(raised.value) == message


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_requires_reinitialization_and_parent_remains_usable() -> None:
    import traceback

    _core.init(_core.Role.Job)
    pid = os.fork()
    if pid == 0:
        try:
            # pure helpers hold no runtime state and work before reinitialization
            assert _core.ExpFilter(0.5, initial=8.0).apply(1, 0.0) == 4.0
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


async def test_run_handle_keeps_loaded_snapshot_and_commits_state(tmp_path: Path) -> None:
    storage = SQLiteStorage(directory=tmp_path)
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
            with pytest.raises(ValueError, match='RunOutcome: "invalid" is not one of'):
                await last_run.abandon("invalid", "reason")
            await last_run.abandon("failed", "finished testing")
            assert await thread.task("missing") is None
            assert await thread.partition_info("missing") is None
            with pytest.raises(ValueError, match='TaskOrder: "invalid" is not one of'):
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


async def test_a_provider_facade_reaches_the_core_as_its_native_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wherever a provider facade reaches the core, the core runs the native object it is: the
    gateway TTS's engine, never the facade as a plugin (whose provider hooks it never calls)."""
    from livekit.agents import APIConnectOptions, APIError, inference

    proxied: list[str] = []
    for hook in ("_core_stream", "_core_synthesize"):
        monkeypatch.setattr(
            inference.TTS, hook, lambda *_a, hook=hook, **_k: proxied.append(hook), raising=True
        )
    model = inference.TTS(
        "cartesia/sonic-3", base_url="http://127.0.0.1:9/v1", api_key="k", api_secret="s"
    )
    stream = _core.TtsStream(tts=model, conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()
    with pytest.raises(APIError):
        while await stream.recv() is not None:
            pass
    await stream.aclose()
    await model.aclose()

    assert proxied == [], "the core ran the wrapper as a plugin"


async def test_a_tokenizer_subclass_replacing_tokenize_still_runs_through_python() -> None:
    """A tokenizer subclass deciding its own sentences is taken behind the host's proxy, decided
    when its class was defined: the core splits by its `tokenize`, the built-in natively."""
    from livekit.agents import APIConnectOptions, tokenize
    from tests.fakes import FakeTTS

    calls: list[str] = []

    class Pipes(tokenize.SentenceTokenizer):
        def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
            calls.append(text)
            return [part for part in text.split("|") if part]

    assert getattr(Pipes, "_core_proxied", False)
    assert not getattr(tokenize.SentenceTokenizer, "_core_proxied", False)
    provider = FakeTTS(streaming=False)
    provider.update_options(sentence_tokenizer=Pipes(), pacing=None)
    stream = _core.TtsStream(tts=provider, conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("one two|three")
    stream.end_input()
    while await stream.recv() is not None:
        pass
    await stream.aclose()
    assert calls, "the core split natively"
    assert provider.texts and all("|" not in text for text in provider.texts)


async def test_realtime_chain_accepts_host_and_native_members() -> None:
    class Provider(RealtimeModel):
        def session(self, *, turn_detection_disabled: bool = False) -> NoReturn:
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
    assert chain._core_active_index == 0
    assert chain.capabilities == capabilities
    await chain._core_restart(switch_model=True)


async def test_http_route_kind_is_validated_before_the_host_runs() -> None:
    from livekit.agents._generated.worker_options import ProcessOptions, ServerOptions
    from livekit.agents._generated.worker_server import EndpointRoute

    runtime = _core.ServerRuntime(
        _core.ServerConfig(ServerOptions(), ProcessOptions(), "0.0.0"), object()
    )
    with pytest.raises(TypeError, match="EndpointRoute.kind") as failure:
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=False,
            routes=[EndpointRoute(path="/", methods=["GET"], kind="grpc", public=False)],
            has_load_fnc=False,
            endpoints=object(),
            text_doors=[],
        )
    assert isinstance(failure.value.__cause__, ValueError)
    assert "not one of 'http', 'text'" in str(failure.value.__cause__)


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

    from livekit.agents._bindings import APITimeoutError
    from livekit.agents.types import APIConnectOptions

    with pytest.raises(APITimeoutError) as failure:
        async with APIConnectOptions(total_timeout=0).total_deadline():
            await asyncio.Future()
    assert failure.value.timeout == "total"
    with pytest.raises(TimeoutError, match="provider timeout"):
        async with APIConnectOptions(total_timeout=None).total_deadline():
            raise TimeoutError("provider timeout")


def test_generated_overlap_event_retains_arrays_but_serializes_them_as_null() -> None:
    import numpy as np

    from livekit.agents._generated.events import OverlappingSpeechEvent as GeneratedEvent
    from livekit.agents.voice.events import OverlappingSpeechEvent

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


def test_generated_room_options_preserve_sdk_defaults_and_their_legs() -> None:
    from livekit.agents.voice.room_io import (
        AudioInputOptions,
        AudioOutputOptions,
        RoomOptions,
        TextInputOptions,
        VideoInputOptions,
    )

    first, second = AudioOutputOptions(), AudioOutputOptions()
    first.track_publish_options.source = rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO
    assert second.track_publish_options.source == rtc.TrackSource.SOURCE_MICROPHONE
    assert TextInputOptions().text_input_cb is TextInputOptions().text_input_cb
    options = RoomOptions()
    assert options.get_video_input_options() is None
    assert options.get_audio_input_options() == AudioInputOptions()
    microphone = AudioInputOptions(sample_rate=48000)
    options = RoomOptions(audio_input=microphone, audio_output=False, video_input=True)
    assert options.get_audio_input_options() is microphone
    assert options.get_audio_output_options() is None
    assert options.get_video_input_options() == VideoInputOptions()
    assert options.get_transcription_output_options() is not None


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
    from livekit.agents.voice.io import SpeechHandle

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


@pytest.mark.asyncio
async def test_realtime_generation_preserves_plain_text_and_transcript_chunk_stamps() -> None:
    import asyncio
    from types import SimpleNamespace

    from livekit.agents._generated import realtime_wire
    from livekit.agents.llm import GenerationCreatedEvent, MessageGeneration, RealtimeSession
    from livekit.agents.types import TranscriptChunk

    async def text() -> AsyncIterator[str | TranscriptChunk]:
        yield "plain "
        yield TranscriptChunk("timed", start_time=0.25, end_time=0.5)

    async def empty() -> AsyncIterator[Never]:
        if False:
            yield

    async def modalities() -> list[Literal["text", "audio"]]:
        return ["text"]

    async def messages() -> AsyncIterator[MessageGeneration]:
        yield MessageGeneration("message", text(), empty(), modalities())

    session = SimpleNamespace(_core_events=asyncio.Queue())
    generation = GenerationCreatedEvent(messages(), empty(), False, response_id="response")
    await RealtimeSession._core_drain(session, generation, None)
    events = []
    while not session._core_events.empty():
        events.append(session._core_events.get_nowait())
    deltas = [event for event in events if isinstance(event, realtime_wire.TextDelta)]
    assert [event.delta for event in deltas] == ["plain ", "timed"]
    assert [event.start_time for event in deltas] == [None, 0.25]


async def test_refused_amd_handle_cannot_replace_or_close_the_attached_detector() -> None:
    from livekit.agents import Agent, VoiceSession
    from livekit.agents.voice.amd import AMD
    from tests.fakes import FakeLLM

    async with VoiceSession(llm=FakeLLM(), vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Welcome the caller.", name="desk"))
        first, second = AMD(session, llm=None, stt=None), AMD(session, llm=None, stt=None)
        try:
            await first.__aenter__()
            assert first.enabled
            await second.__aenter__()
            assert not second.enabled and session._amd is first
            await second.aclose()
            assert first.enabled and session._amd is first
            await first.aclose()
            await second.__aenter__()
            assert second.enabled and session._amd is second
        finally:
            await first.aclose()
            await second.aclose()


def test_generated_grant_preserves_sequence_authoring_and_frozen_public_identity() -> None:
    from dataclasses import FrozenInstanceError

    from livekit.agents.a2a import AgentThreadGrant, Credentials, Principal

    original = ["AT_existing"]
    grant = AgentThreadGrant(thread_ids=original, read=True)
    original.append("AT_other")
    assert grant.thread_ids == ["AT_existing"]
    assert AgentThreadGrant(thread_ids=("AT_existing",)).thread_ids == ["AT_existing"]
    assert AgentThreadGrant().thread_ids == []
    assert AgentThreadGrant.__module__ == Principal.__module__ == "livekit.agents.a2a"
    with pytest.raises(FrozenInstanceError):
        grant.scope = "changed"

    class CustomGrant(AgentThreadGrant):
        pass

    assert CustomGrant(thread_ids=("AT_existing",)).thread_ids == ["AT_existing"]

    credentials = Credentials("binding-test", "a-secret-of-at-least-thirty-two-bytes")
    verified = credentials.verify(
        credentials.thread_grant_token(AgentThreadGrant(scope="user:test", create=True))
    )
    assert type(verified) is Principal and type(verified.grant) is AgentThreadGrant
    assert verified.grant.scope == "user:test" and verified.grant.can_create()
    project = credentials.verify(credentials.project_token())
    assert type(project) is Principal and project.grant is None and project.endpoint_call


@pytest.mark.parametrize("provider", [False, True])
async def test_voice_failure_keeps_input_requirements_on_the_raised_exception(
    provider: bool,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from livekit.agents import APIError
    from livekit.agents._generated.agent_run import RunInputRequirement
    from livekit.agents._generated.session_wire import RunFailedWire
    from livekit.agents._generated.storage_turn import PendingCall
    from livekit.agents.voice._session_callbacks import _on_run_failed

    requirements = [
        RunInputRequirement(
            invocation_id="voice-run",
            call=PendingCall(name="flight", call_id="flight-call", answer_tool=None),
            question="Which date?",
            verbatim=True,
        )
    ]
    run: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    original = APIError("tts failed", kind="server") if provider else None
    failed = RunFailedWire(
        run=run,
        error=APIError("the model request failed: tts failed", kind="invalid_request"),
        input_requirements=requirements,
        provider_error=original,
    )
    _on_run_failed(SimpleNamespace(), failed)  # type: ignore[arg-type]
    with pytest.raises(APIError) as raised:
        await run
    if original is not None:
        assert raised.value is original
    assert raised.value.__dict__["input_requirements"] == requirements
    assert raised.value.__dict__["input_requirements"] is not requirements

    late = RunFailedWire(
        run=run,
        error=RuntimeError("the session closed before the run settled"),
        input_requirements=[],
    )
    _on_run_failed(SimpleNamespace(), late)  # type: ignore[arg-type]
    assert raised.value.__dict__["input_requirements"] == requirements


def test_a_task_started_while_the_loop_shuts_down_is_cancelled_before_it_closes() -> None:
    import asyncio
    from collections.abc import AsyncIterator

    from livekit.agents.utils.aio import close_loop

    loop = asyncio.new_event_loop()
    started: list[asyncio.Task[None]] = []

    async def generator() -> AsyncIterator[None]:
        try:
            yield
        finally:
            # finalized by `shutdown_asyncgens`, after the first cancel pass
            started.append(asyncio.get_running_loop().create_task(_forever()))

    async def _forever() -> None:
        await asyncio.Event().wait()

    agen = generator()

    async def first() -> None:
        # inside the loop, so the loop registers the generator to finalize
        await agen.__anext__()

    loop.run_until_complete(first())
    close_loop(loop)
    assert loop.is_closed()
    assert started and started[0].cancelled(), "a task left pending in the closed loop"


async def test_the_end_of_the_core_events_settles_what_only_they_would() -> None:
    import asyncio

    from livekit.agents import Agent, VoiceSession
    from livekit.agents.voice.amd import AMD
    from tests.fakes import FakeLLM

    async with VoiceSession(llm=FakeLLM(), vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Welcome the caller.", name="desk"))
        amd = AMD(session, llm=None, stt=None)
        await amd.__aenter__()
        verdict = asyncio.ensure_future(amd.execute())
        # what `_run` and an expert's question register
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        session._settled_by_events(answer)
        await asyncio.sleep(0)
        assert not verdict.done()

        session._core_events_ended()
        with pytest.raises(RuntimeError, match="amd closed before a result"):
            await verdict
        with pytest.raises(RuntimeError, match="the voice session ended"):
            await answer
        assert not session._event_settled and session._amd is None
