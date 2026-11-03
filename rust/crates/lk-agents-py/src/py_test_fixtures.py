"""Host fixtures for Rust boundary tests; native assertions remain in the Rust targets."""

import asyncio
import threading
from types import SimpleNamespace

_generated_fixture_lock = threading.Lock()


def room_options_for_native(case):
    from livekit import rtc
    from livekit.agents.voice.room_io import (
        AudioInputOptions,
        AudioOutputOptions,
        RoomOptions,
        TextOutputOptions,
    )

    if case == "configured":
        options = RoomOptions(
            participant_identity="caller",
            audio_input=AudioInputOptions(
                sample_rate=48000,
                noise_cancellation=rtc.NoiseCancellationOptions(module_id="test", options={}),
                pre_connect_audio_timeout=-1,
            ),
            audio_output=AudioOutputOptions(
                track_name="speaker",
                track_publish_options=rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO
                ),
            ),
            text_output=TextOutputOptions(sync_transcription=False),
        )
    elif case == "disabled":
        options = RoomOptions(
            audio_input=False, audio_output=False, text_input=False, text_output=False
        )
    elif case == "invalid_participant_kind":
        options = RoomOptions(participant_kinds=[-1])
    else:
        assert case == "default"
        options = RoomOptions()
    return options._to_core()


def generated_module(name):
    import importlib.util
    import sys
    from pathlib import Path

    module_name = f"livekit.agents._generated.{name}"
    with _generated_fixture_lock:
        module = sys.modules.get(module_name)
        if module is None:
            path = (
                Path(__file__).parents[4]
                / f"livekit-agents-next/livekit/agents/_generated/{name}.py"
            )
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        return module


def invocation():
    import asyncio
    import contextvars
    import threading

    var = contextvars.ContextVar("var", default="unset")
    seen = {}
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()

    def capture(value):
        var.set(value)
        return (loop, contextvars.copy_context())

    def sync_call(tag):
        seen[tag] = (var.get(), threading.current_thread() is not threading.main_thread())
        var.set("changed by " + tag)
        return tag

    async def async_call(tag):
        var.set("before await")
        await asyncio.sleep(0)
        seen[tag] = var.get()
        return tag

    async def failing():
        raise ValueError("boom")

    blocked = {}

    async def blocking():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            blocked["cancelled"] = True
            raise

    return SimpleNamespace(
        var=var,
        seen=seen,
        loop=loop,
        capture=capture,
        sync_call=sync_call,
        async_call=async_call,
        failing=failing,
        blocked=blocked,
        blocking=blocking,
    )


def parentage():
    import asyncio
    import contextvars
    import threading

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    previous_provider = trace.get_tracer_provider()
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer("host-callback", tracer_provider=provider)
    job = contextvars.ContextVar("job", default="outside")
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    cancelled = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(timeout=2)

    def capture(identity, trace_id, span_id, sampled):
        # The logical parent exists even when span-start observation was dropped or never recorded.
        context = contextvars.copy_context()

        def enter():
            job.set(identity)
            parent = trace.NonRecordingSpan(
                trace.SpanContext(trace_id, span_id, False, trace.TraceFlags(1 if sampled else 0))
            )
            from opentelemetry import context as otel_context

            otel_context.attach(trace.set_span_in_context(parent))

        context.run(enter)
        return (loop, context)

    async def callback(mode):
        with tracer.start_as_current_span("tool-or-node") as span:
            span.set_attribute("job", job.get())
            await asyncio.sleep(0)
            if mode == "failure":
                raise ValueError("callback failure")
            if mode == "cancel":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            job.set("changed in callback")
            observed = (span.get_span_context().trace_id, span.is_recording())
        return observed

    def verify():
        spans = exporter.get_finished_spans()
        assert len(spans) == 4, [s.name for s in spans]
        for span in spans:
            expected = {"one": (101, 11), "two": (202, 22)}[span.attributes["job"]]
            assert (span.context.trace_id, span.parent.span_id) == expected
        assert job.get() == "outside"
        assert not trace.get_current_span().get_span_context().is_valid
        assert trace.get_tracer_provider() is previous_provider

    def close():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        assert not thread.is_alive()
        loop.close()
        provider.shutdown()

    return SimpleNamespace(
        job=job,
        loop=loop,
        cancelled=cancelled,
        capture=capture,
        callback=callback,
        verify=verify,
        close=close,
    )


def streams():
    import asyncio
    import contextvars
    import threading

    var = contextvars.ContextVar("var", default="unset")
    log = []
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()

    def capture(value):
        var.set(value)
        return (loop, contextvars.copy_context())

    gate = asyncio.Event()

    async def scoped():
        token = var.set("inside")
        try:
            yield var.get()
            await gate.wait()
            yield var.get()
        finally:
            log.append(("closed in", var.get()))
            var.reset(token)

    async def cancelled():
        yield 1
        raise asyncio.CancelledError()

    class Unclosable:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return "x"

        async def aclose(self):
            raise RuntimeError("cannot close")

    def unclosable():
        return Unclosable()

    class BrokenCleanup(Unclosable):
        @property
        def aclose(self):
            raise RuntimeError("cannot look up cleanup")

    class Duplex:
        def __init__(self):
            self.got = []
            self.ended = asyncio.Event()

        def __aiter__(self):
            var.set("stream")
            return self.output()

        async def output(self):
            await self.ended.wait()
            yield str(self.got)

        def _core_push(self, v):
            if v == "bad":
                raise ValueError("refused")
            self.got.append((v, var.get()))

        def _core_flush(self):
            self.got.append(("flush", var.get()))

        def _core_end_input(self):
            self.got.append(("end", var.get()))
            self.ended.set()

    return SimpleNamespace(
        var=var,
        log=log,
        loop=loop,
        capture=capture,
        gate=gate,
        scoped=scoped,
        cancelled=cancelled,
        unclosable=unclosable,
        BrokenCleanup=BrokenCleanup,
        Duplex=Duplex,
    )


def agent_host():
    import contextvars

    from livekit.agents._generated import agent_run as models, tool_report as reports
    from livekit.agents.voice.agent_host import ToolArgumentError

    var = contextvars.ContextVar("var", default="unset")
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()

    def capture(value):
        var.set(value)
        return (loop, contextvars.copy_context())

    class Host:
        def __init__(self):
            self.seen = {}
            self.cancelled = threading.Event()

        async def render_instructions(self, invocation):
            self.seen["render"] = (var.get(), invocation)
            return "be brief"

        async def prepare_tool(self, invocation, arguments):
            if arguments.get("bad"):
                raise ToolArgumentError("city is required")
            if arguments.get("crash"):
                raise KeyError("crash")
            return models.PreparedArguments(
                prepared=arguments, identity=models.ArgumentIdentityJson(value=arguments)
            )

        async def run_tool(self, invocation, prepared):
            report = reports.ToolReport(outcome=reports.ToolOutcomeValue(text="answered"))
            if prepared.get("bad_arity"):
                return (report,)
            if prepared.get("bad_metadata"):
                return report, ["not a map"]
            return report, {"lk.delegation.task_id": "original-task"}

        async def validate_output(self, invocation, candidate):
            self.seen["candidate"] = candidate
            return models.ValidatedOutputRetry(reason="not a number")

        def cancel(self, invocation):
            self.seen["cancel"] = (var.get(), invocation.operation)
            self.cancelled.set()

    host = Host()
    return SimpleNamespace(var=var, loop=loop, capture=capture, host=host)


def telemetry_host():
    import contextvars

    from livekit.agents import _telemetry

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait()
    tracer = _telemetry.TracerProvider().get_tracer("app")

    def observe():
        span = tracer.start_span("app.child")
        context = span.get_span_context()
        span.end()
        return _telemetry._current(), context.trace_id

    return SimpleNamespace(pair=(loop, contextvars.copy_context()), observe=observe)


def notifications_reach_the_host_in_call_order_in_their_context(state):
    var = state.var
    order = []

    def note(i):
        order.append((i, var.get(), threading.current_thread() is not threading.main_thread()))

    state.order = order
    state.note = note


def a_notification_behind_a_stalled_loop_is_delivered_and_a_lossy_one_dropped(state):
    import time

    seen = []

    def stall(i):
        time.sleep(0.3)

    def note(i):
        seen.append(i)

    state.seen = seen
    state.stall = stall
    state.note = note


def manual_loop():
    import asyncio
    import contextvars

    loop = asyncio.new_event_loop()
    context = contextvars.copy_context()
    entered = []

    def callback():
        entered.append("entered")

    return SimpleNamespace(loop=loop, context=context, entered=entered, callback=callback)


def tick_loop(state):
    loop = state.loop
    loop.call_soon(loop.stop)
    loop.run_forever()


def cancellation_during_task_creation_is_applied_when_the_task_is_published(state):
    cancel_during_creation = state.cancel_during_creation
    entered = state.entered
    loop = state.loop
    original_create_task = loop.create_task

    def create_task(*args, **kwargs):
        task = original_create_task(*args, **kwargs)
        cancel_during_creation()
        return task

    loop.create_task = create_task

    async def callback():
        entered.append("entered")

    state.callback = callback


def bridge_context():
    import contextvars

    job = contextvars.ContextVar("async_bridge_job")
    return SimpleNamespace(job=job)


def verify_async_bridge(state):
    import asyncio

    conversion_failure = state.conversion_failure
    failure = state.failure
    job = state.job
    read_context = state.read_context
    value = state.value

    async def verify():
        assert await value(True) == 42
        assert await value(False) is None
        for function, message in (
            (failure, "bridge failure"),
            (conversion_failure, "conversion failure"),
        ):
            try:
                await function()
            except ValueError as exc:
                assert str(exc) == message
            else:
                raise AssertionError("the Rust error was lost")
        job.set("first")
        first = read_context()
        job.set("second")
        second = read_context()
        assert await first == "first"
        assert await second == "second"
        assert job.get() == "second"

    asyncio.run(verify())


def cancelling_the_python_future_drops_the_pending_rust_future(state):
    dropped = state.dropped
    pending = state.pending
    started = state.started
    import asyncio

    async def verify():
        future = pending()
        while not started():
            await asyncio.sleep(0)
        assert not dropped()
        future.cancel()
        try:
            await future
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation did not reach Python")
        while not dropped():
            await asyncio.sleep(0)

    async def main():
        await asyncio.wait_for(verify(), timeout=5)

    asyncio.run(main())


def synthesis_metadata_is_a_complete_conversion_and_reads_the_frame_once():
    import time

    from livekit import rtc
    from livekit.agents.types import (
        USERDATA_TIMED_TRANSCRIPT,
        USERDATA_TTS_STARTED_TIME,
        TranscriptChunk,
    )

    frame = rtc.AudioFrame.create(16000, 1, 160)
    frame.userdata[USERDATA_TIMED_TRANSCRIPT] = [
        TranscriptChunk("hello", start_time=0.1, end_time=0.2),
    ]
    frame.userdata[USERDATA_TTS_STARTED_TIME] = time.perf_counter() - 0.25

    class Chunk:
        request_id = "request"
        is_final = True
        segment_id = "segment"
        delta_text = "hello"
        usage = None
        characters_count = 5
        reads = 0

        @property
        def frame(self):
            self.reads += 1
            return frame

    chunk = Chunk()
    return SimpleNamespace(chunk=chunk)


def generated_recorder_exposes_the_native_tap_and_finishes_its_output(recorder):
    import asyncio

    async def finish():
        await recorder.ready()
        await recorder.finish()

    asyncio.run(finish())


def generated_recorder_reports_filesystem_open_failures_as_oserror(recorder):
    import asyncio

    async def open_recording():
        await recorder.ready()

    asyncio.run(open_recording())


def a_tool_callback_without_a_tool_call_scope_is_refused_before_the_host(host):

    async def entered(invocation, arguments):
        host.seen["entered"] = True

    host.prepare_tool = entered


def install_stub_package():
    import sys
    import types

    for name in ("livekit", "livekit.agents", "livekit.agents.types", "livekit.rtc"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m

    generated = generated_module("types")
    sys.modules["livekit.agents.types"].NotGiven = generated.NotGiven
    sys.modules["livekit.agents.types"].NOT_GIVEN = generated.NOT_GIVEN

    class AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self._data = bytearray(data)
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel

        @property
        def data(self):
            return memoryview(self._data).cast("h")

    sys.modules["livekit.rtc"].AudioFrame = AudioFrame


def defaults_preserve_lookup_exception_identity_and_cause():
    cause = OSError("disk")
    failure = RuntimeError("lookup failed")
    failure.__cause__ = cause

    class Broken:
        @property
        def field(self):
            raise failure

        def __getitem__(self, key):
            raise failure

    broken = Broken()
    key_failure = KeyError("backend lookup failed")
    key_failure.__cause__ = cause

    class BrokenDict(dict):
        def __getitem__(self, key):
            raise key_failure

    broken_dict = BrokenDict(field=1, value="")
    return SimpleNamespace(
        cause=cause,
        failure=failure,
        broken=broken,
        key_failure=key_failure,
        broken_dict=broken_dict,
    )


def empty_object():
    class Empty:
        pass

    empty = Empty()
    return SimpleNamespace(empty=empty)


def large_video_snapshots_remain_isolated_from_mutable_host_aliases():
    owner = bytearray(b"x" * (1920 * 1080 * 4))
    alias = memoryview(owner).toreadonly()
    return SimpleNamespace(owner=owner, alias=alias)


def large_buffer_allocation_lets_other_python_threads_progress_before_the_snapshot():
    import threading
    import time

    state = SimpleNamespace(
        owner=bytearray(b"x" * (32 * 1024 * 1024)), armed=False, progressed=False
    )
    state.alias = memoryview(state.owner).toreadonly()
    started = threading.Event()

    def run():
        started.set()
        while not state.armed:
            time.sleep(0.001)
        state.progressed = True

    state.thread = threading.Thread(target=run)
    state.thread.start()
    started.wait()
    return state


def bounded_buffers_refuse_oversized_host_buffers_before_copying():
    immutable = b"x" * (1920 * 1080 * 4)
    mutable = memoryview(bytearray(immutable)).toreadonly()
    return SimpleNamespace(immutable=immutable, mutable=mutable)


def retained_buffers_keep_immutable_owners_and_snapshot_writable_aliases():
    from array import array
    from ctypes import c_short

    immutable = b"1234"
    owner = bytearray(b"1234")
    alias = memoryview(owner).toreadonly()
    samples = memoryview(array("h", [1, -2]))
    samples_le = memoryview((c_short.__ctype_le__ * 2)(1, -2))
    wrong_format = memoryview(array("f", [1.0]))
    wrong_layout = memoryview(owner)[::2]
    return SimpleNamespace(
        immutable=immutable,
        owner=owner,
        alias=alias,
        samples=samples,
        samples_le=samples_le,
        wrong_format=wrong_format,
        wrong_layout=wrong_layout,
    )


def a_failing_property_is_not_an_absent_attribute():
    failure = AttributeError("descriptor failed")

    class Broken:
        @property
        def field(self):
            raise failure

    broken = Broken()
    return SimpleNamespace(failure=failure, broken=broken)


def unicode_float_and_newtype_boundaries_keep_their_declared_semantics():
    value = "\ud800"
    return SimpleNamespace(value=value)


def a_date_crosses_both_ways_and_an_impossible_one_is_refused():
    from types import SimpleNamespace

    out = SimpleNamespace(year=2024, month=2, day=30)
    return SimpleNamespace(out=out)


def an_audio_frame_snapshots_its_buffer_both_ways():
    from types import SimpleNamespace

    class Samples(bytearray):
        def tobytes(self):
            raise AssertionError("must use buffer protocol")

    samples = Samples(b"\x01\x00\x02\x00")
    frame = SimpleNamespace(data=samples, sample_rate=16000, num_channels=1, samples_per_channel=2)
    return SimpleNamespace(samples=samples, frame=frame)


def a_video_frame_snapshots_a_sliced_mutable_buffer():
    from types import SimpleNamespace

    owner = bytearray(b"01234567")
    frame = SimpleNamespace(data=memoryview(owner)[2:6], width=1, height=1, type=0)
    return SimpleNamespace(owner=owner, frame=frame)


def logging_handler():
    import logging

    seen = []

    class H(logging.Handler):
        def emit(self, r):
            seen.append(r.getMessage())

    logging.getLogger("livekit.agents.logtest").addHandler(H())
    logging.getLogger("livekit.agents.logtest").setLevel(logging.INFO)
    return SimpleNamespace(seen=seen)


def install_node_types():
    import sys
    import types

    transcript = generated_module("transcript")

    class AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self.data = data
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel
            self.userdata = {}

    sys.modules.setdefault("livekit", types.ModuleType("livekit"))
    sys.modules.setdefault("livekit.agents", types.ModuleType("livekit.agents"))
    public = sys.modules.setdefault(
        "livekit.agents.types", types.ModuleType("livekit.agents.types")
    )
    public.TranscriptChunk = transcript.TranscriptChunk
    rtc = sys.modules.setdefault("livekit.rtc", types.ModuleType("livekit.rtc"))
    rtc.AudioFrame = AudioFrame


def text_node_cancellation_retains_its_binding_until_python_finally_finishes(state):
    entered = threading.Event()
    closing = threading.Event()
    cleaned = threading.Event()
    close_gate = asyncio.Event()

    class Processing:
        async def transcription(self, text):
            try:
                while (word := await text.recv()) is not None:
                    yield word
                entered.set()
                await asyncio.Future()
            finally:
                closing.set()
                await close_gate.wait()
                cleaned.set()

    processing = Processing()
    state.entered = entered
    state.closing = closing
    state.cleaned = cleaned
    state.close_gate = close_gate
    state.processing = processing


def coroutine_node_receives_its_input_before_returning(state):
    consumed = []

    async def node(audio):
        while (frame := await audio.recv()) is not None:
            consumed.append(frame.samples_per_channel)

    class Processing:
        async def stt(self, audio):
            if (await node(audio)) is not None:
                yield

    processing = Processing()
    state.consumed = consumed
    state.processing = processing


def plugin_error():
    try:
        from livekit.agents import APIStatusError
    except ImportError:
        APIStatusError = Exception

    class MyStatusError(APIStatusError):
        def __init__(self):
            super().__init__("rate limited")
            self.retryable = True
            self.status_code = 429

        def __str__(self):
            return "rate limited"

    return MyStatusError()


def hostile_error():
    class Hostile(Exception):
        def __str__(self):
            raise ValueError("nope")

    return Hostile()


def divide_by_zero():
    return 1 / 0


def steering_case(case):
    return {
        "default": {},
        "null_fields": {"disfluencies": None, "nonverbal_sounds": None, "pace": None},
        "full": {
            "disfluencies": False,
            "nonverbal_sounds": {"laughing": False, "other": 1},
            "pace": "slow",
        },
        "enabled": {"nonverbal_sounds": True},
        "disabled": {"nonverbal_sounds": False},
        "invalid_pace": {"pace": "brisk"},
        "invalid_sounds": {"nonverbal_sounds": 3},
    }[case]


def agent_record_case(case):
    from livekit.agents._generated import agent_run as models

    if case == "accepted":
        return models.ValidatedOutputAccepted(text=None, value=None)
    if case == "dict":
        return {"kind": "retry", "reason": "no"}
    if case == "null_question":
        return models.ReadBackAnswerInputRequired(question=None, verbatim=False)
    if case == "default_retries":
        return models.Retries()
    if case == "zero_output":
        return models.Retries(output=0)
    if case == "null_tools":
        return models.Retries(tools=None)
    raise AssertionError(case)


def omission_value(case):
    from livekit.agents.types import NotGiven

    return {"not_given": NotGiven(), "none": None, "zero": 0.0, "invalid": "x"}[case]


def tool_spec(case):
    spec = {
        "name": "lookup",
        "description": "look something up",
        "json_schema": {"type": "object", "properties": {}},
        "ignore_on_enter": True,
    }
    if case.startswith("null_"):
        key = {
            "null_schema": "json_schema",
            "null_description": "description",
            "null_ignore": "ignore_on_enter",
        }[case]
        spec[key] = None
    elif case.startswith("missing_"):
        key = {
            "missing_schema": "json_schema",
            "missing_description": "description",
            "missing_ignore": "ignore_on_enter",
        }[case]
        del spec[key]
    else:
        assert case == "full"
    return spec


def vad_event(kind, count):
    class AudioFrame:
        def __init__(self):
            self.sample_rate = 24000
            self.num_channels = 1
            self.samples_per_channel = 4
            self.data = memoryview(bytes(8))

    return SimpleNamespace(
        type=kind,
        frames=[AudioFrame() for _ in range(count)],
        samples_index=0,
        timestamp=0.0,
        speech_duration=0.0,
        silence_duration=0.0,
        probability=0.9,
        inference_duration=0.0,
        speaking=False,
        raw_accumulated_silence=0.0,
        raw_accumulated_speech=0.0,
    )


def detector_prediction(models_path):
    import importlib.util
    import sys
    import types

    for name in ("livekit", "livekit.agents", "livekit.agents._generated"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location(
        "livekit.agents._generated.turn_events", models_path
    )
    models = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = models
    spec.loader.exec_module(models)
    prediction = models.EotPrediction(
        end_of_turn_probability=0.0,
        last_speaking_time=12.5,
        detection_delay=0.0,
        inference_duration=None,
        backchannel_probability=0.0,
    )
    return prediction
