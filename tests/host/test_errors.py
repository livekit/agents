"""Exceptions crossing the Python <-> native boundary: a Python failure reaches the caller as
itself with its cause, a native failure as its documented type, and cleanup failures never hide
the primary one."""

from __future__ import annotations

import asyncio
import multiprocessing
import pickle
import traceback
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    Agent,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    JobContext,
    ModelSettings,
    RunContext,
    TextSession,
    VoiceSession,
    _core,
    inference,
    llm,
    stt,
    tokenize,
    tts,
    utils,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.ipc import JobLauncher, JobRunnerOptions
from livekit.agents.llm import Toolset
from livekit.agents.storage import StorageUnavailable
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.events import ErrorEvent
from livekit.agents.voice.io import AudioOutputError
from tests.fakes import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeAudioStage,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    audio_frame,
    tool_call,
    until,
)

from .a2a_door import message_send
from .conftest import A2AHandler

REASON = "https://livekit.io/a2a/ext/agents/v1/reason"
NO_RETRY = APIConnectOptions(max_retry=1, retry_interval=0)


# -- Python failures reach the caller as themselves


async def test_a_provider_exception_reaches_the_caller_as_itself_with_its_cause() -> None:
    """The exception a Python LLM raises comes out of `agent.run` as the same object, its cause
    kept and rendered. https://github.com/livekit/agents/pull/6894"""
    failure = APIConnectionError("request failed", kind="invalid_request")
    failure.__cause__ = ValueError("upstream refused the request")

    with pytest.raises(APIConnectionError) as raised:
        await Agent(instructions="Help.", llm=FakeLLM([failure])).run("hello")

    assert raised.value is failure
    assert "ValueError: upstream refused the request" in str(raised.value)
    assert "upstream refused" in "".join(traceback.format_exception(raised.value))
    assert pickle.loads(pickle.dumps(raised.value)).message == "request failed"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "auth"),
        (403, "auth"),
        (402, "quota"),
        (408, "timeout"),
        (429, "rate_limit"),
        (400, "invalid_request"),
        (404, "invalid_request"),
        (500, "server"),
        (503, "server"),
    ],
)
def test_a_status_names_the_kind_that_decides_the_retry(status: int, kind: str) -> None:
    """A 401 or 402 is final however the plugin built it (they used to be retried), a 408, 429
    or 5xx is retried, a provider's own meaning overrides the status's, and a broken provider
    contract is final."""
    error = APIStatusError("refused", status_code=status)

    assert (error.kind, error.retryable) == (kind, kind in ("timeout", "rate_limit", "server"))
    overridden = APIStatusError("out of credit", status_code=400, kind="quota")
    assert (overridden.kind, overridden.retryable) == ("quota", False)
    broken = APIError("the provider sent a malformed frame", kind="protocol", detail="{")
    assert (broken.retryable, broken.detail) == (False, "{")


async def test_a_text_session_preserves_the_provider_status() -> None:
    """A provider's 401 surfaces from a `TextSession` turn as an `APIStatusError` with its code."""
    denied = APIStatusError("denied", status_code=401)
    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM([denied]))

    with pytest.raises(APIStatusError) as raised:
        await TextSession(agent=agent).run("Start")

    assert raised.value.status_code == 401


def test_a_tokenizer_exception_comes_back_as_the_same_object() -> None:
    """A Python tokenizer called from the native token buffer raises its own exception."""
    failure = ValueError("tokenizer failed")

    def tokenizer(text: str) -> list[str]:
        raise failure

    stream = tokenize.BufferedSentenceStream(tokenizer=tokenizer, min_token_len=1, min_ctx_len=1)

    with pytest.raises(ValueError) as raised:
        stream.push_text("abc def. ghi")

    assert raised.value is failure


async def test_exhausted_llm_retries_fail_with_the_last_provider_error_as_cause() -> None:
    """Retries that all fail end in `APIConnectionError` naming the attempts, caused by the last
    failure; each failure is reported, only the last as unrecoverable."""
    failure = APIError("the provider failed", kind="server")
    model = FakeLLM([failure, failure])
    reports: list[bool] = []
    model.on("error", lambda event: reports.append(event.recoverable))

    with pytest.raises(APIConnectionError, match="after 2 attempts") as raised:
        async with model.chat(chat_ctx=llm.ChatContext.empty(), conn_options=NO_RETRY) as stream:
            await stream.collect()

    assert raised.value.__cause__ is failure
    assert reports == [True, False]


async def test_exhausted_stt_stream_retries_fail_with_the_last_provider_error_as_cause() -> None:
    """A streaming Python STT that fails every attempt ends in `APIConnectionError` caused by
    its own last failure."""
    failure = APIError("the provider failed", kind="server")
    provider = FakeSTT([failure, failure])

    with pytest.raises(APIConnectionError, match="after 2 attempts") as raised:
        async with provider.stream(conn_options=NO_RETRY) as stream:
            stream.end_input()
            async for _ in stream:
                pass

    assert raised.value.__cause__ is failure


def test_a_connection_error_names_the_transport_failure_behind_it() -> None:
    """A provider's connection error raised from a socket failure names that failure.

    https://github.com/livekit/agents/pull/6696
    """
    with pytest.raises(APIConnectionError) as raised:
        try:
            raise ConnectionResetError("connection reset by peer")
        except OSError as error:
            raise APIConnectionError(kind="invalid_request") from error

    assert str(raised.value) == (
        "Connection error. (caused by ConnectionResetError: connection reset by peer)"
    )


async def test_a_timeout_keeps_its_phase_through_retries() -> None:
    """A provider timeout that exhausts the retries is raised as itself, with its phase."""
    timeout = APITimeoutError("response stalled", timeout="read")
    model = FakeLLM([timeout, timeout])

    with pytest.raises(APITimeoutError) as raised:
        async with model.chat(chat_ctx=llm.ChatContext.empty(), conn_options=NO_RETRY) as stream:
            await stream.collect()

    assert raised.value is timeout
    assert raised.value.timeout == "read"


async def test_a_failure_after_output_is_final_and_not_retried() -> None:
    """Once text reached the caller, a provider failure is raised as is, never retried."""
    failure = APIError("failed after output", kind="server")
    model = FakeLLM([["partial", failure], "a retry must not happen"])
    texts: list[str] = []

    with pytest.raises(APIError) as raised:
        async with model.chat(
            chat_ctx=llm.ChatContext.empty(), conn_options=APIConnectOptions(max_retry=2)
        ) as stream:
            async for chunk in stream:
                texts.append(chunk.delta.content if chunk.delta and chunk.delta.content else "")

    assert raised.value is failure
    assert texts == ["partial"]


async def test_a_request_retried_to_success_reports_each_failure_as_recoverable() -> None:
    """A Python LLM that fails twice and then answers gives its answer; both failures are
    reported, as recoverable."""
    failure = APIError("the provider failed", kind="server")
    model = FakeLLM([failure, failure, "Hello."])
    reports: list[bool] = []
    model.on("error", lambda event: reports.append(event.recoverable))

    retries = APIConnectOptions(max_retry=2, retry_interval=0)
    async with model.chat(chat_ctx=llm.ChatContext.empty(), conn_options=retries) as stream:
        reply = await stream.collect()

    assert reply.text == "Hello."
    assert reports == [True, True]


async def test_a_retried_tts_stream_is_fed_again_the_text_pushed_before_its_failure() -> None:
    """A Python TTS stream that fails is retried with the text pushed before the failure, and
    that retry speaks."""
    provider = FakeTTS(failures=[APIError("the provider failed", kind="server")])

    async with provider.stream(conn_options=NO_RETRY) as stream:
        stream.push_text("Hello ")
        stream.push_text("there.")
        stream.end_input()
        frames = [event.frame async for event in stream]

    assert provider.texts == ["Hello there."]
    assert sum(frame.duration for frame in frames) > 0


async def test_a_recognition_that_fails_once_is_retried() -> None:
    """A Python STT's `recognize` that fails once is retried and returns the transcript."""
    provider = FakeSTT([APIError("the provider failed", kind="server"), "hello"], streaming=False)

    event = await provider.recognize([audio_frame(0.2)], conn_options=NO_RETRY)

    assert event.alternatives[0].text == "hello"


async def test_a_recognition_that_fails_for_good_is_not_retried() -> None:
    """A Python STT's `recognize` that raises a non-retryable error raises it at once instead of
    asking the provider again."""
    failure = APIError("the provider refused", kind="invalid_request")
    provider = FakeSTT([failure, "hello"], streaming=False)

    with pytest.raises(APIError) as raised:
        await provider.recognize([audio_frame(0.2)], conn_options=NO_RETRY)

    assert raised.value is failure


async def test_exhausted_recognition_retries_fail_with_the_last_provider_error_as_cause() -> None:
    """A Python STT's `recognize` that fails every attempt ends in `APIConnectionError` caused
    by its own last failure."""
    failure = APIError("the provider failed", kind="server")
    provider = FakeSTT([failure, failure], streaming=False)

    with pytest.raises(APIConnectionError, match="after 2 attempts") as raised:
        await provider.recognize([audio_frame(0.2)], conn_options=NO_RETRY)

    assert raised.value.__cause__ is failure


async def test_a_fallback_reports_and_raises_the_providers_own_exception() -> None:
    """A failure after a transcript is final: the STT fallback raises the original exception
    and reports that same object once."""
    failure = APIError("the provider's own failure", kind="invalid_request")
    provider = FakeSTT(["hello", failure])
    reports: list[Exception] = []
    provider.on("error", lambda event: reports.append(event.error))
    fallback = stt.FallbackAdapter([provider], max_retry_per_stt=0)

    with pytest.raises(APIError) as raised:
        async with fallback.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_frame(audio_frame(0.2))
            stream.push_frame(audio_frame(0.3, level=0))
            stream.end_input()
            async for _ in stream:
                pass
    await fallback.aclose()

    assert raised.value is failure
    assert reports == [failure]


# -- native failures reach Python as their documented type


@pytest.fixture
async def rejecting_gateway() -> AsyncIterator[str]:
    """An inference gateway that refuses every request with a 401."""

    async def reject(request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid api key"}, status=401)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", reject)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
    await runner.cleanup()


async def test_a_native_provider_rejection_is_an_api_status_error(rejecting_gateway: str) -> None:
    """The native inference client's HTTP failure reaches Python as `APIStatusError`."""
    model = inference.LLM(
        "openai/gpt-4.1-mini", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )

    with pytest.raises(APIStatusError) as raised:
        await Agent(instructions="Help.", llm=model).run("hello")
    await model.aclose()

    assert raised.value.status_code == 401


async def test_a_corrupt_container_fails_the_synthesis_at_join() -> None:
    """Audio the native decoder cannot read fails `join()` instead of ending in silence, so a
    fallback chain can act on it."""
    emitter = tts.AudioEmitter(label="corrupt", dst_ch=utils.aio.Chan[tts.SynthesizedAudio]())
    emitter.initialize(request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/ogg")
    emitter.push(b"\x5a" * 4096)
    emitter.end_input()

    with pytest.raises(RuntimeError, match="decode failed"):
        await emitter.join()
    await emitter.aclose()


async def test_a_storage_failure_in_a_turn_fails_its_task_as_storage_unavailable(
    a2a_door: Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]],
) -> None:
    """A turn whose save cannot be written fails its A2A task with the `storage-unavailable`
    reason, so the client resends."""

    async def handler(ctx: A2ASessionContext) -> str:
        raise StorageUnavailable("the database went away")

    async with a2a_door(handler) as door:
        response = await door.post("/v1", json=message_send("hi"))

    status = response.json()["result"]["task"]["status"]
    assert status["state"] == "TASK_STATE_FAILED"
    assert status["message"]["metadata"][REASON] == "storage-unavailable"


async def _entrypoint(ctx: JobContext) -> None:
    raise AssertionError("the launch fails before a job runs")


async def test_an_unpicklable_entrypoint_fails_the_process_launch_with_its_cause() -> None:
    """Under `spawn`, a job process launch whose entrypoint cannot be pickled raises the
    pickling error itself, and the runner still closes."""

    async def local_entrypoint(ctx: JobContext) -> None:
        await _entrypoint(ctx)

    launcher = JobLauncher(
        job_entrypoint_fnc=local_entrypoint,
        session_end_fnc=None,
        simulation_end_fnc=None,
        session_end_timeout=0.25,
        mp_ctx=multiprocessing.get_context("spawn"),
        loop=asyncio.get_running_loop(),
    )
    runner = _core.JobSupervisor(launcher, JobRunnerOptions(initialize_timeout=10, close_timeout=1))

    # Python 3.14 raises PicklingError where earlier versions raise AttributeError.
    with pytest.raises((AttributeError, pickle.PicklingError), match="local object"):
        await runner.start()
    await runner.aclose()
    await runner.join()

    assert runner.pid is None


# -- a primary failure survives cleanup failures


class FailingClose(Toolset):
    """A toolset whose close raises."""

    async def aclose(self) -> None:
        await super().aclose()
        raise ValueError("cleanup failed")


@pytest.mark.parametrize("entrypoint", ["agent.run", "TextSession.run"])
async def test_a_setup_error_survives_a_toolset_cleanup_error(entrypoint: str) -> None:
    """A run that fails to start (no LLM) raises that error even when its toolset's close
    fails too."""
    agent = Agent(name="agent", instructions="Help.", tools=[FailingClose(id="resource")])

    with pytest.raises(RuntimeError, match="has no LLM"):
        if entrypoint == "agent.run":
            await agent.run("Hello")
        else:
            await TextSession(agent=agent).run("Hello")


async def test_an_exception_in_the_stream_block_stays_primary_over_cleanup_errors() -> None:
    """Raising inside `run_stream` re-raises that exception after cancelling the tool body; the
    toolset's close failure is what `result()` reports."""
    entered, cancelled = asyncio.Event(), asyncio.Event()
    agent = Agent(
        instructions="Help.",
        tools=[FailingClose(id="resource")],
        llm=FakeLLM([tool_call("wait")]),
    )

    @agent.tool
    async def wait(ctx: RunContext) -> str:
        """Wait for the application."""
        entered.set()
        try:
            await asyncio.Future[None]()
        finally:
            cancelled.set()
        return "unreachable"

    body = LookupError("body failed")
    with pytest.raises(LookupError) as raised:
        async with agent.run_stream("start") as stream:
            await entered.wait()
            raise body

    assert raised.value is body
    assert cancelled.is_set()
    with pytest.raises(ValueError, match="cleanup failed"):
        await stream.result()


async def test_a_failing_toolset_close_lets_its_siblings_close_and_is_reported_once() -> None:
    """One toolset failing to close does not cut its siblings' closes short; `result()` raises
    that same exception each time it is asked."""
    closed: list[str] = []
    failed = asyncio.Event()

    class Failing(FailingClose):
        async def aclose(self) -> None:
            try:
                await super().aclose()
            finally:
                failed.set()

    class SlowClose(Toolset):
        async def aclose(self) -> None:
            # still closing when its sibling's close fails
            await until(failed.is_set, "the failing toolset closed")
            closed.append(self.id)

    agent = Agent(
        instructions="Help.",
        tools=[Failing(id="failing"), SlowClose(id="slow")],
        llm=FakeLLM(["Hello"]),
    )

    async with agent.run_stream("hi") as stream:
        with pytest.raises(ValueError, match="cleanup failed") as first:
            await stream.result()
        with pytest.raises(ValueError) as second:
            await stream.result()

    assert second.value is first.value
    assert closed == ["slow"]


async def test_a_failed_session_start_releases_the_toolsets_already_set_up() -> None:
    """A start that fails on one toolset's setup raises that failure and closes the toolsets
    that were already set up."""
    failure = ValueError("the database is down")
    closed: list[str] = []

    class Database(Toolset):
        async def setup(self) -> Database:
            if self.id == "down":
                raise failure
            return await super().setup()

        async def aclose(self) -> None:
            closed.append(self.id)
            await super().aclose()

    session = VoiceSession(vad=None, tools=[Database(id="up"), Database(id="down")])

    with pytest.raises(ValueError) as raised:
        await session.start(Agent(instructions="Help."))
    await session.aclose()

    assert raised.value is failure
    assert closed == ["up"]


async def test_an_on_exit_failure_does_not_abort_the_session_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An `on_exit` that raises is logged; the session still closes. https://github.com/livekit/agents/pull/4826"""
    agent = Agent(instructions="Help.")
    exited: list[bool] = []

    @agent.on_exit
    async def on_exit(session: VoiceSession) -> None:
        exited.append(True)
        raise ValueError("exit callback failed")

    async with VoiceSession(llm=FakeLLM()) as session:
        await session.start(agent)

    assert exited == [True]
    assert "exit callback failed" in caplog.text


# -- a failing audio output


async def test_an_audio_output_raising_mid_speech_fails_only_that_speech() -> None:
    """A Python audio output that raises while playing ends that speech interrupted with the
    exception, reported as an `AudioOutputError`; the next speech plays."""
    output = FakeAudioOutput(fail_at_frame=3)
    session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.5))
    session.output.audio = output
    errors: list[ErrorEvent] = []
    session.on("error", errors.append)

    async with session:
        await session.start(Agent(instructions="Help."))
        first = await session.say("The first sentence here.")
        second = await session.say("The second one.")

    failure = first.exception()
    assert first.interrupted and isinstance(failure, RuntimeError)
    [error] = errors
    assert isinstance(error.error, AudioOutputError) and error.error.error is failure
    assert second.exception() is None and not second.interrupted


async def test_an_output_failing_below_a_python_stage_is_reported_at_the_stage() -> None:
    """An output below a user's stage that raises on its first frame is reported as the stage's
    recoverable error, and the next speech plays through both."""
    sink = FakeAudioOutput(fail_at_frame=0)
    stage = FakeAudioStage(sink)
    session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.5))
    session.output.audio = stage
    errors: list[ErrorEvent] = []
    session.on("error", errors.append)

    async with session:
        await session.start(Agent(instructions="Help."))
        first = await session.say("The first sentence here.")
        second = await session.say("The second one.")

    assert first.interrupted and isinstance(first.exception(), RuntimeError)
    [error] = errors
    assert isinstance(error.error, AudioOutputError) and error.error.recoverable
    assert error.source is stage
    assert second.exception() is None and not second.interrupted
    assert sink.frames > 1


async def test_a_native_provider_failure_in_a_session_is_its_error_event(
    rejecting_gateway: str,
) -> None:
    """A native inference LLM's failed request in a session emits the provider's ``error``."""
    model = inference.LLM(
        "openai/gpt-4.1-mini", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[llm.LLMError] = []
    model.on("error", errors.append)

    async with VoiceSession(llm=model, vad=None) as session:
        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="hello")
    await model.aclose()

    [error] = errors
    assert isinstance(error.error, APIStatusError) and error.error.status_code == 401
    assert not error.recoverable


async def test_a_native_stt_failure_in_a_session_is_its_error_event(rejecting_gateway: str) -> None:
    """A native inference STT's refused connection in a session emits the provider's ``error``."""
    model = inference.STT(
        "deepgram/nova-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[stt.STTError] = []
    model.on("error", errors.append)
    microphone = FakeAudioInput()

    async with VoiceSession(stt=model, llm=FakeLLM(["Hi!"])) as session:
        session.input.audio = microphone
        await session.start(Agent(instructions="Help."))
        microphone.speak(0.5)
        await until(lambda: any(not e.recoverable for e in errors), "no unrecoverable STT error")
    await model.aclose()

    assert all(isinstance(e.error, APIStatusError) and e.error.status_code == 401 for e in errors)


async def test_a_native_tts_failure_in_a_session_is_its_error_event(rejecting_gateway: str) -> None:
    """A native inference TTS's refused synthesis in a session emits the provider's ``error``."""
    model = inference.TTS(
        "cartesia/sonic-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[tts.TTSError] = []
    model.on("error", errors.append)

    async with VoiceSession(tts=model, vad=None) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="Help."))
        await session.say("Hello there.")
    await model.aclose()

    assert errors and not errors[-1].recoverable
    assert all(isinstance(e.error, APIStatusError) and e.error.status_code == 401 for e in errors)


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("streamed", [False, True])
async def test_a_native_tts_request_reports_its_final_error_once(
    status: int, streamed: bool
) -> None:
    """``synthesize()`` and ``stream()`` over a native TTS report each retried attempt as a
    recoverable ``error`` and the error they raise as the one final event."""
    requests = 0

    async def reject(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.json_response({"error": "refused"}, status=status)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", reject)
    async with TestServer(app) as server:
        model = inference.TTS(
            "cartesia/sonic-3",
            base_url=str(server.make_url("/v1")),
            api_key="key",
            api_secret="secret",
        )
        events: list[tts.TTSError] = []
        model.on("error", events.append)
        if streamed:
            streaming = model.stream(conn_options=NO_RETRY)
            streaming.push_text("Hello there.")
            streaming.end_input()
            stream: tts.SynthesizeStream | tts.ChunkedStream = streaming
        else:
            stream = model.synthesize("Hello there.", conn_options=NO_RETRY)
        with pytest.raises(APIStatusError) as raised:
            _ = [audio async for audio in stream]
        await stream.aclose()
        await model.aclose()

    assert (raised.value.status_code, raised.value.retryable) == (status, status == 429)
    assert requests == (2 if status == 429 else 1)
    assert [e.recoverable for e in events] == [True] * (requests - 1) + [False]
    assert all(isinstance(e.error, APIStatusError) for e in events)


# -- input after a provider's failure raises that failure


def _refused(failure: type[APIStatusError], *calls: Callable[[], object]) -> None:
    for call in calls:
        with pytest.raises(failure) as refused:
            call()
        assert refused.value.status_code == 401, call


@pytest.mark.parametrize("native", [True, False], ids=["inference", "plugin"])
async def test_input_after_a_tts_failure_raises_that_failure(
    native: bool, rejecting_gateway: str
) -> None:
    """Text pushed, flushed or ended after the voice failed raises the voice's own error with its
    status, never a generic closed-input error, and is never swallowed."""
    model: tts.TTS = (
        inference.TTS(
            "cartesia/sonic-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
        )
        if native
        else FakeTTS(error=APIStatusError("refused", status_code=401))
    )

    async with model.stream(conn_options=NO_RETRY) as stream:
        stream.push_text("Hello there.")
        stream.flush()
        with pytest.raises(APIStatusError):
            _ = [audio async for audio in stream]
        _refused(APIStatusError, lambda: stream.push_text("Again."), stream.flush, stream.end_input)
    await model.aclose()


@pytest.mark.parametrize("native", [True, False], ids=["inference", "plugin"])
async def test_input_after_an_stt_failure_raises_that_failure(
    native: bool, rejecting_gateway: str
) -> None:
    """Audio pushed, flushed or ended after the recognizer failed raises the recognizer's own
    error with its status, never a generic closed-input error."""
    model: stt.STT = (
        inference.STT(
            "deepgram/nova-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
        )
        if native
        else FakeSTT([APIStatusError("refused", status_code=401)])
    )

    async with model.stream(conn_options=NO_RETRY) as stream:
        with pytest.raises(APIStatusError):
            _ = [event async for event in stream]
        _refused(
            APIStatusError,
            lambda: stream.push_frame(audio_frame(0.1)),
            stream.flush,
            stream.end_input,
        )
    await model.aclose()


async def test_input_after_a_vad_failure_raises_that_failure() -> None:
    """Audio pushed, flushed or ended after the detector failed raises the detector's own error,
    not that the stream is closed."""
    detector = FakeVAD(error=APIStatusError("refused", status_code=401))

    async with detector.stream() as stream:
        with pytest.raises(APIStatusError):
            _ = [event async for event in stream]
        _refused(
            APIStatusError,
            lambda: stream.push_frame(audio_frame(0.1)),
            stream.flush,
            stream.end_input,
        )


# A node that calls the session's default must still run a native provider natively: handed its
# Python wrapper instead, the core runs the wrapper as a plugin, whose provider hooks a native
# provider does not have (a `tts_node` over `inference.TTS` failed every reply with
# "streaming is not supported by this TTS"). The rejecting gateway's 401 proves the native
# provider made the request.


async def test_a_tts_node_over_the_default_runs_the_native_tts(rejecting_gateway: str) -> None:
    model = inference.TTS(
        "cartesia/sonic-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[tts.TTSError] = []
    model.on("error", errors.append)

    async with VoiceSession(tts=model, vad=None) as session:
        session.output.audio = FakeAudioOutput()

        @session.tts_node
        def node(ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings) -> Any:
            return session.default.tts_node(ctx, text, model_settings)

        await session.start(Agent(instructions="Help."))
        await session.say("Hello there.")
    await model.aclose()

    assert errors
    assert all(isinstance(e.error, APIStatusError) and e.error.status_code == 401 for e in errors)


async def test_agent_run_runs_the_native_llm(rejecting_gateway: str) -> None:
    """`agent.run()` hands the core the gateway LLM itself, not its Python wrapper."""
    model = inference.LLM(
        "openai/gpt-4.1-mini", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    with pytest.raises(APIStatusError) as raised:
        await Agent(instructions="Help.", llm=model).run("hello")
    await model.aclose()

    assert raised.value.status_code == 401


@pytest.mark.parametrize("wrap", ["fallback", "stream_adapter"])
async def test_an_adapter_over_the_native_tts_runs_it_natively(wrap: str) -> None:
    """Behind an adapter, and through a `tts_node` over the default, the gateway TTS still makes
    its own request: the gateway sees it."""
    requests: list[str] = []

    async def reject(request: web.Request) -> web.Response:
        requests.append(request.path)
        return web.json_response({"error": "invalid api key"}, status=401)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", reject)
    async with TestServer(app) as gateway:
        model = inference.TTS(
            "cartesia/sonic-3", base_url=str(gateway.make_url("/v1")), api_key="k", api_secret="s"
        )
        voice = tts.FallbackAdapter([model]) if wrap == "fallback" else tts.StreamAdapter(tts=model)
        async with VoiceSession(tts=voice, vad=None) as session:
            session.output.audio = FakeAudioOutput()

            @session.tts_node
            def node(
                ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
            ) -> Any:
                return session.default.tts_node(ctx, text, model_settings)

            await session.start(Agent(instructions="Help."))
            await session.say("Hello there.")
        await model.aclose()

    assert requests, "the gateway TTS never made a request"


@pytest.mark.parametrize(
    ("base", "hook", "node"),
    [
        (inference.LLM, "chat", "session.llm_node"),
        (inference.STT, "stream", "session.stt_node"),
        (inference.TTS, "_stream", "session.tts_node"),
    ],
)
def test_a_native_provider_subclass_replacing_a_hook_the_core_bypasses_is_refused(
    base: type, hook: str, node: str
) -> None:
    """The core runs a native provider itself: a subclass replacing a hook it would never call
    is refused as the class is defined, pointed to the node that customizes it."""

    def replaced(self: object, **kwargs: Any) -> Any:
        raise AssertionError("never called")

    with pytest.raises(TypeError, match=node):
        type("Replacing", (base,), {hook: replaced})


async def test_an_stt_node_over_the_default_runs_the_native_stt(rejecting_gateway: str) -> None:
    model = inference.STT(
        "deepgram/nova-3", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[stt.STTError] = []
    model.on("error", errors.append)
    microphone = FakeAudioInput()

    async with VoiceSession(stt=model, llm=FakeLLM(["Hi!"])) as session:
        session.input.audio = microphone

        @session.stt_node
        def node(
            ctx: RunContext, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> Any:
            return session.default.stt_node(ctx, audio, model_settings)

        await session.start(Agent(instructions="Help."))
        microphone.speak(0.5)
        await until(lambda: any(not e.recoverable for e in errors), "no unrecoverable STT error")
    await model.aclose()

    assert all(isinstance(e.error, APIStatusError) and e.error.status_code == 401 for e in errors)


async def test_an_llm_node_over_the_default_runs_the_native_llm(rejecting_gateway: str) -> None:
    model = inference.LLM(
        "openai/gpt-4.1-mini", base_url=rejecting_gateway, api_key="key", api_secret="secret"
    )
    errors: list[llm.LLMError] = []
    model.on("error", errors.append)

    async with VoiceSession(llm=model, vad=None) as session:

        @session.llm_node
        def node(
            ctx: RunContext,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings: ModelSettings,
        ) -> Any:
            return session.default.llm_node(ctx, chat_ctx, tools, model_settings)

        await session.start(Agent(instructions="Help."))
        await session.generate_reply(user_input="hello")
    await model.aclose()

    assert errors
    assert all(isinstance(e.error, APIStatusError) and e.error.status_code == 401 for e in errors)


# -- one retry owner per request, each failure reported once


async def test_a_native_stt_in_a_session_connects_once_per_attempt_and_reports_each_once() -> None:
    """The session owns the gateway STT's retries: it connects once per attempt of the session's
    policy (never the session's retries times the gateway's own), each retried attempt is one
    recoverable ``error`` and the give-up one final one."""
    requests = 0

    async def refuse(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.json_response({"error": "busy"}, status=429)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", refuse)
    async with TestServer(app) as server:
        model = inference.STT(
            "deepgram/nova-3",
            base_url=str(server.make_url("/v1")),
            api_key="key",
            api_secret="secret",
        )
        events: list[stt.STTError] = []
        model.on("error", events.append)
        microphone = FakeAudioInput()
        conn = SessionConnectOptions(
            stt_conn_options=APIConnectOptions(max_retry=2, retry_interval=0)
        )
        async with VoiceSession(stt=model, llm=FakeLLM(["Hi!"]), conn_options=conn) as session:
            session.input.audio = microphone
            await session.start(Agent(instructions="Help."))
            microphone.speak(0.5)
            await until(lambda: any(not e.recoverable for e in events), "no final STT error")
            assert requests == 3
        await model.aclose()

    assert [e.recoverable for e in events] == [True, True, False]


async def test_an_stt_fallback_runs_once_and_each_provider_reports_one_final_error() -> None:
    """``recognize()`` over a fallback runs the chain once: each member's attempt the chain moves
    past is a recoverable ``error`` on that member, the attempt it gives up after a final one, and
    the adapter reports the give-up once."""
    down = APIError("the provider is down", kind="server")
    members = [FakeSTT([down] * 8), FakeSTT([down] * 8)]
    reports: list[list[bool]] = [[], [], []]
    fallback = stt.FallbackAdapter(members, max_retry_per_stt=1, retry_interval=0)
    for provider, seen in zip([*members, fallback], reports, strict=True):
        provider.on("error", lambda event, seen=seen: seen.append(event.recoverable))

    with pytest.raises(APIConnectionError, match="all backends failed"):
        await fallback.recognize(
            [audio_frame(0.2)], conn_options=APIConnectOptions(max_retry=3, retry_interval=0)
        )
    await fallback.aclose()

    assert reports == [[True, True], [True, False], [False]]


async def test_an_llm_fallback_runs_once_and_each_provider_reports_one_final_error() -> None:
    """A run over an LLM fallback runs the chain once: each member's attempt the chain moves past
    is a recoverable ``error`` on that member, the attempt it gives up after a final one, and the
    adapter reports the give-up once."""
    down = APIError("the provider is down", kind="server")
    members = [FakeLLM([down] * 8), FakeLLM([down] * 8)]
    reports: list[list[bool]] = [[], [], []]
    fallback = llm.FallbackAdapter(members, max_retry_per_llm=1, retry_interval=0)
    for provider, seen in zip([*members, fallback], reports, strict=True):
        provider.on("error", lambda event, seen=seen: seen.append(event.recoverable))

    with pytest.raises(APIConnectionError, match="all backends failed"):
        await Agent(instructions="Help.", llm=fallback).run("hello")
    await fallback.aclose()

    assert reports == [[True, True], [True, False], [False]]
