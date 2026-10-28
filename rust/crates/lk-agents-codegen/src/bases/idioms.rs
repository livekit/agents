//! The Python idioms the provider bases are written from: each a template whose `{Name}`
//! placeholders the writer fills in.

pub(super) const ACONTEXT: &str = r#"
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
"#;

/// The recognition stream's idiom. `{Stream}`, `{Provider}` and `{Event}` are filled in.
pub(super) const RECOGNIZE: &str = r#"
    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(
        self,
        *,
        stt: {Provider},
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            stt: The provider this stream recognizes for.
            conn_options: The timeout and retries of the connection: a failed connection is
                reopened, the audio it had not taken read by the next.
            sample_rate: The rate the audio is resampled to before recognition; the input's own
                rate when not given.
        """
        self._stt = stt
        self._conn_options = conn_options
        self.input = aio.Chan[rtc.AudioFrame | {Stream}.Flush]()
        """The audio ``_run`` reads, a ``Flush`` at the end of each segment."""
        self._event_ch = aio.Chan[{Event}]()
        self._start_time = time.time()
        self._start_time_offset = 0.0
        self._context: BaseModel | None = None
        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None
        self._core_attempts = _bindings.Attempts("stt_stream", conn_options, self._emit_error)
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def context(self) -> BaseModel | None:
        """Speaker metadata the plugin updates as it learns about the speaker."""
        return self._context

    @context.setter
    def context(self, value: BaseModel | None) -> None:
        self._context = value

    @property
    def start_time_offset(self) -> float:
        """Where this stream's timestamps sit on the caller's input timeline, in seconds."""
        return self._start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._start_time_offset = value

    @property
    def start_time(self) -> float:
        """Wall-clock anchor of the stream; plugins may move it to when the first audio reached
        the provider, so stream-relative times convert accurately."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time must be non-negative")
        self._start_time = value

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        started = time.time()

        # each attempt moves the timeline on by the time the last one ran, so a reconnected
        # stream's timestamps stay linear
        async def attempt() -> None:
            nonlocal started
            now = time.time()
            self._start_time_offset += now - started
            self._start_time = started = now
            try:
                await self._run()
            except Exception:
                if _core.api_attempt_connected(
                    self._conn_options.connect_timeout, time.time() - started
                ):
                    self._core_attempts.count = 0
                raise

        await self._core_attempts.run(attempt)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt._emit_error(api_error, recoverable)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized."""
        self._check_input_not_ended()
        self._check_not_closed()
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")
        self._pushed_sr = frame.sample_rate
        if self._needed_sr and self._needed_sr != frame.sample_rate and not self._resampler:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._needed_sr, quality=rtc.AudioResamplerQuality.HIGH
            )
        for f in self._resampler.push(frame) if self._resampler else [frame]:
            self.input.send_nowait(f)

    def flush(self) -> None:
        """Mark the end of the current segment."""
        self._check_input_not_ended()
        self._check_not_closed()
        if self._resampler:
            for frame in self._resampler.flush():
                self.input.send_nowait(frame)
        self.input.send_nowait(self.Flush())

    def end_input(self) -> None:
        """Mark the end of input: no more audio will be pushed."""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately."""
        self.input.close()
        await aio.cancel_and_wait(self._task)

    async def __anext__(self) -> {Event}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        return val

    def __aiter__(self) -> AsyncIterator[{Event}]:
        return self

    def send(self, event: {Event}) -> None:
        """Hand ``event`` to the reader."""
        record, final = _core.stt_stream_metrics(event, _bindings.metrics_context(self._stt))
        if final:
            # a recognition succeeded: the retry budget starts over
            self._core_attempts.count = 0
        _bindings.emit_metrics(self._stt, "STTMetrics", record)
        self._event_ch.send_nowait(event)

    @property
    def closed(self) -> bool:
        """Whether the stream is closed: its reader takes nothing more."""
        return self._event_ch.closed

    def _check_not_closed(self) -> None:
        if self.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self.input.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
"#;

/// What both syntheses share: each attempt run against a fresh `AudioEmitter`, retried by
/// `_core_attempts`.
pub(super) const SYNTHESIS: &str = r#"
    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._tts._emit_error(api_error, recoverable)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None: ...

    async def _attempt(self) -> None:
        self._emitter = output_emitter = _core.AudioEmitter(
            label=self._tts.label, dst_ch=self._event_ch
        )
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if problem := self._invalid(output_emitter):
                raise APIError(problem)
        finally:
            await output_emitter.aclose()

    def _produced(self) -> bool:
        return self._emitter is not None and self._emitter.pushed_duration() > 0.0
"#;

/// The one-shot synthesis's idiom.
pub(super) const CHUNKED: &str = r#"
    def __init__(self, *, tts: {Provider}, input_text: str, conn_options: APIConnectOptions) -> None:
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        # the full text is submitted to the provider at creation time
        self._started_time: float = time.perf_counter()
        self._event_ch = aio.Chan[{Audio}]()
        self._emitter: _core.AudioEmitter | None = None
        self._metrics: _core.TtsMetricState | None = _core.TtsMetricState(False)
        self._core_attempts = _bindings.Attempts("tts_batch", conn_options, self._emit_error)
        self._synthesize_task = asyncio.create_task(
            self._core_attempts.run(self._attempt, produced=self._produced),
            name="TTS._synthesize_task",
        )
        self._synthesize_task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def input_text(self) -> str:
        return self._input_text

    @property
    def done(self) -> bool:
        return self._synthesize_task.done()

    @property
    def exception(self) -> BaseException | None:
        return self._synthesize_task.exception()
{SYNTHESIS}
    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(self._input_text, output_emitter.pushed_duration(), None)  # type: ignore[no-any-return]

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        return rtc.combine_audio_frames([ev.frame async for ev in self])

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.cancel_and_wait(self._synthesize_task)
        self._event_ch.close()

    async def __anext__(self) -> {Audio}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (exc := self._synthesize_task.exception()):
                raise exc  # noqa: B904
            if self._metrics is not None:
                record = self._metrics.finish(
                    time.perf_counter(),
                    self._started_time,
                    False,
                    self._input_text,
                    0,
                    0,
                    0.0,
                    False,
                    _bindings.metrics_context(self._tts),
                )
                self._metrics = None
                _bindings.emit_metrics(self._tts, "TTSMetrics", record)
            raise StopAsyncIteration from None
        if self._metrics is not None:
            self._metrics.observe(time.perf_counter(), self._started_time, val)
        val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        return val

    def __aiter__(self) -> AsyncIterator[{Audio}]:
        return self
"#;

/// The live synthesis's idiom.
pub(super) const SYNTHESIZE: &str = r#"
    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(self, *, tts: {Provider}, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self.input = aio.Chan[str | {Stream}.Flush]()
        """The text ``_run`` reads, a ``Flush`` at the end of each segment; a retry reads it
        again from the start."""
        self._input_log: list[str | {Stream}.Flush] = []
        self._event_ch = aio.Chan[{Audio}]()
        self._emitter: _core.AudioEmitter | None = None
        self._metrics = _core.TtsMetricState(True)
        self._segment_text = ""
        self._core_attempts = _bindings.Attempts("tts_stream", conn_options, self._emit_error)
        self._task = asyncio.create_task(
            self._core_attempts.run(self._attempt, produced=self._produced, again=self._replay),
            name="TTS._main_task",
        )
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._started_time: float = 0
        self._pushed_text: str = ""
        self._segment_open = False
        self._num_segments = 0
{SYNTHESIS}
    def _replay(self) -> None:
        ended = self.input.closed
        self.input = aio.Chan[str | {Stream}.Flush]()
        for item in self._input_log:
            self.input.send_nowait(item)
        if ended:
            self.input.close()

    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(  # type: ignore[no-any-return]
            self._pushed_text,
            output_emitter.pushed_duration(),
            (self._num_segments, output_emitter.num_segments),
        )

    def mark_started(self) -> None:
        """Note that the first text went to the provider: the time its audio is measured from."""
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if not token or self.input.closed:
            return
        if not self._segment_open:
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += token
        self._segment_text += token
        self._input_log.append(token)
        self.input.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self.input.closed:
            return
        self._segment_open = False
        if self._segment_text:
            self._metrics.queue_text(self._segment_text)
            self._segment_text = ""
        self._input_log.append(self.Flush())
        self.input.send_nowait(self._input_log[-1])

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
        self.input.close()

    async def __anext__(self) -> {Audio}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if self._started_time:
            val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        # a segment's last frame closes its metrics; the next segment is timed from its own start
        if self._metrics.observe(time.perf_counter(), self._started_time, val):
            record = self._metrics.finish(
                time.perf_counter(),
                self._started_time,
                False,
                "",
                0,
                0,
                0.0,
                False,
                _bindings.metrics_context(self._tts),
            )
            _bindings.emit_metrics(self._tts, "TTSMetrics", record)
            self._started_time = 0
        return val

    def __aiter__(self) -> AsyncIterator[{Audio}]:
        return self
"#;

/// `synthesize()` over the provider's stream: one push, then the end of input.
pub(super) const FROM_STREAM: &str = r#"

class _ChunkedStreamFromStream({Chunked}):
    """``synthesize()`` over a provider that only streams."""

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        async with self._tts.stream(conn_options=self._conn_options) as stream:
            stream._core_attempts.core = self._core_attempts.core
            stream.push_text(self._input_text)
            stream.end_input()
            async for ev in stream:
                output_emitter.push(ev.frame.data.tobytes())
                if timed_transcripts := ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(timed_transcripts)
        output_emitter.flush()
"#;

/// The completion's idiom.
pub(super) const CHAT: &str = r#"
    def __init__(
        self,
        llm: {Provider},
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options
        self._event_ch = _{Provider}EventChannel()
        self._metrics = _core.LlmMetricState(time.perf_counter())
        self._core_attempts = _bindings.Attempts("llm", conn_options, self._emit_error)
        self._task = asyncio.create_task(self._main_task(), name="LLM._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _run(self) -> None: ...

    async def _attempt(self) -> None:
        self._metrics = _core.LlmMetricState(time.perf_counter())
        await self._run()

    async def _main_task(self) -> None:
        # a failure after output reached the reader is final: the core never retries it
        await self._core_attempts.run(self._attempt, produced=lambda: self._event_ch.output_sent)
        record = self._metrics.finish(
            time.perf_counter(), False, _bindings.metrics_context(self._llm)
        )
        _bindings.emit_metrics(self._llm, "LLMMetrics", record)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._llm._emit_error(api_error, recoverable)

    def send(self, chunk: {Chunk}) -> None:
        """Hand ``chunk`` to the reader."""
        self._metrics.observe(time.perf_counter(), chunk)
        self._event_ch.send_nowait(chunk)

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> {Chunk}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        return val

    def __aiter__(self) -> AsyncIterator[{Chunk}]:
        return self

    def to_str_iterable(self) -> AsyncIterable[str]:
        """The stream as its text, for a completion that calls no tool."""

        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()

    async def collect(self) -> {Collected}:
        """Collect the entire stream into a single response.

        Example:
            ```python
            from livekit.agents import llm

            response = await my_llm.chat(chat_ctx=ctx, tools=tools).collect()

            for tc in response.tool_calls:
                result = await llm.execute_function_call(tc, tool_ctx)
                ctx.insert(result.fnc_call)
                ctx.insert(result.fnc_call_out)
            ```
        """
        text_parts: list[str] = []
        tool_calls: list[{Call}] = []
        usage: {Usage} | None = None
        extra: dict[str, Any] = {}

        async with self:
            async for chunk in self:
                if chunk.delta:
                    if chunk.delta.content:
                        text_parts.append(chunk.delta.content)
                    if chunk.delta.tool_calls:
                        tool_calls.extend(chunk.delta.tool_calls)
                    if chunk.delta.extra:
                        extra.update(chunk.delta.extra)
                if chunk.usage is not None:
                    usage = chunk.usage

        return {Collected}(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=usage,
            extra=extra,
        )
"#;

/// The detection stream's idiom.
pub(super) const VAD: &str = r#"
    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(self, vad: {Provider}) -> None:
        self._vad = vad
        self.input = aio.Chan[rtc.AudioFrame | {Stream}.Flush]()
        """The audio ``_main_task`` reads, a ``Flush`` at the end of each segment."""
        self._event_ch = aio.Chan[{Event}]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push some audio frame to be analyzed"""
        self._check_input_not_ended()
        self._check_not_closed()
        self.input.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment.

        Implementations MUST treat this as a hard segment boundary: drop any accumulated
        speech/silence state so the next pushed frame starts a fresh segment. Used by the
        pipeline to recover from out-of-band end-of-turn signals (e.g. STT EOS) without
        tearing down and recreating the stream.
        """
        self._check_input_not_ended()
        self._check_not_closed()
        self.input.send_nowait(self.Flush())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately"""
        self.input.close()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> {Event}:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        return val

    def __aiter__(self) -> AsyncIterator[{Event}]:
        return self

    def send(self, event: {Event}) -> None:
        """Hand ``event`` to the reader."""
        self._event_ch.send_nowait(event)

    @property
    def closed(self) -> bool:
        """Whether the stream is closed: its reader takes nothing more."""
        return self._event_ch.closed

    def _check_not_closed(self) -> None:
        if self.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self.input.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
"#;

/// The realtime session's idiom: its face, then the core's surface.
pub(super) const REALTIME: &str = r#"EventTypes = Literal[
    "input_speech_started",  # serverside VAD (also used for interruptions)
    "input_speech_stopped",  # serverside VAD
    "input_audio_transcription_completed",
    "generation_created",
    "session_reconnected",
    "metrics_collected",
    "remote_item_added",
    "error",
]

{TEVENT}
class {Stream}(ABC, rtc.EventEmitter[EventTypes | TEvent], Generic[TEvent]):
{DOC}
    def __init__(self, realtime_model: {Provider}) -> None:
        super().__init__()
        self._realtime_model = realtime_model

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report connection timing as a RealtimeModelMetrics event with zero usage."""
        from ..metrics import Metadata, RealtimeModelMetrics

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                acquire_time=acquire_time,
                connection_reused=False,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    @property
    def realtime_model(self) -> {Provider}:
        return self._realtime_model

    @property
    def capabilities(self) -> {Caps}:
        """Capabilities of the session.

        Defaults to the parent model's capabilities. Adapters that swap the underlying model
        mid-session override this to report the currently active model's capabilities.
        """
        return self._realtime_model.capabilities

    @property
    @abstractmethod
    def chat_ctx(self) -> ChatContext: ...

    @property
    @abstractmethod
    def tools(self) -> ToolContext: ...

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def update_chat_ctx(
        self, chat_ctx: ChatContext
    ) -> None: ...  # can raise RealtimeError on Timeout

    @abstractmethod
    async def update_tools(self, tools: list[Tool]) -> None: ...

    @abstractmethod
    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    @abstractmethod
    def push_video(self, frame: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]: ...  # can raise RealtimeError on Timeout

    # commit the input audio buffer to the server
    @abstractmethod
    def commit_audio(self) -> None: ...

    # clear the input audio buffer to the server
    @abstractmethod
    def clear_audio(self) -> None: ...

    # cancel the current generation (do nothing if no generation is in progress)
    @abstractmethod
    def interrupt(self) -> None: ...

    # message_id is the ID of the message to truncate (inside the ChatCtx)
    @abstractmethod
    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        from ..llm.realtime import RealtimeError

        if is_given(instructions):
            try:
                await self.update_instructions(instructions)
            except RealtimeError:
                logger.exception("failed to update the instructions")

        if is_given(chat_ctx):
            try:
                await self.update_chat_ctx(chat_ctx)
            except RealtimeError:
                logger.exception("failed to update the chat_ctx")

        if is_given(tools):
            try:
                await self.update_tools(tools)
            except RealtimeError:
                logger.exception("failed to update the tools")

    def start_user_activity(self) -> None:
        """notifies the model that user activity has started"""
        pass

    def say(
        self,
        text: str | AsyncIterable[str],
    ) -> asyncio.Future[GenerationCreatedEvent]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement say(). use a TTS model instead"
        )

    def _core_attach(self, tools: Callable[[], Sequence[Tool | Toolset]]) -> None:
        """Report this session's events to the framework, which offers it the tools ``tools``
        names."""
        from .._core_view import to_core_item

        self._core_tools = tools
        self._core_events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        put = self._core_events.put_nowait
        self.on("input_speech_started", lambda _: put({"kind": "input_speech_started"}))
        self.on(
            "input_speech_stopped",
            lambda ev: put(
                {
                    "kind": "input_speech_stopped",
                    "user_transcription_enabled": ev.user_transcription_enabled,
                }
            ),
        )
        self.on(
            "input_audio_transcription_completed",
            lambda ev: put(
                {
                    "kind": "input_transcription",
                    "item_id": ev.item_id,
                    "transcript": ev.transcript,
                    "is_final": ev.is_final,
                    "confidence": ev.confidence,
                    "turn_started_at": ev.turn_started_at,
                }
            ),
        )
        self.on("session_reconnected", lambda _: put({"kind": "session_reconnected"}))
        self.on("metrics_collected", lambda ev: put({"kind": "metrics", "metrics": ev}))
        self.on(
            "error",
            lambda ev: put(
                {
                    "kind": "error",
                    "label": ev.label,
                    "message": str(ev.error),
                    "recoverable": ev.recoverable,
                    "timestamp": ev.timestamp,
                }
            ),
        )
        self.on(
            "generation_created",
            lambda ev: (
                None if ev.user_initiated else put({"kind": "server_generation", "generation": ev})
            ),
        )
        self.on(
            "remote_item_added",
            lambda ev: put(
                {
                    "kind": "remote_item_added",
                    "previous_item_id": ev.previous_item_id,
                    "item": to_core_item(ev.item),
                }
            ),
        )

    def _core_send(self, cmd: dict[str, Any]) -> None:
        match cmd["kind"]:
            case "push_audio":
                self.push_audio(cmd["frame"])
            case "interrupt":
                self.interrupt()
            case "commit_audio":
                self.commit_audio()
            case "clear_audio":
                self.clear_audio()
            case "truncate":
                transcript = cmd["audio_transcript"]
                self.truncate(
                    message_id=cmd["message_id"],
                    modalities=["audio", "text"] if cmd["has_audio"] else ["text"],
                    audio_end_ms=cmd["audio_end_ms"],
                    audio_transcript=NOT_GIVEN if transcript is None else transcript,
                )
            case kind:
                raise ValueError(f"unknown session command {kind!r}")

    async def _core_update(self, cmd: dict[str, Any]) -> None:
        from .._core_view import from_core_item, offered_tools
        from ..llm.chat_context import ChatContext

        match cmd["kind"]:
            case "update_instructions":
                await self.update_instructions(cmd["instructions"])
            case "update_chat_ctx":
                await self.update_chat_ctx(ChatContext([from_core_item(i) for i in cmd["items"]]))
            case "push_item":
                ctx = self.chat_ctx.copy()
                ctx.insert(from_core_item(cmd["item"]))
                await self.update_chat_ctx(ctx)
            case "update_tools":
                await self.update_tools(offered_tools(self._core_tools(), cmd["names"]))
            case kind:
                raise ValueError(f"unknown session update {kind!r}")

    async def _core_request(self, cmd: dict[str, Any]) -> asyncio.Future[Any]:
        if cmd["kind"] == "say":
            return self.say(cmd["text"])
        choice = cmd["tool_choice"]
        return self.generate_reply(
            instructions=cmd["instructions"] or NOT_GIVEN,
            tool_choice=NOT_GIVEN if choice is None else choice,
        )

    async def _core_generation(
        self, request: asyncio.Future[GenerationCreatedEvent]
    ) -> GenerationCreatedEvent:
        return await request

    async def _core_cancel(self, request: asyncio.Future[Any]) -> None:
        request.cancel()

    async def _core_drain(self, ev: GenerationCreatedEvent, req_id: str | None) -> None:
        from .._core_view import to_core_item

        put = self._core_events.put_nowait
        response_id = ev.response_id or ""
        put({"kind": "generation_created", "response_id": response_id, "req_id": req_id})
        ids = {"response_id": ev.response_id}

        async def _message(msg: Any) -> None:
            mid = {"message_id": msg.message_id, **ids}
            put({"kind": "message_started", **mid})

            async def _text() -> None:
                async for delta in msg.text_stream:
                    put({"kind": "text_delta", "delta": str(delta), **mid})

            async def _audio() -> None:
                async for frame in msg.audio_stream:
                    put({"kind": "audio_delta", "frame": frame, **mid})

            async def _modalities() -> None:
                modalities = await msg.modalities
                kind = "audio" if "audio" in modalities else "text"
                put({"kind": "message_modalities", "modalities": kind, **mid})

            await asyncio.gather(_text(), _audio(), _modalities())
            put({"kind": "message_done", **mid})

        async def _messages() -> None:
            # each message forwards as it arrives, not once the server has sent them all
            started = []
            async for m in ev.message_stream:
                started.append(asyncio.ensure_future(_message(m)))
            await asyncio.gather(*started)

        async def _calls() -> None:
            async for call in ev.function_stream:
                put({"kind": "function_call", "call": to_core_item(call)})

        await asyncio.gather(_messages(), _calls())
        put({"kind": "generation_done", "response_id": response_id})

    async def _core_next_event(self) -> dict[str, Any] | None:
        return await self._core_events.get()

    async def _core_aclose(self) -> None:
        await self.aclose()
        self._core_events.put_nowait(None)
"#;
