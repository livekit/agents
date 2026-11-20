from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from opentelemetry import trace

from livekit import rtc

from .. import utils
from .._exceptions import APIConnectionError
from ..log import logger
from ..telemetry import trace_types
from ..types import DEFAULT_API_CONNECT_OPTIONS, USERDATA_TIMED_TRANSCRIPT, APIConnectOptions
from ..utils import aio
from .stream_adapter import StreamAdapter
from .tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata

# don't retry when using the fallback adapter; the adapter fails over to the next instance
# instead, and each instance retries up to `max_retry_per_tts` times on its own
DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)

# set inside recovery probes; the child streams they open inherit it, so their notices are
# told apart from a user attempt's on the same instance
_in_recovery_probe: ContextVar[bool] = ContextVar("tts_fallback_in_recovery_probe", default=False)


@dataclass
class _TTSStatus:
    available: bool
    # One slot per recovery path, mirroring `_STTStatus`. A single shared slot let a probe in
    # flight on one path suppress probing on the other, and let the second path to run drop the
    # first path's task so `aclose()` could no longer cancel it.
    recovering_synthesize_task: asyncio.Task[None] | None
    recovering_stream_task: asyncio.Task[None] | None
    needs_resampling: bool
    # user attempts in flight; only their notices are forwarded (a probe's audio is discarded)
    serving: int = 0


@dataclass
class AvailabilityChangedEvent:
    tts: TTS
    available: bool


class FallbackAdapter(
    TTS[Literal["tts_availability_changed", "fallback_activated"]],
):
    """Agent Fallback Adapter for TTS. Manages multiple TTS instances with automatic fallback
    when the primary provider fails.

    A request goes to the instances in the order given, skipping any that is marked unavailable.
    An instance that fails before any audio is produced is marked unavailable and the request
    moves to the next one. When every instance that was tried has failed, the instances that
    were skipped as unavailable are tried, in order, as a last resort. A request only fails
    once every instance has been tried, and then it raises one unrecoverable error.

    An instance that fails after its audio started ends the utterance: it is not replayed on
    another instance. The adapter emits one ``error`` event for it, recoverable when the adapter
    has another instance to serve the next utterance.

    A recovery probe tests an unavailable instance in the background and marks it available when
    it succeeds. An instance that serves a request is also marked available, but only when it
    produced at least one audio frame, so a request with no speakable text does not restore it.
    """

    def __init__(
        self,
        tts: list[TTS],
        *,
        max_retry_per_tts: int = 2,
        sample_rate: int | None = None,
    ) -> None:
        """
        Initialize a FallbackAdapter that manages multiple TTS instances.

        Args:
            tts (list[TTS]): A list of TTS instances to use for fallback.
            max_retry_per_tts (int, optional): Maximum number of retries per TTS instance. Defaults to 2. This is the only retry control: the ``max_retry`` of the ``conn_options`` passed to ``stream()`` or ``synthesize()`` is ignored (its ``timeout`` and ``retry_interval`` still apply).
            sample_rate (int | None, optional): Desired sample rate for the synthesized audio. If None, uses the maximum sample rate among the TTS instances.

        Raises:
            ValueError: If less than one TTS instance is provided.
            ValueError: If TTS instances have different numbers of channels.
        """  # noqa: E501

        if len(tts) < 1:
            raise ValueError("at least one TTS instance must be provided.")

        if len({t.num_channels for t in tts}) != 1:
            raise ValueError("all TTS must have the same number of channels")

        if sample_rate is None:
            sample_rate = max(t.sample_rate for t in tts)

        num_channels = tts[0].num_channels

        super().__init__(
            capabilities=TTSCapabilities(
                streaming=any(t.capabilities.streaming for t in tts),
                aligned_transcript=all(t.capabilities.aligned_transcript for t in tts),
            ),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._tts_instances = tts
        self._max_retry_per_tts = max_retry_per_tts
        self._closed = False

        # the instance that most recently served a request; used to label metrics & traces
        self._active_instance: TTS = self._tts_instances[0]

        self._status: list[_TTSStatus] = []
        self._fallback_forwarders: list[Callable[..., None]] = []
        for t in tts:
            needs_resampling = sample_rate != t.sample_rate
            if needs_resampling:
                logger.info(f"resampling {t.label} from {t.sample_rate}Hz to {sample_rate}Hz")

            self._status.append(
                _TTSStatus(
                    available=True,
                    recovering_synthesize_task=None,
                    recovering_stream_task=None,
                    needs_resampling=needs_resampling,
                )
            )

            t.on("metrics_collected", self._on_metrics_collected)
            forwarder = self._fallback_forwarder(self._status[-1])
            self._fallback_forwarders.append(forwarder)
            t.on("fallback_activated", forwarder)

    def _next_instance(self) -> TTS:
        """The instance the next request goes to first: the first one marked available, or
        the primary once all are down (they are then all tried, primary first). An instance
        goes back to available when its recovery probe succeeds or when it serves a request,
        so a recovered primary is reported again before the next request reaches it."""
        for instance, status in zip(self._tts_instances, self._status, strict=True):
            if status.available:
                return instance
        return self._tts_instances[0]

    @property
    def model(self) -> str:
        """The model of the instance that serves next (see :meth:`_next_instance`). Spans and
        metrics read this, so a failover shows the model expected to answer rather than the
        adapter; the instance that actually served is stamped per request by the stream."""
        return self._next_instance().model

    @property
    def provider(self) -> str:
        """The provider of the instance that serves next (see :attr:`model`)."""
        return self._next_instance().provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata of the instance that most recently served a request (the primary before any traffic)."""  # noqa: E501
        return self._active_instance.metrics_metadata

    def _has_fallback(self) -> bool:
        """Whether the next utterance can be served by another instance. Every instance is
        tried before a request fails, whatever it is marked, so this is true for any adapter
        with more than one instance. An error is recoverable on that condition, as in
        ``RealtimeFallbackAdapter`` (recoverable while a fallback remains)."""
        return len(self._tts_instances) > 1

    def _set_available(self, tts: TTS, status: _TTSStatus, available: bool) -> bool:
        """Mark an instance available or unavailable and emit ``tts_availability_changed``.
        Returns False, emitting nothing, when the instance is already in that state, so a
        probe that finishes after the instance served a request does not report it twice."""
        if status.available == available:
            return False

        status.available = available
        self.emit(
            "tts_availability_changed", AvailabilityChangedEvent(tts=tts, available=available)
        )
        return True

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackChunkedStream:
        """Synthesize ``text``, falling back through the instances.

        The ``max_retry`` of ``conn_options`` is ignored. The adapter does not retry a request
        itself: that would rerun every instance's retries on each attempt, and AgentSession
        passes ``max_retry=3``. Each instance retries ``max_retry_per_tts`` times on its own.
        The ``timeout`` and ``retry_interval`` of ``conn_options`` still reach the instances.
        """
        return FallbackChunkedStream(
            tts=self, input_text=text, conn_options=dataclasses.replace(conn_options, max_retry=0)
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS
    ) -> FallbackSynthesizeStream:
        """Open a stream, falling back through the instances.

        The ``max_retry`` of ``conn_options`` is ignored, as in :meth:`synthesize`: each
        instance retries ``max_retry_per_tts`` times on its own, and the adapter does not retry
        the stream as a whole. The ``timeout`` and ``retry_interval`` still reach the instances.
        """
        return FallbackSynthesizeStream(
            tts=self, conn_options=dataclasses.replace(conn_options, max_retry=0)
        )

    def prewarm(self) -> None:
        if self._tts_instances:
            self._tts_instances[0].prewarm()

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    def _fallback_forwarder(self, status: _TTSStatus) -> Callable[..., None]:
        def forward(*args: Any, **kwargs: Any) -> None:
            if status.serving and not _in_recovery_probe.get():
                self.emit("fallback_activated", *args, **kwargs)

        return forward

    async def aclose(self) -> None:
        # set before the sweep: _try_recovery is synchronous, so a probe is
        # either already in a slot (and cancelled below) or refused by this
        # flag. A stream still in flight runs its finally after this returns,
        # and must not start a probe that nothing is left to cancel
        self._closed = True

        for tts_status in self._status:
            if tts_status.recovering_synthesize_task is not None:
                await aio.cancel_and_wait(tts_status.recovering_synthesize_task)

            if tts_status.recovering_stream_task is not None:
                await aio.cancel_and_wait(tts_status.recovering_stream_task)

        for t, forwarder in zip(self._tts_instances, self._fallback_forwarders, strict=True):
            t.off("metrics_collected", self._on_metrics_collected)
            t.off("fallback_activated", forwarder)


def _record_fallback_served(
    tts: TTS, index: int, request_span: trace.Span | None, caller_span: trace.Span
) -> None:
    """The instance that served: its label and position on the adapter's request span, and
    the response side there and on the caller's (tts_node). From ``tts``, not the adapter:
    concurrent requests may be served by different instances."""
    response_attrs: dict[str, Any] = {trace_types.ATTR_GEN_AI_RESPONSE_MODEL: tts.model}
    if (normalized := trace_types.gen_ai_provider_name(tts.provider)) is not None:
        response_attrs[trace_types.ATTR_GEN_AI_PROVIDER_NAME] = normalized
    if request_span is not None:
        request_span.set_attributes(
            {
                trace_types.ATTR_FALLBACK_LABEL: tts.label,
                trace_types.ATTR_FALLBACK_INDEX: index,
                **response_attrs,
            }
        )
    caller_span.set_attributes(response_attrs)


def _mark_recovery(stream: ChunkedStream | SynthesizeStream) -> None:
    # the probe nests beside the request's real attempts; tell them apart
    if stream._tts_request_span is not None:
        stream._tts_request_span.set_attribute(trace_types.ATTR_FALLBACK_RECOVERY, True)


class FallbackChunkedStream(ChunkedStream):
    _tts_request_span_name: ClassVar[str] = "tts_fallback_adapter"
    # each instance's request span nests directly under the adapter's
    _tts_attempt_span_name: ClassVar[str | None] = None

    def __init__(
        self, *, tts: FallbackAdapter, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._fallback_adapter = tts
        # the span this request was made under (tts_node); see _record_fallback_served
        self._caller_span = trace.get_current_span()

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        async for _ in event_aiter:
            pass

    async def _try_synthesize(
        self, *, tts: TTS, recovering: bool = False
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        try:
            async with tts.synthesize(
                self._input_text,
                conn_options=dataclasses.replace(
                    self._conn_options, max_retry=self._fallback_adapter._max_retry_per_tts
                ),
            ) as stream:
                should_set_active = not recovering
                try:
                    async for audio in stream:
                        if should_set_active:
                            should_set_active = False
                            self._fallback_adapter._active_instance = tts
                        yield audio
                finally:
                    if recovering:
                        _mark_recovery(stream)

        except Exception as e:
            if recovering:
                logger.warning("%s recovery failed: %s", tts.label, e, extra={"streamed": False})
                raise

            logger.warning(
                f"{tts.label} error, switching to next TTS",
                extra={"streamed": False},
            )
            raise

    def _try_recovery(self, tts: TTS) -> None:
        adapter = self._fallback_adapter
        if adapter._closed:
            # nothing would cancel a probe started from here
            return

        tts_status = adapter._status[adapter._tts_instances.index(tts)]
        if tts_status.available:
            # it served, or another probe restored it, since it was marked unavailable
            return

        recovering_task = tts_status.recovering_synthesize_task
        if recovering_task is None or recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                _in_recovery_probe.set(True)
                try:
                    async for _ in self._try_synthesize(tts=tts, recovering=True):
                        pass

                    if adapter._set_available(tts, tts_status, True):
                        logger.info(f"tts.FallbackAdapter, {tts.label} recovered")
                except Exception:  # exceptions already logged inside _try_synthesize
                    return

            tts_status.recovering_synthesize_task = asyncio.create_task(_recover_tts_task(tts))

    async def _serve(self, output_emitter: AudioEmitter, index: int, tts: TTS) -> bool:
        """Run the request on one instance. Returns True when the request is over, because the
        instance served it or failed after its audio started, and False when it failed before
        any audio, in which case it is marked unavailable and its recovery probe is started."""
        adapter = self._fallback_adapter
        tts_status = adapter._status[index]
        tts_status.serving += 1
        yielded_audio = False
        try:
            resampler = (
                rtc.AudioResampler(
                    input_rate=tts.sample_rate,
                    output_rate=adapter.sample_rate,
                    num_channels=adapter.num_channels,
                )
                if tts_status.needs_resampling
                else None
            )
            async for synthesized_audio in self._try_synthesize(tts=tts, recovering=False):
                yielded_audio = True
                if texts := synthesized_audio.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(texts)

                if resampler is not None:
                    for rf in resampler.push(synthesized_audio.frame):
                        output_emitter.push_frame(rf)
                else:
                    output_emitter.push_frame(synthesized_audio.frame)

            if resampler is not None:
                for rf in resampler.flush():
                    output_emitter.push_frame(rf)

            # an instance that produced audio is available, whatever it was marked before. One
            # that made no provider call (a request with no speakable text) proved nothing
            if yielded_audio:
                adapter._set_available(tts, tts_status, True)
            _record_fallback_served(tts, index, self._tts_request_span, self._caller_span)
            return True
        except Exception as e:  # exceptions already logged inside _try_synthesize
            adapter._set_available(tts, tts_status, False)

            if output_emitter.pushed_duration() > 0.0:
                logger.warning(f"{tts.label} already synthesized of audio, ignoring fallback")
                # the caller heard this instance's audio: it served, partially. The utterance
                # ends here, so say so, and probe the instance at once rather than when a
                # later request skips it
                self._emit_error(e, recoverable=adapter._has_fallback())
                self._try_recovery(tts)
                _record_fallback_served(tts, index, self._tts_request_span, self._caller_span)
                return True
        finally:
            tts_status.serving -= 1

        self._try_recovery(tts)
        return False

    async def _run(self, output_emitter: AudioEmitter) -> None:
        adapter = self._fallback_adapter
        start_time = time.time()

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=adapter.sample_rate,
            num_channels=adapter.num_channels,
            mime_type="audio/pcm",
        )

        # instances marked unavailable are skipped at first and tried last, if every other
        # instance fails. Their probes start once the request is over, so a probe does not
        # run beside the last-resort request on the same instance
        skipped: list[tuple[int, TTS]] = []
        try:
            for i, tts in enumerate(adapter._tts_instances):
                if not adapter._status[i].available:
                    skipped.append((i, tts))
                    continue

                if await self._serve(output_emitter, i, tts):
                    return

            if len(skipped) == len(adapter._tts_instances):
                logger.error("all TTSs are unavailable, retrying..")
            elif skipped:
                logger.warning("all available TTSs failed, trying the unavailable ones")

            while skipped:
                i, tts = skipped.pop(0)
                if await self._serve(output_emitter, i, tts):
                    return

            raise APIConnectionError(
                f"all TTSs failed ({[tts.label for tts in adapter._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
            )
        finally:
            for _, tts in skipped:
                self._try_recovery(tts)


class FallbackSynthesizeStream(SynthesizeStream):
    _tts_request_span_name: ClassVar[str] = "tts_fallback_adapter"
    # each instance's request span nests directly under the adapter's
    _tts_attempt_span_name: ClassVar[str | None] = None

    def __init__(self, *, tts: FallbackAdapter, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._fallback_adapter = tts
        self._caller_span = trace.get_current_span()
        self._pushed_tokens: list[str] = []

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        async for _ in event_aiter:
            pass

    async def _try_synthesize(
        self,
        *,
        tts: TTS,
        input_ch: aio.ChanReceiver[str | SynthesizeStream._FlushSentinel],
        conn_options: APIConnectOptions,
        recovering: bool = False,
    ) -> AsyncGenerator[SynthesizedAudio, None]:
        # If TTS doesn't support streaming, wrap it with StreamAdapter
        temporary_adapter: StreamAdapter | None = None
        if tts.capabilities.streaming:
            stream = tts.stream(conn_options=conn_options)
        else:
            from .. import tokenize

            temporary_adapter = StreamAdapter(
                tts=tts,
                sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True),
            )
            stream = temporary_adapter.stream(conn_options=conn_options)

        @utils.log_exceptions(logger=logger)
        async def _forward_input_task() -> None:
            try:
                async for data in input_ch:
                    if isinstance(data, str):
                        stream.push_text(data)
                    elif isinstance(data, self._FlushSentinel):
                        stream.flush()
            finally:
                stream.end_input()

        input_task = asyncio.create_task(_forward_input_task())

        def _capture_started_time() -> None:
            # ttfb measures the fallback adapter as a whole: anchor on the first time a
            # sentence was handed to any underlying TTS — even one that failed before
            # emitting audio — and never overwrite it when falling back to another TTS
            if not recovering and not self._started_time and stream._started_time:
                self._started_time = stream._started_time

        try:
            async with stream:
                should_set_active = not recovering
                try:
                    async for audio in stream:
                        _capture_started_time()
                        if should_set_active:
                            should_set_active = False
                            self._fallback_adapter._active_instance = tts
                        yield audio
                finally:
                    if recovering:
                        _mark_recovery(stream)
        except Exception as e:
            if recovering:
                logger.warning(
                    "%s recovery failed: %s",
                    tts.label,
                    e,
                    extra={"streamed": True},
                )
                raise

            logger.exception(
                f"{tts.label} error, switching to next TTS",
                extra={"streamed": True},
            )
            raise
        finally:
            _capture_started_time()
            try:
                await utils.aio.cancel_and_wait(input_task)
            finally:
                if temporary_adapter is not None:
                    await temporary_adapter.aclose()

    async def _run(self, output_emitter: AudioEmitter) -> None:
        adapter = self._fallback_adapter
        start_time = time.time()

        # a retry (SynthesizeStream._main_task) replays the whole input into a fresh
        # _input_ch, which _forward_input_task below records again: start empty, or the
        # text an earlier attempt read is sent to the TTS a second time. FallbackAdapter
        # never retries (stream() sets max_retry=0), so this only matters for a stream
        # built directly with its own max_retry
        self._pushed_tokens = []

        new_input_ch: aio.Chan[str | SynthesizeStream._FlushSentinel] | None = None
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=adapter.sample_rate,
            num_channels=adapter.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        async def _forward_input_task() -> None:
            nonlocal new_input_ch

            async for data in self._input_ch:
                if new_input_ch:
                    new_input_ch.send_nowait(data)

                if isinstance(data, str) and data:
                    self._pushed_tokens.append(data)

            if new_input_ch:
                new_input_ch.close()

        input_task = asyncio.create_task(_forward_input_task())

        # instances that failed, and instances skipped as unavailable and never tried. A probe
        # needs text to synthesize, and _pushed_tokens is only filled once _forward_input_task
        # has run: an instance that is skipped because it is already unavailable is reached
        # before that happens, so the probes are started in the finally below, after the input
        # has been cancelled. That also keeps a probe from running beside the last-resort
        # request on the same instance
        pending_recovery: list[TTS] = []
        skipped: list[tuple[int, TTS]] = []

        async def _serve(index: int, tts: TTS) -> bool:
            """Run the request on one instance. Returns True when the request is over, because
            the instance served it or failed after its audio started, and False when it failed
            before any audio, in which case it is marked unavailable and probed later."""
            nonlocal new_input_ch

            tts_status = adapter._status[index]
            tts_status.serving += 1
            yielded_audio = False
            try:
                new_input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()

                for text in self._pushed_tokens:
                    new_input_ch.send_nowait(text)

                if input_task.done():
                    new_input_ch.close()

                resampler = (
                    rtc.AudioResampler(
                        input_rate=tts.sample_rate,
                        output_rate=adapter.sample_rate,
                        num_channels=adapter.num_channels,
                    )
                    if tts_status.needs_resampling
                    else None
                )
                async for synthesized_audio in self._try_synthesize(
                    tts=tts,
                    input_ch=new_input_ch,
                    conn_options=dataclasses.replace(
                        self._conn_options, max_retry=adapter._max_retry_per_tts
                    ),
                    recovering=False,
                ):
                    yielded_audio = True
                    if texts := synthesized_audio.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                        output_emitter.push_timed_transcript(texts)

                    if resampler is not None:
                        for resampled_frame in resampler.push(synthesized_audio.frame):
                            output_emitter.push_frame(resampled_frame)

                        if synthesized_audio.is_final:
                            for resampled_frame in resampler.flush():
                                output_emitter.push_frame(resampled_frame)
                    else:
                        output_emitter.push_frame(synthesized_audio.frame)

                # an instance that produced audio is available, whatever it was marked before.
                # One that made no provider call (a request with no speakable text) proved
                # nothing
                if yielded_audio:
                    adapter._set_available(tts, tts_status, True)
                _record_fallback_served(tts, index, self._tts_request_span, self._caller_span)
                return True
            except Exception as e:
                adapter._set_available(tts, tts_status, False)
                pending_recovery.append(tts)

                if output_emitter.pushed_duration() > 0.0:
                    logger.warning(
                        f"{tts.label} already synthesized of audio, ignoring the current segment for the tts fallback"  # noqa: E501
                    )
                    # the caller heard this instance's audio: it served, partially. The
                    # utterance ends here, so say so. The probe starts in the finally below,
                    # once input_task is cancelled, and re-synthesizes the text read so far
                    # rather than when a later request skips the instance
                    self._emit_error(e, recoverable=adapter._has_fallback())
                    _record_fallback_served(tts, index, self._tts_request_span, self._caller_span)
                    return True
            finally:
                tts_status.serving -= 1

            return False

        try:
            for i, tts in enumerate(adapter._tts_instances):
                if not adapter._status[i].available:
                    skipped.append((i, tts))
                    continue

                if await _serve(i, tts):
                    return

            # every instance that was tried has failed: try the skipped ones, in order
            if len(skipped) == len(adapter._tts_instances):
                logger.error("all TTSs are unavailable, retrying..")
            elif skipped:
                logger.warning("all available TTSs failed, trying the unavailable ones")

            while skipped:
                i, tts = skipped.pop(0)
                if await _serve(i, tts):
                    return

            raise APIConnectionError(
                f"all TTSs failed ({[tts.label for tts in adapter._tts_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
            )
        finally:
            await utils.aio.cancel_and_wait(input_task)

            for tts in [*pending_recovery, *(tts for _, tts in skipped)]:
                self._try_recovery(tts)

    def _try_recovery(self, tts: TTS) -> None:
        adapter = self._fallback_adapter
        if adapter._closed:
            # a stream still in flight when the adapter closed runs its finally
            # after the sweep, and nothing would cancel a probe started here
            return

        tts_status = adapter._status[adapter._tts_instances.index(tts)]
        if tts_status.available:
            # it served, or another probe restored it, since it was marked unavailable
            return

        retry_text = self._pushed_tokens.copy()
        if not retry_text:
            return

        recovering_task = tts_status.recovering_stream_task
        if recovering_task is None or recovering_task.done():

            async def _recover_tts_task(tts: TTS) -> None:
                _in_recovery_probe.set(True)
                try:
                    input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
                    for t in retry_text:
                        input_ch.send_nowait(t)

                    input_ch.close()

                    async for _ in self._try_synthesize(
                        tts=tts,
                        input_ch=input_ch,
                        recovering=True,
                        conn_options=dataclasses.replace(self._conn_options, max_retry=0),
                    ):
                        pass

                    if adapter._set_available(tts, tts_status, True):
                        logger.info(f"tts.FallbackAdapter, {tts.label} recovered")
                except Exception:
                    return

            tts_status.recovering_stream_task = asyncio.create_task(_recover_tts_task(tts))
