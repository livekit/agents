# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import weakref
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, cast

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechData
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import (
    ENCODINGS,
    ERROR_MESSAGE_HEADER,
    FILLER_MODES,
    MAX_PHRASES,
    PRERECORDED_PATH,
    TURNS_PATH,
    Encoding,
    FillerMode,
    auth_headers,
    build_speech_data,
    build_url,
    check_channels,
    check_comma_joined,
    check_probability,
    integration_headers,
    normalize_languages,
    problem_parts,
    resolve_base_url,
    status_error,
)
from .log import logger

KEEPALIVE_INTERVAL = 30.0
_SEND_CHUNK_MS = 100


@dataclass(frozen=True)
class TurnOptions:
    """
    When Reson8 ends a turn, the main lever on end-of-turn latency.

    ``None`` leaves the server's default. See
    https://docs.reson8.dev/speech-to-text/turns/.

    Args:
        eager_probability: Confidence at which the preflight transcript is
            emitted, so the agent can start generating speculatively.
        final_probability: Confidence at which the turn commits. The server
            default is tuned for conversational speech and is slow to commit a
            one-word answer; lower it to commit sooner, at the risk of cutting
            off longer utterances.
    """

    # /turns does not report its effective config, so the defaults are mirrored
    # here to check one threshold when only the other is set. Keep in sync with
    # https://docs.reson8.dev/api/speech-to-text/turns/
    SERVER_DEFAULT_EAGER: ClassVar[float] = 0.5
    SERVER_DEFAULT_FINAL: ClassVar[float] = 0.92

    eager_probability: float | None = None
    final_probability: float | None = None

    def __post_init__(self) -> None:
        check_probability("eager_probability", self.eager_probability)
        check_probability("final_probability", self.final_probability)

        eager = self.eager_probability
        final = self.final_probability
        eager = self.SERVER_DEFAULT_EAGER if eager is None else eager
        final = self.SERVER_DEFAULT_FINAL if final is None else final

        if eager > final:
            raise ValueError(
                f"eager_probability ({eager}) must be below final_probability ({final}); "
                f"raise final_probability or lower eager_probability"
            )

        if eager == final:
            logger.warning(
                "eager_probability and final_probability are both %s, so the turn commits "
                "on the same event as the preflight transcript and preemptive generation "
                "gets no lead time",
                final,
            )

    def query_params(self) -> dict[str, str]:
        params: dict[str, str] = {}

        if self.eager_probability is not None:
            params["eager_turn_probability"] = str(self.eager_probability)

        if self.final_probability is not None:
            params["final_turn_probability"] = str(self.final_probability)

        return params


@dataclass(frozen=True)
class AudioOptions:
    """
    How to describe the audio sent to Reson8.

    These label the stream rather than convert it, so they have to match the
    frames actually sent. Streaming input is resampled to ``sample_rate``, but
    nothing remixes channels or transcodes samples: a pushed frame whose
    channel count disagrees with ``num_channels`` is rejected instead of being
    relabelled. See
    https://docs.reson8.dev/speech-to-text/features/audio-formats/.

    Args:
        sample_rate: Rate in Hz. Streaming input is resampled to this.
        encoding: Encoding of the audio sent to Reson8. ``rtc.AudioFrame``
            carries signed 16-bit PCM, which this plugin forwards unchanged,
            so ``"pcm_s16le"`` is the only value there is.
        num_channels: Channel count of the frames you push, 1 to 10.
    """

    sample_rate: int = 16000
    encoding: Encoding = "pcm_s16le"
    num_channels: int = 1

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"unsupported encoding: {self.encoding}. "
                f"Reson8 accepts: {', '.join(sorted(ENCODINGS))}."
            )

        check_channels(self.num_channels)

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    def query_params(self) -> dict[str, str]:
        return {
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
            "channels": str(self.num_channels),
        }


@dataclass(frozen=True)
class TranscriptOptions:
    """
    How the transcript is produced, and what detail accompanies it.

    Args:
        words: Word-level results, each with its own timing.
        language: The detected language code, on by default so
            ``SpeechData.language`` is populated when the language is
            auto-detected rather than pinned.
        confidence: Per-word confidence. Batch recognition only.
        filler_mode: What to do with filler words: ``"clean"`` removes them,
            ``"natural"`` lets the model decide, ``"verbatim"`` preserves them.
            ``None`` leaves the server's default.
    """

    words: bool = False
    language: bool = True
    confidence: bool = False
    filler_mode: FillerMode | None = None

    def __post_init__(self) -> None:
        if self.filler_mode is not None and self.filler_mode not in FILLER_MODES:
            raise ValueError(
                f"unsupported filler_mode: {self.filler_mode}. "
                f"Reson8 accepts: {', '.join(sorted(FILLER_MODES))}."
            )

    def query_params(self, *, streaming: bool) -> dict[str, str]:
        params: dict[str, str] = {"include_timestamps": "true"}

        if self.words:
            params["include_words"] = "true"

        if self.language:
            params["include_language"] = "true"

        if self.confidence and not streaming:
            params["include_confidence"] = "true"

        if self.filler_mode is not None:
            params["filler_mode"] = self.filler_mode

        return params


@dataclass(frozen=True)
class BiasingOptions:
    """
    How to bias recognition toward terminology you expect.

    See https://docs.reson8.dev/speech-to-text/features/custom-models/
    and https://docs.reson8.dev/speech-to-text/features/patterns/.

    Args:
        custom_model_id: A custom model to bias toward, for a vocabulary too
            large for ``phrases`` or one reused across requests. Build the
            model in Reson8 and pass its id here.
        phrases: Terms to bias toward, at most 250.
        strength: Additive boost on top of the model's trained calibration,
            non-negative and unbounded. The server default suits most
            requests; raise it only when expected terminology is not being
            recovered.
        patterns: Regex-style shapes for short alphanumeric tokens to recover,
            such as ``"AMZ[0-9]{6}"`` for an order code,
            ``"[0-9]{4,6}"`` for a variable-length one, or
            ``"[A-Z]{2}[0-9]{2} [A-Z]{3}"`` for a licence plate. Set these only
            when the token is likely to be spoken.
    """

    custom_model_id: str | None = None
    phrases: Sequence[str] | None = None
    strength: float | None = None
    patterns: Sequence[str] | None = None

    def __post_init__(self) -> None:
        check_comma_joined("phrases", self.phrases, limit=MAX_PHRASES)
        check_comma_joined("patterns", self.patterns, allow_braced_commas=True)

        if self.strength is not None and self.strength < 0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")

    def query_params(self) -> dict[str, str]:
        params: dict[str, str] = {}

        if self.custom_model_id:
            params["custom_model_id"] = self.custom_model_id

        if self.phrases:
            params["phrases"] = ",".join(self.phrases)

        if self.patterns:
            params["patterns"] = ",".join(self.patterns)

        if self.strength is not None:
            params["bias_strength"] = str(self.strength)

        return params


@dataclass(frozen=True)
class STTOptions:
    """The resolved configuration behind an ``STT`` instance."""

    language: str | None = None
    turn: TurnOptions = field(default_factory=TurnOptions)
    audio: AudioOptions = field(default_factory=AudioOptions)
    transcript: TranscriptOptions = field(default_factory=TranscriptOptions)
    biasing: BiasingOptions = field(default_factory=BiasingOptions)

    def merged(
        self,
        *,
        language: NotGivenOr[str | Sequence[str] | None] = NOT_GIVEN,
        turn: NotGivenOr[TurnOptions] = NOT_GIVEN,
        transcript: NotGivenOr[TranscriptOptions] = NOT_GIVEN,
        biasing: NotGivenOr[BiasingOptions] = NOT_GIVEN,
    ) -> STTOptions:
        """
        A copy with the given sections replaced.

        Each section validates itself on construction, so an invalid value
        raises in the caller's hands and never reaches a live stream.
        """

        changes: dict[str, Any] = {}
        if is_given(language):
            changes["language"] = normalize_languages(language)

        if is_given(turn):
            changes["turn"] = turn

        if is_given(transcript):
            changes["transcript"] = transcript

        if is_given(biasing):
            changes["biasing"] = biasing

        return replace(self, **changes)

    def query_params(self, *, streaming: bool) -> dict[str, str]:
        params = {
            **self.audio.query_params(),
            **self.transcript.query_params(streaming=streaming),
            **self.biasing.query_params(),
        }

        # When omitted, Reson8 auto-detects the spoken language; otherwise this
        # pins recognition to the given code(s) (comma-joined for multiple).
        if self.language:
            params["language"] = self.language

        if streaming:
            params.update(self.turn.query_params())

        return params


class STT(stt.STT):
    """
    Reson8 speech-to-text.

    A single model that adapts to how LiveKit uses it:

    * **Streaming** (:meth:`stream`) connects to the turn-aware
      ``/v1/speech-to-text/turns`` endpoint. Reson8 detects conversational turn
      boundaries server-side and emits a turn-end *candidate* once it believes a
      turn is complete. That candidate surfaces as a preflight transcript the
      agent can act on speculatively, and is then either confirmed as a final
      transcript or replaced by a later candidate. Ideal for
      low-latency voice agents.
    * **Batch** (:meth:`recognize`) sends pre-recorded audio to
      ``/v1/speech-to-text/prerecorded`` and returns the full transcript.

    Leave ``language`` as ``None`` (the default) to auto-detect the spoken
    language, or pass one or more :data:`SupportedLanguage` codes to pin
    recognition.

    :class:`TurnOptions` is the main lever on end-of-turn latency, and a
    stream can commit a turn on demand without touching it::

        # the server's 0.92 default is tuned for conversational speech and is
        # slow to commit a one-word answer
        stt = reson8.STT(language="es", turn=reson8.TurnOptions(final_probability=0.7))

        # or leave it alone and commit when you already know they are done
        stream = stt.stream()
        stream.flush()

    Reson8 detects turns server-side, so hand turn-taking to it rather than
    letting LiveKit run its own detector::

        session = AgentSession(
            stt=reson8.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(),
            turn_handling={
                "turn_detection": "stt",
                "preemptive_generation": {"enabled": True},
            },
        )

    See https://docs.reson8.dev/speech-to-text/turns/
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        language: str | Sequence[str] | None = None,
        turn: TurnOptions | None = None,
        audio: AudioOptions | None = None,
        transcript: TranscriptOptions | None = None,
        biasing: BiasingOptions | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            api_key: Reson8 API key. Falls back to the ``RESON8_API_KEY`` env var.
            base_url: Reson8 API base URL. Falls back to ``RESON8_BASE_URL``,
                then ``https://api.reson8.dev``.
            language: One or more :data:`SupportedLanguage` codes to pin
                recognition to. Pass a single code (``"nl"``), a comma-string
                (``"nl,de"``), or a list (``["nl", "de"]``). Leave as ``None`` to
                auto-detect. Raises ``ValueError`` for unsupported codes.
            turn: When Reson8 ends a turn. See :class:`TurnOptions`.
            audio: How the audio sent to Reson8 is described. See
                :class:`AudioOptions`.
            transcript: Which extra detail to report. See
                :class:`TranscriptOptions`.
            biasing: How to bias recognition. See :class:`BiasingOptions`.
            http_session: Optional session to use for requests. Defaults to the
                shared session managed by the agent framework.
        """

        transcript = transcript or TranscriptOptions()

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=True,
                aligned_transcript="word" if transcript.words else False,
            ),
        )

        api_key = api_key or os.environ.get("RESON8_API_KEY")
        if not api_key:
            raise ValueError(
                "Reson8 API key is required, either as argument or RESON8_API_KEY env var"
            )

        self._api_key = api_key
        self._base_url = resolve_base_url(base_url)
        self._opts = STTOptions(
            language=normalize_languages(language),
            turn=turn or TurnOptions(),
            audio=audio or AudioOptions(),
            transcript=transcript,
            biasing=biasing or BiasingOptions(),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    @property
    def model(self) -> str:
        return self._opts.biasing.custom_model_id or "default"

    @property
    def provider(self) -> str:
        return "Reson8"

    def update_options(
        self,
        *,
        language: NotGivenOr[str | Sequence[str] | None] = NOT_GIVEN,
        turn: NotGivenOr[TurnOptions] = NOT_GIVEN,
        transcript: NotGivenOr[TranscriptOptions] = NOT_GIVEN,
        biasing: NotGivenOr[BiasingOptions] = NOT_GIVEN,
    ) -> None:
        """
        Change settings at runtime.

        Reson8 takes its configuration from the query string, so a live stream
        applies the new settings by reconnecting immediately.

        Reson8 holds turn state server-side, so a redial mid-utterance
        abandons the audio already sent with the session it belongs to: that
        utterance is transcribed from wherever the new connection picks up.
        Change options between turns, or accept losing the one in progress.
        Waiting for a safe moment is not something the client can decide --
        nothing in the protocol ties a ``turn_end`` to a position in the audio
        that was sent, so there is no way to know what the server still owes.

        This reaches every live stream, so it also replaces a ``language``
        that was passed to :meth:`stream` for one of them.

        :class:`AudioOptions` is deliberately absent: the input resampler is
        built when a stream opens, so changing the rate mid-stream would
        describe the audio to Reson8 as something it is not.

        Turning word timings on or off also moves
        ``capabilities.aligned_transcript``. A running ``AgentSession`` reads
        that when the STT is attached or swapped, so it keeps the value it saw
        until then.
        """

        self._opts = self._opts.merged(
            language=language, turn=turn, transcript=transcript, biasing=biasing
        )
        self._capabilities.aligned_transcript = "word" if self._opts.transcript.words else False

        for stream in self._streams:
            stream.update_options(
                language=language, turn=turn, transcript=transcript, biasing=biasing
            )

    async def aclose(self) -> None:
        """
        Close every stream created by :meth:`stream`.

        The HTTP session is left open: it is either supplied by the caller or
        owned by the shared HTTP context, so it is not ours to close.
        """

        streams = list(self._streams)
        self._streams.clear()

        await asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True)

    def stream(
        self,
        *,
        language: NotGivenOr[str | Sequence[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = self._opts.merged(language=language) if is_given(language) else self._opts
        stream = SpeechStream(
            stt=self,
            opts=opts,
            api_key=self._api_key,
            base_url=self._base_url,
            conn_options=conn_options,
            http_session=self._session,
        )
        self._streams.add(stream)
        return stream

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str | Sequence[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = normalize_languages(language) if is_given(language) else self._opts.language
        frames = rtc.combine_audio_frames(buffer)

        opts = replace(
            self._opts,
            language=lang,
            audio=replace(
                self._opts.audio,
                encoding="pcm_s16le",
                sample_rate=frames.sample_rate,
                num_channels=frames.num_channels,
            ),
        )

        url = build_url(self._base_url, PRERECORDED_PATH, opts.query_params(streaming=False))

        try:
            async with self._ensure_session().post(
                url,
                data=frames.data.tobytes(),
                headers={
                    **auth_headers(self._api_key),
                    **integration_headers(),
                    "Content-Type": "application/octet-stream",
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    code, detail = problem_parts(text)
                    raise status_error(resp.status, code=code, detail=detail)
        except asyncio.TimeoutError:
            raise APITimeoutError("Reson8 did not respond in time") from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to reach Reson8 ({type(e).__name__})") from None

        try:
            body = json.loads(text)
        except ValueError:
            body = None

        if not isinstance(body, dict):
            raise APIConnectionError("Reson8 returned a malformed response body") from None

        try:
            alternative = build_speech_data(body, language=lang)
        except (AttributeError, TypeError, ValueError):
            raise APIConnectionError("Reson8 returned a malformed response body") from None

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[alternative],
        )


class SpeechStream(stt.RecognizeStream):
    """
    Turn-aware streaming session against the ``/turns`` endpoint.

    A reconnect -- whether the framework retrying this stream or an option
    change redialling -- cannot resume the turn it interrupted. Frames are
    consumed from the input channel as they are sent, so audio already handed
    to a dead socket is gone, and Reson8 holds the turn state that would have
    interpreted it. The replacement connection therefore starts clean: any
    speech already announced is closed out with END_OF_SPEECH so the caller is
    not left waiting, the pending candidate is dropped rather than promoted by
    an unrelated ``turn_end``, and the utterance in progress is transcribed
    from wherever the new connection picks it up. Buffering audio for replay
    would be the alternative, at the cost of holding every utterance in memory
    for a failure that is rare.
    """

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        api_key: str,
        base_url: str,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.audio.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._base_url = base_url
        self._session = http_session
        self._request_id = str(uuid.uuid4())
        self._reconnect_event = asyncio.Event()

        self._turn_settled = asyncio.Event()
        self._turn_settled.set()

        self._speaking = False
        # the most recent turn-end candidate, promoted to a final transcript
        # once the server confirms the turn ended
        self._candidate: SpeechData | None = None
        self._speech_duration = 0.0
        # outlives a reconnect on purpose: a fresh one would strand the bytes
        # already taken off _input_ch but not yet big enough to send
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=opts.audio.sample_rate,
            num_channels=opts.audio.num_channels,
            samples_per_channel=opts.audio.sample_rate * _SEND_CHUNK_MS // 1000,
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[str | Sequence[str] | None] = NOT_GIVEN,
        turn: NotGivenOr[TurnOptions] = NOT_GIVEN,
        transcript: NotGivenOr[TranscriptOptions] = NOT_GIVEN,
        biasing: NotGivenOr[BiasingOptions] = NOT_GIVEN,
    ) -> None:
        self._opts = self._opts.merged(
            language=language, turn=turn, transcript=transcript, biasing=biasing
        )

        self._reconnect_event.set()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if frame.num_channels != self._opts.audio.num_channels:
            raise ValueError(
                f"expected {self._opts.audio.num_channels}-channel frames, got "
                f"{frame.num_channels}; set AudioOptions(num_channels="
                f"{frame.num_channels}) or push audio in the configured shape"
            )

        super().push_frame(frame)

    async def _await_final_turn(self) -> None:
        """
        Wait for Reson8 to answer the flush that ``end_input`` queued.

        ``end_input`` queues a flush sentinel and then closes the input
        channel, so the last thing the send task does is ask Reson8 to finalise
        the audio it holds. Hanging up on the socket at that point would throw
        away the turn_end that answers it, and the caller would see the stream
        finish with no transcript at all.

        ``aclose`` cancels the run task instead of closing the input channel
        cleanly, so it never reaches this and stays immediate.

        What this cannot do is tell one turn's confirmation from another's.
        Reson8 confirms a turn while later audio is already on its way, and no
        ``turn_end`` says how much of what was sent it covers -- so a
        confirmation that arrives for an earlier turn releases the wait, and a
        final turn still being transcribed can be missed. Requiring a
        ``turn_end`` after the flush instead would stall every stream whose
        flush had nothing left to finalise. Closing this properly needs a turn
        id on the wire.

        Running out of time is a lost turn, not a clean finish, so it raises.
        The audio is already consumed from the input channel, which is why the
        error is not retryable: another attempt has nothing left to send.
        """

        try:
            await asyncio.wait_for(self._turn_settled.wait(), self._conn_options.timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError(
                "Reson8 did not finalise the last turn before input closed",
                retryable=False,
            ) from None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = build_url(
            self._base_url, TURNS_PATH, self._opts.query_params(streaming=True), websocket=True
        )

        connect = self._ensure_session().ws_connect(
            url,
            headers={**auth_headers(self._api_key), **integration_headers()},
            heartbeat=KEEPALIVE_INTERVAL,
        )

        try:
            return await asyncio.wait_for(
                cast("Awaitable[aiohttp.ClientWebSocketResponse]", connect),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            reason = e.headers.get(ERROR_MESSAGE_HEADER) if e.headers else None
            raise status_error(e.status, detail=reason) from None
        except asyncio.TimeoutError:
            raise APITimeoutError("Timed out connecting to Reson8") from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to connect to Reson8 ({type(e).__name__})") from None

    async def _run(self) -> None:
        closing_ws = False
        input_ended = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws, input_ended

            try:
                async for data in self._input_ch:
                    flushing = isinstance(data, self._FlushSentinel)
                    if isinstance(data, rtc.AudioFrame):
                        frames = self._audio_bstream.write(data.data.tobytes())
                    else:
                        frames = self._audio_bstream.flush()

                    for frame in frames:
                        self._speech_duration += frame.duration
                        self._turn_settled.clear()
                        await ws.send_bytes(frame.data.tobytes())

                    if flushing:
                        await ws.send_str(json.dumps({"type": "flush_request"}))
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._ensure_session().closed:
                    return

                raise APIConnectionError(
                    f"Failed to send audio to Reson8 ({type(e).__name__})"
                ) from None

            input_ended = True
            await self._await_final_turn()

            closing_ws = True
            await ws.close()

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._ensure_session().closed:
                        return

                    if input_ended:
                        if self._turn_settled.is_set():
                            return

                        self._turn_settled.set()
                        raise APIConnectionError(
                            "Reson8 closed before finalising the last turn",
                            retryable=False,
                        )

                    raise APIConnectionError(
                        f"Reson8 connection closed unexpectedly (code={ws.close_code})"
                    )

                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(
                        f"Reson8 connection failed ({type(msg.data).__name__})"
                    ) from None

                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    parsed = json.loads(msg.data)
                except (ValueError, TypeError):
                    parsed = None

                if not isinstance(parsed, dict):
                    logger.warning(
                        "Ignoring unparseable Reson8 message",
                        extra={"lk.pii.message": msg.data},
                    )
                    continue

                self._process_message(parsed)

        connection_started: float | None = None

        while True:
            ws: aiohttp.ClientWebSocketResponse | None = None
            now = time.time()

            if connection_started is not None:
                self.start_time_offset += now - connection_started

            self.start_time = now
            connection_started = now

            self._end_speaking()
            self._candidate = None
            self._turn_settled.set()

            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task is not wait_reconnect:
                            task.result()

                    if wait_reconnect not in done:
                        break

                    self._reconnect_event.clear()
                    logger.debug("Reconnecting to Reson8 to apply updated options")
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect)
                    tasks_group.cancel()
                    tasks_group.exception()
            finally:
                if ws is not None:
                    await ws.close()

    def _process_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        logger.debug(
            "received turn event",
            extra={"lk.pii.type": msg_type, "lk.pii.text": msg.get("text")},
        )

        if msg_type == "turn_start":
            self._candidate = None
            self._start_speaking()

        elif msg_type == "turn_end_candidate":
            self._start_speaking()

            try:
                current = build_speech_data(
                    msg,
                    language=self._opts.language,
                    start_time_offset=self.start_time_offset,
                )
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(
                    "Ignoring malformed Reson8 turn payload",
                    extra={"error": type(e).__name__, "lk.pii.message": msg},
                )
                return

            previous = self._candidate
            self._candidate = current

            repeated = previous is not None and previous.text == self._candidate.text
            if self._candidate.text and not repeated:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[self._candidate],
                    )
                )

        elif msg_type == "turn_end":
            candidate = self._candidate
            self._candidate = None

            if candidate is not None:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[candidate],
                    )
                )

            self._end_speaking()

            if self._speech_duration > 0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._speech_duration
                        ),
                    )
                )
                self._speech_duration = 0.0

            self._turn_settled.set()

        else:
            logger.debug("ignoring unhandled Reson8 message", extra={"lk.pii.type": msg_type})

    def _start_speaking(self) -> None:
        if self._speaking:
            return

        self._speaking = True
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _end_speaking(self) -> None:
        if not self._speaking:
            return

        self._speaking = False
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
