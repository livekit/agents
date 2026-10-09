# Copyright 2025 LiveKit, Inc.
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
import dataclasses
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, cast
from urllib.parse import urlparse

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    LanguageCode,
    stt,
    utils,
    vad,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from speechmatics.agent_stt import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    AdditionalVocabEntry,
    AgentSttAsyncClient,
    AudioEncoding,
    AudioFormat,
    ClientMessageType,
    ConnectionError as SMConnectionError,
    Model,
    Segment,
    ServerMessageType,
    SessionError,
    SpeakerDiarizationConfig,
    SpeakerIdentifier,
    TimeoutError as SMTimeoutError,
    TranscriptionConfig,
    TranscriptionError,
    TransportError,
    TurnConfig,
    TurnDetectionMode as AgentTurnDetectionMode,
)

from .log import logger
from .version import __version__ as lk_version

# Endpoint resolution. The default is the EU real-time Agent STT host; the env var
# overrides it (e.g. to target another region or a self-hosted endpoint), and an explicit
# `base_url` argument overrides both. Every Agent STT endpoint ends in `AGENT_URL_PATH`;
# anything else speaks a protocol this plugin cannot read.
DEFAULT_BASE_URL = "wss://eu2.rt.speechmatics.com/v2/agent"
BASE_URL_ENV_VAR = "SPEECHMATICS_RT_URL"
AGENT_URL_PATH = "/v2/agent"

# Audio format we can actually send. The service fixes the sample rate at 16 kHz, and
# LiveKit frames are 16-bit PCM which we forward unconverted, so declaring any other
# encoding would describe the bytes wrongly rather than change them.
SUPPORTED_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_ENCODINGS = (AudioEncoding.PCM_S16LE,)

# Why each part of the pre-Agent-STT surface is inert, reported wherever a caller still
# reaches for it. See `_DROPPED_ARGS` and `STT.update_speakers`.
_SPEAKER_FOCUS_REMOVED = (
    "`SpeakerFocusMode` is not supported by Agent STT, so speaker focus is dropped from the "
    "config; it is expected to be reintroduced in a future release"
)
_EOU_REMOVED = "end-of-utterance timing is the service's own and is not configurable"
_LEGACY_ARG = "the argument predates Agent STT and was already unused"


class TurnDetectionMode(str, Enum):
    """How turn boundaries (end of speech) are detected.

    `VAD`: the STT service runs its own VAD and closes turns itself.

    `EXTERNAL`: turn boundaries are controlled by the caller — the service does not
    endpoint on its own, and the caller drives turns by calling `finalize()` (for
    example from an external VAD).

    The member names and values mirror the Agent STT SDK's own turn-detection modes so
    the two never drift.

    The plugin offered four modes before Agent STT: `EXTERNAL`, `FIXED`, `ADAPTIVE` and
    `SMART_TURN`. `EXTERNAL` carries over unchanged. The other three each named a
    service-side endpointing strategy, and Agent STT exposes exactly one, so they are kept
    only so existing code still runs and `_resolve_turn_detection_mode` reconciles them to
    `DEFAULT_TURN_DETECTION_MODE`.
    """

    VAD = AgentTurnDetectionMode.VAD.value
    EXTERNAL = AgentTurnDetectionMode.EXTERNAL.value

    # Deprecated, resolved to the default mode. This enum has always been the plugin's own
    # — no SDK declares these three — so the values are carried over verbatim from the
    # modes the plugin accepted before Agent STT. Remove after 2026-10-05.
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    SMART_TURN = "smart_turn"


# The mode a caller gets when they ask for none, and where the deprecated modes land.
# `EXTERNAL` for parity with the plugin's pre-Agent-STT default. Single source of truth:
# changing it moves the default and the deprecated modes together.
DEFAULT_TURN_DETECTION_MODE = TurnDetectionMode.EXTERNAL


@dataclasses.dataclass
class STTOptions:
    """Configuration parameters for Speechmatics STT service."""

    # Service configuration
    language: LanguageCode = LanguageCode("en")
    output_locale: str | None = None
    domain: str | None = None

    # Endpointing mode
    turn_detection_mode: TurnDetectionMode = DEFAULT_TURN_DETECTION_MODE

    # Output formatting
    speaker_format: str | None = None

    # Speakers
    known_speakers: list[SpeakerIdentifier] = dataclasses.field(default_factory=list)

    # Custom dictionary
    additional_vocab: list[AdditionalVocabEntry] = dataclasses.field(default_factory=list)

    # -------------------
    # Advanced features
    # -------------------

    # Features
    # The resolved model name (operating point). See `_resolve_model`.
    model: str = DEFAULT_MODEL.value
    include_partials: bool | None = None

    # Diarization
    enable_diarization: bool | None = None
    speaker_sensitivity: float | None = None
    max_speakers: int | None = None
    prefer_current_speaker: bool | None = None


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection_mode: TurnDetectionMode = DEFAULT_TURN_DETECTION_MODE,
        model: NotGivenOr[Model | str] = NOT_GIVEN,
        operating_point: NotGivenOr[Model | str] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        language: str = "en",
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        include_partials: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        speaker_sensitivity: NotGivenOr[float] = NOT_GIVEN,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
        speaker_format: NotGivenOr[str] = NOT_GIVEN,
        prefer_current_speaker: NotGivenOr[bool] = NOT_GIVEN,
        known_speakers: NotGivenOr[list[SpeakerIdentifier]] = NOT_GIVEN,
        sample_rate: int = 16000,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        **kwargs: Any,
    ):
        """Create a new instance of Speechmatics STT using Agent STT SDK.

        Args:
            api_key: Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable.

            base_url: Custom base URL for the API. Use this to target a specific
                Speechmatics real-time host (e.g. another region or a self-hosted
                endpoint). Can be set via `base_url` argument or `SPEECHMATICS_RT_URL`
                environment variable. Falls back to `DEFAULT_BASE_URL` (the EU real-time
                Agent STT host, `wss://eu2.rt.speechmatics.com/v2/agent`) when neither is
                set.

            turn_detection_mode: How end-of-speech turns are detected. `EXTERNAL` (the
                default) hands turn control to the caller, who drives it via `finalize()`
                — in practice from the `vad` passed below, since LiveKit does not call
                `finalize()` itself. With no `vad` and none loadable, `EXTERNAL` never
                closes a turn, so nothing is finalized. `VAD` instead lets the service
                and close turns itself; pair it with `turn_detection="stt"` on the
                `AgentSession`, which otherwise ignores the end-of-speech events this
                plugin emits. The deprecated `FIXED`, `ADAPTIVE` and `SMART_TURN` modes
                all resolve to the default mode with a warning. Defaults to
                `DEFAULT_TURN_DETECTION_MODE`.

            model: The transcription model to use, e.g. `"linden-1"`. A name agent-STT
                does not accept falls back to the default model with a warning. Defaults
                to the SDK's default model. Preferred over `operating_point`.

            operating_point: Deprecated alias for `model`; `model` wins if both are given.
                The old `enhanced` / `standard` operating points are not agent-STT models,
                so they fall back to the default model too. Optional.

            domain: Domain to use. Optional.

            language: Language code for the STT model. Defaults to `en`.

            output_locale: Output locale for the STT model, e.g. `en-GB`. Optional.

            include_partials: Whether the service sends partial segments, emitted as
                interim transcripts. Defaults to True.

            enable_diarization: Attribute words to distinct speakers. Defaults to True.

            additional_vocab: Vocabulary entries that bias the model toward specific
                words. Defaults to [].

            speaker_sensitivity: Diarization sensitivity between 0.0 and 1.0; higher
                values separate similar-sounding speakers more aggressively. Optional.

            max_speakers: Upper bound on the number of distinct speakers (2–100).
                Optional.

            speaker_format: Formatter for speaker-attributed text, with `text` and
                `speaker_id` placeholders, e.g. `@{speaker_id}: {text}`. Replaces the
                deprecated `speaker_active_format`. Defaults to the raw transcript.

            prefer_current_speaker: Bias closely-spaced words toward the same speaker.
                Optional.

            known_speakers: Known speaker identifiers, used to attribute words to the
                same speakers across sessions. Defaults to [].

            sample_rate: Audio sample rate in Hz. The service accepts 16000 only, so any
                other value raises a `ValueError`. Defaults to 16000.

            audio_encoding: Audio encoding format. LiveKit frames are 16-bit PCM and are
                forwarded unconverted, so `AudioEncoding.PCM_S16LE` is the only value that
                describes what is actually sent; anything else raises a `ValueError`.
                Defaults to `AudioEncoding.PCM_S16LE`.

            vad: External Voice Activity Detector, used only in `EXTERNAL` turn-detection
                mode where its end-of-speech drives `finalize()`. Ignored in `VAD` mode.
                When omitted in `EXTERNAL` mode, `livekit-plugins-silero` is loaded if it
                is installed, so a bare `STT()` still closes turns. Pass `vad=None` to opt
                out and drive `finalize()` yourself. Defaults to NOT_GIVEN.

            **kwargs: Catches deprecated parameters. A warning is logged for every name,
                whether it is a recognised deprecation or not.
        """

        # Normalize the deprecated modes away before anything reads the mode.
        turn_detection_mode = _resolve_turn_detection_mode(turn_detection_mode)

        # `EXTERNAL` is the default and the service does not endpoint in it, so a bare
        # `STT()` has nothing to close a turn with. Load a local VAD to drive `finalize()`,
        # as the plugin did before Agent STT, so zero-config still transcribes. An explicit
        # `vad=None` opts out: that caller drives `finalize()` themselves.
        if turn_detection_mode == TurnDetectionMode.EXTERNAL and not is_given(vad):
            vad = _load_default_vad()

        self._vad = vad if is_given(vad) else None

        # EXTERNAL mode needs something to close turns. The service does not endpoint on its own,
        # and LiveKit never calls STT.finalize() itself, so with no `vad` turns close only if the
        # caller drives finalize() by hand — otherwise nothing is ever finalized. Warn loudly
        # instead of silently producing no transcripts.
        if turn_detection_mode == TurnDetectionMode.EXTERNAL and self._vad is None:
            logger.warning(
                "Speechmatics STT is in EXTERNAL turn-detection mode with no `vad`: the service "
                "will not endpoint on its own and LiveKit does not call finalize() for you, so "
                "turns close only if you call STT.finalize() yourself. Pass a `vad` to drive "
                "finalize() from end-of-speech, or use turn_detection_mode=VAD for service-side "
                "endpointing."
            )

        # Set STT options
        def _set(value: Any) -> Any:
            return value if is_given(value) else None

        # Create STT options from parameters
        opts = STTOptions(
            language=LanguageCode(language),
            output_locale=_set(output_locale),
            domain=_set(domain),
            turn_detection_mode=turn_detection_mode,
            speaker_format=_set(speaker_format),
            known_speakers=_set(known_speakers) or [],
            additional_vocab=_normalize_additional_vocab(_set(additional_vocab) or []),
            model=_resolve_model(model, operating_point),
            include_partials=_set(include_partials),
            enable_diarization=enable_diarization if is_given(enable_diarization) else True,
            speaker_sensitivity=_set(speaker_sensitivity),
            max_speakers=_set(max_speakers),
            prefer_current_speaker=_set(prefer_current_speaker),
        )

        # Migrate / warn about any deprecated kwargs before anything reads the options
        _check_deprecated_args(kwargs, opts)

        # Capabilities mirror the resolved options, so a value that arrived through a
        # deprecated alias is reflected too. Unset partials means the service default,
        # which is on.
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=opts.include_partials is not False,
                diarization=opts.enable_diarization is not False,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )

        self._stt_options = opts

        # Validate config options
        errors = self._validate_stt_options()
        if errors:
            raise ValueError("Invalid STT options: " + ", ".join(errors))

        # Set API key
        self._api_key: str = api_key if is_given(api_key) else os.getenv("SPEECHMATICS_API_KEY", "")

        # Set base URL
        self._base_url: str = _resolve_base_url(base_url)

        # Validate API key and base URL
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Set audio parameters. Neither is a free choice: an unsupported rate is rejected
        # by the service, and an encoding we do not actually send would mislabel the audio.
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SUPPORTED_SAMPLE_RATE}")
        if audio_encoding not in SUPPORTED_AUDIO_ENCODINGS:
            supported = ", ".join(e.value for e in SUPPORTED_AUDIO_ENCODINGS)
            raise ValueError(f"audio_encoding must be one of: {supported}")

        self._sample_rate = sample_rate
        self._audio_encoding = audio_encoding

        # Initialize list of streams
        self._streams: list[SpeechStream] = []

    @property
    def provider(self) -> str:
        return "Speechmatics"

    @property
    def model(self) -> str:
        return self._stt_options.model

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        """Create a new SpeechStream."""

        # Create the stream
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            config=self._prepare_config(language),
            turn_detection_mode=self._stt_options.turn_detection_mode,
            id=len(self._streams),
            vad_instance=self._vad,
        )

        # Add to the list of streams
        self._streams.append(stream)

        # Return the stream
        return stream

    def _validate_stt_options(self) -> list[str]:
        """Validate options in STTOptions."""
        errors: list[str] = []
        opts = self._stt_options

        # server rejects speaker counts outside 2–100
        if opts.max_speakers is not None and not (1 < opts.max_speakers <= 100):
            errors.append("max_speakers must be between 2 and 100")

        # diarization sensitivity range enforced by the engine
        if opts.speaker_sensitivity is not None and not (0.0 <= opts.speaker_sensitivity <= 1.0):
            errors.append("speaker_sensitivity must be between 0.0 and 1.0")

        return errors

    def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN) -> TranscriptionConfig:
        """Prepare an Agent STT TranscriptionConfig from STTOptions.

        This is the only place the config crosses from the plugin's public options into
        the Agent STT session driver. Only the fields agent-STT accepts on the wire are set.

        Turn detection is configured separately: it is a top-level `turn_config` sibling of
        `transcription_config` on the wire, passed to the client as a `TurnConfig` (see
        `SpeechStream._run`), not a member of this config.
        """

        # Reference to STT options
        opts = self._stt_options

        return TranscriptionConfig(
            language=language if is_given(language) else opts.language,
            # The SDK types `model` as the `Model` enum, but the proxy resolves any model
            # string (see its docstring), and `opts.model` is the resolved wire string.
            model=cast(Model, opts.model),
            diarization="speaker" if opts.enable_diarization else None,
            speaker_diarization_config=_build_diarization_config(opts),
            # `additional_vocab` accepts entries or raw dicts; our list is entries only, and
            # list invariance makes the narrower type not assignable — widen it for the SDK.
            additional_vocab=cast(
                "list[AdditionalVocabEntry | dict[str, Any]] | None",
                opts.additional_vocab or None,
            ),
            output_locale=opts.output_locale,
            domain=opts.domain,
            enable_partials=opts.include_partials,
        )

    def finalize(self) -> None:
        """Force the current turn to end, flushing buffered words as final segments.

        Only takes effect in `EXTERNAL` turn-detection mode; in `VAD` mode the
        service endpoints on its own and this is a no-op.
        """

        # Iterate over the streams
        for stream in self._streams:
            # Do not finalize if being handled by a client
            if not stream._client or not stream._client.is_connected:
                continue

            # Only finalize() if EXTERNAL turn_detection_mode is selected
            if stream._turn_detection_mode == TurnDetectionMode.EXTERNAL:
                stream._client.finalize()

    async def get_speaker_ids(
        self,
    ) -> list[SpeakerIdentifier] | list[list[SpeakerIdentifier]]:
        """Get the list of speakers from the current STT session.

        If diarization is enabled, then this will use the GET_SPEAKERS message
        to retrieve the list of speakers for the current session. This should
        be used once speakers have said at least 5 words to improve the results.

        Returns:
            list[SpeakerIdentifier]: List of speakers in the session.
        """

        # Results
        results: list[list[SpeakerIdentifier]] = []

        # Iterate over all streams
        for idx, stream in enumerate(self._streams):
            # Skip streams that aren't actively connected
            if stream._client is None or not stream._client.is_connected:
                logger.warning(f"Not connected in stream {idx}")
                results.append([])
                continue

            # Return if diarization is not enabled
            if not stream._config.diarization:
                logger.warning(f"Diarization is not enabled in stream {idx}")
                results.append([])
                continue

            # Clear the speaker result
            stream._speaker_result_event.clear()

            # Send message to client
            await stream._client.send_message({"message": ClientMessageType.GET_SPEAKERS.value})

            # Wait the result (5 second timeout)
            try:
                await asyncio.wait_for(
                    stream._speaker_result_event.wait(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"GetSpeakers timed-out for stream {idx}")
                results.append([])
                continue

            # Return the list of speakers
            results.append(stream._speaker_result or [])

        # Return the list of speakers
        if len(results) == 1:
            return results[0]
        return results

    def update_speakers(
        self,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: Any = NOT_GIVEN,
    ) -> None:
        """Deprecated no-op, kept so code written against the previous plugin still runs.

        Agent STT does not support speaker focus, so there is nothing to update mid-session
        and the arguments are ignored. Diarization is unaffected — see `get_speaker_ids`.
        """
        logger.warning(f"`STT.update_speakers()` is deprecated: {_SPEAKER_FOCUS_REMOVED}")


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions,
        config: TranscriptionConfig,
        turn_detection_mode: TurnDetectionMode,
        id: int,
        vad_instance: vad.VAD | None = None,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=stt._sample_rate,
        )

        self._stt: STT = stt
        self._id: int = id
        self._config: TranscriptionConfig = config
        self._turn_detection_mode: TurnDetectionMode = turn_detection_mode
        self._client: AgentSttAsyncClient | None = None
        self._msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._speech_duration: float = 0

        self._vad: vad.VAD | None = vad_instance
        self._vad_stream: vad.VADStream | None = None

        self._tasks: list[asyncio.Task] = []

        # Speaker result event
        self._speaker_result_event: asyncio.Event = asyncio.Event()
        self._speaker_result: list[SpeakerIdentifier] | None = None

    @property
    def session_id(self) -> str | None:
        """The service-assigned session id, set once `RecognitionStarted` arrives.

        Read from the client's `session_info`. Returns `None` before the session starts
        and after the stream is closed (which drops the client).
        """
        if self._client is None:
            return None
        info = self._client.session_info
        return info.session_id if info is not None else None

    async def _run(self) -> None:
        """Run the STT stream."""
        logger.debug("Connecting to Speechmatics STT service")

        # Config is required
        if not self._config:
            raise ValueError("Config is required")

        # Create the Agent STT client. Turn detection is a top-level `turn_config` (a sibling
        # of the transcription config on the wire), not a field on the transcription config;
        # audio encoding / sample rate go via AudioFormat.
        self._client = AgentSttAsyncClient(
            api_key=self._stt._api_key,
            url=self._stt._base_url,
            app=f"livekit/{lk_version}",
            transcription_config=self._config,
            turn_config=TurnConfig(
                turn_detection_mode=_handle_turn_detection_mode(self._turn_detection_mode)
            ),
            audio_format=AudioFormat(
                encoding=self._stt._audio_encoding,
                sample_rate=self._stt._sample_rate,
                chunk_size=DEFAULT_CHUNK_SIZE,
            ),
        )

        # Add message handlers
        def add_message(message: dict[str, Any]) -> None:
            self._msg_queue.put_nowait(message)

        # Default messages to listen to
        messages: list[ServerMessageType] = [
            ServerMessageType.RECOGNITION_STARTED,
            ServerMessageType.INFO,
            ServerMessageType.ERROR,
            ServerMessageType.WARNING,
            ServerMessageType.ADD_PARTIAL_SEGMENT,
            ServerMessageType.ADD_SEGMENT,
            ServerMessageType.START_OF_TURN,
            ServerMessageType.END_OF_TURN,
        ]

        # Speaker IDs message handler
        if self._config.diarization:
            messages.append(ServerMessageType.SPEAKERS_RESULT)

        # Add message handlers
        for event in messages:
            self._client.on(event, add_message)  # type: ignore[arg-type]

        # Connect to the service. A rejected session (e.g. invalid config) surfaces as an
        # APIError so it reaches the caller instead of dying in a log; transient transport
        # failures surface as a (retryable) APIConnectionError.
        try:
            await self._client.connect()
        except (TranscriptionError, SessionError) as e:
            raise APIError(f"Speechmatics rejected the session: {e}", retryable=False) from e
        except (SMConnectionError, SMTimeoutError, TransportError) as e:
            raise APIConnectionError(f"failed to connect to Speechmatics: {e}") from e
        logger.debug("Connected to Speechmatics STT service")

        # Open external VAD stream (if provided) before tasks start pushing frames.
        # Only in EXTERNAL mode: the VAD exists solely to drive finalize(), and in VAD
        # mode the server endpoints itself, so finalizing here would double-endpoint.
        if self._vad is not None and self._turn_detection_mode == TurnDetectionMode.EXTERNAL:
            self._vad_stream = self._vad.stream()

        # Audio and messaging tasks
        audio_task = asyncio.create_task(self._process_audio())
        message_task = asyncio.create_task(self._process_messages())

        # Tasks
        self._tasks = [audio_task, message_task]

        # Optional VAD task: calls `client.finalize()` on end of speech
        vad_task: asyncio.Task | None = None
        if self._vad_stream is not None:
            vad_task = asyncio.create_task(self._process_vad(self._vad_stream))
            self._tasks.append(vad_task)

        # Wait for tasks to complete
        try:
            done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()

        # Disconnect the client
        finally:
            # Cancel audio first — stops sending audio to the STT engine
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass

            # Close the VAD stream so its task drains and exits
            if self._vad_stream is not None:
                await self._vad_stream.aclose()
                self._vad_stream = None

            if vad_task is not None:
                await utils.aio.cancel_and_wait(vad_task)

            # Disconnect flushes final messages from the STT engine
            await self._client.disconnect()

            # Cancel message task after disconnect — final messages have been processed
            message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass

            # Remove from active streams so stale streams aren't iterated
            if self in self._stt._streams:
                self._stt._streams.remove(self)

    async def _process_audio(self) -> None:
        """Process audio from the input channel."""
        try:
            # Input audio stream
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._stt._sample_rate,
                num_channels=1,
            )

            # Process input audio
            async for data in self._input_ch:
                # Handle flush sentinel
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    # Forward the original frame to the VAD before resampling/repacking
                    if self._vad_stream is not None:
                        self._vad_stream.push_frame(data)
                    frames = audio_bstream.write(data.data.tobytes())

                if self._client:
                    for frame in frames:
                        await self._client.send_audio(frame.data.tobytes())

                        # send_audio never raises: it closes the audio gate and drops
                        # this and every later frame. A session error is already raised
                        # as fatal in _handle_message, so only a clean gate close is a
                        # lost connection.
                        if not self._client.is_ready_for_audio:
                            if self._client.session_error is None:
                                raise APIConnectionError(
                                    "lost connection to Speechmatics while sending audio"
                                )
                            break

                        # Only audio the service accepted counts towards usage.
                        self._speech_duration += frame.duration

            # No more input — let the VAD flush any pending event
            if self._vad_stream is not None:
                self._vad_stream.end_input()

        except asyncio.CancelledError:
            pass

    async def _process_vad(self, vad_stream: vad.VADStream) -> None:
        """Call `client.finalize()` whenever the external VAD reports end of speech."""
        try:
            async for ev in vad_stream:
                if ev.type == vad.VADEventType.END_OF_SPEECH:
                    if self._client and self._client.is_connected:
                        self._client.finalize()
        except asyncio.CancelledError:
            pass

    async def _process_messages(self) -> None:
        """Process messages from the STT client."""
        try:
            while True:
                message = await self._msg_queue.get()
                self._handle_message(message)
        except asyncio.CancelledError:
            pass

    def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the STT client."""

        # Get the message type
        event = message.get("message", None)

        # Only handle valid messages
        if event is None:
            return

        # Log info, error and warning messages
        elif event in [
            ServerMessageType.RECOGNITION_STARTED,
            ServerMessageType.INFO,
        ]:
            logger.info(f"received {event} message", extra={"lk.pii.message": message})
        elif event == ServerMessageType.WARNING:
            logger.warning(f"received {event} message", extra={"lk.pii.message": message})
        elif event == ServerMessageType.ERROR:
            # An agent-STT `Error` means the session is over. Raise it as an APIError so the
            # stream fails with the reason instead of silently hanging until timeout. It
            # propagates out of the message task and is re-raised by `_run`.
            reason = message.get("reason", "unknown")
            error_type = message.get("type", "error")
            raise APIError(
                f"Speechmatics returned an error [{error_type}]: {reason}",
                body=message,
                retryable=False,
            )

        # Handle the messages
        elif event == ServerMessageType.ADD_PARTIAL_SEGMENT:
            self._handle_segment(message, is_final=False)
        elif event == ServerMessageType.ADD_SEGMENT:
            self._handle_segment(message, is_final=True)
        elif event == ServerMessageType.START_OF_TURN:
            self._handle_start_of_turn(message)
        elif event == ServerMessageType.END_OF_TURN:
            self._handle_end_of_turn(message)

        # Handle the speaker result message
        elif event == ServerMessageType.SPEAKERS_RESULT:
            self._handle_speakers_result(message)

        # Log all other messages
        else:
            logger.debug(f"received {event} message", extra={"lk.pii.message": message})

    def _handle_segment(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle AddSegment / AddPartialSegment events.

        agent-STT sends a singular `segment` (`transcript`/`speaker`) plus message-level
        `metadata`; `Segment.from_message` parses it into a SpeechData event.
        """
        opts = self._stt._stt_options

        # Parse the singular agent-STT segment
        seg = Segment.from_message(message)

        # Segments the service did not attribute are labelled UU
        speaker_id = seg.speaker or "UU"
        format_str = opts.speaker_format or "{text}"
        text = format_str.format(speaker_id=speaker_id, text=seg.transcript)

        # Create speech event. Label with the stream's own configured language (which honors a
        # per-stream `stream(language=...)` override), not the constructor default, so the labeled
        # language matches the one recognition actually ran with.
        speech_data = stt.SpeechData(
            language=LanguageCode(self._config.language),
            text=text,
            speaker_id=speaker_id,
            start_time=seg.start_time + self.start_time_offset,
            end_time=seg.end_time + self.start_time_offset,
        )

        # Determine the event type
        event_type = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )

        # Send the event
        self._event_ch.send_nowait(stt.SpeechEvent(type=event_type, alternatives=[speech_data]))

    def _handle_start_of_turn(self, message: dict[str, Any]) -> None:
        """Handle StartOfTurn events."""
        logger.debug("StartOfTurn received")
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events."""
        logger.debug("EndOfTurn received")
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        if self._speech_duration > 0.0:
            usage_event = stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=self._speech_duration),
            )
            self._event_ch.send_nowait(usage_event)
            self._speech_duration = 0

    def _handle_speakers_result(self, message: dict[str, Any]) -> None:
        """Handle SpeakersResult events."""
        logger.debug("SpeakersResult received")
        self._speaker_result = message.get("speakers", [])
        self._speaker_result_event.set()

    async def aclose(self) -> None:
        """Close the STT stream."""
        await super().aclose()

        # Cancel message processing task
        if self._tasks:
            for task in self._tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close the VAD stream if it's still open
        if self._vad_stream is not None:
            await self._vad_stream.aclose()
            self._vad_stream = None

        # Close the client
        if self._client and self._client.is_connected:
            await self._client.disconnect()
        self._client = None

        # Remove from active streams
        if self in self._stt._streams:
            self._stt._streams.remove(self)


def _build_diarization_config(opts: STTOptions) -> SpeakerDiarizationConfig | None:
    """Build the wire `speaker_diarization_config` from the diarization options.

    Returns `None` when diarization is off or no diarization knob was set, so an empty
    config is never sent. Only the fields the caller actually set are included.
    """
    if not opts.enable_diarization:
        return None

    fields: dict[str, Any] = {}
    if opts.max_speakers is not None:
        fields["max_speakers"] = opts.max_speakers
    if opts.speaker_sensitivity is not None:
        fields["speaker_sensitivity"] = opts.speaker_sensitivity
    if opts.prefer_current_speaker is not None:
        fields["prefer_current_speaker"] = opts.prefer_current_speaker
    if opts.known_speakers:
        fields["speakers"] = opts.known_speakers

    return SpeakerDiarizationConfig(**fields) if fields else None


# The pre-Agent-STT modes, all of which named a service-side endpointing strategy.
# Remove after 2026-10-05.
_DEPRECATED_TURN_DETECTION_MODES = frozenset(
    {
        TurnDetectionMode.FIXED,
        TurnDetectionMode.ADAPTIVE,
        TurnDetectionMode.SMART_TURN,
    }
)


def _load_default_vad() -> vad.VAD | None:
    """Load a local VAD to drive `finalize()` in `EXTERNAL` mode, or `None` if unavailable.

    The pre-Agent-STT plugin loaded Silero here and raised `ImportError` when it was
    missing, which made the plugin unusable wherever the optional package was not
    installed. Returning `None` instead lets construction succeed and leaves the caller
    with the `EXTERNAL`-without-a-`vad` warning, which says what to do about it.

    Deliberately not `livekit.agents.inference.VAD`: that one is bundled, so it would load
    for everybody and run a second VAD over the same audio as the one `AgentSession` fills
    in for itself, with independent thresholds.
    """
    try:
        from livekit.plugins.silero import VAD as SileroVAD
    except ImportError:
        return None

    return SileroVAD.load()


def _normalize_additional_vocab(entries: list[Any]) -> list[AdditionalVocabEntry]:
    """Accept vocab entries from either SDK, returning entries agent-STT can serialize.

    The pre-Agent-STT plugin took `speechmatics.voice`'s pydantic `AdditionalVocabEntry`.
    The Agent STT config carries an unrecognised object straight into
    `transcription_config.additional_vocab`, where the JSON encode then refuses it — so
    without this the old class fails when the session starts rather than at construction.
    Both classes have the same two fields, so the value survives the copy.

    Entries are duck-typed on `content` instead of matched against the voice class, which
    keeps the voice SDK off the runtime path. Dicts pass through: the SDK accepts them.

    No warning: the two classes hold the same two fields, so the entry is copied without
    losing anything and the word reaches the engine exactly as asked. There is nothing for
    the caller to do differently.

    Raises:
        ValueError: if an entry is neither a mapping nor an object with `content`.
    """
    normalized: list[AdditionalVocabEntry] = []

    for entry in entries:
        if isinstance(entry, (AdditionalVocabEntry, dict)):
            normalized.append(cast(AdditionalVocabEntry, entry))
            continue

        content = getattr(entry, "content", None)
        if content is None:
            raise ValueError(
                f"`additional_vocab` entry {entry!r} is not an `AdditionalVocabEntry`: no "
                "`content` to read"
            )

        normalized.append(
            AdditionalVocabEntry(content=content, sounds_like=getattr(entry, "sounds_like", None))
        )

    return normalized


def _resolve_turn_detection_mode(mode: TurnDetectionMode) -> TurnDetectionMode:
    """Reconcile the pre-Agent-STT turn detection modes with the two that remain.

    `FIXED`, `ADAPTIVE` and `SMART_TURN` each selected one of the old engine's
    service-side endpointing strategies. Agent-STT exposes none of them by name, so all
    three resolve to `DEFAULT_TURN_DETECTION_MODE`; `EXTERNAL` keeps its meaning and
    passes through.

    `FIXED` is the lossy case: it timed turns from `end_of_utterance_silence_trigger`,
    which agent-STT does not support, so the service's own timing applies instead. It
    warns separately rather than resolving quietly.

    Returns:
        `VAD` or `EXTERNAL` — the only modes the rest of the plugin handles.
    """
    if mode not in _DEPRECATED_TURN_DETECTION_MODES:
        return mode

    logger.warning(
        f"`TurnDetectionMode.{mode.name}` is deprecated and will be removed after 2026-10-05; "
        f"it resolves to `TurnDetectionMode.{DEFAULT_TURN_DETECTION_MODE.name}`, the default mode"
    )
    if mode is TurnDetectionMode.FIXED:
        logger.warning(
            "`TurnDetectionMode.FIXED` timed turns with `end_of_utterance_silence_trigger`, "
            "which agent-STT does not support: the service's own endpointing timing applies"
        )
    return DEFAULT_TURN_DETECTION_MODE


def _handle_turn_detection_mode(mode: TurnDetectionMode) -> AgentTurnDetectionMode:
    """Map the plugin's turn detection mode onto the session driver's.

    - `VAD`      -> the service runs its own VAD and closes turns.
    - `EXTERNAL` -> the caller closes turns by calling `finalize()`.

    The plugin's enum values are the SDK's values, so this is a direct lookup — but it
    is still required: `TurnConfig.to_dict()` reads the enum member's value, so the
    `TurnConfig` must carry the SDK's own enum member, not the plugin's.
    """
    return AgentTurnDetectionMode(mode.value)


def _resolve_base_url(base_url: NotGivenOr[str]) -> str:
    """Resolve the STT endpoint URL.

    Precedence (highest first):
        1. the explicit `base_url` argument
        2. the ``SPEECHMATICS_RT_URL`` environment variable
        3. ``DEFAULT_BASE_URL``

    Warns if the result is not an Agent STT endpoint. The check covers all three sources,
    which matters most for the environment variable: it survives an upgrade untouched, so
    a value left pointing at `/v2` silently keeps the caller on the old protocol.
    """
    resolved = base_url if is_given(base_url) else os.getenv(BASE_URL_ENV_VAR, DEFAULT_BASE_URL)

    if not urlparse(resolved).path.rstrip("/").endswith(AGENT_URL_PATH):
        logger.warning(
            f"{resolved!r} is not an Agent STT endpoint: those end in {AGENT_URL_PATH!r}. This "
            "plugin only reads the Agent STT protocol, so the session will connect and then "
            "produce no transcripts — if your transcripts are missing, this is why."
        )

    return resolved


def _model_name(value: Model | str) -> str:
    """Normalize a model value to its wire string.

    Accepts an enum member (`Model`, or any future `str` enum) or a plain string,
    and returns the string the service reads.
    """
    return value.value if isinstance(value, Enum) else str(value)


def _resolve_model(
    model: NotGivenOr[Model | str],
    operating_point: NotGivenOr[Model | str],
) -> str:
    """Reconcile the preferred `model` with its deprecated `operating_point` alias.

    Rules:
        - neither given          -> the SDK's default model
        - only `operating_point` -> use it, with a deprecation warning
        - only `model`           -> use it
        - both given             -> `model` wins, with a warning if the two differ

    This never raises. A name agent-STT does not accept — a typo, or one of the `enhanced` /
    `standard` operating points the plugin took before Agent STT — falls back to the default
    model with a warning, so old code keeps transcribing instead of dying at construction.

    Returns:
        The resolved model name as a string.
    """
    resolved_model = _model_name(model) if is_given(model) else None
    resolved_op = _model_name(operating_point) if is_given(operating_point) else None

    if resolved_op is not None:
        logger.warning(
            "`operating_point` is deprecated and will be removed in a future release; "
            "use `model` instead"
        )

    if resolved_model and resolved_op and resolved_model != resolved_op:
        logger.warning(
            f"`model` ({resolved_model!r}) and `operating_point` ({resolved_op!r}) name "
            f"different options; using {resolved_model!r}."
        )

    resolved = resolved_model or resolved_op
    if not resolved:
        return DEFAULT_MODEL.value

    supported = [m.value for m in Model]
    if resolved in supported:
        return resolved

    logger.warning(
        f"{resolved!r} is not a model agent-STT accepts ({', '.join(supported)}); using "
        f"{DEFAULT_MODEL.value!r} instead. The old `enhanced` and `standard` operating points "
        f"belong to the previous service and have no Agent STT equivalent."
    )
    return DEFAULT_MODEL.value


# Deprecated arguments with no agent-STT equivalent, each with the reason it is gone.
# Accepted so upgrading does not break construction, but they reach neither the config nor
# the wire.
_DROPPED_ARGS: dict[str, str] = {
    "audio_settings": _LEGACY_ARG,
    "chunk_size": _LEGACY_ARG,
    "end_of_turn_config": _EOU_REMOVED,
    "end_of_utterance_max_delay": _EOU_REMOVED,
    "end_of_utterance_mode": _EOU_REMOVED,
    "end_of_utterance_silence_trigger": _EOU_REMOVED,
    "focus_mode": _SPEAKER_FOCUS_REMOVED,
    "focus_speakers": _SPEAKER_FOCUS_REMOVED,
    "http_session": _LEGACY_ARG,
    "ignore_speakers": _SPEAKER_FOCUS_REMOVED,
    "max_delay": "Agent STT rejects it",
    "punctuation_overrides": "Agent STT rejects it",
    "speaker_passive_format": "a segment has no active/passive split to format",
    "transcription_config": _LEGACY_ARG,
    "vad_config": "turn detection is set with `turn_detection_mode`",
}

# Deprecated arguments that were renamed: old name -> (STTOptions field, value coercion).
_MIGRATED_ARGS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "diarization_sensitivity": ("speaker_sensitivity", float),
    "enable_partials": ("include_partials", bool),
    "speaker_active_format": ("speaker_format", str),
}


def _check_deprecated_args(kwargs: dict[str, Any], opts: STTOptions) -> None:
    """Warn about deprecated kwargs, and migrate the ones that still have an equivalent.

    Anything in `_DROPPED_ARGS` is reported and discarded; anything in `_MIGRATED_ARGS`
    is carried over to the option that replaced it.
    """

    for name, reason in _DROPPED_ARGS.items():
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and ignored: {reason}")

    for name, (replacement, coerce) in _MIGRATED_ARGS.items():
        if name not in kwargs:
            continue

        # An explicit new-style argument always wins over the deprecated alias.
        if getattr(opts, replacement) is not None:
            logger.warning(f"Both `{name}` and `{replacement}` provided; using `{replacement}`")
            continue

        try:
            value = coerce(kwargs[name])
        except (TypeError, ValueError) as e:
            raise ValueError(f"`{name}` has an invalid value: {e}") from e

        logger.warning(f"`{name}` is deprecated, migrated to `{replacement}`")
        setattr(opts, replacement, value)

    # Anything left is a typo or an argument from some other plugin. `**kwargs` swallows it,
    # so without this the caller never learns why their setting had no effect.
    unrecognized = sorted(set(kwargs) - _DROPPED_ARGS.keys() - _MIGRATED_ARGS.keys())
    if unrecognized:
        logger.warning(
            f"unrecognized argument(s) ignored: {', '.join(unrecognized)}. Check for a typo — "
            "nothing reads them."
        )
