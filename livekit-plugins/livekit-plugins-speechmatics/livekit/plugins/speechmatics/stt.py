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
import warnings
from enum import Enum
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
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

# Session driver: the Agent STT SDK. Only the transport/config/message shapes come from
# here — the plugin's public API (below) is unchanged and still speaks the voice-SDK types.
from speechmatics.agent_stt import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    REMOVE,
    AgentSttAsyncClient,
    AudioFormat,
    ClientMessageType,
    Model,
    Segment,
    ServerMessageType,
    TranscriptionConfig,
)
from speechmatics.agent_stt import (
    TurnDetectionMode as AgentTurnDetectionMode,
)

# Public-API types — unchanged from the voice-SDK plugin surface.
from speechmatics.voice import (
    AdditionalVocabEntry,
    AudioEncoding,
    OperatingPoint,
    SpeakerFocusMode,
    SpeakerIdentifier,
)

from ._debug import dd  # noqa: F401  # debug-only dump-and-die helper
from .log import logger
from .version import __version__ as lk_version


class TurnDetectionMode(str, Enum):
    """How turn boundaries (end of speech) are detected.

    `DEFAULT`: the STT service runs its own VAD and closes turns itself.

    `EXTERNAL`: turn boundaries are controlled by the caller — the service does not
    endpoint on its own, and the caller drives turns by calling `finalize()` (for
    example from an external VAD).

    The values mirror the Agent STT SDK's own turn-detection modes so the two never
    drift; only the member names differ (DEFAULT=VAD)
    """

    DEFAULT = AgentTurnDetectionMode.VAD.value
    EXTERNAL = AgentTurnDetectionMode.EXTERNAL.value


@dataclasses.dataclass
class STTOptions:
    """Configuration parameters for Speechmatics STT service."""

    # Service configuration
    language: LanguageCode = LanguageCode("en")
    output_locale: str | None = None
    domain: str | None = None

    # Endpointing mode
    turn_detection_mode: TurnDetectionMode = TurnDetectionMode.DEFAULT

    # Output formatting
    speaker_active_format: str | None = None
    speaker_passive_format: str | None = None

    # Speakers
    focus_speakers: list[str] = dataclasses.field(default_factory=list)
    ignore_speakers: list[str] = dataclasses.field(default_factory=list)
    focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN
    known_speakers: list[SpeakerIdentifier] = dataclasses.field(default_factory=list)

    # Custom dictionary
    additional_vocab: list[AdditionalVocabEntry] = dataclasses.field(default_factory=list)

    # -------------------
    # Advanced features
    # -------------------

    # Features
    # The resolved model name (operating point). See `_resolve_model`.
    model: str = DEFAULT_MODEL.value
    max_delay: float | None = None
    end_of_utterance_silence_trigger: float | None = None
    end_of_utterance_max_delay: float | None = None
    punctuation_overrides: dict | None = None
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
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.DEFAULT,
        model: NotGivenOr[Model | str] = NOT_GIVEN,
        operating_point: NotGivenOr[OperatingPoint | Model | str] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        language: str = "en",
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        include_partials: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        max_delay: NotGivenOr[float] = NOT_GIVEN,
        end_of_utterance_silence_trigger: NotGivenOr[float] = NOT_GIVEN,
        end_of_utterance_max_delay: NotGivenOr[float] = NOT_GIVEN,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        punctuation_overrides: NotGivenOr[dict] = NOT_GIVEN,
        speaker_sensitivity: NotGivenOr[float] = NOT_GIVEN,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
        speaker_active_format: NotGivenOr[str] = NOT_GIVEN,
        speaker_passive_format: NotGivenOr[str] = NOT_GIVEN,
        prefer_current_speaker: NotGivenOr[bool] = NOT_GIVEN,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN,
        known_speakers: NotGivenOr[list[SpeakerIdentifier]] = NOT_GIVEN,
        sample_rate: int = 16000,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        **kwargs: Any,
    ):
        """Create a new instance of Speechmatics STT.

        Args:
            api_key: Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable.

            base_url: Custom base URL for the API. Can be set via `base_url`
                argument or `SPEECHMATICS_RT_URL` environment variable. Optional.

            turn_detection_mode: How end-of-speech turns are detected. `DEFAULT` lets
                the STT service run its own VAD and close turns itself. `EXTERNAL`
                hands turn control to the caller, who drives it via `finalize()` (e.g.
                from an external VAD). Defaults to `TurnDetectionMode.DEFAULT`.

            model: The transcription model (operating point) to use, e.g. `"linden-1"`.
                Defaults to the SDK's default model. Preferred over `operating_point`.

            operating_point: Deprecated alias for `model`. If both are given they must
                name the same value, otherwise a `ValueError` is raised. Optional.

            domain: Domain to use. Optional.

            language: Language code for the STT model. Defaults to `en`.

            output_locale: Output locale for the STT model, e.g. `en-GB`. Optional.

            include_partials: Include partial segment fragments (words) in the output
                of AddPartialSegment messages. Partial fragments from the STT will
                always be used for speaker activity detection. This setting is used
                only for the formatted text output of individual segments. Optional.

            enable_diarization: Enable speaker diarization. When enabled, the STT
                engine will determine and attribute words to unique speakers.
                Overrides preset if provided. Defaults to True.

            max_delay: Maximum delay in seconds for transcription. This forces the
                STT engine to speed up the processing of transcribed words and reduces
                the interval between partial and final results. Lower values can have
                an impact on accuracy. Overrides preset if provided. Optional.

            end_of_utterance_silence_trigger: Silence duration in seconds that
                triggers end of utterance. The delay is used to wait for any further
                transcribed words before emitting the `FINAL_TRANSCRIPT` events.
                Overrides preset if provided. Optional.

            end_of_utterance_max_delay: Maximum delay in seconds for end of utterance.
                Must be greater than `end_of_utterance_silence_trigger`.
                Overrides preset if provided. Optional.

            additional_vocab: List of additional vocabulary entries to increase the
                weight of specific words in the transcription model. Defaults to [].

            punctuation_overrides: Punctuation overrides. Allows overriding the
                punctuation behaviour in the STT engine. Overrides preset if provided.
                Optional.

            speaker_sensitivity: Diarization sensitivity. A higher value increases the
                sensitivity of diarization and helps when two or more speakers have
                similar voices. Overrides preset if provided. Optional.

            max_speakers: Maximum number of speakers to detect during diarization.
                When set, the STT engine will limit the number of unique speakers
                identified. Overrides preset if provided. Optional.

            speaker_active_format: Formatter for active speaker output. The attributes
                `text` and `speaker_id` are available. Example: `@{speaker_id}: {text}`.
                Defaults to transcription output.

            speaker_passive_format: Formatter for passive speaker output. The attributes
                `text` and `speaker_id` are available. Example:
                `@{speaker_id} [background]: {text}`. Defaults to transcription output.

            prefer_current_speaker: When True, groups of words close together are given
                extra weight to be identified as the same speaker. Overrides preset if
                provided. Optional.

            focus_speakers: List of speaker IDs to focus on. Only these speakers are
                emitted as `FINAL_TRANSCRIPT` events; others are treated as passive.
                Words from passive speakers are still processed but only emitted when a
                focused speaker has also said new words. Defaults to [].

            ignore_speakers: List of speaker IDs to ignore. These speakers are excluded
                from transcription and their speech will not trigger VAD or end of
                utterance detection. By default, any speaker with a label wrapped in
                double underscores (e.g. `__ASSISTANT__`) is excluded. Defaults to [].

            focus_mode: Controls what happens to words from non-focused speakers. When
                `RETAIN`, non-ignored speakers are processed as passive frames. When
                `IGNORE`, their words are discarded entirely. Defaults to
                `SpeakerFocusMode.RETAIN`.

            known_speakers: List of known speaker labels and identifiers. When supplied,
                the STT engine uses them to attribute words to specific speakers across
                sessions. Defaults to [].

            sample_rate: Audio sample rate in Hz. Defaults to 16000.

            audio_encoding: Audio encoding format. Defaults to `AudioEncoding.PCM_S16LE`.

            vad: Optional external Voice Activity Detector. When provided, each audio
                frame is forwarded to the VAD and `finalize()` is called whenever the
                VAD reports end of speech. None is auto-loaded when omitted.
                Defaults to NOT_GIVEN.

            **kwargs: Catches deprecated parameters. A warning is logged for any
                recognised deprecated name.
        """

        # An external VAD, if provided, drives finalize(); none is auto-loaded.
        self._vad = vad if is_given(vad) else None

        # Set default values for optional parameters
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=enable_diarization if is_given(enable_diarization) else True,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )

        # Set STT options
        def _set(value: Any) -> Any:
            return value if is_given(value) else None

        # Create STT options from parameters
        self._stt_options = STTOptions(
            language=LanguageCode(language),
            output_locale=_set(output_locale),
            domain=_set(domain),
            turn_detection_mode=turn_detection_mode,
            speaker_active_format=_set(speaker_active_format),
            speaker_passive_format=_set(speaker_passive_format),
            focus_speakers=_set(focus_speakers) or [],
            ignore_speakers=_set(ignore_speakers) or [],
            focus_mode=focus_mode,
            known_speakers=_set(known_speakers) or [],
            additional_vocab=_set(additional_vocab) or [],
            model=_resolve_model(model, operating_point),
            max_delay=_set(max_delay),
            end_of_utterance_silence_trigger=_set(end_of_utterance_silence_trigger),
            end_of_utterance_max_delay=_set(end_of_utterance_max_delay),
            punctuation_overrides=_set(punctuation_overrides),
            include_partials=_set(include_partials),
            enable_diarization=_set(enable_diarization),
            speaker_sensitivity=_set(speaker_sensitivity),
            max_speakers=_set(max_speakers),
            prefer_current_speaker=_set(prefer_current_speaker),
        )

        # Migrate / warn about any deprecated kwargs
        _check_deprecated_args(kwargs, self._stt_options)

        # Validate config options
        errors = self._validate_stt_options()
        if errors:
            raise ValueError("Invalid STT options: " + ", ".join(errors))

        # Set API key
        self._api_key: str = api_key if is_given(api_key) else os.getenv("SPEECHMATICS_API_KEY", "")

        # Set base URL
        self._base_url: str = (
            base_url
            if is_given(base_url)
            else os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")
        )

        # Validate API key and base URL
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Set audio parameters
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

        # end_of_utterance_silence_trigger must be between 0 and 2
        if opts.end_of_utterance_silence_trigger is not None and not (
            0 < opts.end_of_utterance_silence_trigger < 2
        ):
            errors.append("end_of_utterance_silence_trigger must be between 0 and 2")

        # end_of_utterance_max_delay must exceed end_of_utterance_silence_trigger so the engine has time to detect silence
        if (
            opts.end_of_utterance_max_delay is not None
            and opts.end_of_utterance_silence_trigger is not None
            and opts.end_of_utterance_max_delay <= opts.end_of_utterance_silence_trigger
        ):
            errors.append(
                "end_of_utterance_max_delay must be greater than end_of_utterance_silence_trigger"
            )

        # server rejects speaker counts outside 2–100
        if opts.max_speakers is not None and not (1 < opts.max_speakers <= 100):
            errors.append("max_speakers must be between 2 and 100")

        # latency budget: below 0.7s is unsupported
        if opts.max_delay is not None and not (0.7 <= opts.max_delay <= 4.0):
            errors.append("max_delay must be between 0.7 and 4.0")

        # diarization sensitivity range enforced by the engine
        if opts.speaker_sensitivity is not None and not (0.0 < opts.speaker_sensitivity < 1.0):
            errors.append("speaker_sensitivity must be between 0.0 and 1.0")

        return errors

    def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN) -> TranscriptionConfig:
        """Prepare an Agent STT TranscriptionConfig from STTOptions.

        This is the only place the config crosses from the plugin's (voice-SDK-shaped)
        public options into the Agent STT session driver. Only the fields agent-STT
        accepts on the wire are set; the voice-only / unsupported knobs (max_delay,
        punctuation_overrides, end_of_utterance_*, speaker focus, speaker sensitivity,
        max_speakers, prefer_current_speaker) are silently dropped.
        """

        # Reference to STT options
        opts = self._stt_options

        config = TranscriptionConfig(
            language=language if is_given(language) else opts.language,
            model=opts.model,
            turn_detection_mode=_handle_turn_detection_mode(opts.turn_detection_mode),
            diarization="speaker" if opts.enable_diarization else None,
            additional_vocab=opts.additional_vocab or None,
            output_locale=opts.output_locale,
            domain=opts.domain,
            enable_partials=opts.include_partials,
        )

        # TEMPORARY (spec↔SDK drift): the SDK's to_dict() always emits
        # `transcription_config.vad_config`, but the deployed agent-STT input spec rejects it
        # config.override_config({"vad_config": REMOVE})

        return config

    def update_speakers(
        self,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: NotGivenOr[SpeakerFocusMode] = NOT_GIVEN,
    ) -> None:
        """Updates the speaker configuration.

        Records the new speaker focus configuration on the options. Note that
        speaker focus is a voice-SDK concept with no agent-STT wire equivalent yet,
        so the update is not pushed to an in-flight session.
        """
        if is_given(focus_speakers):
            self._stt_options.focus_speakers = focus_speakers
        if is_given(ignore_speakers):
            self._stt_options.ignore_speakers = ignore_speakers
        if is_given(focus_mode):
            self._stt_options.focus_mode = focus_mode

        logger.info("update_speakers has no agent-STT equivalent; not pushed to the session")

    def finalize(self) -> None:
        """Finalize the turn (from external VAD).

        When using an external VAD, such as Silero, this should be called
        when the VAD detects the end of a speech turn. This will force the
        finalization of the words in the STT buffer and emit them as final
        segments.
        """

        # Iterate over the streams
        for stream in self._streams:
            # Do not finalize if being handled by a client
            if not stream._client or not stream._client.is_connected:
                continue

            # Only finalize() if EXTERNAL turn_detection_mode is selected
            if stream._config.turn_detection_mode == AgentTurnDetectionMode.EXTERNAL:
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


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions,
        config: TranscriptionConfig,
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
        self._client: AgentSttAsyncClient | None = None
        self._msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._speech_duration: float = 0

        self._vad: vad.VAD | None = vad_instance
        self._vad_stream: vad.VADStream | None = None

        self._tasks: list[asyncio.Task] = []

        # Speaker result event
        self._speaker_result_event: asyncio.Event = asyncio.Event()
        self._speaker_result: list[SpeakerIdentifier] | None = None

    async def _run(self) -> None:
        """Run the STT stream."""
        logger.debug("Connecting to Speechmatics STT service")

        # Config is required
        if not self._config:
            raise ValueError("Config is required")

        # Create the Agent STT client
        self._client = AgentSttAsyncClient(
            api_key=self._stt._api_key,
            url=self._stt._base_url,
            app=f"livekit/{lk_version}",
            config=self._config,
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

        # Connect to the service
        await self._client.connect()
        logger.debug("Connected to Speechmatics STT service")

        # Open external VAD stream (if provided) before tasks start pushing frames
        if self._vad is not None:
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

                # Send audio frames
                if self._client:
                    for frame in frames:
                        self._speech_duration += frame.duration
                        await self._client.send_audio(frame.data.tobytes())

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
            logger.info(f"{event} -> {message}")
        elif event == ServerMessageType.WARNING:
            logger.warning(f"{event} -> {message}")
        elif event == ServerMessageType.ERROR:
            logger.error(f"{event} -> {message}")

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
            logger.debug(f"{event} -> {message}")

    def _handle_segment(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle AddSegment / AddPartialSegment events.

        agent-STT sends a singular `segment` (`transcript`/`speaker`) plus message-level
        `metadata`, unlike the voice SDK's `segments` list — `Segment.from_message`
        bridges that shape.
        """
        opts = self._stt._stt_options

        # Parse the singular agent-STT segment
        seg = Segment.from_message(message)

        # Format the text (active formatter only — agent-STT has no passive/active split)
        speaker_id = seg.speaker or "UU"
        format_str = opts.speaker_active_format or "{text}"
        text = format_str.format(speaker_id=speaker_id, text=seg.transcript)

        # Create speech event
        speech_data = stt.SpeechData(
            language=LanguageCode(opts.language),
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


def _handle_turn_detection_mode(mode: TurnDetectionMode) -> AgentTurnDetectionMode:
    """Map the plugin's turn detection mode onto the session driver's.

    - `DEFAULT`  -> the service runs its own VAD and closes turns.
    - `EXTERNAL` -> the caller closes turns by calling `finalize()`.

    The plugin's enum values are the SDK's values, so this is a direct lookup — but it
    is still required: `TranscriptionConfig.to_dict()` compares by identity, so the
    config must carry the SDK's own enum member, not the plugin's.
    """
    return AgentTurnDetectionMode(mode.value)


def _model_name(value: Model | OperatingPoint | str) -> str:
    """Normalize a model / operating-point value to its wire string.

    Accepts an enum member (`Model`, `OperatingPoint`, or any future `str` enum) or a
    plain string, and returns the string the service reads.
    """
    return value.value if isinstance(value, Enum) else str(value)


def _resolve_model(
    model: NotGivenOr[Model | str],
    operating_point: NotGivenOr[OperatingPoint | Model | str],
) -> str:
    """Reconcile the preferred `model` with its deprecated `operating_point` alias.

    Rules:
        - neither given          -> the SDK's default model
        - only `operating_point` -> use it, with a `DeprecationWarning`
        - only `model`           -> use it
        - both given             -> they must name the same value; if they differ a
                                    `ValueError` is raised, otherwise `model` is used

    Returns:
        The resolved model name as a string.

    Raises:
        ValueError: if `model` and `operating_point` are both given but differ.
    """
    resolved_model = _model_name(model) if is_given(model) else None
    resolved_op = _model_name(operating_point) if is_given(operating_point) else None

    if resolved_model is not None and resolved_op is not None:
        if resolved_model != resolved_op:
            raise ValueError(
                f"`model` ({resolved_model!r}) and `operating_point` ({resolved_op!r}) name "
                "different options. Pass only `model` (`operating_point` is deprecated)."
            )
        return resolved_model

    if resolved_op is not None:
        warnings.warn(
            "`operating_point` is deprecated and will be removed in a future release; "
            "use `model` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return resolved_op

    if resolved_model is not None:
        return resolved_model

    return DEFAULT_MODEL.value


def _check_deprecated_args(kwargs: dict[str, Any], opts: STTOptions) -> None:
    """Warn about deprecated kwargs and migrate values where possible."""

    # Removed — no replacement
    for name in (
        "end_of_utterance_mode",
        "chunk_size",
        "transcription_config",
        "audio_settings",
        "http_session",
    ):
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used")

    # Partials
    if "enable_partials" in kwargs:
        if opts.include_partials is None:
            logger.warning("`enable_partials` is deprecated, migrated to `include_partials`")
            opts.include_partials = bool(kwargs["enable_partials"])
        else:
            logger.warning(
                "Both `enable_partials` and `include_partials` provided; using `include_partials`"
            )

    # Diarization
    if "diarization_sensitivity" in kwargs and isinstance(
        kwargs["diarization_sensitivity"], (int, float)
    ):
        if opts.speaker_sensitivity is None:
            logger.warning(
                "`diarization_sensitivity` is deprecated, migrated to `speaker_sensitivity`"
            )
            opts.speaker_sensitivity = kwargs["diarization_sensitivity"]
        else:
            logger.warning(
                "Both `diarization_sensitivity` and `speaker_sensitivity` provided;"
                " using `speaker_sensitivity`"
            )