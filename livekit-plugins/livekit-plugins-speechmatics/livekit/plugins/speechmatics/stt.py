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
from enum import Enum
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from speechmatics.rt import ClientMessageType
from speechmatics.voice import (
    AdditionalVocabEntry,
    AgentServerMessageType,
    AudioEncoding,
    EndOfUtteranceMode,
    OperatingPoint,
    SpeakerFocusConfig,
    SpeakerFocusMode,
    SpeakerIdentifier,
    VoiceAgentClient,
    VoiceAgentConfig,
    VoiceAgentConfigPreset,
)

from .log import logger
from .version import __version__ as lk_version


class TurnDetectionMode(str, Enum):
    """Endpoint and turn detection handling mode.

    How the STT engine handles the endpointing of speech. Use `TurnDetectionMode.EXTERNAL` when
    turn boundaries are controlled manually, for example via an external VAD or the `finalize()`
    method.

    To use the STT engine's built-in endpointing, use `TurnDetectionMode.ADAPTIVE` for simple
    voice activity detection or `TurnDetectionMode.SMART_TURN` for more advanced ML-based
    endpointing.

    The default is `ADAPTIVE` which uses voice activity detection to determine end of speech.
    """

    EXTERNAL = "external"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    SMART_TURN = "smart_turn"


@dataclasses.dataclass
class STTOptions:
    """Configuration parameters for Speechmatics STT service."""

    # Service configuration
    language: str = "en"
    output_locale: str | None = None
    domain: str | None = None

    # Endpointing mode
    turn_detection_mode: TurnDetectionMode = TurnDetectionMode.ADAPTIVE

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
    operating_point: OperatingPoint | None = None
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
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.ADAPTIVE,
        operating_point: NotGivenOr[OperatingPoint] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        language: str = "en",
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        include_partials: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: bool = True,
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
        **kwargs: Any,
    ):
        """Create a new instance of Speechmatics STT using the Voice SDK.

        Args:
            api_key: Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable.

            base_url: Custom base URL for the API. Can be set via `base_url`
                argument or `SPEECHMATICS_RT_URL` environment variable. Optional.

            turn_detection_mode: Controls how the STT engine detects end of speech
                turns. Use `EXTERNAL` when turn boundaries are controlled manually,
                for example via an external VAD or the `finalize()` method. Use
                `ADAPTIVE` for simple VAD or `SMART_TURN` for ML-based endpointing.
                Defaults to `TurnDetectionMode.ADAPTIVE`.

            operating_point: Operating point for transcription accuracy vs. latency
                tradeoff. Overrides preset if provided. Optional.

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

            **kwargs: Catches deprecated parameters. A warning is logged for any
                recognised deprecated name.
        """

        # Set default values for optional parameters
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=enable_diarization,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )

        # Set STT options
        def _set(value: Any) -> Any:
            return value if is_given(value) else None

        # Create STT options from parameters
        self._stt_options = STTOptions(
            language=language,
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
            operating_point=_set(operating_point),
            max_delay=_set(max_delay),
            end_of_utterance_silence_trigger=_set(end_of_utterance_silence_trigger),
            end_of_utterance_max_delay=_set(end_of_utterance_max_delay),
            punctuation_overrides=_set(punctuation_overrides),
            include_partials=_set(include_partials),
            enable_diarization=enable_diarization,
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

        # Show warning for external
        if self._stt_options.turn_detection_mode == TurnDetectionMode.EXTERNAL:
            logger.info("STT under external turn detection control")

    @property
    def provider(self) -> str:
        return "Speechmatics"

    @property
    def model(self) -> str:
        op = self._stt_options.operating_point
        return str(op.value) if op is not None else "enhanced"

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
        )

        # Add to the list of streams
        self._streams.append(stream)

        # Return the stream
        return stream

    def _validate_stt_options(self) -> list[str]:
        """Validate options in STTOptions."""
        errors: list[str] = []
        opts = self._stt_options

        # end_of_utterance_silence_trigger must be between 0 and 1
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

    def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN) -> VoiceAgentConfig:
        """Prepare VoiceAgentConfig from STTOptions."""

        # Reference to STT options
        opts = self._stt_options

        # Preset taken from `FIXED`, `EXTERNAL`, `ADAPTIVE` or `SMART_TURN`
        config = VoiceAgentConfigPreset.load(opts.turn_detection_mode.value)

        # Set sample rate and encoding
        config.sample_rate = self._sample_rate
        config.audio_encoding = self._audio_encoding

        # Language and domain
        config.language = language if is_given(language) else opts.language
        config.domain = opts.domain
        config.output_locale = opts.output_locale

        # Speaker configuration
        config.speaker_config = SpeakerFocusConfig(
            focus_speakers=opts.focus_speakers,
            ignore_speakers=opts.ignore_speakers,
            focus_mode=opts.focus_mode,
        )
        config.known_speakers = opts.known_speakers

        # Additional vocabulary
        config.additional_vocab = opts.additional_vocab

        # Override preset parameters if provided
        advanced_params = [
            "enable_diarization",
            "end_of_utterance_max_delay",
            "end_of_utterance_silence_trigger",
            "include_partials",
            "max_delay",
            "max_speakers",
            "operating_point",
            "prefer_current_speaker",
            "punctuation_overrides",
            "speaker_sensitivity",
        ]

        # Override preset parameters if provided
        for param in advanced_params:
            value = getattr(opts, param)
            if value is not None:
                setattr(config, param, value)

        # Return the config
        return config

    def update_speakers(
        self,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: NotGivenOr[SpeakerFocusMode] = NOT_GIVEN,
    ) -> None:
        """Updates the speaker configuration.

        This can update the speakers to listen to or ignore during an in-flight
        transcription. Only available if diarization is enabled.

        This will be applied to *all* streams (typically only one).

        Args:
            focus_speakers: List of speakers to focus on.
            ignore_speakers: List of speakers to ignore.
            focus_mode: Focus mode to use.
        """
        # Do this for each stream
        for stream in self._streams:
            # Check if diarization is enabled
            if not stream._config.enable_diarization:
                raise ValueError("Diarization is not enabled")

            # Update the configuration
            if is_given(focus_speakers):
                self._stt_options.focus_speakers = focus_speakers
                stream._config.speaker_config.focus_speakers = focus_speakers
            if is_given(ignore_speakers):
                self._stt_options.ignore_speakers = ignore_speakers
                stream._config.speaker_config.ignore_speakers = ignore_speakers
            if is_given(focus_mode):
                self._stt_options.focus_mode = focus_mode
                stream._config.speaker_config.focus_mode = focus_mode

            # Send update to client if stream is active
            if stream._client and stream._client._is_connected:
                stream._client.update_diarization_config(stream._config.speaker_config)

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
            if not stream._client or not stream._client._is_connected:
                continue

            # Check that VAD is not being handled by the client
            if stream._config.vad_config is None or not stream._config.vad_config.enabled:
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
            if stream._client is None or not stream._client._is_connected:
                logger.warning(f"Not connected in stream {idx}")
                results.append([])
                continue

            # Return if diarization is not enabled
            if not stream._config.enable_diarization:
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
        config: VoiceAgentConfig,
        id: int,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=stt._sample_rate,
        )

        self._stt: STT = stt
        self._id: int = id
        self._config: VoiceAgentConfig = config
        self._client: VoiceAgentClient | None = None
        self._msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._speech_duration: float = 0

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

        # Create the Voice Agent client
        self._client = VoiceAgentClient(
            api_key=self._stt._api_key,
            url=self._stt._base_url,
            app=f"livekit/{lk_version}",
            config=self._config,
        )

        # Add message handlers
        def add_message(message: dict[str, Any]) -> None:
            self._msg_queue.put_nowait(message)

        # Default messages to listen to
        messages: list[AgentServerMessageType] = [
            AgentServerMessageType.RECOGNITION_STARTED,
            AgentServerMessageType.INFO,
            AgentServerMessageType.ERROR,
            AgentServerMessageType.WARNING,
            AgentServerMessageType.ADD_PARTIAL_SEGMENT,
            AgentServerMessageType.ADD_SEGMENT,
            AgentServerMessageType.START_OF_TURN,
            AgentServerMessageType.END_OF_TURN,
        ]

        # Speaker IDs message handler
        if self._config.enable_diarization:
            messages.append(AgentServerMessageType.SPEAKERS_RESULT)

        # Optional debug messages to log
        # messages.append(AgentServerMessageType.END_OF_UTTERANCE)
        # messages.append(AgentServerMessageType.END_OF_TURN_PREDICTION)
        # messages.append(AgentServerMessageType.DIAGNOSTICS)

        # Add message handlers
        for event in messages:
            self._client.on(event, add_message)  # type: ignore[arg-type]

        # Connect to the service
        await self._client.connect()
        logger.debug("Connected to Speechmatics STT service")

        # Audio and messaging tasks
        audio_task = asyncio.create_task(self._process_audio())
        message_task = asyncio.create_task(self._process_messages())

        # Tasks
        self._tasks = [audio_task, message_task]

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
                    frames = audio_bstream.write(data.data.tobytes())

                # Send audio frames
                if self._client:
                    for frame in frames:
                        self._speech_duration += frame.duration
                        await self._client.send_audio(frame.data.tobytes())

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
            AgentServerMessageType.RECOGNITION_STARTED,
            AgentServerMessageType.INFO,
        ]:
            logger.info(f"{event} -> {message}")
        elif event == AgentServerMessageType.WARNING:
            logger.warning(f"{event} -> {message}")
        elif event == AgentServerMessageType.ERROR:
            logger.error(f"{event} -> {message}")

        # Handle the messages
        elif event == AgentServerMessageType.ADD_PARTIAL_SEGMENT:
            self._handle_partial_segment(message)
        elif event == AgentServerMessageType.ADD_SEGMENT:
            self._handle_segment(message)
        elif event == AgentServerMessageType.START_OF_TURN:
            self._handle_start_of_turn(message)
        elif event == AgentServerMessageType.END_OF_TURN:
            self._handle_end_of_turn(message)

        # Handle the speaker result message
        elif event == AgentServerMessageType.SPEAKERS_RESULT:
            self._handle_speakers_result(message)

        # Log all other messages
        else:
            logger.debug(f"{event} -> {message}")

    def _handle_partial_segment(self, message: dict[str, Any]) -> None:
        """Handle AddPartialSegment events."""
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            self._send_frames(segments, is_final=False)

    def _handle_segment(self, message: dict[str, Any]) -> None:
        """Handle AddSegment events."""
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            self._send_frames(segments, is_final=True)

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

    def _send_frames(self, segments: list[dict[str, Any]], is_final: bool) -> None:
        """Send frames to the pipeline."""

        # Check for empty segments
        if not segments:
            return

        # Get the options
        opts = self._stt._stt_options

        # Determine the event type
        event_type = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )

        # Process each segment
        for segment in segments:
            # Format the text based on speaker activity
            is_active = segment.get("is_active", True)
            format_str = (
                opts.speaker_active_format if is_active else opts.speaker_passive_format
            ) or "{text}"
            text = format_str.format(
                speaker_id=segment.get("speaker_id", "UU"),
                text=segment.get("text", ""),
            )

            # Create speech event
            speech_data = stt.SpeechData(
                language=segment.get("language", opts.language),
                text=text,
                speaker_id=segment.get("speaker_id", "UU"),
                start_time=segment.get("metadata", {}).get("start_time", 0)
                + self.start_time_offset,
                end_time=segment.get("metadata", {}).get("end_time", 0) + self.start_time_offset,
                confidence=1.0,
            )

            # Create speech event
            event = stt.SpeechEvent(
                type=event_type,
                alternatives=[speech_data],
            )

            # Send the event
            self._event_ch.send_nowait(event)

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

        # Close the client
        if self._client and self._client._is_connected:
            await self._client.disconnect()
        self._client = None

        # Remove from active streams
        if self in self._stt._streams:
            self._stt._streams.remove(self)


def _check_deprecated_args(kwargs: dict[str, Any], opts: STTOptions) -> None:
    """Warn about deprecated kwargs and migrate values where possible."""

    # Removed — no replacement
    for name in (
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

    # Turn detection — "none" is not a valid TurnDetectionMode, map to ADAPTIVE
    if "end_of_utterance_mode" in kwargs and isinstance(
        kwargs["end_of_utterance_mode"], (str, EndOfUtteranceMode)
    ):
        value = kwargs["end_of_utterance_mode"]
        opts.turn_detection_mode = (
            TurnDetectionMode.ADAPTIVE if value == "none" else TurnDetectionMode(value)
        )
        logger.warning("`end_of_utterance_mode` is deprecated, migrated to `turn_detection_mode`")
