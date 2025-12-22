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

    How the STT engine handles the endpointing of speech. If using Pipecat's built-in endpointing,
    then use `TurnDetectionMode.EXTERNAL`.

    Using LiveKit's own VAD, the default is `TurnDetectionMode.FIXED`.

    To use the STT engine's built-in endpointing, then use `TurnDetectionMode.ADAPTIVE` for simple
    voice activity detection or `TurnDetectionMode.SMART_TURN` for more advanced ML-based
    endpointing.
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
    turn_detection_mode: TurnDetectionMode = TurnDetectionMode.FIXED

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
        language: str = "en",
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        domain: NotGivenOr[str] = NOT_GIVEN,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.FIXED,
        speaker_active_format: NotGivenOr[str] = NOT_GIVEN,
        speaker_passive_format: NotGivenOr[str] = NOT_GIVEN,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN,
        known_speakers: NotGivenOr[list[SpeakerIdentifier]] = NOT_GIVEN,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        operating_point: NotGivenOr[OperatingPoint] = NOT_GIVEN,
        max_delay: NotGivenOr[float] = NOT_GIVEN,
        end_of_utterance_silence_trigger: NotGivenOr[float] = NOT_GIVEN,
        end_of_utterance_max_delay: NotGivenOr[float] = NOT_GIVEN,
        punctuation_overrides: NotGivenOr[dict] = NOT_GIVEN,
        include_partials: NotGivenOr[bool] = NOT_GIVEN,
        enable_diarization: bool = True,
        speaker_sensitivity: NotGivenOr[float] = NOT_GIVEN,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
        prefer_current_speaker: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: int = 16000,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        **kwargs: Any,
    ):
        """
        Create a new instance of Speechmatics STT using the Voice SDK.

        Args:
            api_key (str): Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable

            base_url (str): Custom base URL for the API. Can be set via `base_url`
                argument or `SPEECHMATICS_RT_URL` environment variable. Optional.

            language (str): Language code for the STT model. Defaults to `en`. Optional.

            output_locale (str): Output locale for the STT model, e.g. `en-GB`. Optional.

            domain (str): Domain to use. Optional.

            turn_detection_mode (TurnDetectionMode): Endpoint handling, one of
                `TurnDetectionMode.EXTERNAL`, `TurnDetectionMode.ADAPTIVE` and
                `TurnDetectionMode.SMART_TURN`. Defaults to `TurnDetectionMode.EXTERNAL`.

            speaker_active_format (str): Formatter for active speaker ID. This formatter is used
                to format the text output for individual speakers and ensures that the context is
                clear for language models further down the pipeline. The attributes `text` and
                `speaker_id` are available. Example: `@{speaker_id}: {text}`.
                Defaults to transcription output.

            speaker_passive_format (str): Formatter for passive speaker ID. As with the
                speaker_active_format, the attributes `text` and `speaker_id` are available.
                Example: `@{speaker_id} [background]: {text}`.
                Defaults to transcription output.

            focus_speakers (list[str]): List of speaker IDs to focus on. When enabled, only these
                speakers are emitted as `FINAL_TRANSCRIPT` events and other speakers are considered
                passive. Words from other speakers are still processed, but only emitted when a
                focussed speaker has also said new words. A list of labels (e.g. `S1`, `S2`) or
                identifiers of known speakers (e.g. `speaker_1`, `speaker_2`) can be used.
                Defaults to [].

            ignore_speakers (list[str]): List of speaker IDs to ignore. When enabled, these speakers
                are excluded from the transcription and their words are not processed. Their speech
                will not trigger any VAD or end of utterance detection. By default, any speaker
                with a label starting and ending with double underscores will be excluded (e.g.
                `__ASSISTANT__`).
                Defaults to [].

            focus_mode (SpeakerFocusMode): Speaker focus mode for diarization. When set to
                `SpeakerFocusMode.RETAIN`, the STT engine will retain words spoken by other speakers
                (not listed in `ignore_speakers`) and process them as passive speaker frames. When set to
                `SpeakerFocusMode.IGNORE`, the STT engine will ignore words spoken by other speakers
                and they will not be processed. Defaults to `SpeakerFocusMode.RETAIN`.

            known_speakers (list[SpeakerIdentifier]): List of known speaker labels and identifiers.
                If you supply a list of labels and identifiers for speakers, then the STT engine will
                use them to attribute any spoken words to that speaker. This is useful when you want to
                attribute words to a specific speaker, such as the assistant or a specific user.
                Defaults to [].

            additional_vocab (list[AdditionalVocabEntry]): List of additional vocabulary entries.
                If you supply a list of additional vocabulary entries, this will increase the
                weight of the words in the vocabulary and help the STT engine to better transcribe
                the words. Defaults to [].

            operating_point (OperatingPoint): Operating point for transcription accuracy
                vs. latency tradeoff. Overrides preset if provided. Optional.

            max_delay (float): Maximum delay in seconds for transcription. This forces the
                STT engine to speed up the processing of transcribed words and reduces the
                interval between partial and final results. Lower values can have an impact on
                accuracy. Overrides preset if provided. Optional.

            end_of_utterance_silence_trigger (float): Maximum delay in seconds for end of
                utterance trigger. The delay is used to wait for any further transcribed
                words before emitting the `FINAL_TRANSCRIPT` events.
                Overrides preset if provided. Optional.

            end_of_utterance_max_delay (float): Maximum delay in seconds for end of utterance
                delay. Must be greater than end_of_utterance_silence_trigger.
                Overrides preset if provided. Optional.

            punctuation_overrides (dict): Punctuation overrides. This allows you to override
                the punctuation in the STT engine. Overrides preset if provided. Optional.

            include_partials (bool): Include partial segment fragments (words) in the output of
                AddPartialSegment messages. Partial fragments from the STT will always be used for
                speaker activity detection. This setting is used only for the formatted text output
                of individual segments. Defaults to True.

            enable_diarization (bool): Enable speaker diarization. When enabled, the STT
                engine will determine and attribute words to unique speakers.
                Overrides preset if provided. Defaults to True.

            speaker_sensitivity (float): Diarization sensitivity. A higher value increases
                the sensitivity of diarization and helps when two or more speakers have similar voices.
                Overrides preset if provided. Optional.

            max_speakers (int): Maximum number of speakers to detect during diarization. When set,
                the STT engine will limit the number of unique speakers identified in the transcription.
                Overrides preset if provided. Optional.

            prefer_current_speaker (bool): Prefer current speaker ID. When set to true, groups of
                words close together are given extra weight to be identified as the same speaker.
                Overrides preset if provided. Optional.

            sample_rate (int): Sample rate for the audio. Optional. Defaults to 16000.

            audio_encoding (AudioEncoding): Audio encoding for the audio. Optional.
                Defaults to `AudioEncoding.PCM_S16LE`.

            kwargs (Any): Additional keyword arguments used to validate deprecated parameters.
                Optional.
        """

        # Set default values for optional parameters
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=is_given(enable_diarization) and enable_diarization,
            ),
        )

        # Set STT options
        def _set(value: Any) -> Any:
            return value if is_given(value) else None

        # Create STT options from parameters
        self._stt_options = STTOptions(
            language=_set(language),
            output_locale=_set(output_locale),
            domain=_set(domain),
            turn_detection_mode=_set(turn_detection_mode),
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
            enable_diarization=_set(enable_diarization),
            speaker_sensitivity=_set(speaker_sensitivity),
            max_speakers=_set(max_speakers),
            prefer_current_speaker=_set(prefer_current_speaker),
        )

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

        # Initialize config and stream
        self._config: VoiceAgentConfig | None = None
        self._stream: SpeechStream | None = None

    @property
    def model(self) -> str:
        return str(self._stt_options.turn_detection_mode) if self._stt_options else "UNKNOWN"

    @property
    def provider(self) -> str:
        return "Speechmatics"

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

        # Prepare the config
        self._config = self._prepare_config(language)

        # Create the stream
        self._stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
        )

        # Return the new stream
        return self._stream

    def _prepare_config(self, language: NotGivenOr[str] = NOT_GIVEN) -> VoiceAgentConfig:
        """Prepare VoiceAgentConfig from STTOptions."""

        # Reference to STT options
        opts = self._stt_options

        # Preset taken from `EXTERNAL`, `ADAPTIVE` or `SMART_TURN`
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
            "operating_point",
            "max_delay",
            "end_of_utterance_silence_trigger",
            "end_of_utterance_max_delay",
            "punctuation_overrides",
            "enable_diarization",
            "speaker_sensitivity",
            "max_speakers",
            "prefer_current_speaker",
        ]

        # Override preset parameters if provided
        for param in advanced_params:
            value = getattr(opts, param)
            if value is not None:
                setattr(config, param, value)

        # Handle partials
        if not opts.include_partials:
            config.include_partials = False

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

        Args:
            focus_speakers: List of speakers to focus on.
            ignore_speakers: List of speakers to ignore.
            focus_mode: Focus mode to use.
        """
        # Check if diarization is enabled
        if not self._config or not self._config.enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the configuration
        if is_given(focus_speakers):
            self._stt_options.focus_speakers = focus_speakers
            self._config.speaker_config.focus_speakers = focus_speakers
        if is_given(ignore_speakers):
            self._stt_options.ignore_speakers = ignore_speakers
            self._config.speaker_config.ignore_speakers = ignore_speakers
        if is_given(focus_mode):
            self._stt_options.focus_mode = focus_mode
            self._config.speaker_config.focus_mode = focus_mode

        # Send update to client if stream is active
        if self._stream and self._stream._client:
            self._stream._client.update_diarization_config(self._config.speaker_config)

    async def get_speaker_ids(self) -> list[SpeakerIdentifier]:
        """Get the list of speakers from the current STT session.

        If diarization is enabled, then this will use the GET_SPEAKERS message
        to retrieve the list of speakers for the current session. This should
        be used once speakers have said at least 5 words to improve the results.

        Returns:
            list[SpeakerIdentifier]: List of speakers in the session.
        """

        # Return if diarization is not enabled
        if (
            self._stream is None
            or self._stream._client is None
            or self._config is None
            or not self._config.enable_diarization
        ):
            logger.warning("Diarization is not enabled")
            return []

        # Clear the speaker result
        self._stream._speaker_result_event.clear()

        # Send message to client
        await self._stream._client.send_message({"message": ClientMessageType.GET_SPEAKERS.value})

        # Wait the result (5 second timeout)
        try:
            await asyncio.wait_for(
                self._stream._speaker_result_event.wait(),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Speaker result timed-out")
            return []

        # Return the list of speakers
        return self._stream._speaker_result or []


class SpeechStream(stt.RecognizeStream):
    def __init__(self, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=stt._sample_rate,
        )

        # redefine types
        self._stt: STT = stt
        self._client: VoiceAgentClient | None = None
        self._msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._msg_task: asyncio.Task | None = None

        # Speaker result event
        self._speaker_result_event: asyncio.Event = asyncio.Event()
        self._speaker_result: list[SpeakerIdentifier] | None = None

    async def _run(self) -> None:
        """Run the STT stream."""
        logger.debug("Connecting to Speechmatics STT service")

        # Config is required
        if not self._stt._config:
            raise ValueError("Config is required")

        # Create the Voice Agent client
        self._client = VoiceAgentClient(
            api_key=self._stt._api_key,
            url=self._stt._base_url,
            app=f"livekit/{lk_version}",
            config=self._stt._config,
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
        if (
            self._stt._config.enable_diarization is not None
            and self._stt._config.enable_diarization
        ):
            messages.append(AgentServerMessageType.SPEAKERS_RESULT)

        # Optional debug messages to log
        if True:
            messages.append(AgentServerMessageType.END_OF_UTTERANCE)
            messages.append(AgentServerMessageType.END_OF_TURN_PREDICTION)
            messages.append(AgentServerMessageType.DIAGNOSTICS)

        # Add message handlers
        for event in messages:
            self._client.on(event, add_message)  # type: ignore[arg-type]

        # Connect to the service
        await self._client.connect()
        logger.debug("Connected to Speechmatics STT service")

        # Start message processing task
        self._msg_task = asyncio.create_task(self._process_messages())

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
            for frame in frames:
                await self._client.send_audio(frame.data.tobytes())

        # Close the connection
        await self._client.disconnect()

    async def _process_messages(self) -> None:
        """Process messages from the STT client."""
        try:
            while True:
                message = await self._msg_queue.get()
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass

    async def _handle_message(self, message: dict[str, Any]) -> None:
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
            await self._handle_partial_segment(message)
        elif event == AgentServerMessageType.ADD_SEGMENT:
            await self._handle_segment(message)
        elif event == AgentServerMessageType.START_OF_TURN:
            await self._handle_start_of_turn(message)
        elif event == AgentServerMessageType.END_OF_TURN:
            await self._handle_end_of_turn(message)

        # Handle the speaker result message
        elif event == AgentServerMessageType.SPEAKERS_RESULT:
            await self._handle_speakers_result(message)

        # Log all other messages
        else:
            logger.debug(f"{event} -> {message}")

    async def _handle_partial_segment(self, message: dict[str, Any]) -> None:
        """Handle AddPartialSegment events."""
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments, is_final=False)

    async def _handle_segment(self, message: dict[str, Any]) -> None:
        """Handle AddSegment events."""
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments, is_final=True)

    async def _handle_start_of_turn(self, message: dict[str, Any]) -> None:
        """Handle StartOfTurn events."""
        logger.debug("StartOfTurn received")
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

    async def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events."""
        logger.debug("EndOfTurn received")
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    async def _handle_speakers_result(self, message: dict[str, Any]) -> None:
        """Handle SpeakersResult events."""
        logger.debug("SpeakersResult received")
        self._speaker_result = message.get("speakers", [])
        self._speaker_result_event.set()

    async def _send_frames(self, segments: list[dict[str, Any]], is_final: bool) -> None:
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
                start_time=segment.get("metadata", {}).get("start_time", 0),
                end_time=segment.get("metadata", {}).get("end_time", 0),
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
        if self._msg_task:
            self._msg_task.cancel()
            try:
                await self._msg_task
            except asyncio.CancelledError:
                pass

        # Close the client
        if self._client:
            await self._client.disconnect()
            self._client = None
