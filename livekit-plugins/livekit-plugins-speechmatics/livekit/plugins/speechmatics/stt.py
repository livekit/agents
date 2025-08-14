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
import datetime
import logging
import os
import re
from typing import Any

import aiohttp

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
from speechmatics.rt import (  # type: ignore
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    ClientMessageType,
    ConversationConfig,
    OperatingPoint,
    ServerMessageType,
    TranscriptionConfig,
)

from .log import logger
from .types import (
    AdditionalVocabEntry,
    AudioSettings,
    DiarizationFocusMode,
    DiarizationKnownSpeaker,
    EndOfUtteranceMode,
    SpeakerFragments,
    SpeechFragment,
)
from .utils import get_endpoint_url


@dataclasses.dataclass
class STTOptions:
    operating_point: OperatingPoint = OperatingPoint.ENHANCED
    domain: str | None = None
    language: str = "en"
    output_locale: str | None = None
    enable_vad: bool = False
    enable_partials: bool = True
    enable_diarization: bool = False
    max_delay: float = 1.0
    end_of_utterance_silence_trigger: float = 0.5
    end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED
    additional_vocab: list[AdditionalVocabEntry] = dataclasses.field(default_factory=list)
    punctuation_overrides: dict = dataclasses.field(default_factory=dict)
    diarization_sensitivity: float = 0.5
    speaker_active_format: str = "{text}"
    speaker_passive_format: str = "{text}"
    prefer_current_speaker: bool = False
    focus_speakers: list[str] = dataclasses.field(default_factory=list)
    ignore_speakers: list[str] = dataclasses.field(default_factory=list)
    focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN
    known_speakers: list[DiarizationKnownSpeaker] = dataclasses.field(default_factory=list)


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        operating_point: OperatingPoint = OperatingPoint.ENHANCED,
        domain: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        output_locale: NotGivenOr[str] = NOT_GIVEN,
        enable_vad: bool = False,
        enable_partials: bool = True,
        enable_diarization: bool = False,
        max_delay: float = 1.0,
        end_of_utterance_silence_trigger: float = 0.5,
        end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        punctuation_overrides: NotGivenOr[dict] = NOT_GIVEN,
        diarization_sensitivity: float = 0.5,
        speaker_active_format: str = "{text}",
        speaker_passive_format: str = "{text}",
        prefer_current_speaker: bool = False,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN,
        known_speakers: NotGivenOr[list[DiarizationKnownSpeaker]] = NOT_GIVEN,
        sample_rate: int = 16000,
        chunk_size: int = 160,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        transcription_config: NotGivenOr[TranscriptionConfig] = NOT_GIVEN,  # Deprecated
        audio_settings: NotGivenOr[AudioSettings] = NOT_GIVEN,  # Deprecated
        http_session: NotGivenOr[aiohttp.ClientSession] = NOT_GIVEN,
    ):
        """
        Create a new instance of Speechmatics STT.

        Args:
            api_key (str): Speechmatics API key. Can be set via `api_key` argument or `SPEECHMATICS_API_KEY` environment variable
            base_url (str): Custom base URL for the API. Can be set via `base_url` argument or `SPEECHMATICS_RT_URL` environment variable. Optional.
            operating_point (OperatingPoint): Operating point to use. Optional. Defaults to `OperatingPoint.ENHANCED`.
            domain (str): Domain to use. Optional.
            language (str): Language code for the STT model. Optional.
            output_locale (str): Output locale for the STT model. Optional.
            enable_vad (bool): Whether to enable VAD. Optional. Defaults to False.
            enable_partials (bool): Whether to enable partials. Optional. Defaults to True.
            enable_diarization (bool): Whether to enable diarization. Optional. Defaults to False.
            max_delay (float): Maximum delay for partials. Optional. Defaults to 1.0.
            end_of_utterance_silence_trigger (float): End of utterance silence trigger. Optional. Defaults to 0.5.
            end_of_utterance_mode (EndOfUtteranceMode): End of utterance mode. Optional. Defaults to `EndOfUtteranceMode.FIXED`.
            additional_vocab (list[AdditionalVocabEntry]): Additional vocabulary. Optional.
            punctuation_overrides (dict): Punctuation overrides. Optional.
            diarization_sensitivity (float): Diarization sensitivity. Optional. Defaults to 0.5.
            speaker_active_format (str): Speaker active format. Optional. Defaults to `{text}`.
            speaker_passive_format (str): Speaker passive format. Optional. Defaults to `{text}`.
            prefer_current_speaker (bool): Whether to prefer the current speaker. Optional. Defaults to False.
            focus_speakers (list[str]): List of speakers to focus on. Optional.
            ignore_speakers (list[str]): List of speakers to ignore. Optional.
            focus_mode (DiarizationFocusMode): Focus mode. Optional. Defaults to `DiarizationFocusMode.RETAIN`.
            known_speakers (list[DiarizationKnownSpeaker]): List of known speakers. Optional.
            sample_rate (int): Sample rate for the audio. Optional. Defaults to 16000.
            chunk_size (int): Chunk size for the audio. Optional. Defaults to 160.
            audio_encoding (AudioEncoding): Audio encoding for the audio. Optional. Defaults to `AudioEncoding.PCM_S16LE`.
            transcription_config (TranscriptionConfig): Transcription configuration (Deprecated). Optional.
            audio_settings (AudioSettings): Audio settings (Deprecated). Optional.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            follow_redirects (bool): Whether to follow redirects in HTTP requests. Defaults to True.
        """

        # Initialize the base class
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
        )

        # Parse deprecated `transcription_config`
        if is_given(transcription_config):
            # Show deprecation warning
            logger.warning(
                "`transcription_config` is deprecated. Use individual arguments instead (which override this argument)."
            )

            # Fix the type
            config: TranscriptionConfig = transcription_config

            # Migrate settings over
            language = language if is_given(language) else config.language
            output_locale = output_locale if is_given(output_locale) else config.output_locale
            domain = domain if is_given(domain) else config.domain
            operating_point = operating_point or config.operating_point
            enable_diarization = enable_diarization or config.diarization == "speaker"
            enable_partials = enable_partials or config.enable_partials
            max_delay = max_delay or config.max_delay
            additional_vocab = (
                additional_vocab if is_given(additional_vocab) else config.additional_vocab
            )
            punctuation_overrides = (
                punctuation_overrides
                if is_given(punctuation_overrides)
                else config.punctuation_overrides
            )

        # Parse deprecated `audio_settings`
        if is_given(audio_settings):
            # Show deprecation warning
            logger.warning(
                "`audio_settings` is deprecated. Use individual arguments instead (which override this argument)."
            )

            # Fix the type
            audio: AudioSettings = audio_settings

            # Migrate settings over
            sample_rate = sample_rate or audio.sample_rate
            audio_encoding = audio_encoding or audio.encoding

        # STT options
        self._stt_options = STTOptions(
            operating_point=operating_point,
            domain=domain if is_given(domain) else None,
            language=language if is_given(language) else "en",
            output_locale=output_locale if is_given(output_locale) else None,
            enable_vad=enable_vad,
            enable_partials=enable_partials,
            enable_diarization=enable_diarization,
            max_delay=max_delay,
            end_of_utterance_silence_trigger=end_of_utterance_silence_trigger,
            end_of_utterance_mode=end_of_utterance_mode,
            additional_vocab=additional_vocab if is_given(additional_vocab) else [],
            punctuation_overrides=punctuation_overrides if is_given(punctuation_overrides) else {},
            diarization_sensitivity=diarization_sensitivity,
            speaker_active_format=speaker_active_format,
            speaker_passive_format=speaker_passive_format,
            prefer_current_speaker=prefer_current_speaker,
            focus_speakers=focus_speakers if is_given(focus_speakers) else [],
            ignore_speakers=ignore_speakers if is_given(ignore_speakers) else [],
            focus_mode=focus_mode,
            known_speakers=known_speakers if is_given(known_speakers) else [],
        )

        # Service parameters
        self._api_key: str = api_key if is_given(api_key) else os.getenv("SPEECHMATICS_API_KEY", "")
        self._base_url: str = (
            base_url
            if is_given(base_url)
            else os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Complete configuration objects
        self._transcription_config: TranscriptionConfig | None = None
        self._process_config()

        # Set the audio settings
        self._audio_format = AudioFormat(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            encoding=audio_encoding,
        )

        # Set of active stream
        self._stream: stt.RecognizeStream | None = None

        # HTTP session
        self._http_session: aiohttp.ClientSession | None = None

        # Lower logging of the SMX module
        logging.getLogger("speechmatics.rt.transport").setLevel(logging.WARNING)

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

        # Create a copy of the transcription config
        if self._transcription_config is None:
            raise RuntimeError("Transcription config not initialized")
        transcription_config = dataclasses.replace(self._transcription_config)

        # Set the language if given
        if is_given(language):
            transcription_config.language = language

        # Create the stream
        self._stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
        )

        # Return the stream
        return self._stream

    def _process_config(self) -> None:
        """Create a formatted STT transcription config.

        Creates a transcription config object based on the service parameters. Aligns
        with the Speechmatics RT API transcription config.
        """
        # Transcription config
        transcription_config = TranscriptionConfig(
            language=self._stt_options.language,
            domain=self._stt_options.domain,
            output_locale=self._stt_options.output_locale,
            operating_point=self._stt_options.operating_point,
            diarization="speaker" if self._stt_options.enable_diarization else None,
            enable_partials=self._stt_options.enable_partials,
            max_delay=self._stt_options.max_delay,
        )

        # Additional vocab
        if self._stt_options.additional_vocab:
            transcription_config.additional_vocab = [
                {
                    "content": e.content,
                    "sounds_like": e.sounds_like,
                }
                for e in self._stt_options.additional_vocab
            ]

        # Diarization
        if self._stt_options.enable_diarization:
            dz_cfg = {}
            if self._stt_options.diarization_sensitivity is not None:
                dz_cfg["speaker_sensitivity"] = self._stt_options.diarization_sensitivity
            if self._stt_options.prefer_current_speaker is not None:
                dz_cfg["prefer_current_speaker"] = self._stt_options.prefer_current_speaker
            if self._stt_options.known_speakers:
                dz_cfg["speakers"] = {
                    s.label: s.speaker_identifiers for s in self._stt_options.known_speakers
                }
            if dz_cfg:
                transcription_config.speaker_diarization_config = dz_cfg

        # End of Utterance (for fixed)
        if (
            self._stt_options.end_of_utterance_silence_trigger
            and self._stt_options.end_of_utterance_mode == EndOfUtteranceMode.FIXED
        ):
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._stt_options.end_of_utterance_silence_trigger,
            )

        # Punctuation overrides
        if self._stt_options.punctuation_overrides:
            transcription_config.punctuation_overrides = self._stt_options.punctuation_overrides

        # Set config
        self._transcription_config = transcription_config

    def update_speakers(
        self,
        focus_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        ignore_speakers: NotGivenOr[list[str]] = NOT_GIVEN,
        focus_mode: NotGivenOr[DiarizationFocusMode] = NOT_GIVEN,
    ) -> None:
        """Updates the speaker configuration.

        This can update the speakers to listen to or ignore during an in-flight
        transcription. Only available if diarization is enabled.

        Args:
            focus_speakers: List of speakers to focus on.
            ignore_speakers: List of speakers to ignore.
            focus_mode: Focus mode to use.
        """
        # Check possible
        if not self._stt_options.enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the diarization configuration
        if is_given(focus_speakers):
            self._stt_options.focus_speakers = focus_speakers
        if is_given(ignore_speakers):
            self._stt_options.ignore_speakers = ignore_speakers
        if is_given(focus_mode):
            self._stt_options.focus_mode = focus_mode


class SpeechStream(stt.RecognizeStream):
    def __init__(self, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=stt._audio_format.sample_rate
        )

        # Reference to STT object
        self._stt = stt

        # Session
        self._speech_duration: float = 0
        self._start_time: datetime.datetime | None = None

        # Client
        self._client: AsyncClient | None = None

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []

        # EndOfUtterance fallback timer
        self._end_of_utterance_timer: asyncio.Task | None = None

    async def _run(self) -> None:
        """Run the STT stream."""

        # Create Speechmatics client
        self._client = AsyncClient(
            api_key=self._stt._api_key,
            url=get_endpoint_url(self._stt._base_url),
        )

        # Log the event
        logger.debug("Connected to Speechmatics STT service")

        # Config
        opts = self._stt._stt_options

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]) -> None:
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        if opts.enable_partials:

            @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def _evt_on_partial_transcript(message: dict[str, Any]) -> None:
                self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]) -> None:
            self._handle_transcript(message, is_final=True)

        # End of Utterance
        if opts.end_of_utterance_mode == EndOfUtteranceMode.FIXED:

            @self._client.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]) -> None:
                logger.debug("End of utterance received from STT")
                asyncio.create_task(self._handle_end_of_utterance())

        # Speaker Result
        if opts.enable_diarization:

            @self._client.on(ServerMessageType.SPEAKERS_RESULT)
            def _evt_on_speakers_result(message: dict[str, Any]) -> None:
                logger.debug("Speakers result received from STT")
                logger.debug(message)

        # Start session
        await self._client.start_session(
            transcription_config=self._stt._transcription_config,
            audio_format=self._stt._audio_format,
        )

        # Create an audio byte stream
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._stt._audio_format.sample_rate,
            num_channels=1,
        )

        async for data in self._input_ch:
            """Send audio data to the STT client."""

            # If the data is a flush sentinel, flush the audio byte stream
            if isinstance(data, self._FlushSentinel):
                frames = audio_bstream.flush()
            else:
                frames = audio_bstream.write(data.data.tobytes())

            # Send the audio frames to the STT client
            for frame in frames:
                self._speech_duration += frame.duration
                await self._client.send_audio(frame.data.tobytes())

        # TODO - handle the closing of the stream?

    async def send_message(self, message: NotGivenOr[ClientMessageType], **kwargs: Any) -> None:
        """Send a message to the STT service.

        This sends a message to the STT service via the underlying transport. If the session
        is not running, this will raise an exception. Messages in the wrong format will also
        cause an error.

        Args:
            message: Message to send to the STT service.
            **kwargs: Additional arguments passed to the underlying transport.
        """
        try:
            payload = {"message": message}
            payload.update(kwargs)
            logger.debug(f"Sending message to STT: {payload}")
            if self._client:
                asyncio.create_task(self._client.send_message(payload))
        except Exception as e:
            raise RuntimeError(f"error sending message to STT: {e}") from e

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Add the speech fragments
        has_changed = self._add_speech_fragments(
            message=message,
            is_final=is_final,
        )

        # Skip if unchanged
        if not has_changed:
            return

        # Set a timer for the end of utterance
        self._end_of_utterance_timer_start()

        # Send frames
        asyncio.create_task(self._send_frames())

    def _end_of_utterance_timer_start(self) -> None:
        """Start the timer for the end of utterance.

        This will use the STT's `end_of_utterance_silence_trigger` value and set
        a timer to send the latest transcript to the pipeline. It is used as a
        fallback from the EnfOfUtterance messages from the STT.

        Note that the `end_of_utterance_silence_trigger` will be from when the
        last updated speech was received and this will likely be longer in
        real world time to that inside of the STT engine.
        """
        # Reset the end of utterance timer
        if self._end_of_utterance_timer is not None:
            self._end_of_utterance_timer.cancel()

        # Send after a delay
        async def send_after_delay(delay: float) -> None:
            await asyncio.sleep(delay)
            logger.debug("Fallback EndOfUtterance triggered.")
            asyncio.create_task(self._handle_end_of_utterance())

        # Start the timer
        self._end_of_utterance_timer = asyncio.create_task(
            send_after_delay(self._stt._stt_options.end_of_utterance_silence_trigger * 2)
        )

    async def _handle_end_of_utterance(self) -> None:
        """Handle the end of utterance event.

        This will check for any running timers for end of utterance, reset them,
        and then send a finalized frame to the pipeline.
        """
        # Send the frames
        await self._send_frames(finalized=True)

        # Reset the end of utterance timer
        if self._end_of_utterance_timer is not None:
            self._end_of_utterance_timer.cancel()
            self._end_of_utterance_timer = None

    async def _send_frames(self, finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Send speech frames to the pipeline. If VAD is enabled, then this will
        also send an interruption and user started speaking frames. When the
        final transcript is received, then this will send a user stopped speaking
        and stop interruption frames.

        Args:
            finalized: Whether the data is final or partial.
        """
        # Get speech frames (InterimTranscriptionFrame)
        speech_frames = self._get_frames_from_fragments()

        # Skip if no frames
        if not speech_frames:
            return

        # Check at least one frame is active
        if not any(frame.is_active for frame in speech_frames):
            return

        # Event type to send
        if not finalized:
            event_type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        else:
            event_type = stt.SpeechEventType.FINAL_TRANSCRIPT

        # Get the speech data and send
        for item in speech_frames:
            final_event = stt.SpeechEvent(
                type=event_type,
                alternatives=[
                    stt.SpeechData(
                        **item._as_speech_data_attributes(
                            self._stt._stt_options.speaker_active_format,
                            self._stt._stt_options.speaker_passive_format,
                        )
                    )
                ],
            )
            self._event_ch.send_nowait(final_event)

        # Send end of speech
        if finalized:
            # Send End of Speech
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

            # Reset the data
            self._speech_fragments.clear()

            # Send the recognition usage event
            if self._speech_duration > 0:
                usage_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._speech_duration),
                )
                self._event_ch.send_nowait(usage_event)
                self._speech_duration = 0

    def _add_speech_fragments(self, message: dict[str, Any], is_final: bool = False) -> bool:
        """Takes a new Partial or Final from the STT engine.

        Accumulates it into the _speech_data list. As new final data is added, all
        partials are removed from the list.

        Note: If a known speaker is `__[A-Z0-9_]{2,}__`, then the words are skipped,
        as this is used to protect against self-interruption by the assistant or to
        block out specific known voices.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.

        Returns:
            bool: True if the speech data was updated, False otherwise.
        """
        # Options
        opts = self._stt._stt_options

        # Parsed new speech data from the STT engine
        fragments: list[SpeechFragment] = []

        # Current length of the speech data
        current_length = len(self._speech_fragments)

        # Iterate over the results in the payload
        for result in message.get("results", []):
            alt = result.get("alternatives", [{}])[0]
            if alt.get("content", None):
                # Create the new fragment
                fragment = SpeechFragment(
                    start_time=result.get("start_time", 0),
                    end_time=result.get("end_time", 0),
                    language=alt.get("language", "en"),
                    is_eos=alt.get("is_eos", False),
                    is_final=is_final,
                    attaches_to=result.get("attaches_to", ""),
                    content=alt.get("content", ""),
                    speaker=alt.get("speaker", None),
                    confidence=alt.get("confidence", 1.0),
                    result=result,
                )

                # Speaker filtering
                if fragment.speaker:
                    # Drop `__XX__` speakers
                    if re.match(r"^__[A-Z0-9_]{2,}__$", fragment.speaker):
                        continue

                    # Drop speakers not focussed on
                    if (
                        opts.focus_mode == DiarizationFocusMode.IGNORE
                        and opts.focus_speakers
                        and fragment.speaker not in opts.focus_speakers
                    ):
                        continue

                    # Drop ignored speakers
                    if opts.ignore_speakers and fragment.speaker in opts.ignore_speakers:
                        continue

                # Add the fragment
                fragments.append(fragment)

        # Remove existing partials, as new partials and finals are provided
        self._speech_fragments = [frag for frag in self._speech_fragments if frag.is_final]

        # Return if no new fragments and length of the existing data is unchanged
        if not fragments and len(self._speech_fragments) == current_length:
            return False

        # Add the fragments to the speech data
        self._speech_fragments.extend(fragments)

        # Data was updated
        return True

    def _get_frames_from_fragments(self) -> list[SpeakerFragments]:
        """Get speech data objects for the current fragment list.

        Each speech fragments is grouped by contiguous speaker and then
        returned as internal SpeakerFragments objects with the `speaker_id` field
        set to the current speaker (string). An utterance may contain speech from
        more than one speaker (e.g. S1, S2, S1, S3, ...), so they are kept
        in strict order for the context of the conversation.

        Returns:
            List[SpeakerFragments]: The list of objects.
        """
        # Speaker groups
        current_speaker: str | None = None
        speaker_groups: list[list[SpeechFragment]] = [[]]

        # Group by speakers
        for frag in self._speech_fragments:
            if frag.speaker != current_speaker:
                current_speaker = frag.speaker
                if speaker_groups[-1]:
                    speaker_groups.append([])
            speaker_groups[-1].append(frag)

        # Create SpeakerFragments objects
        speaker_fragments: list[SpeakerFragments] = []
        for group in speaker_groups:
            sd = self._get_speaker_fragments_from_fragment_group(group)
            if sd:
                speaker_fragments.append(sd)

        # Return the grouped SpeakerFragments objects
        return speaker_fragments

    def _get_speaker_fragments_from_fragment_group(
        self,
        group: list[SpeechFragment],
    ) -> SpeakerFragments | None:
        """Take a group of fragments and piece together into SpeakerFragments.

        Each fragment for a given speaker is assembled into a string,
        taking into consideration whether words are attached to the
        previous or next word (notably punctuation). This ensures that
        the text does not have extra spaces. This will also check for
        any straggling punctuation from earlier utterances that should
        be removed.

        Args:
            group: List of SpeechFragment objects.

        Returns:
            SpeakerFragments: The object for the group.
        """

        # Options
        opts = self._stt._stt_options

        # Check for starting fragments that are attached to previous
        if group and group[0].attaches_to == "previous":
            group = group[1:]

        # Check for trailing fragments that are attached to next
        if group and group[-1].attaches_to == "next":
            group = group[:-1]

        # Check there are results
        if not group:
            return None

        # Get the timing extremes
        start_time = min(frag.start_time for frag in group)

        # Timestamp
        ts = (self._start_time + datetime.timedelta(seconds=start_time)).isoformat(
            timespec="milliseconds"
        )

        # Determine if the speaker is considered active
        is_active = True
        if opts.enable_diarization and opts.focus_speakers:
            is_active = group[0].speaker in opts.focus_speakers

        # Return the SpeakerFragments object
        return SpeakerFragments(
            speaker_id=group[0].speaker,
            timestamp=ts,
            language=group[0].language,
            fragments=group,
            is_active=is_active,
        )

    async def aclose(self) -> None:
        """
        End input to the STT engine.

        This will close the STT engine and the WebSocket connection, if established, and
        release any resources.
        """
        await super().aclose()

        # Close the STT session cleanly
        if self._client:
            await self._client.close()
            self._client = None
