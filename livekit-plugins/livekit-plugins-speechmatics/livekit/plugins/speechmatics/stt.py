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
import os
import re
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
from speechmatics.rt import (
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
    DiarizationFocusMode,
    DiarizationKnownSpeaker,
    EndOfUtteranceMode,
    SpeakerFragments,
    SpeechFragment,
)
from .utils import get_endpoint_url


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        operating_point: OperatingPoint = OperatingPoint.ENHANCED,
        domain: str | None = None,
        language: str = "en",
        output_locale: str | None = None,
        enable_vad: bool = False,
        enable_partials: bool = True,
        enable_diarization: bool = False,
        max_delay: float = 1.0,
        end_of_utterance_silence_trigger: float = 0.5,
        end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED,
        additional_vocab: list[AdditionalVocabEntry] | None = None,
        diarization_sensitivity: float = 0.5,
        speaker_active_format: str = "{text}",
        speaker_passive_format: str = "{text}",
        prefer_current_speaker: bool = False,
        focus_speakers: list[str] | None = None,
        ignore_speakers: list[str] | None = None,
        focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN,
        known_speakers: list[DiarizationKnownSpeaker] | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 160,
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
        )

        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = (
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )
        self._operating_point: OperatingPoint = operating_point
        self._domain: str | None = domain

        # Language
        self._language: str | None = language
        self._output_locale: str | None = output_locale

        # Features
        self._enable_vad: bool = enable_vad
        self._enable_partials: bool = enable_partials
        self._enable_diarization: bool = enable_diarization

        # STT parameters
        self._max_delay: float = max_delay
        self._end_of_utterance_silence_trigger: float = end_of_utterance_silence_trigger
        self._end_of_utterance_mode: EndOfUtteranceMode = end_of_utterance_mode
        self._additional_vocab: list[AdditionalVocabEntry] = additional_vocab or []

        # Diarization
        self._diarization_sensitivity: float = diarization_sensitivity
        self._speaker_active_format: str = speaker_active_format
        self._speaker_passive_format: str = speaker_passive_format
        self._prefer_current_speaker: bool = prefer_current_speaker
        self._focus_speakers: list[str] = focus_speakers or []
        self._ignore_speakers: list[str] = ignore_speakers or []
        self._focus_mode: DiarizationFocusMode = focus_mode
        self._known_speakers: list[DiarizationKnownSpeaker] = known_speakers or []

        # Audio settings
        self._sample_rate: int = sample_rate
        self._chunk_size: int = chunk_size
        self._audio_encoding: AudioEncoding = audio_encoding

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Complete configuration objects
        self._transcription_config: TranscriptionConfig = None
        self._process_config()

        # Set the audio settings
        self._audio_format = AudioFormat(
            sample_rate=self._sample_rate,
            chunk_size=self._chunk_size,
            encoding=self._audio_encoding,
        )

        # Set of active stream
        self._stream: stt.RecognizeStream | None = None

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
            language=self._language,
            domain=self._domain,
            output_locale=self._output_locale,
            operating_point=self._operating_point,
            diarization="speaker" if self._enable_diarization else None,
            enable_partials=self._enable_partials,
            max_delay=self._max_delay,
        )

        # Additional vocab
        if self._additional_vocab:
            transcription_config.additional_vocab = [
                {
                    "content": e.content,
                    "sounds_like": e.sounds_like,
                }
                for e in self._additional_vocab
            ]

        # Diarization
        if self._enable_diarization:
            dz_cfg = {}
            if self._diarization_sensitivity is not None:
                dz_cfg["speaker_sensitivity"] = self._diarization_sensitivity
            if self._prefer_current_speaker is not None:
                dz_cfg["prefer_current_speaker"] = self._prefer_current_speaker
            if self._known_speakers:
                dz_cfg["speakers"] = {s.label: s.speaker_identifiers for s in self._known_speakers}
            if dz_cfg:
                transcription_config.speaker_diarization_config = dz_cfg

        # End of Utterance (for fixed)
        if (
            self._end_of_utterance_silence_trigger
            and self._end_of_utterance_mode == EndOfUtteranceMode.FIXED
        ):
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._end_of_utterance_silence_trigger,
            )

        # Set config
        self._transcription_config = transcription_config


class SpeechStream(stt.RecognizeStream):
    def __init__(self, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)

        # Reference to STT object
        self._stt = stt

        # Speakers
        self._focus_speakers: list[str] = stt._focus_speakers.copy()
        self._ignore_speakers: list[str] = stt._ignore_speakers.copy()
        self._focus_mode: DiarizationFocusMode = stt._focus_mode

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

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        if self._stt._enable_partials:

            @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def _evt_on_partial_transcript(message: dict[str, Any]):
                self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

        # End of Utterance
        if self._stt._end_of_utterance_mode == EndOfUtteranceMode.FIXED:

            @self._client.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]):
                logger.debug("End of utterance received from STT")
                asyncio.create_task(self._send_frames(finalized=True))

        # Speaker Result
        if self._stt._enable_diarization:

            @self._client.on(ServerMessageType.SPEAKERS_RESULT)
            def _evt_on_speakers_result(message: dict[str, Any]):
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

    def update_speakers(
        self,
        focus_speakers: list[str] | None = None,
        ignore_speakers: list[str] | None = None,
        focus_mode: DiarizationFocusMode | None = None,
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
        if not self._enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the diarization configuration
        if focus_speakers is not None:
            self._focus_speakers = focus_speakers
        if ignore_speakers is not None:
            self._ignore_speakers = ignore_speakers
        if focus_mode is not None:
            self._focus_mode = focus_mode

    async def send_message(self, message: ClientMessageType | str, **kwargs: Any) -> None:
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
            asyncio.run_coroutine_threadsafe(
                self._client.send_message(payload), self.get_event_loop()
            )
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

        # Send frames
        asyncio.create_task(self._send_frames())

    def _end_of_utterance_timer_start(self):
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
        async def send_after_delay(delay: float):
            await asyncio.sleep(delay)
            logger.debug("Fallback EndOfUtterance triggered.")
            asyncio.create_task(self._handle_end_of_utterance())

        # Start the timer
        self._end_of_utterance_timer = asyncio.create_task(
            send_after_delay(self._stt._end_of_utterance_silence_trigger * 2)
        )

    async def _handle_end_of_utterance(self):
        """Handle the end of utterance event.

        This will check for any running timers for end of utterance, reset them,
        and then send a finalized frame to the pipeline.
        """
        # Send the frames
        await self._send_frames(finalized=True)

        # Reset the end of utterance timer
        if self._end_of_utterance_timer:
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
                            self._stt._speaker_active_format, self._stt._speaker_passive_format
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

                    # Drop ignored speakers
                    if self._ignore_speakers and fragment.speaker in self._ignore_speakers:
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
        if self._stt._enable_diarization and self._focus_speakers:
            is_active = group[0].speaker in self._focus_speakers

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
