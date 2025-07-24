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
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

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
from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    ConversationConfig,
    OperatingPoint,
    ServerMessageType,
    SpeakerDiarizationConfig,
    TranscriptionConfig,
)

from .log import logger
from .types import SpeakerFragments, SpeechFragment
from .version import __version__

# Default transcription configuration
DEFAULT_TRANSCRIPTION_CONFIG = TranscriptionConfig(
    language="en",
    operating_point=OperatingPoint.ENHANCED,
    enable_partials=True,
    enable_entities=True,
    max_delay=1.5,
    max_delay_mode="fixed",
    diarization="speaker",
    speaker_diarization_config=SpeakerDiarizationConfig(max_speakers=4),
    conversation_config=ConversationConfig(
        end_of_utterance_silence_trigger=0.5,
    ),
)


@dataclass
class SpeakerSpeechData(stt.SpeechData):
    def text_formatted(self) -> str:
        """Wrap with speaker id XML tags."""

        # Wrap in XML tags
        if self.speaker_id:
            return f"<{self.speaker_id}>{self.text.strip()}</{self.speaker_id}>"

        # Simply return the unformatted text
        return self.text


class STT(stt.STT):
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        transcription_config: NotGivenOr[TranscriptionConfig] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_headers: NotGivenOr[dict] = NOT_GIVEN,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
        )

        # Set the transcription config
        if not is_given(transcription_config):
            transcription_config = DEFAULT_TRANSCRIPTION_CONFIG
        else:
            # Merge the default and given transcription config
            merged_config = {
                **DEFAULT_TRANSCRIPTION_CONFIG.asdict(),
                **transcription_config.asdict(),
            }

            # Convert nested RTSpeakerDiarizationConfig if present
            if "speaker_diarization_config" in merged_config:
                merged_config["speaker_diarization_config"] = SpeakerDiarizationConfig(
                    **merged_config["speaker_diarization_config"]
                )

            # Convert nested RTConversationConfig if present
            if "conversation_config" in merged_config:
                merged_config["conversation_config"] = ConversationConfig(
                    **merged_config["conversation_config"]
                )

            # Create the transcription config
            transcription_config = TranscriptionConfig(**merged_config)

        # Set the connection settings
        self._base_url = base_url

        # Set the audio settings
        if not is_given(audio_format):
            audio_format = AudioFormat(
                sample_rate=16000, chunk_size=160, encoding=AudioEncoding.PCM_S16LE
            )
        self._audio_format = audio_format

        # Session configuration
        self._transcription_config = transcription_config
        self._extra_headers = extra_headers or {}

        # Set of active stream
        self._stream: stt.RecognizeStream | None = None

        # Lower logging level for AsyncClient
        # logging.getLogger("speechmatics.rt.transport").setLevel(logging.INFO)

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
            transcription_config=transcription_config,
            audio_format=self._audio_format,
            base_url=self._base_url,
            conn_options=conn_options,
            extra_headers=self._extra_headers,
        )

        # Return the stream
        return self._stream


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        transcription_config: TranscriptionConfig,
        audio_format: AudioFormat,
        base_url: str,
        conn_options: APIConnectOptions,
        extra_headers: dict,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=audio_format.sample_rate)

        # Session configuration
        self._transcription_config = transcription_config
        self._audio_format = audio_format
        self._extra_headers = extra_headers

        # Endpoint URL
        self._base_url = _get_endpoint_url(
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )

        # Uses EndOfUtterance detection
        self._uses_eou_detection = (
            transcription_config.conversation_config
            and transcription_config.conversation_config.end_of_utterance_silence_trigger
        )

        # Session
        self._speech_duration: float = 0
        self._start_time: datetime.datetime | None = None

        # Client
        self._client: AsyncClient | None = None

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []

    async def _run(self) -> None:
        """Run the STT stream."""

        # Create Speechmatics client
        self._client = AsyncClient(url=self._base_url)

        # Recognition started event
        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def _evt_on_partial_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

        # End of Utterance
        if self._uses_eou_detection:

            @self._client.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]):
                logger.debug("End of utterance received from STT")
                asyncio.run_coroutine_threadsafe(
                    self._send_frames(finalized=True), self.get_event_loop()
                )

        # Start session
        await self._client.start_session(
            transcription_config=self._transcription_config,
            audio_format=self._audio_format,
            ws_headers=self._extra_headers,
        )

        # Create an audio byte stream
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._audio_format.sample_rate,
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

    def _process_stream_event(self, data: dict, closing_ws: bool) -> None:
        """
        Process a stream event from the STT engine.

        Events we expect to receive:
            RecognitionStarted: Received once the STT engine has started and ready to receive audio.
            AddPartialTranscript: Partial transcript messages from the STT engine.
            AddTranscript: Final transcript messages from the STT engine.
            EndOfUtterance: End of utterance message from the STT engine.
            EndOfTranscript: End of transcript message from the STT engine at the session end.

        Args:
            data: The stream event data.
            closing_ws: Whether the WebSocket is closing.
        """

        # Get the message type
        message_type = data["message"]

        if message_type == ServerMessageType.RecognitionStarted:
            """Received once the STT engine has started and is ready to receive audio."""
            self._recognition_started.set()
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_ch.send_nowait(start_event)

        elif message_type in (
            ServerMessageType.AddPartialTranscript,
            ServerMessageType.AddTranscript,
        ):
            """Partial and Final transcript messages from the STT engine."""

            # Add the new speech fragments to the list
            has_changed = self._add_speech_fragments(
                message=data,
                is_final=message_type == ServerMessageType.AddTranscript,
            )

            # Skip if unchanged
            if not has_changed:
                return

            # Get the speech data
            speech_data = self._get_speech_data_from_fragments()
            if speech_data:
                self._send_result(speech_data, is_final=False)

        elif message_type == ServerMessageType.EndOfUtterance:
            """End of utterance message from the STT engine."""

            # Get the speech data
            speech_data = self._get_speech_data_from_fragments()
            if speech_data:
                self._send_result(speech_data, is_final=True, is_eou=True)

        elif message_type == ServerMessageType.EndOfTranscript:
            """End of transcript message from the STT engine at the end of the session."""

            if closing_ws:
                pass
            else:
                raise Exception("Speechmatics connection closed unexpectedly")

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

        frags = self._get_frames_from_fragments()
        if frags:
            print(frags)

        # Send frames
        # asyncio.run_coroutine_threadsafe(self._send_frames(), self.get_event_loop())

    def _send_result(
        self,
        speech_data: list[SpeakerSpeechData],
        is_final: bool = False,
        is_eou: bool = False,
    ) -> None:
        """
        Send an interim or final transcript to LiveKit.

        Process the new partial and final data from the STT. With ever new
        payload, all previous partials are removed, retaining any finals.
        The STT will emit repeat partials until they are finalised by the
        engine.

        Args:
            speech_data: The SpeechData objects to send.
            is_final: Whether the transcript is final.
            is_eou: Whether the transcript is an end of utterance.
        """

        # Event type to send
        if not is_final:
            event_type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        else:
            event_type = stt.SpeechEventType.FINAL_TRANSCRIPT

        # Get the speech data and send
        for item in speech_data:
            final_event = stt.SpeechEvent(
                type=event_type,
                alternatives=[item],
            )
            self._event_ch.send_nowait(final_event)

        # Send End of Speech
        if is_eou:
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        # Reset the accumulator and update LiveKit with timing info
        if is_final:
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
                    # if (
                    #     self._diarization_config.ignore_speakers
                    #     and fragment.speaker in self._diarization_config.ignore_speakers
                    # ):
                    #     continue

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
            list[SpeakerFragments]: The list of objects.
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
        # if self._diarization_config.enable and self._diarization_config.focus_speakers:
        #     is_active = group[0].speaker in self._diarization_config.focus_speakers

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


def _get_endpoint_url(url: str) -> str:
    """Format the endpoint URL with the SDK and app versions.

    Args:
        url: The base URL for the endpoint.

    Returns:
        str: The formatted endpoint URL.
    """
    query_params = {}
    query_params["sm-app"] = f"livekit/{__version__}"
    query = urlencode(query_params)

    return f"{url}?{query}"
