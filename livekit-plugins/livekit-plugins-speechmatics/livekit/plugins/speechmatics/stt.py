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
import json
import os
import re
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .types import (
    AudioSettings,
    ClientMessageType,
    ConnectionSettings,
    RTConversationConfig,
    RTSpeakerDiarizationConfig,
    ServerMessageType,
    SpeechFragment,
    TranscriptionConfig,
)
from .utils import get_access_token, sanitize_url

# Default transcription configuration
DEFAULT_TRANSCRIPTION_CONFIG = TranscriptionConfig(
    language="en",
    operating_point="enhanced",
    enable_partials=True,
    enable_entities=True,
    max_delay=2.0,
    max_delay_mode="fixed",
    diarization="speaker",
    speaker_diarization_config=RTSpeakerDiarizationConfig(max_speakers=4),
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
        transcription_config: NotGivenOr[TranscriptionConfig] = NOT_GIVEN,
        connection_settings: NotGivenOr[ConnectionSettings] = NOT_GIVEN,
        audio_settings: NotGivenOr[AudioSettings] = NOT_GIVEN,
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
                merged_config["speaker_diarization_config"] = RTSpeakerDiarizationConfig(
                    **merged_config["speaker_diarization_config"]
                )

            # Convert nested RTConversationConfig if present
            if "conversation_config" in merged_config:
                merged_config["conversation_config"] = RTConversationConfig(
                    **merged_config["conversation_config"]
                )

            # Create the transcription config
            transcription_config = TranscriptionConfig(**merged_config)

        # Set the connection settings
        if not is_given(connection_settings):
            connection_settings = ConnectionSettings(  # noqa: B008
                url="wss://eu2.rt.speechmatics.com/v2",
            )

        # Set the audio settings
        if not is_given(audio_settings):
            audio_settings = AudioSettings()  # noqa: B008

        # Session configuration
        self._transcription_config = transcription_config
        self._audio_settings = audio_settings
        self._connection_settings = connection_settings
        self._extra_headers = extra_headers or {}

        # Current session
        self._session = http_session

        # Set of active stream
        self._stream: stt.RecognizeStream | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

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
            audio_settings=self._audio_settings,
            connection_settings=self._connection_settings,
            conn_options=conn_options,
            http_session=self.session,
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
        audio_settings: AudioSettings,
        connection_settings: ConnectionSettings,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
        extra_headers: dict,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=audio_settings.sample_rate)

        # Session configuration
        self._transcription_config = transcription_config
        self._audio_settings = audio_settings
        self._connection_settings = connection_settings
        self._extra_headers = extra_headers

        # Uses EndOfUtterance detection
        self._uses_eou_detection = (
            transcription_config.conversation_config
            and transcription_config.conversation_config.end_of_utterance_silence_trigger
        )

        # Session
        self._session = http_session
        self._speech_duration: float = 0

        # Events
        self._reconnect_event = asyncio.Event()
        self._recognition_started = asyncio.Event()

        # Sequence number for audio frames to STT
        self._seq_no = 0

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []

    async def _run(self) -> None:
        """Run the STT stream."""

        # Flag for when the WebSocket is closing
        closing_ws = False

        async def recv_from_lk_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive audio data from LiveKit and send over WebSocket."""

            # Nonlocal flag for when the WebSocket is closing
            nonlocal closing_ws

            # Full message to start recognition
            start_recognition_msg = {
                "message": ClientMessageType.StartRecognition,
                "audio_format": self._audio_settings.asdict(),
                "transcription_config": self._transcription_config.asdict(),
            }

            # Send the start recognition message
            await ws.send_str(json.dumps(start_recognition_msg))

            # Wait for recognition to start
            await self._recognition_started.wait()

            # Create an audio byte stream
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._audio_settings.sample_rate,
                num_channels=1,
            )

            async for data in self._input_ch:
                """Send audio data to the WebSocket."""

                # If the data is a flush sentinel, flush the audio byte stream
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                # Send the audio frames to the WebSocket
                for frame in frames:
                    self._seq_no += 1
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

            # Mark the end of stream message
            closing_ws = True

            # Send the end of stream message to close the session in the STT engine
            await ws.send_str(
                json.dumps(
                    {
                        "message": ClientMessageType.EndOfStream,
                        "last_seq_no": self._seq_no,
                    }
                )
            )

        async def recv_from_stt_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Receive messages from the WebSocket."""

            # Nonlocal flag for when the WebSocket is closing
            nonlocal closing_ws

            # Receive messages from the WebSocket
            while True:
                msg = await ws.receive()

                # Check if the WebSocket is closed
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # Close is expected, see SpeechStream.aclose
                    if closing_ws:
                        return

                    # This will trigger a reconnection, see the _run loop
                    raise APIStatusError(message="Speechmatics connection closed unexpectedly")

                try:
                    # Process the JSON message
                    data = json.loads(msg.data)
                    self._process_stream_event(data, closing_ws)

                except Exception:
                    logger.exception("failed to process Speechmatics message")

        # WebSocket connection
        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            """Loop [re]connects to the WebSocket and runs the send and receive tasks."""

            try:
                # [Re]connect to the WebSocket
                ws = await self._connect_ws()

                # Run the main WebSocket send and receive tasks
                tasks = [
                    asyncio.create_task(recv_from_lk_task(ws)),
                    asyncio.create_task(recv_from_stt_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)

                # Additional task for reconnection
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    # Wait for the first task to complete
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel the other tasks (unless the reconnection task)
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    # If the reconnection task is not done, break
                    if wait_reconnect_task not in done:
                        break

                    # Clear the reconnection event
                    self._reconnect_event.clear()

                finally:
                    # Cancel any running tasks
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    await tasks_group

            finally:
                # Close the WebSocket (if it's open)
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the Speechmatics WebSocket."""

        # Get the API key
        api_key = self._connection_settings.api_key or os.environ.get("SPEECHMATICS_API_KEY")

        # Check we have a valid API key
        if api_key is None:
            raise ValueError(
                "Speechmatics API key is required. "
                "Pass one in via ConnectionSettings.api_key parameter, "
                "or set `SPEECHMATICS_API_KEY` environment variable"
            )

        # Get the access token if required
        if self._connection_settings.get_access_token:
            api_key = await get_access_token(api_key)

        # Create the request headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            **self._extra_headers,
        }

        # Create the WebSocket URL
        url = sanitize_url(self._connection_settings.url, self._transcription_config.language)

        # Connect to the WebSocket
        return await self._session.ws_connect(
            url,
            ssl=self._connection_settings.ssl_context,
            headers=headers,
        )

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
                data=data,
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

    def _add_speech_fragments(self, data: dict, is_final: bool) -> bool:
        """
        Takes a new Partial or Final from the STT engine and accumulates it into the
        _speech_data list. As new final data is added, all partials are removed from
        the list.

        Note: If a known speaker is `__[A-Z0-9]{2,}__`, then the words are skipped,
        as this is used to protect against self-interruption by the assistant or to
        block out specific voices.

        Returns:
            bool: True if the speech data was updated, False otherwise.
        """

        # Parsed new speech data from the STT engine
        fragments: list[SpeechFragment] = []

        # Current length of the speech data
        current_length = len(self._speech_fragments)

        # Iterate over the results in the payload
        for result in data.get("results", []):
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
                )

                # Drop `__XX__` speakers
                if fragment.speaker and re.match(r"^__[A-Z0-9]{2,}__$", fragment.speaker):
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

    def _get_speech_data_from_fragments(self) -> list[SpeakerSpeechData]:
        """
        Get speech data objects for the current fragment list.

        Each speech fragments is grouped by contiguous speaker and then
        returned as a SpeakerSpeechData object with the `speaker_id` field set to
        the current speaker (string). An utterance may contain speech from
        more than one speaker (e.g. S1, S2, S1, S3, ...), so they are kept
        in strict order for the context of the conversation.

        Returns:
            list[SpeakerSpeechData]: The list of SpeakerSpeechData grouped fragments.
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

        # Create SpeechData objects
        speech_data: list[SpeakerSpeechData] = []
        for group in speaker_groups:
            sd = self._get_speech_data_from_fragment_group(group)
            if sd:
                speech_data.append(sd)

        # Return the grouped SpeechData objects
        return speech_data

    def _get_speech_data_from_fragment_group(
        self,
        group: list[SpeechFragment],
    ) -> SpeakerSpeechData | None:
        """
        Take a group of fragments and piece together into SpeakerSpeechData.

        Each fragment for a given speaker is assembled into a string,
        taking into consideration whether words are attached to the
        previous or next word (notably punctuation). This ensures that
        the text does not have extra spaces. This will also check for
        any straggling punctuation from earlier utterances that should
        be removed.

        Returns:
            SpeakerSpeechData: The SpeakerSpeechData object for the group.
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
        end_time = max(frag.end_time for frag in group)
        avg_confidence = sum(frag.confidence for frag in group) / len(group)

        # Cumulative contents
        content = ""

        # Assemble the text
        for frag in group:
            if content == "" or frag.attaches_to == "previous":
                content += frag.content
            else:
                content += " " + frag.content

        # Return the SpeechData object
        return SpeakerSpeechData(
            language=group[0].language,
            text=content,
            start_time=start_time,
            end_time=end_time,
            speaker_id=group[0].speaker,
            confidence=avg_confidence,
        )
