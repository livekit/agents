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
from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
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
from .utils import get_stt_url


@dataclasses.dataclass
class STTOptions:
    operating_point: OperatingPoint = OperatingPoint.ENHANCED
    domain: str | None = None
    language: str = "en"
    output_locale: str | None = None
    enable_partials: bool = True
    enable_diarization: bool = False
    max_delay: float = 0.7
    end_of_utterance_silence_trigger: float = 0.3
    end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED
    additional_vocab: list[AdditionalVocabEntry] = dataclasses.field(default_factory=list)
    punctuation_overrides: dict = dataclasses.field(default_factory=dict)
    diarization_sensitivity: float = 0.5
    max_speakers: int | None = None
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
        enable_partials: bool = True,
        enable_diarization: bool = False,
        max_delay: float = 1.0,
        end_of_utterance_silence_trigger: float = 0.5,
        end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED,
        additional_vocab: NotGivenOr[list[AdditionalVocabEntry]] = NOT_GIVEN,
        punctuation_overrides: NotGivenOr[dict] = NOT_GIVEN,
        diarization_sensitivity: float = 0.5,
        max_speakers: NotGivenOr[int] = NOT_GIVEN,
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
            api_key (str): Speechmatics API key. Can be set via `api_key` argument
                or `SPEECHMATICS_API_KEY` environment variable

            base_url (str): Custom base URL for the API. Can be set via `base_url`
                argument or `SPEECHMATICS_RT_URL` environment variable. Optional.

            operating_point (OperatingPoint): Operating point for transcription accuracy
                vs. latency tradeoff. It is recommended to use OperatingPoint.ENHANCED
                for most use cases. Defaults to OperatingPoint.ENHANCED.

            domain (str): Domain to use. Optional.

            language (str): Language code for the STT model. Defaults to `en`. Optional.

            output_locale (str): Output locale for the STT model, e.g. `en-GB`. Optional.

            enable_partials (bool): Enable partial transcriptions. When enabled, the STT
                engine will emit `INTERIM_TRANSCRIPT` events - useful for the visualisation
                of real-time transcription. Defaults to True.

            enable_diarization (bool): Enable speaker diarization. When enabled, the STT
                engine will determine and attribute words to unique speakers. The
                speaker_sensitivity parameter can be used to adjust the sensitivity of
                diarization. Defaults to False.

            max_delay (float): Maximum delay in seconds for transcription. This forces the
                STT engine to speed up the processing of transcribed words and reduces the
                interval between partial and final results. Lower values can have an impact on
                accuracy. Defaults to 1.0.

            end_of_utterance_silence_trigger (float): Maximum delay in seconds for end of
                utterance trigger. The delay is used to wait for any further transcribed
                words before emitting the `FINAL_TRANSCRIPT` events. The value must be
                lower than `max_delay`. Defaults to 0.5.

            end_of_utterance_mode (EndOfUtteranceMode): End of utterance delay mode. When
                ADAPTIVE is used, the delay can be adjusted on the content of what the most
                recent speaker has said, such as rate of speech and whether they have any
                pauses or disfluencies. When FIXED is used, the delay is fixed to the value of
                `end_of_utterance_silence_trigger`. Use of NONE disables end of utterance detection and
                uses a fallback timer. Defaults to `EndOfUtteranceMode.FIXED`.

            additional_vocab (list[AdditionalVocabEntry]): List of additional vocabulary entries.
                If you supply a list of additional vocabulary entries, the this will increase the
                weight of the words in the vocabulary and help the STT engine to better transcribe
                the words. Defaults to [].

            punctuation_overrides (dict): Punctuation overrides. This allows you to override
                the punctuation in the STT engine. This is useful for languages that use different
                punctuation than English. See documentation for more information.
                Defaults to None.

            diarization_sensitivity (float): Diarization sensitivity. A higher value increases
                the sensitivity of diarization and helps when two or more speakers have similar voices.
                Defaults to 0.5.

            max_speakers (int): Maximum number of speakers to detect during diarization. When set,
                the STT engine will limit the number of unique speakers identified in the transcription.
                This is useful for scenarios where you know the maximum number of participants (e.g.,
                2-person interviews, small group meetings). Optional.

            speaker_active_format (str): Formatter for active speaker ID. This formatter is used
                to format the text output for individual speakers and ensures that the context is
                clear for language models further down the pipeline. The attributes `text` and
                `speaker_id` are available. The system instructions for the language model may need
                to include any necessary instructions to handle the formatting.
                Example: `@{speaker_id}: {text}`.
                Defaults to transcription output.

            speaker_passive_format (str): Formatter for passive speaker ID. As with the
                speaker_active_format, the attributes `text` and `speaker_id` are available.
                Example: `@{speaker_id} [background]: {text}`.
                Defaults to transcription output.

            prefer_current_speaker (bool): Prefer current speaker ID. When set to true, groups of
                words close together are given extra weight to be identified as the same speaker.
                Defaults to False.

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

            focus_mode (DiarizationFocusMode): Speaker focus mode for diarization. When set to
                `DiarizationFocusMode.RETAIN`, the STT engine will retain words spoken by other speakers
                (not listed in `ignore_speakers`) and process them as passive speaker frames. When set to
                `DiarizationFocusMode.IGNORE`, the STT engine will ignore words spoken by other speakers
                and they will not be processed. Defaults to `DiarizationFocusMode.RETAIN`.

            known_speakers (list[DiarizationKnownSpeaker]): List of known speaker labels and identifiers.
                If you supply a list of labels and identifiers for speakers, then the STT engine will
                use them to attribute any spoken words to that speaker. This is useful when you want to
                attribute words to a specific speaker, such as the assistant or a specific user. Labels
                and identifiers can be obtained from a running STT session and then used in subsequent
                sessions. Identifiers are unique to each Speechmatics account and cannot be used across
                accounts. Refer to our examples on the format of the known_speakers parameter.
                Defaults to [].

            sample_rate (int): Sample rate for the audio. Optional. Defaults to 16000.

            chunk_size (int): Chunk size for the audio. Optional. Defaults to 160.

            audio_encoding (AudioEncoding): Audio encoding for the audio. Optional.
                Defaults to `AudioEncoding.PCM_S16LE`.

            transcription_config (TranscriptionConfig): Transcription configuration (Deprecated). Optional.

            audio_settings (AudioSettings): Audio settings (Deprecated). Optional.

            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=enable_diarization,
                aligned_transcript="chunk",
                offline_recognize=False,
            ),
        )

        if is_given(transcription_config):
            logger.warning(
                "`transcription_config` is deprecated. Use individual arguments instead (which override this argument)."
            )

        if is_given(audio_settings):
            logger.warning(
                "`audio_settings` is deprecated. Use individual arguments instead (which override this argument)."
            )

        self._stt_options = STTOptions(
            operating_point=operating_point,
            domain=domain if is_given(domain) else None,
            language=language if is_given(language) else "en",
            output_locale=output_locale if is_given(output_locale) else None,
            enable_partials=enable_partials,
            enable_diarization=enable_diarization,
            max_delay=max_delay,
            end_of_utterance_silence_trigger=end_of_utterance_silence_trigger,
            end_of_utterance_mode=end_of_utterance_mode,
            additional_vocab=additional_vocab if is_given(additional_vocab) else [],
            punctuation_overrides=punctuation_overrides if is_given(punctuation_overrides) else {},
            diarization_sensitivity=diarization_sensitivity,
            max_speakers=max_speakers if is_given(max_speakers) else None,
            speaker_active_format=speaker_active_format,
            speaker_passive_format=speaker_passive_format,
            prefer_current_speaker=prefer_current_speaker,
            focus_speakers=focus_speakers if is_given(focus_speakers) else [],
            ignore_speakers=ignore_speakers if is_given(ignore_speakers) else [],
            focus_mode=focus_mode,
            known_speakers=known_speakers if is_given(known_speakers) else [],
        )

        self._api_key: str = api_key if is_given(api_key) else os.getenv("SPEECHMATICS_API_KEY", "")
        self._base_url: str = (
            base_url
            if is_given(base_url)
            else os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")
        )

        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        self._transcription_config: TranscriptionConfig | None = None
        self._process_config()
        self._audio_format = AudioFormat(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            encoding=audio_encoding,
        )

        self._stream: stt.RecognizeStream | None = None
        self._http_session: aiohttp.ClientSession | None = None

        # Lower logging of the SMX module
        logging.getLogger("speechmatics.rt.transport").setLevel(logging.WARNING)

    @property
    def model(self) -> str:
        return "unknown"

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
        if self._transcription_config is None:
            raise RuntimeError("Transcription config not initialized")
        transcription_config = dataclasses.replace(self._transcription_config)
        if is_given(language):
            transcription_config.language = language
        self._stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
        )

        return self._stream

    def _process_config(self) -> None:
        """Create a formatted STT transcription config.

        Creates a transcription config object based on the service parameters. Aligns
        with the Speechmatics RT API transcription config.
        """
        transcription_config = TranscriptionConfig(
            language=self._stt_options.language,
            domain=self._stt_options.domain,
            output_locale=self._stt_options.output_locale,
            operating_point=self._stt_options.operating_point,
            diarization="speaker" if self._stt_options.enable_diarization else None,
            enable_partials=self._stt_options.enable_partials,
            max_delay=self._stt_options.max_delay,
        )

        if self._stt_options.additional_vocab:
            # API expects list of dicts, not dict format
            transcription_config.additional_vocab = [
                {
                    "content": e.content,
                    **({"sounds_like": e.sounds_like} if e.sounds_like else {}),
                }
                for e in self._stt_options.additional_vocab
            ]

        if self._stt_options.enable_diarization:
            # Use dict for speaker diarization config to support all fields including speakers
            dz_cfg: dict[str, Any] = {
                "speaker_sensitivity": self._stt_options.diarization_sensitivity,
                "prefer_current_speaker": self._stt_options.prefer_current_speaker,
            }

            # Add max_speakers if provided
            if self._stt_options.max_speakers is not None:
                dz_cfg["max_speakers"] = self._stt_options.max_speakers

            # Add speakers mapping from known speakers
            if self._stt_options.known_speakers:
                dz_cfg["speakers"] = {
                    s.label: s.speaker_identifiers for s in self._stt_options.known_speakers
                }

            transcription_config.speaker_diarization_config = dz_cfg  # type: ignore[assignment]
        if (
            self._stt_options.end_of_utterance_silence_trigger
            and self._stt_options.end_of_utterance_mode == EndOfUtteranceMode.FIXED
        ):
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=self._stt_options.end_of_utterance_silence_trigger,
            )

        if self._stt_options.punctuation_overrides:
            transcription_config.punctuation_overrides = self._stt_options.punctuation_overrides

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
            stt=stt,
            conn_options=conn_options,
            sample_rate=stt._audio_format.sample_rate,
        )

        # redefine types
        self._stt: STT = stt
        self._speech_duration: float = 0
        # fill in with default value, it'll be reset when `RECOGNITION_STARTED` is received
        self._client: AsyncClient | None = None
        self._speech_fragments: list[SpeechFragment] = []

        # EndOfUtterance fallback timer
        self._end_of_utterance_timer: asyncio.TimerHandle | None = None

    async def _run(self) -> None:
        """Run the STT stream."""
        self._client = AsyncClient(
            api_key=self._stt._api_key,
            url=get_stt_url(self._stt._base_url),
        )

        logger.debug("Connected to Speechmatics STT service")

        opts = self._stt._stt_options

        @self._client.on(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]) -> None:
            logger.debug("Recognition started", extra={"data": message})

        if opts.enable_partials:

            @self._client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def _evt_on_partial_transcript(message: dict[str, Any]) -> None:
                self._handle_transcript(message, is_final=False)

        @self._client.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]) -> None:
            self._handle_transcript(message, is_final=True)

        if opts.end_of_utterance_mode == EndOfUtteranceMode.FIXED:

            @self._client.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]) -> None:
                self._handle_end_of_utterance()

        await self._client.start_session(
            transcription_config=self._stt._transcription_config,
            audio_format=self._stt._audio_format,
        )

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._stt._audio_format.sample_rate,
            num_channels=1,
        )

        async for data in self._input_ch:
            # If the data is a flush sentinel, flush the audio byte stream
            if isinstance(data, self._FlushSentinel):
                frames = audio_bstream.flush()
            else:
                frames = audio_bstream.write(data.data.tobytes())

            for frame in frames:
                self._speech_duration += frame.duration
                await self._client.send_audio(frame.data.tobytes())

        # TODO - handle the closing of the stream?

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        has_changed = self._add_speech_fragments(
            message=message,
            is_final=is_final,
        )

        if not has_changed:
            return

        self._end_of_utterance_timer_start()
        self._send_frames()

    def _end_of_utterance_timer_start(self) -> None:
        """Start the timer for the end of utterance.

        This will use the STT's `end_of_utterance_silence_trigger` value and set
        a timer to send the latest transcript to the pipeline. It is used as a
        fallback from the EnfOfUtterance messages from the STT. Majority of the times,
        the server should be sending the end of utterance messages. In the rare case
        that it doesn't, we'll still time it out so that the pipeline doesn't hang.

        Note that the `end_of_utterance_silence_trigger` will be from when the
        last updated speech was received and this will likely be longer in
        real world time to that inside of the STT engine.
        """
        if self._end_of_utterance_timer is not None:
            self._end_of_utterance_timer.cancel()

        def send_after_delay() -> None:
            logger.debug("Fallback EndOfUtterance triggered.")
            self._handle_end_of_utterance()

        delay = self._stt._stt_options.end_of_utterance_silence_trigger * 4
        self._end_of_utterance_timer = asyncio.get_event_loop().call_later(delay, send_after_delay)

    def _handle_end_of_utterance(self) -> None:
        """Handle the end of utterance event.

        This will check for any running timers for end of utterance, reset them,
        and then send a finalized frame to the pipeline.
        """
        self._send_frames(finalized=True)
        if self._end_of_utterance_timer is not None:
            self._end_of_utterance_timer.cancel()
            self._end_of_utterance_timer = None

    def _send_frames(self, finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Send speech frames to the pipeline. If VAD is enabled, then this will
        also send an interruption and user started speaking frames. When the
        final transcript is received, then this will send a user stopped speaking
        and stop interruption frames.

        Args:
            finalized: Whether the data is final or partial.
        """
        speech_frames = self._get_frames_from_fragments()
        if not speech_frames:
            return

        if not any(frame.is_active for frame in speech_frames):
            return

        if not finalized:
            event_type = stt.SpeechEventType.INTERIM_TRANSCRIPT
        else:
            event_type = stt.SpeechEventType.FINAL_TRANSCRIPT

        for item in speech_frames:
            final_event = stt.SpeechEvent(
                type=event_type,
                alternatives=[
                    item._as_speech_data(
                        self._stt._stt_options.speaker_active_format,
                        self._stt._stt_options.speaker_passive_format,
                    ),
                ],
            )
            self._event_ch.send_nowait(final_event)

        if finalized:
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            self._speech_fragments.clear()

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
        opts = self._stt._stt_options
        fragments: list[SpeechFragment] = []
        current_length = len(self._speech_fragments)

        for result in message.get("results", []):
            alt = result.get("alternatives", [{}])[0]
            if alt.get("content", None):
                fragment = SpeechFragment(
                    start_time=result.get("start_time", 0) + self.start_time_offset,
                    end_time=result.get("end_time", 0) + self.start_time_offset,
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

                fragments.append(fragment)

        self._speech_fragments = [frag for frag in self._speech_fragments if frag.is_final]
        if not fragments and len(self._speech_fragments) == current_length:
            return False

        self._speech_fragments.extend(fragments)
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
        current_speaker: str | None = None
        speaker_groups: list[list[SpeechFragment]] = [[]]
        for frag in self._speech_fragments:
            if frag.speaker != current_speaker:
                current_speaker = frag.speaker
                if speaker_groups[-1]:
                    speaker_groups.append([])
            speaker_groups[-1].append(frag)

        speaker_fragments: list[SpeakerFragments] = []
        for group in speaker_groups:
            sd = self._get_speaker_fragments_from_fragment_group(group)
            if sd:
                speaker_fragments.append(sd)

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
        opts = self._stt._stt_options

        # Check for starting fragments that are attached to previous
        if group and group[0].attaches_to == "previous":
            group = group[1:]

        # Check for trailing fragments that are attached to next
        if group and group[-1].attaches_to == "next":
            group = group[:-1]

        if not group:
            return None

        start_time = min(frag.start_time for frag in group)
        end_time = max(frag.end_time for frag in group)

        # Determine if the speaker is considered active
        is_active = True
        if opts.enable_diarization and opts.focus_speakers:
            is_active = group[0].speaker in opts.focus_speakers

        return SpeakerFragments(
            speaker_id=group[0].speaker,
            start_time=start_time,
            end_time=end_time,
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
