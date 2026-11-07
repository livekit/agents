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
import contextlib
import json
import os
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType
from livekit.agents.tts._provider_format import drop_bracket_cues
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
    TimedString,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.io import (
    AudioOutput,
    PlaybackFinishedEvent,
    PlaybackProgressedEvent,
    PlaybackStartedEvent,
    TextOutput,
)

from .log import logger

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# WebSocket messages and tokens.
KEEPALIVE_MESSAGE = '{"type": "keepalive"}'
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


def is_end_token(token: dict) -> bool:
    """Return True if the given token marks an end or finalized event."""
    return token.get("text") in (END_TOKEN, FINALIZED_TOKEN)


@dataclass
class ContextGeneralItem:
    key: str
    value: str


@dataclass
class ContextTranslationTerm:
    source: str
    target: str


@dataclass
class ContextObject:
    """Context object for models with context_version 2, for Soniox stt-rt-v3-preview and higher.

    Learn more about context in the documentation:
    https://soniox.com/docs/stt/concepts/context
    """

    general: list[ContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[ContextTranslationTerm] | None = None


@dataclass
class TranslationConfig:
    """Translation configuration for the Soniox Speech-to-Text API.

    See: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    type: Literal["one_way", "two_way"]
    target_language: str | None = None
    """Target language for one-way translation."""
    language_a: str | None = None
    """First language for two-way translation."""
    language_b: str | None = None
    """Second language for two-way translation."""

    def __post_init__(self) -> None:
        if self.type == "one_way" and not self.target_language:
            raise ValueError("target_language is required for one_way translation")
        if self.type == "two_way" and not (self.language_a and self.language_b):
            raise ValueError("language_a and language_b are both required for two_way translation")


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: str = "stt-rt-v6"

    language_hints: list[str] | None = None
    language_hints_strict: bool = False
    context: ContextObject | str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_speaker_diarization: bool = False
    enable_language_identification: bool = True

    max_endpoint_delay_ms: int = 2000
    """Maximum delay in milliseconds between speech cessation and endpoint detection.
    Range: 500–3000.
    See: https://soniox.com/docs/stt/rt/endpoint-detection"""

    endpoint_sensitivity: float | None = None
    """How readily the model emits speech endpoints. Range: -1.0 to 1.0.
    Higher values make endpoints more likely (finalize sooner); lower values make them
    less likely. Leave as None to use the server-side default.
    Introduced in the Soniox v5 model; earlier models reject it."""

    endpoint_latency_adjustment_level: int | None = None
    """How aggressively the model reduces endpoint latency. Range: 0 to 3.
    Higher values reduce latency but may emit more endpoints and slightly reduce accuracy.
    Leave as None to use the server-side default.
    Introduced in the Soniox v5 model; earlier models reject it."""

    client_reference_id: str | None = None
    translation: TranslationConfig | None = None

    enable_agent_text_input: bool = False
    """Send each agent reply added to the conversation to Soniox as agent text, so words
    the agent says (e.g. a corrected product name) are recognized when the user says them.
    The model must support it."""

    enable_echo_cancellation: bool = False
    """Send the agent's speech to Soniox as agent text while it plays, which also lets Soniox
    cancel the agent's echo in the user's audio. Implies ``enable_agent_text_input``. Put
    ``STT.agent_text_output()`` in front of the session's transcription output so it receives
    the agent's words with their Text-to-Speech timestamps."""

    agent_text_lead_ms: int = 700
    """How far ahead of playback each of the agent's words is sent, in milliseconds.

    Soniox places agent text where it sits among the audio frames and recognizes the caller's
    reply better when the words arrive a little before the caller hears them; 500 to 1000 is
    the documented range. Only used with ``enable_echo_cancellation``."""

    def __post_init__(self) -> None:
        if not (500 <= self.max_endpoint_delay_ms <= 3000):
            raise ValueError("max_endpoint_delay_ms must be between 500 and 3000")
        if self.agent_text_lead_ms < 0:
            raise ValueError("agent_text_lead_ms cannot be negative")
        if self.endpoint_sensitivity is not None and not (-1.0 <= self.endpoint_sensitivity <= 1.0):
            raise ValueError("endpoint_sensitivity must be between -1.0 and 1.0")
        if self.endpoint_latency_adjustment_level is not None and not (
            0 <= self.endpoint_latency_adjustment_level <= 3
        ):
            raise ValueError("endpoint_latency_adjustment_level must be between 0 and 3")


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        params: STTOptions | None = None,
    ):
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            http_session: Optional aiohttp.ClientSession to use for requests.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        params = params or STTOptions()
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
                diarization=params.enable_speaker_diarization,
                # with echo cancellation each reply is already sent while it plays
                chat_context=params.enable_agent_text_input and not params.enable_echo_cancellation,
            )
        )

        self._api_key = api_key or os.getenv("SONIOX_API_KEY")
        if not self._api_key:
            raise ValueError("Soniox API key is required. Set SONIOX_API_KEY or pass api_key")
        self._base_url = base_url
        self._http_session = http_session
        self._params = params
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._params.model

    @property
    def provider(self) -> str:
        return "Soniox"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Raise error since single-frame recognition is not supported
        by Soniox Speech-to-Text API."""
        raise NotImplementedError(
            "Soniox Speech-to-Text API does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Return a new LiveKit streaming speech-to-text session."""
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        # With echo cancellation each reply is already sent while it plays.
        if not self._params.enable_agent_text_input or self._params.enable_echo_cancellation:
            return
        if (
            (item := ev.item).type == "message"
            and item.role == "assistant"
            and (text := item.text_content)
        ):
            for stream in self._streams:
                stream._push_agent_text(text, end_of_segment=True)

    def agent_text_output(
        self,
        audio_output: AudioOutput,
        next_in_chain: TextOutput | None = None,
        transform: Callable[..., list[TimedString]] = drop_bracket_cues,
    ) -> TextOutput:
        """Return a text output that sends the agent's speech to Soniox as it is spoken.

        It paces the agent's words against playback itself, sending each one
        ``agent_text_lead_ms`` before the caller hears it, so it goes *ahead* of the
        transcript synchronizer rather than after it - that is where the words still
        carry the Text-to-Speech timestamps - and forwards everything on unchanged::

            session.output.transcription = stt.agent_text_output(
                session.output.audio, next_in_chain=session.output.transcription
            )

        Requires ``STTOptions.enable_echo_cancellation``, and, for the pacing to have
        anything to work from, ``use_tts_aligned_transcript=True`` on the session with a
        TTS that reports word timestamps (Soniox TTS does with ``word_timestamps=True``).

        One service per session: the agent's words go to every recognition stream the
        service has open, so a service shared between sessions would send each session
        the other's speech.

        Soniox cancels the echo in the transcript only. A VAD still hears the agent's
        voice in the microphone and interrupts it, so without acoustic echo
        cancellation on the audio, turn detection and interruptions need to come
        from the STT (``turn_detection="stt"`` and no VAD).

        Args:
            audio_output: The session's audio output, whose playback reports say how much
                of the reply the caller has heard.
            next_in_chain: Text output that receives the transcript after this one,
                normally whatever was on the session already.
            transform: Decides which of the Text-to-Speech's words are sent to Soniox,
                called as ``transform(words, held, final=False)`` with the words that
                arrived and a list it keeps between calls for words it holds back,
                such as the start of a tag that has not closed yet. It returns the
                words to send, each keeping its timestamps; ``final=True`` at the end
                of the reply releases what it held. Defaults to dropping bracket
                cues such as ``[laughs]``, which the voice performs rather than says.

        Returns:
            The text output to put in front of the transcript synchronizer.
        """
        return _AgentTextOutput(
            self, audio_output=audio_output, next_in_chain=next_in_chain, transform=transform
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Set up state and queues for a WebSocket-based transcription stream."""
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._params.sample_rate)
        self._stt: STT = stt
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reconnect_event = asyncio.Event()

        self.audio_queue: asyncio.Queue[bytes | _AgentText] = asyncio.Queue()
        # Whether this connection has been sent agent text whose segment is still open.
        self._agent_segment_open = False

        self._reported_duration_ms = 0
        self._num_output_tokens = 0
        self._reported_output_tokens = 0

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession for WebSocket connections."""
        if not self._stt._http_session:
            self._stt._http_session = utils.http_context.http_session()

        return self._stt._http_session

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection to the Soniox Speech-to-Text API and send the
        initial configuration."""
        context_raw = self._stt._params.context
        context_value: dict[str, Any] | str | None
        if isinstance(context_raw, ContextObject):
            context_value = asdict(context_raw)
        else:
            context_value = context_raw

        # Create initial config object.
        config: dict[str, Any] = {
            "model": self._stt._params.model,
            "audio_format": "pcm_s16le",
            "num_channels": self._stt._params.num_channels or 1,
            "enable_endpoint_detection": True,
            "sample_rate": self._stt._params.sample_rate,
            "language_hints": self._stt._params.language_hints,
            "language_hints_strict": self._stt._params.language_hints_strict,
            "context": context_value,
            "enable_speaker_diarization": self._stt._params.enable_speaker_diarization,
            "enable_language_identification": self._stt._params.enable_language_identification,
            "client_reference_id": self._stt._params.client_reference_id,
        }
        config["max_endpoint_delay_ms"] = self._stt._params.max_endpoint_delay_ms
        if self._stt._params.endpoint_sensitivity is not None:
            config["endpoint_sensitivity"] = self._stt._params.endpoint_sensitivity
        if self._stt._params.endpoint_latency_adjustment_level is not None:
            config["endpoint_latency_adjustment_level"] = (
                self._stt._params.endpoint_latency_adjustment_level
            )
        if self._stt._params.enable_agent_text_input or self._stt._params.enable_echo_cancellation:
            config["enable_agent_text_input"] = True
        if self._stt._params.translation is not None:
            tr = self._stt._params.translation
            translation_dict: dict[str, Any] = {"type": tr.type}
            if tr.type == "one_way":
                translation_dict["target_language"] = tr.target_language
            elif tr.type == "two_way":
                translation_dict["language_a"] = tr.language_a
                translation_dict["language_b"] = tr.language_b
            config["translation"] = translation_dict
        # Connect to the Soniox Speech-to-Text API.
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._stt._base_url,
                headers={"Authorization": f"Bearer {self._stt._api_key}"},
                autoping=False,
            ),
            timeout=self._conn_options.timeout,
        )
        # Set initial configuration message.
        await ws.send_str(json.dumps(config))
        logger.debug("Soniox Speech-to-Text API connection established!")

        self._reported_duration_ms = 0
        self._num_output_tokens = 0
        self._reported_output_tokens = 0
        self._agent_segment_open = False
        # Audio queued for the old connection is stale, but agent text still has to
        # reach Soniox: it may have been queued before the first connection opened.
        stale = self.audio_queue
        self.audio_queue = asyncio.Queue()
        while not stale.empty():
            if isinstance(item := stale.get_nowait(), _AgentText):
                self.audio_queue.put_nowait(item)
        return ws

    def _push_agent_text(self, text: str, *, end_of_segment: bool) -> None:
        # rides the audio queue: Soniox places agent text by its position among the audio
        self.audio_queue.put_nowait(_AgentText(text, end_of_segment))

    def _report_processed_audio_duration(self, total_audio_proc_ms: float) -> None:
        """Report the audio duration and output tokens processed by the STT engine."""
        to_report_ms = total_audio_proc_ms - self._reported_duration_ms
        to_report_output_tokens = self._num_output_tokens - self._reported_output_tokens
        if to_report_ms <= 0 and to_report_output_tokens <= 0:
            return

        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=max(to_report_ms, 0) / 1000,
                output_tokens=max(to_report_output_tokens, 0),
            ),
        )
        self._event_ch.send_nowait(usage_event)
        self._reported_duration_ms = max(int(total_audio_proc_ms), self._reported_duration_ms)
        self._reported_output_tokens = self._num_output_tokens

    async def _run(self) -> None:
        """Manage connection lifecycle, spawning tasks and handling reconnection."""
        while True:
            try:
                ws = await self._connect_ws()
                self._input_ch.restart_provider_clock()
                self._ws = ws
                # Create task for audio processing, voice turn detection and message handling.
                tasks: list[asyncio.Task[None]] = [
                    asyncio.create_task(self._prepare_audio_task()),
                    asyncio.create_task(self._send_audio_task()),
                    asyncio.create_task(self._recv_messages_task()),
                    asyncio.create_task(self._keepalive_task()),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                tasks_group: asyncio.Future[Any] = asyncio.gather(*tasks)
                try:
                    done, _ = await asyncio.wait(
                        [tasks_group, wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                    if ws.closed:
                        if self._input_ch.closed:
                            return
                        raise APIConnectionError(
                            "Soniox STT connection closed unexpectedly"
                        ) from ws.exception()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()

            except APIError:
                raise
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout during Soniox Speech-to-Text API connection/initialization: {e}"
                )
                raise APITimeoutError(
                    "Timeout connecting to or initializing Soniox Speech-to-Text API session"
                ) from e

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "Soniox Speech-to-Text API status error during session init:"
                    + f"{e.status} {e.message}"
                )
                raise APIStatusError(
                    message=e.message, status_code=e.status, request_id=None, body=None
                ) from e

            except aiohttp.ClientError as e:
                logger.error(f"Soniox Speech-to-Text API connection error: {e}")
                raise APIConnectionError(f"Soniox Speech-to-Text API connection error: {e}") from e

            except Exception as e:
                logger.exception(f"Unexpected error occurred: {e}")
                raise APIConnectionError(f"An unexpected error occurred: {e}") from e
            # Close the WebSocket connection on finish.
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None

    async def _keepalive_task(self) -> None:
        """Periodically send keepalive messages (while no audio is being sent)
        to maintain the WebSocket connection."""
        try:
            while self._ws:
                await self._ws.send_str(KEEPALIVE_MESSAGE)
                await asyncio.sleep(5)
        except (aiohttp.ClientError, ConnectionError) as e:
            if self._input_ch.closed:
                return
            raise APIConnectionError("Soniox STT keepalive write failed") from e
        except Exception as e:
            logger.error(f"Error while sending keep alive message: {e}")

    async def _prepare_audio_task(self) -> None:
        """Read audio frames and enqueue PCM data for sending."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                # Get the raw bytes from the audio frame.
                pcm_data = data.data.tobytes()
                self.audio_queue.put_nowait(pcm_data)

    async def _send_audio_task(self) -> None:
        """Take queued audio data and transmit it over the WebSocket."""
        if not self._ws:
            logger.error("WebSocket connection to Soniox Speech-to-Text API is not established")
            return

        while self._ws:
            try:
                data = await self.audio_queue.get()

                if isinstance(data, bytes):
                    await self._ws.send_bytes(data)
                elif data.text or self._agent_segment_open:
                    # A bare end_of_segment closes a segment that began before a
                    # reconnect, which this connection never saw, so it is not sent.
                    self._agent_segment_open = not data.end_of_segment
                    await self._ws.send_str(json.dumps({"type": "agent_text", **data._asdict()}))
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, ConnectionError) as e:
                if self._input_ch.closed:
                    return
                raise APIConnectionError("Soniox STT audio write failed") from e
            except Exception as e:
                logger.error(f"Error while sending audio data: {e}")
                break

    async def _recv_messages_task(self) -> None:
        """Receive transcription messages, handle tokens, errors, and dispatch events."""

        # Translation routes original-language tokens to `final_original` and translated
        # tokens to `final`. In non-translation mode, all tokens go to `final` and
        # `final_original` stays empty (so `final` IS the source side there).
        is_translation_mode = self._stt._params.translation is not None

        # final tokens are accumulated across messages until an endpoint is detected.
        final = _TokenAccumulator()
        final_original = _TokenAccumulator()
        is_speaking = False

        def send_endpoint_transcript() -> None:
            nonlocal is_speaking
            if final.text:
                # Translation mode determines the role of each accumulator:
                # when on, `final_original` carries the source side and
                # `final` carries the target side -- even across flush windows
                # where the originals were finalized in a prior message and
                # only translation tokens land in this one. When translation
                # is off, `final` IS the source side and `final_original`
                # stays empty.
                src_segs, tgt_segs = (
                    (final_original._lang_segments, final._lang_segments)
                    if is_translation_mode
                    else (final._lang_segments, [])
                )
                source_languages, source_texts = _lang_segments_to_fields(src_segs)
                target_languages, target_texts = _lang_segments_to_fields(tgt_segs)
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            final.to_speech_data(
                                self.start_time_offset,
                                source_languages=source_languages,
                                source_texts=source_texts,
                                target_languages=target_languages,
                                target_texts=target_texts,
                            )
                        ],
                    )
                )
                # Translation tokens can omit timing; the source tokens carry the speech span.
                endpoint_tokens = final_original if final_original.end_time > 0 else final
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        alternatives=[endpoint_tokens.to_speech_data(self.start_time_offset)]
                        if endpoint_tokens.end_time > 0
                        else [],
                    )
                )

                # Reset buffers.
                final.reset()
                final_original.reset()

                # Reset speaking state, so the next transcript will send START_OF_SPEECH again.
                is_speaking = False
            else:
                final_original.reset()
                if is_speaking:
                    # All tokens of the segment were retracted (non-final
                    # tokens are not retained across messages): the endpoint
                    # must still close the utterance, otherwise the session
                    # stays stuck in speaking and the next utterance never
                    # gets its own START_OF_SPEECH.
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=SpeechEventType.END_OF_SPEECH,
                        )
                    )
                    is_speaking = False

        if not self._ws:
            return

        try:
            async for msg in utils.aio.WebSocketMonitor(
                self._ws, liveness_timeout=self._conn_options.liveness_timeout
            ):
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break

                if msg.type == aiohttp.WSMsgType.ERROR:
                    # Preserve the underlying transport error.
                    logger.warning("Soniox STT WebSocket error frame: %r", self._ws.exception())
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(
                        f"Unexpected message type from Soniox Speech-to-Text API: {msg.type}"
                    )
                    continue

                try:
                    content = json.loads(msg.data)
                    has_error = bool(content.get("error_code") or content.get("error_message"))
                    tokens = content.get("tokens", []) if has_error else content["tokens"]

                    non_final = _TokenAccumulator()
                    non_final_original = _TokenAccumulator()
                    total_audio_proc_ms = content.get("total_audio_proc_ms", 0)

                    # 1) process tokens: accumulate final/non-final,
                    #    flush immediately on endpoint tokens.
                    for token in tokens:
                        # Non-final tokens are re-sent as final later, so only
                        # final tokens count toward output usage. <end>/<fin>
                        # markers are control tokens, not transcription output.
                        if token["is_final"] and not is_end_token(token):
                            self._num_output_tokens += 1
                        is_translated = token.get("translation_status") == "translation"
                        if is_translation_mode and not is_end_token(token) and not is_translated:
                            # Original-language token: capture text for source_text only.
                            if token["is_final"]:
                                final_original.update(token)
                            else:
                                non_final_original.update(token)
                            continue
                        if token["is_final"]:
                            if is_end_token(token):
                                send_endpoint_transcript()
                            else:
                                final.update(token)
                        else:
                            non_final.update(token)

                    # 2) emit START_OF_SPEECH + transcript for remaining content.
                    if final.text or non_final.text:
                        if not is_speaking:
                            is_speaking = True
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                            )
                        # Same source/target classification as in
                        # `send_endpoint_transcript`: in translation mode the
                        # `_original` buckets carry the source side and `final` /
                        # `non_final` carry the translation; in non-translation
                        # mode the `_original` buckets are empty and `final` /
                        # `non_final` ARE the source.
                        merged_originals = _merge_lang_segments(
                            final_original._lang_segments,
                            non_final_original._lang_segments,
                        )
                        merged_primary = _merge_lang_segments(
                            final._lang_segments, non_final._lang_segments
                        )
                        interim_src_segs, interim_tgt_segs = (
                            (merged_originals, merged_primary)
                            if is_translation_mode
                            else (merged_primary, [])
                        )
                        interim_src_langs, interim_src_texts = _lang_segments_to_fields(
                            interim_src_segs
                        )
                        interim_tgt_langs, interim_tgt_texts = _lang_segments_to_fields(
                            interim_tgt_segs
                        )

                        # When all tokens in this batch are final (no non-final pending),
                        # speech has reached a stable state — emit PREFLIGHT_TRANSCRIPT to
                        # allow preemptive LLM generation. This mirrors Deepgram v2's
                        # EagerEndOfTurn behavior.
                        event_type = (
                            SpeechEventType.PREFLIGHT_TRANSCRIPT
                            if final.text and not non_final.text
                            else SpeechEventType.INTERIM_TRANSCRIPT
                        )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=event_type,
                                alternatives=[
                                    final.merged_speech_data(
                                        non_final,
                                        self.start_time_offset,
                                        source_languages=interim_src_langs,
                                        source_texts=interim_src_texts,
                                        target_languages=interim_tgt_langs,
                                        target_texts=interim_tgt_texts,
                                    )
                                ],
                            )
                        )

                    # 3) on error or finish, flush any remaining final tokens.
                    if content.get("finished") or has_error:
                        send_endpoint_transcript()

                    # 4) report processed audio for every frame; gating on
                    #    endpoint/finish frames dropped whatever never reached one.
                    self._report_processed_audio_duration(total_audio_proc_ms)
                    if not has_error and not content.get("finished"):
                        self._mark_provider_progress()

                    if has_error:
                        err_code = content.get("error_code")
                        err_msg = content.get("error_message", "Unknown Soniox STT error")
                        logger.error(f"WebSocket error: {err_code} - {err_msg}")
                        status_code = int(err_code) if isinstance(err_code, int) else -1
                        if isinstance(err_code, str) and err_code.isdigit():
                            status_code = int(err_code)
                        raise APIStatusError(
                            f"Soniox STT error: {err_code} - {err_msg}",
                            status_code=status_code,
                            body=content,
                        )

                    if content.get("finished"):
                        logger.debug("Transcription finished")

                except APIError:
                    raise
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")

        except asyncio.CancelledError:
            # Normal shutdown — don't trigger reconnect.
            raise
        except APIError:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"WebSocket error while receiving: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while receiving messages: {e}")

        # Request reconnect if STT silently dies on WS drop.
        if not self._reconnect_event.is_set():
            logger.warning("Soniox STT WebSocket closed; requesting reconnect")
            self._reconnect_event.set()


def _merge_lang_segments(
    a: list[tuple[str, str]], b: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Merge two (language, text) segment lists, combining adjacent segments of the same language."""
    result = list(a)
    for lang, text in b:
        if result and result[-1][0] == lang:
            lang, t = result[-1]
            result[-1] = (lang, t + text)
        else:
            result.append((lang, text))
    return result


def _lang_segments_to_fields(
    segments: list[tuple[str, str]],
) -> tuple[list[LanguageCode] | None, list[str] | None]:
    """Convert `(lang, text)` runs to the parallel `SpeechData` field pair,
    or `(None, None)` when empty."""
    if not segments:
        return None, None
    return (
        [LanguageCode(lang) for lang, _ in segments],
        [t for _, t in segments],
    )


class _AgentText(NamedTuple):
    text: str
    end_of_segment: bool


@dataclass
class _Playback:
    """How much of the reply being spoken the caller has heard.

    Fed by the audio output's own playback reports, which are only made once the
    audio can no longer be discarded, so the position is never revised
    backwards. A sink that reports no progress at all leaves the time since
    playback began as the estimate.
    """

    began_at: float = 0.0
    offset: float = 0.0
    duration: float = 0.0

    def reset(self) -> None:
        """Forget the last reply, so the next one is not paced against where it ended."""
        self.began_at = self.offset = self.duration = 0.0

    def on_started(self, ev: PlaybackStartedEvent) -> None:
        self.began_at = time.time()
        self.offset = self.duration = 0.0

    def on_progressed(self, ev: PlaybackProgressedEvent) -> None:
        self.began_at = ev.started_at
        self.offset = ev.offset
        self.duration = ev.duration

    def seconds(self) -> float:
        if not self.began_at:
            return 0.0
        elapsed = time.time() - self.began_at
        if not self.duration:
            return max(0.0, elapsed)
        # Never count past the stretch the sink has proven played, so the text
        # cannot run ahead of the voice when the Text-to-Speech falls behind.
        return self.offset + min(elapsed, self.duration)


class _AgentTextOutput(TextOutput):
    """Sends the agent's words to Soniox a little before the caller hears them.

    Soniox places agent text where it sits among the audio frames and recognizes
    the caller's reply better when the words arrive early, so this sits ahead of
    the transcript synchronizer, where the words still carry the timestamps the
    Text-to-Speech gave them, and paces them itself against playback. Everything
    is forwarded on unchanged, so the transcript the caller sees is untouched.

    The text of a reply is complete (``flush``) as soon as the Text-to-Speech has
    finished with it, which is usually well before the caller has heard it, so
    the reply ends with its playback instead: every word is sent when playback
    finishes, or only the words the caller heard when it was interrupted, and
    then ``end_of_segment``. A reply that never plays is closed, unsent, when
    the next one begins.
    """

    def __init__(
        self,
        stt: STT,
        *,
        audio_output: AudioOutput,
        next_in_chain: TextOutput | None,
        transform: Callable[..., list[TimedString]],
    ) -> None:
        super().__init__(label="SonioxAgentText", next_in_chain=next_in_chain)
        self._stt = stt
        self._audio_output = audio_output
        self._segment_open = False
        # The reply's text is all here and only its playback is left.
        self._text_complete = False
        self._transform = transform
        # Words the transform holds back until it can tell what they are.
        self._held: list[TimedString] = []
        self._playback = _Playback()
        # Words still to send, in the order the Text-to-Speech reported them,
        # each with the second of the reply it is heard at.
        self._pending: list[tuple[float, str]] = []
        self._last_start = 0.0
        # Whitespace left on its own, such as the space a dropped tag leaves,
        # waits for the next word; Soniox cancels the echo more reliably
        # without whitespace-only messages.
        self._space = ""
        self._pacing: asyncio.Task[None] | None = None
        self._warned_untimed = False
        # Set when the audio output revises the playback position, so a word
        # waiting on the old estimate is reconsidered against the new one.
        self._revised = asyncio.Event()
        audio_output.on("playback_started", self._playback.on_started)
        audio_output.on("playback_progressed", self._on_playback_progressed)
        audio_output.on("playback_finished", self._on_playback_finished)

    def on_detached(self) -> None:
        self._audio_output.off("playback_started", self._playback.on_started)
        self._audio_output.off("playback_progressed", self._on_playback_progressed)
        self._audio_output.off("playback_finished", self._on_playback_finished)
        self._end_segment()
        super().on_detached()

    async def capture_text(self, text: str) -> None:
        if self._stt._params.enable_echo_cancellation:
            if self._text_complete:
                # The last reply's playback never finished, so nothing says how
                # much of it was heard.
                self._end_segment()
            # The segment is open from the first text of it, not from the first
            # text the transform lets through: a chunk ending mid-tag is held
            # back, and its segment still has to be closed.
            self._segment_open = True
            words = [text if isinstance(text, TimedString) else TimedString(text)]
            self._queue(self._transform(words, self._held))
        if self.next_in_chain:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        if self._segment_open:
            self._text_complete = True
        if self.next_in_chain:
            self.next_in_chain.flush()

    def _on_playback_progressed(self, ev: PlaybackProgressedEvent) -> None:
        """Take the revised position, and let a word waiting on the old one go."""
        self._playback.on_progressed(ev)
        self._revised.set()

    def _on_playback_finished(self, ev: PlaybackFinishedEvent) -> None:
        if self._segment_open:
            if ev.interrupted:
                # The words the caller heard are sent. The rest of the reply is
                # dropped, and so is what the transform still holds back, such
                # as a tag that was cut off.
                self._send_until(ev.playback_position)
            else:
                self._queue(self._transform([], self._held, final=True))
                self._send_until(float("inf"))
            self._end_segment()

    def _end_segment(self) -> None:
        """Close the open segment, dropping whatever of it is still unsent."""
        self._pending.clear()
        self._held.clear()
        self._last_start = 0.0
        self._space = ""
        self._stop_pacing()
        self._playback.reset()
        self._text_complete = False
        if self._segment_open:
            self._segment_open = False
            for stream in self._stt._streams:
                stream._push_agent_text("", end_of_segment=True)

    def _start_of(self, text: TimedString) -> float:
        """The second of the reply this text is heard at."""
        if is_given(text.start_time):
            return text.start_time

        if not self._warned_untimed:
            self._warned_untimed = True
            logger.warning(
                "Soniox agent text has no word timestamps to pace against; set "
                "use_tts_aligned_transcript=True on the session and word timestamps "
                "on the TTS, or the agent's words are sent as they are generated"
            )
        # Nothing to pace against, so this is as late as it gets: a word whose
        # time is unknown is due now.
        return self._playback.seconds()

    def _queue(self, words: list[TimedString]) -> None:
        """Hold spoken words until the caller is about to hear them."""
        for word in words:
            if not word.strip():
                self._space += word
                continue
            start = self._start_of(word)
            if is_given(word.start_time):
                if start < self._last_start:
                    logger.warning(
                        "the Text-to-Speech reported a word timestamp earlier than the "
                        "previous one; Soniox agent text keeps the order the words arrived in"
                    )
                self._last_start = start
            self._pending.append((start, self._space + word))
            self._space = ""
        if not self._pending:
            return
        self._send_due()
        if self._pending and self._pacing is None:
            self._pacing = asyncio.create_task(self._pace())

    def _send_due(self) -> None:
        """Send every word the caller is about to hear."""
        self._send_until(self._horizon())

    def _horizon(self) -> float:
        """How far into the reply the words sent by now have to reach."""
        return self._playback.seconds() + self._stt._params.agent_text_lead_ms / 1000

    def _send_until(self, horizon: float) -> None:
        """Send every word heard by `horizon`, and nothing beyond them."""
        while self._pending and self._pending[0][0] <= horizon:
            _, text = self._pending.pop(0)
            for stream in self._stt._streams:
                stream._push_agent_text(text, end_of_segment=False)

    @utils.log_exceptions(logger=logger)
    async def _pace(self) -> None:
        """Send each word when its moment comes, until none are left.

        The wait is computed from the playback position each time round, as the
        framework's own `TranscriptSynchronizer` recomputes its delay per word.
        A revision of that position cuts the wait short rather than waiting it
        out: the audio output reports a run when it has played, so the position
        it reports can be well ahead of the estimate, and a word that is due by
        then has to go now or the caller hears it first.
        """
        try:
            while self._pending:
                self._revised.clear()
                due = self._pending[0][0] - self._horizon()
                if due > 0:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._revised.wait(), due)
                self._send_due()
        finally:
            # A reply that ends while this one is cancelled has already started
            # the next pacing task.
            if self._pacing is asyncio.current_task():
                self._pacing = None

    def _stop_pacing(self) -> None:
        if self._pacing is not None:
            self._pacing.cancel()
            self._pacing = None


class _LangStats(NamedTuple):
    num_chars: int
    updated_at: float


class _TokenAccumulator:
    """Accumulates token metadata (text, language, speaker, timing, confidence).

    Tokens are assumed to arrive in chronological order, but individual tokens
    may omit timing keys. start_time is the earliest ``start_ms`` seen and
    end_time the latest ``end_ms`` seen, so a token that carries timing late
    (or regresses) cannot collapse the span onto itself.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.language: str = ""
        self.speaker_id: str | None = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self._has_start_time: bool = False
        self._lang_segments: list[tuple[str, str]] = []  # (language, text) pairs
        self._lang_stats: dict[str, _LangStats] = {}

    def _get_language(self) -> str:
        """Language with the most characters; ties go to the one that reached the count first."""
        if not self._lang_stats:
            return ""
        most_chars = max(s.num_chars for s in self._lang_stats.values())
        tied = [
            (lang_code, stats)
            for lang_code, stats in self._lang_stats.items()
            if stats.num_chars == most_chars
        ]
        return min(tied, key=lambda t: t[1].updated_at)[0]

    def update(self, token: dict[str, Any]) -> None:
        text = token["text"]
        lang = token.get("language", "")
        self.text += text
        if lang and text:
            chars, _ = self._lang_stats.get(lang, (0, 0.0))
            self._lang_stats[lang] = _LangStats(chars + len(text), time.monotonic())
            self.language = self._get_language()
        if "speaker" in token and self.speaker_id is None:
            self.speaker_id = str(token["speaker"])
        if "start_ms" in token:
            start_ms = float(token["start_ms"])
            if not self._has_start_time or start_ms < self.start_time:
                self._has_start_time = True
                self.start_time = start_ms
        if "end_ms" in token:
            self.end_time = max(self.end_time, float(token["end_ms"]))
        if "confidence" in token:
            self._confidence_sum += token["confidence"]
            self._confidence_count += 1
        if text:
            if self._lang_segments and self._lang_segments[-1][0] == lang:
                lang, t = self._lang_segments[-1]
                self._lang_segments[-1] = (lang, t + text)
            else:
                self._lang_segments.append((lang, text))

    @property
    def confidence(self) -> float:
        if self._confidence_count == 0:
            return 0.0
        return self._confidence_sum / self._confidence_count

    def reset(self) -> None:
        self.text = ""
        self.language = ""
        self.speaker_id = None
        self.start_time = 0.0
        self.end_time = 0.0
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._has_start_time = False
        self._lang_segments = []
        self._lang_stats = {}

    def to_speech_data(
        self,
        start_time_offset: float = 0.0,
        source_languages: list[LanguageCode] | None = None,
        source_texts: list[str] | None = None,
        target_languages: list[LanguageCode] | None = None,
        target_texts: list[str] | None = None,
    ) -> stt.SpeechData:
        return stt.SpeechData(
            text=self.text,
            language=LanguageCode(self.language),
            source_languages=source_languages,
            source_texts=source_texts,
            target_languages=target_languages,
            target_texts=target_texts,
            speaker_id=self.speaker_id,
            start_time=self.start_time / 1000 + start_time_offset,
            end_time=self.end_time / 1000 + start_time_offset,
            confidence=self.confidence,
        )

    def merged_speech_data(
        self,
        other: _TokenAccumulator,
        start_time_offset: float = 0.0,
        source_languages: list[LanguageCode] | None = None,
        source_texts: list[str] | None = None,
        target_languages: list[LanguageCode] | None = None,
        target_texts: list[str] | None = None,
    ) -> stt.SpeechData:
        """Build a SpeechData combining self (final) with other (non-final)."""
        candidates = [acc.start_time for acc in (self, other) if acc._has_start_time]
        start = min(candidates) if candidates else 0.0
        end = max(self.end_time, other.end_time)
        total_count = self._confidence_count + other._confidence_count
        total_sum = self._confidence_sum + other._confidence_sum
        return stt.SpeechData(
            text=self.text + other.text,
            language=LanguageCode(self.language if self.language else other.language),
            source_languages=source_languages,
            source_texts=source_texts,
            target_languages=target_languages,
            target_texts=target_texts,
            speaker_id=self.speaker_id if self.speaker_id is not None else other.speaker_id,
            start_time=start / 1000 + start_time_offset,
            end_time=end / 1000 + start_time_offset,
            confidence=total_sum / total_count if total_count > 0 else 0.0,
        )
