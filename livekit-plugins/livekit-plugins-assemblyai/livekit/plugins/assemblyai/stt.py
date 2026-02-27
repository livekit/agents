# Copyright 2023 LiveKit, Inc.
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
import weakref
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    Language,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .log import logger


@dataclass
class STTOptions:
    sample_rate: int
    buffer_size_seconds: float
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    speech_model: Literal[
        "universal-streaming-english", "universal-streaming-multilingual", "u3-rt-pro", "u3-pro"
    ] = "universal-streaming-english"
    language_detection: NotGivenOr[bool] = NOT_GIVEN
    end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN
    min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN
    max_turn_silence: NotGivenOr[int] = NOT_GIVEN
    format_turns: NotGivenOr[bool] = NOT_GIVEN
    keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN
    prompt: NotGivenOr[str] = NOT_GIVEN
    vad_threshold: NotGivenOr[float] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le",
        model: Literal[
            "universal-streaming-english",
            "universal-streaming-multilingual",
            "u3-rt-pro",
            "u3-pro",
        ] = "universal-streaming-english",
        language_detection: NotGivenOr[bool] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        format_turns: NotGivenOr[bool] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        buffer_size_seconds: float = 0.05,
        base_url: str = "wss://streaming.assemblyai.com",
    ):
        """
        Args:
            base_url: The AssemblyAI streaming endpoint base URL. Use the EU endpoint
                (wss://streaming.eu.assemblyai.com) for streaming in the EU. Defaults to
                wss://streaming.assemblyai.com.
                See https://www.assemblyai.com/docs/universal-streaming for more details.
            vad_threshold: The threshold for voice activity detection (VAD). A value between
                0 and 1 that determines how sensitive the VAD is. Lower values make the VAD
                more sensitive (detects quieter speech). Higher values make it less sensitive.
                Defaults to 0.4.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            ),
        )
        if model == "u3-pro":
            logger.warning("'u3-pro' is deprecated, use 'u3-rt-pro' instead.")
            model = "u3-rt-pro"

        if is_given(prompt) and model != "u3-rt-pro":
            raise ValueError("The 'prompt' parameter is only supported with the 'u3-rt-pro' model.")

        self._base_url = base_url
        assemblyai_api_key = api_key if is_given(api_key) else os.environ.get("ASSEMBLYAI_API_KEY")
        if not assemblyai_api_key:
            raise ValueError(
                "AssemblyAI API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `ASSEMBLYAI_API_KEY` environment variable"
            )
        self._api_key = assemblyai_api_key

        # we want to minimize latency as much as possible, it's ok if the phrase arrives in multiple final transcripts
        # designed to work with LK's end of turn models
        if not is_given(min_end_of_turn_silence_when_confident):
            min_end_of_turn_silence_when_confident = 100

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            encoding=encoding,
            speech_model=model,
            language_detection=language_detection,
            end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
            min_end_of_turn_silence_when_confident=min_end_of_turn_silence_when_confident,
            max_turn_silence=max_turn_silence,
            format_turns=format_turns,
            keyterms_prompt=keyterms_prompt,
            prompt=prompt,
            vad_threshold=vad_threshold,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.speech_model

    @property
    def provider(self) -> str:
        return "AssemblyAI"

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
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self.session,
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_end_of_turn_silence_when_confident):
            self._opts.min_end_of_turn_silence_when_confident = (
                min_end_of_turn_silence_when_confident
            )
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(keyterms_prompt):
            self._opts.keyterms_prompt = keyterms_prompt
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold

        for stream in self._streams:
            stream.update_options(
                buffer_size_seconds=buffer_size_seconds,
                end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
                min_end_of_turn_silence_when_confident=min_end_of_turn_silence_when_confident,
                max_turn_silence=max_turn_silence,
                prompt=prompt,
                keyterms_prompt=keyterms_prompt,
                vad_threshold=vad_threshold,
            )


class SpeechStream(stt.SpeechStream):
    # Used to close websocket
    _CLOSE_MSG: str = json.dumps({"type": "Terminate"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._base_url = base_url
        self._speech_duration: float = 0
        self._last_preflight_start_time: float = 0
        self._config_update_queue: asyncio.Queue[dict] = asyncio.Queue()

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        end_of_turn_confidence_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_when_confident: NotGivenOr[int] = NOT_GIVEN,
        max_turn_silence: NotGivenOr[int] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        keyterms_prompt: NotGivenOr[list[str]] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(end_of_turn_confidence_threshold):
            self._opts.end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        if is_given(min_end_of_turn_silence_when_confident):
            self._opts.min_end_of_turn_silence_when_confident = (
                min_end_of_turn_silence_when_confident
            )
        if is_given(max_turn_silence):
            self._opts.max_turn_silence = max_turn_silence
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(keyterms_prompt):
            self._opts.keyterms_prompt = keyterms_prompt
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold

        # Send UpdateConfiguration message over the active websocket
        config_msg: dict = {"type": "UpdateConfiguration"}
        if is_given(prompt):
            config_msg["prompt"] = prompt
        if is_given(keyterms_prompt):
            config_msg["keyterms_prompt"] = keyterms_prompt
        if is_given(max_turn_silence):
            config_msg["max_turn_silence"] = max_turn_silence
        if is_given(min_end_of_turn_silence_when_confident):
            config_msg["min_end_of_turn_silence_when_confident"] = (
                min_end_of_turn_silence_when_confident
            )
        if is_given(end_of_turn_confidence_threshold):
            config_msg["end_of_turn_confidence_threshold"] = end_of_turn_confidence_threshold
        if is_given(vad_threshold):
            config_msg["vad_threshold"] = vad_threshold

        if len(config_msg) > 1:
            self._config_update_queue.put_nowait(config_msg)

    def force_endpoint(self) -> None:
        """Force-finalize the current turn immediately."""
        self._config_update_queue.put_nowait({"type": "ForceEndpoint"})

    async def _run(self) -> None:
        """Run a single websocket connection to AssemblyAI."""
        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            samples_per_buffer = self._opts.sample_rate // round(1 / self._opts.buffer_size_seconds)
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            # forward inputs to AssemblyAI
            # if we receive a close message, signal it to AssemblyAI and break.
            # the recv task will then make sure to process the remaining audio and stop
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

            closing_ws = True
            await ws.send_str(SpeechStream._CLOSE_MSG)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    if closing_ws:
                        break
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    raise APIStatusError(
                        "AssemblyAI connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("unexpected AssemblyAI message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process AssemblyAI message")

        async def send_config_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """Send config updates and control messages immediately, independent of audio."""
            while True:
                config_msg = await self._config_update_queue.get()
                await ws.send_str(json.dumps(config_msg))

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            config_task = asyncio.create_task(send_config_task(ws))
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(config_task, *tasks)
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        # u3-rt-pro defaults: min=100, max=min (so both 100 unless overridden)
        min_silence: int | None
        max_silence: int | None
        if self._opts.speech_model == "u3-rt-pro":
            min_silence = (
                self._opts.min_end_of_turn_silence_when_confident
                if is_given(self._opts.min_end_of_turn_silence_when_confident)
                else 100
            )
            max_silence = (
                self._opts.max_turn_silence
                if is_given(self._opts.max_turn_silence)
                else min_silence
            )
        else:
            min_silence = (
                self._opts.min_end_of_turn_silence_when_confident
                if is_given(self._opts.min_end_of_turn_silence_when_confident)
                else None
            )
            max_silence = (
                self._opts.max_turn_silence if is_given(self._opts.max_turn_silence) else None
            )

        live_config = {
            "sample_rate": self._opts.sample_rate,
            "encoding": self._opts.encoding,
            "speech_model": self._opts.speech_model,
            "format_turns": self._opts.format_turns if is_given(self._opts.format_turns) else None,
            "end_of_turn_confidence_threshold": self._opts.end_of_turn_confidence_threshold
            if is_given(self._opts.end_of_turn_confidence_threshold)
            else None,
            "min_end_of_turn_silence_when_confident": min_silence,
            "max_turn_silence": max_silence,
            "keyterms_prompt": json.dumps(self._opts.keyterms_prompt)
            if is_given(self._opts.keyterms_prompt)
            else None,
            "language_detection": self._opts.language_detection
            if is_given(self._opts.language_detection)
            else True
            if "multilingual" in self._opts.speech_model or self._opts.speech_model == "u3-rt-pro"
            else False,
            "prompt": self._opts.prompt if is_given(self._opts.prompt) else None,
            "vad_threshold": self._opts.vad_threshold
            if is_given(self._opts.vad_threshold)
            else None,
        }

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": "AssemblyAI/1.0 (integration=Livekit)",
        }

        filtered_config = {
            k: ("true" if v else "false") if isinstance(v, bool) else v
            for k, v in live_config.items()
            if v is not None
        }
        url = f"{self._base_url}/v3/ws?{urlencode(filtered_config)}"
        ws = await self._session.ws_connect(url, headers=headers)
        return ws

    def _process_stream_event(self, data: dict) -> None:
        message_type = data.get("type")

        if message_type == "SpeechStarted":
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            return

        if message_type != "Turn":
            return
        words = data.get("words", [])
        end_of_turn = data.get("end_of_turn", False)
        turn_is_formatted = data.get("turn_is_formatted", False)
        utterance = data.get("utterance", "")
        transcript = data.get("transcript", "")
        language = Language(data.get("language_code", "en"))

        # transcript (final) and words (interim) are cumulative
        # utterance (preflight) is chunk based
        start_time: float = 0
        end_time: float = 0
        confidence: float = 0
        # word timestamps are in milliseconds
        # https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api#receive.receiveTurn.words
        timed_words: list[TimedString] = [
            TimedString(
                text=word.get("text", ""),
                start_time=word.get("start", 0) / 1000 + self.start_time_offset,
                end_time=word.get("end", 0) / 1000 + self.start_time_offset,
                start_time_offset=self.start_time_offset,
                confidence=word.get("confidence", 0),
            )
            for word in words
        ]

        # words are cumulative
        if timed_words:
            interim_text = " ".join(word for word in timed_words)
            start_time = timed_words[0].start_time or start_time
            end_time = timed_words[-1].end_time or end_time
            confidence = sum(word.confidence or 0.0 for word in timed_words) / len(timed_words)

            interim_event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=language,
                        text=interim_text,
                        start_time=start_time,
                        end_time=end_time,
                        words=timed_words,
                        confidence=confidence,
                    )
                ],
            )
            self._event_ch.send_nowait(interim_event)

        if utterance:
            if self._last_preflight_start_time == 0.0:
                self._last_preflight_start_time = start_time

            # utterance is chunk based so we need to filter the words to
            # only include the ones that are part of the current utterance
            utterance_words = [
                word
                for word in timed_words
                if is_given(word.start_time) and word.start_time >= self._last_preflight_start_time
            ]
            utterance_confidence = sum(word.confidence or 0.0 for word in utterance_words) / max(
                len(utterance_words), 1
            )

            final_event = stt.SpeechEvent(
                type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=language,
                        text=utterance,
                        start_time=self._last_preflight_start_time,
                        end_time=end_time,
                        words=utterance_words,
                        confidence=utterance_confidence,
                    )
                ],
            )
            self._event_ch.send_nowait(final_event)
            self._last_preflight_start_time = end_time

        if end_of_turn and (
            not (is_given(self._opts.format_turns) and self._opts.format_turns) or turn_is_formatted
        ):
            final_event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=language,
                        text=transcript,
                        start_time=start_time,
                        end_time=end_time,
                        words=timed_words,
                        confidence=confidence,
                    )
                ],
            )
            self._event_ch.send_nowait(final_event)

            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

            if self._speech_duration > 0.0:
                usage_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(audio_duration=self._speech_duration),
                )
                self._event_ch.send_nowait(usage_event)
                self._speech_duration = 0
                self._last_preflight_start_time = 0.0
