from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents._exceptions import APIConnectionError, APIError, APITimeoutError
from livekit.agents.log import logger
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import create_access_token, get_default_inference_url

STSModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-1.5",
]

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _STSOptions:
    model: str
    voice: str
    instructions: str
    modalities: list[str]
    temperature: float | None
    base_url: str
    api_key: str
    api_secret: str
    turn_detection: dict[str, Any] | None
    input_audio_transcription: dict[str, Any] | None
    noise_reduction: dict[str, Any] | None


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    messages: dict[str, _MessageGeneration]
    response_id: str
    created_timestamp: float = 0.0
    first_token_timestamp: float | None = None


class STS(llm.RealtimeModel):
    def __init__(
        self,
        model: STSModels | str = "openai/gpt-realtime",
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        instructions: str = "",
        modalities: list[str] | None = None,
        temperature: float | None = None,
        turn_detection: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        noise_reduction: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        td = (
            turn_detection
            if is_given(turn_detection)
            else {"type": "server_vad", "silence_duration_ms": 300}
        )
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=td is not None,
                user_transcription=input_audio_transcription is not None
                if is_given(input_audio_transcription)
                else False,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
            )
        )

        lk_base_url = base_url if is_given(base_url) else get_default_inference_url()
        lk_api_key = (
            api_key
            if is_given(api_key)
            else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
        )
        if not lk_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY env var"
            )

        lk_api_secret = (
            api_secret
            if is_given(api_secret)
            else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
        )
        if not lk_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET env var"
            )

        self._opts = _STSOptions(
            model=model,
            voice=voice if is_given(voice) else "alloy",
            instructions=instructions,
            modalities=modalities or ["text", "audio"],
            temperature=temperature,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            turn_detection=td,
            input_audio_transcription=input_audio_transcription
            if is_given(input_audio_transcription)
            else None,
            noise_reduction=noise_reduction if is_given(noise_reduction) else None,
        )

    @classmethod
    def from_model_string(cls, model: str) -> STS:
        return cls(model=model)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self) -> STSSession:
        return STSSession(self)

    async def aclose(self) -> None:
        pass


class STSSession(llm.RealtimeSession):
    def __init__(self, realtime_model: STS) -> None:
        super().__init__(realtime_model)
        self._model: STS = realtime_model
        self._opts = realtime_model._opts
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._recv_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._connected = False

        self._current_generation: _ResponseGeneration | None = None
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def _connect(self) -> None:
        if self._connected:
            return

        self._http_session = aiohttp.ClientSession()
        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        token = create_access_token(self._opts.api_key, self._opts.api_secret)
        headers = {"Authorization": f"Bearer {token}"}

        try:
            self._ws = await self._http_session.ws_connect(
                f"{base_url}/sts?model={self._opts.model}",
                headers=headers,
                timeout=30,
            )
        except aiohttp.ClientResponseError as e:
            raise APIConnectionError(f"STS connection failed: {e.message}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("STS connection timed out") from e

        session_create: dict[str, Any] = {
            "type": "session.create",
            "model": self._opts.model,
            "voice": self._opts.voice,
            "modalities": self._opts.modalities,
            "instructions": self._opts.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }
        if self._opts.turn_detection is not None:
            session_create["turn_detection"] = self._opts.turn_detection
        if self._opts.temperature is not None:
            session_create["temperature"] = self._opts.temperature
        if self._opts.input_audio_transcription is not None:
            session_create["input_audio_transcription"] = self._opts.input_audio_transcription
        if self._opts.noise_reduction is not None:
            session_create["noise_reduction"] = self._opts.noise_reduction

        await self._ws.send_str(json.dumps(session_create))

        msg = await self._ws.receive(timeout=10)
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get("type") == "error":
                raise APIError(f"STS session creation failed: {data.get('message', '')}")

        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def _recv_loop(self) -> None:
        if not self._ws:
            return

        while True:
            try:
                msg = await self._ws.receive()
            except Exception:
                break

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                break

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "input_audio_buffer.speech_started":
                self.emit("input_speech_started", llm.InputSpeechStartedEvent())

            elif event_type == "input_audio_buffer.speech_stopped":
                self.emit(
                    "input_speech_stopped",
                    llm.InputSpeechStoppedEvent(
                        user_transcription_enabled=self._opts.input_audio_transcription
                        is not None
                    ),
                )

            elif event_type == "response.created":
                self._handle_response_created(data)

            elif event_type == "response.output_item.added":
                self._handle_response_output_item_added(data)

            elif event_type == "response.content_part.added":
                self._handle_response_content_part_added(data)

            elif event_type == "response.output_audio.delta":
                self._handle_response_audio_delta(data)

            elif event_type == "response.output_audio_transcript.delta":
                self._handle_response_text_delta(data)

            elif event_type == "response.output_text.delta":
                self._handle_response_text_delta(data)

            elif event_type == "response.output_item.done":
                self._handle_response_output_item_done(data)

            elif event_type == "response.done":
                self._handle_response_done(data)

            elif event_type == "error":
                err_data = data.get("error", {})
                err_msg = err_data.get("message", str(data))
                logger.warning("STS error: %s", err_msg)
                self.emit(
                    "error",
                    llm.RealtimeModelError(
                        timestamp=time.time(),
                        label="sts_error",
                        error=APIError(err_msg),
                        recoverable=True,
                    ),
                )

        self._close_current_generation()

    def _handle_response_created(self, data: dict[str, Any]) -> None:
        response = data.get("response", {})
        response_id = response.get("id", "")

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
            response_id=response_id,
            created_timestamp=time.time(),
        )

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )

        metadata = response.get("metadata", {})
        if isinstance(metadata, dict):
            client_event_id = metadata.get("client_event_id", "")
            if client_event_id and client_event_id in self._response_created_futures:
                fut = self._response_created_futures.pop(client_event_id)
                if not fut.done():
                    generation_ev.user_initiated = True
                    fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item = data.get("item", {})
        item_id = item.get("id", "")
        item_type = item.get("type", "")

        if item_type == "message":
            item_gen = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=asyncio.get_running_loop().create_future(),
            )
            self._current_generation.messages[item_id] = item_gen

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_gen.text_ch,
                    audio_stream=item_gen.audio_ch,
                    modalities=item_gen.modalities,
                )
            )

        elif item_type == "function_call":
            call_id = item.get("call_id", "")
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            if call_id and name:
                self._current_generation.function_ch.send_nowait(
                    llm.FunctionCall(
                        id=item_id,
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                    )
                )

    def _handle_response_content_part_added(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item_id = data.get("item_id", "")
        part = data.get("part", {})
        part_type = part.get("type", "")

        item_gen = self._current_generation.messages.get(item_id)
        if item_gen and not item_gen.modalities.done():
            if part_type == "audio":
                item_gen.modalities.set_result(["audio", "text"])
            elif part_type == "text":
                item_gen.modalities.set_result(["text"])

    def _handle_response_audio_delta(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item_id = data.get("item_id", "")
        item_gen = self._current_generation.messages.get(item_id)
        if item_gen is None:
            return

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        if not item_gen.modalities.done():
            item_gen.modalities.set_result(["audio", "text"])

        delta = data.get("delta", "")
        if delta:
            audio_data = base64.b64decode(delta)
            item_gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio_data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=len(audio_data) // 2,
                )
            )

    def _handle_response_text_delta(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        item_id = data.get("item_id", "")
        item_gen = self._current_generation.messages.get(item_id)
        if item_gen is None:
            return

        delta = data.get("delta", "")
        if delta:
            item_gen.text_ch.send_nowait(delta)

    def _handle_response_output_item_done(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item = data.get("item", {})
        item_id = item.get("id", "")
        item_gen = self._current_generation.messages.get(item_id)
        if item_gen:
            if not item_gen.text_ch.closed:
                item_gen.text_ch.close()
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.close()
            if not item_gen.modalities.done():
                item_gen.modalities.set_result(self._model._opts.modalities)

    def _handle_response_done(self, data: dict[str, Any]) -> None:
        self._emit_usage_metrics(data)
        self._close_current_generation()

    def _emit_usage_metrics(self, data: dict[str, Any]) -> None:
        response = data.get("response", {})
        usage = response.get("usage", {})
        if not usage:
            return

        gen = self._current_generation
        created_timestamp = gen.created_timestamp if gen else time.time()
        first_token_timestamp = gen.first_token_timestamp if gen else None
        response_id = response.get("id", gen.response_id if gen else "")
        status = response.get("status", "")

        ttft = (
            first_token_timestamp - created_timestamp
            if first_token_timestamp
            else -1
        )
        duration = time.time() - created_timestamp

        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})
        cached_details = input_details.get("cached_tokens_details", {})

        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=response_id,
            ttft=ttft,
            duration=duration,
            cancelled=status == "cancelled",
            label="sts",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=(
                usage.get("output_tokens", 0) / duration if duration > 0 else 0
            ),
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=input_details.get("audio_tokens", 0),
                cached_tokens=input_details.get("cached_tokens", 0),
                text_tokens=input_details.get("text_tokens", 0),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    text_tokens=cached_details.get("text_tokens", 0),
                    audio_tokens=cached_details.get("audio_tokens", 0),
                    image_tokens=cached_details.get("image_tokens", 0),
                ),
                image_tokens=input_details.get("image_tokens", 0),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=output_details.get("text_tokens", 0),
                audio_tokens=output_details.get("audio_tokens", 0),
                image_tokens=output_details.get("image_tokens", 0),
            ),
            metadata=Metadata(
                model_name=self._model._opts.model,
                model_provider="livekit",
            ),
        )
        self.emit("metrics_collected", metrics)

    def _close_current_generation(self) -> None:
        if self._current_generation is None:
            return

        for item_gen in self._current_generation.messages.values():
            if not item_gen.text_ch.closed:
                item_gen.text_ch.close()
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.close()
            if not item_gen.modalities.done():
                item_gen.modalities.set_result(self._model._opts.modalities)

        if not self._current_generation.function_ch.closed:
            self._current_generation.function_ch.close()
        if not self._current_generation.message_ch.closed:
            self._current_generation.message_ch.close()

        self._current_generation = None

    async def _send_loop(self) -> None:
        async for msg in self._msg_ch:
            try:
                if self._ws:
                    await self._ws.send_str(json.dumps(msg))
            except Exception:
                logger.exception("STS: failed to send event")

    def _queue_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    async def _send(self, event: dict[str, Any]) -> None:
        if not self._connected or not self._ws:
            await self._connect()
        self._queue_event(event)

    async def update_instructions(self, instructions: str) -> None:
        self._opts.instructions = instructions
        await self._send({
            "type": "session.update",
            "session": {"type": "realtime", "instructions": instructions},
        })

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        tool_defs: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, llm.FunctionTool):
                tool_defs.append(
                    llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                )
            elif isinstance(tool, llm.RawFunctionTool):
                desc = dict(tool.info.raw_schema)
                desc.pop("meta", None)
                desc["type"] = "function"
                tool_defs.append(desc)
        await self._send({
            "type": "session.update",
            "session": {"type": "realtime", "tools": tool_defs},
        })

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._connected or not self._ws:
            logger.warning("STS push_audio called before session is connected, dropping frame")
            return
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self._queue_event({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(nf.data).decode("utf-8"),
                })

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        self._response_created_futures[event_id] = fut

        response_params: dict[str, Any] = {
            "metadata": {"client_event_id": event_id},
        }
        if is_given(instructions):
            response_params["instructions"] = instructions

        self._queue_event({
            "type": "response.create",
            "event_id": event_id,
            "response": response_params,
        })

        def _on_timeout() -> None:
            self._response_created_futures.pop(event_id, None)
            if not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_running_loop().call_later(10.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())

        return fut

    def commit_audio(self) -> None:
        if self._ws:
            for nf in self._bstream.flush():
                self._queue_event({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(nf.data).decode("utf-8"),
                })
            self._queue_event({"type": "input_audio_buffer.commit"})

    def clear_audio(self) -> None:
        if self._ws:
            self._queue_event({"type": "input_audio_buffer.clear"})

    @property
    def has_active_generation(self) -> bool:
        return self._current_generation is not None or len(self._response_created_futures) > 0

    def interrupt(self) -> None:
        if not self._ws or not self.has_active_generation:
            return
        self._queue_event({"type": "response.cancel"})

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._ws:
            self._queue_event({
                "type": "conversation.item.truncate",
                "item_id": message_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            })

    async def aclose(self) -> None:
        self._close_current_generation()
        self._msg_ch.close()
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._http_session:
            await self._http_session.close()
        self._connected = False
