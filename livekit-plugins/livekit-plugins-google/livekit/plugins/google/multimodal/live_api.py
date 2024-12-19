from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import AsyncIterable, Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.multimodal import MultimodalModel, MultimodalSession

from google import genai
from google.genai.types import (
    GenerationConfigDict,
    LiveConnectConfigDict,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    VoiceConfig,
)

from ..log import logger
from .api_proto import (
    ClientEvents,
    MultimodalModels,
    ResponseModality,
    Voice,
)

EventTypes = Literal[
    "input_speech_started",
    "response_content_added",
    "response_content_done",
    "function_calls_collected",
    "function_calls_finished",
]


@dataclass
class RealtimeContent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    tool_calls: list[Tool]
    content_type: ResponseModality


@dataclass
class ModelOptions:
    model: MultimodalModels | str
    api_key: str | None
    voice: Voice | str
    response_modalities: ResponseModality
    vertexai: bool
    project: str | None
    location: str | None
    candidate_count: int
    temperature: float | None
    max_output_tokens: int | None
    top_p: float | None
    top_k: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    instructions: str


class RealtimeModel(MultimodalModel):
    def __init__(
        self,
        *,
        instructions: str = "",
        model: MultimodalModels | str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        voice: Voice | str = "Puck",
        response_modalities: ResponseModality = "AUDIO",
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        candidate_count: int = 1,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._model = model
        self._http_session = http_session
        self._loop = loop or asyncio.get_event_loop()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._vertexai = vertexai
        self._project_id = project or os.environ.get("GOOGLE_PROJECT")
        self._location = location or os.environ.get("GOOGLE_LOCATION")
        if self._api_key is None and not self._vertexai:
            raise ValueError("GOOGLE_API_KEY is not set")

        self._rt_sessions: list[GeminiRealtimeSession] = []
        self._opts = ModelOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            response_modalities=response_modalities,
            vertexai=vertexai,
            project=project,
            location=location,
            candidate_count=candidate_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            instructions=instructions,
        )

    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
    ) -> MultimodalSession:
        session = GeminiRealtimeSession(
            opts=self._opts,
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            loop=self._loop,
        )
        self._rt_sessions.append(session)

        return session


class GeminiRealtimeSession(utils.EventEmitter[EventTypes], MultimodalSession):
    def __init__(
        self,
        *,
        opts: ModelOptions,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()
        self._loop = loop
        self._opts = opts
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._fnc_tasks = utils.aio.TaskSet()

        self._config = LiveConnectConfigDict(
            model=self._opts.model,
            response_modalities=self._opts.response_modalities,
            generation_config=GenerationConfigDict(
                candidate_count=self._opts.candidate_count,
                temperature=self._opts.temperature,
                max_output_tokens=self._opts.max_output_tokens,
                top_p=self._opts.top_p,
                top_k=self._opts.top_k,
                presence_penalty=self._opts.presence_penalty,
                frequency_penalty=self._opts.frequency_penalty,
            ),
            system_instruction=self._opts.instructions,
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self._opts.voice
                    )
                )
            ),
        )
        self._client = genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
        )
        self._main_atask = asyncio.create_task(
            self._main_task(), name="gemini-realtime-session"
        )
        # dummy task to wait for the session to be initialized # TODO: remove
        self._init_sync_task = asyncio.create_task(
            asyncio.sleep(0), name="gemini-realtime-session-init"
        )
        self._send_ch = utils.aio.Chan[ClientEvents]()
        self._active_response_id = None

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._send_ch.close()
        await self._main_atask

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._fnc_ctx = value

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        data = base64.b64encode(frame.data).decode("utf-8")
        self._queue_msg({"mime_type": "audio/pcm", "data": data})

    def _queue_msg(self, msg: dict) -> None:
        self._send_ch.send_nowait(msg)

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None:
        self._chat_ctx = ctx.copy()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        @utils.log_exceptions(logger=logger)
        async def _send_task():
            async for msg in self._send_ch:
                await self._session.send(msg)

            await self._session.send(".", end_of_turn=True)

        @utils.log_exceptions(logger=logger)
        async def _recv_task():
            while True:
                async for response in self._session.receive():
                    if response.server_content:
                        model_turn = response.server_content.model_turn
                        if model_turn:
                            if self._active_response_id is None:
                                self._active_response_id = utils.shortuuid()
                                text_stream = utils.aio.Chan[str]()
                                audio_stream = utils.aio.Chan[rtc.AudioFrame]()
                                content = RealtimeContent(
                                    response_id=self._active_response_id,
                                    item_id=utils.shortuuid(),
                                    output_index=0,
                                    content_index=0,
                                    text_stream=text_stream,
                                    audio_stream=audio_stream,
                                    tool_calls=[],
                                    content_type="audio",
                                )
                                self.emit("response_content_added", content)

                            for part_index, part in enumerate(model_turn.parts):
                                if part.text:
                                    text_stream.send_nowait(part.text)
                                if part.inline_data:
                                    frame = rtc.AudioFrame(
                                        data=part.inline_data.data,
                                        sample_rate=24000,
                                        num_channels=1,
                                        samples_per_channel=len(part.inline_data.data)
                                        // 2,
                                    )
                                    content.audio_stream.send_nowait(frame)
                    if response.server_content.interrupted:
                        self.emit("input_speech_started")
                    if response.server_content.turn_complete:
                        if isinstance(content.text_stream, utils.aio.Chan):
                            content.text_stream.close()
                        if isinstance(content.audio_stream, utils.aio.Chan):
                            content.audio_stream.close()
                        self.emit("response_content_done", content)
                        self._active_response_id = None

                    # TODO: handle tool calls

        async with self._client.aio.live.connect(
            model=self._opts.model, config=self._config
        ) as session:
            self._session = session
            tasks = [
                asyncio.create_task(_send_task(), name="gemini-realtime-send"),
                asyncio.create_task(_recv_task(), name="gemini-realtime-recv"),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                await self._session.close()

    # TODO: remove
    def _update_conversation_item_content(
        self, item_id: str, content: str | list | None
    ) -> None:
        pass

    def _recover_from_text_response(self, item_id: str | None) -> None:
        pass
