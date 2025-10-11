            from copy import deepcopy
from . import events, protocol
from ..log import logger
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils, get_job_context
from livekit.agents.types import (
from livekit.agents.utils import is_given
from typing import Literal
import aiohttp
import asyncio
import json
import numpy as np
import struct
import time
import uuid
import weakref

"""豆包 Realtime API 适配器"""

    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
SAMPLE_RATE = 16000  # 豆包输入采样率 16kHz
OUTPUT_SAMPLE_RATE = 24000  # 豆包输出采样率 24kHz
NUM_CHANNELS = 1
DOUBAO_BASE_URL = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"

DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


@dataclass
class _RealtimeOptions:
    """豆包 Realtime 配置项"""
    model: str  # O 或 SC
    voice: str
    app_id: str
    access_key: str
    resource_id: str
    app_key: str
    asr_config: events.ASRConfig | None
    dialog_config: events.DialogConfig | None
    tts_config: events.TTSConfig | None
    max_session_duration: float | None
    conn_options: APIConnectOptions


class RealtimeModel(llm.RealtimeModel):
    """豆包 Realtime Model"""

    def __init__(
        self,
        *,
        model: Literal["O", "SC"] = "O",
        voice: str = "zh_female_vv_jupiter_bigtts",
        app_id: str | None = None,
        access_key: str | None = None,
        resource_id: str = "volc.speech.dialog",
        app_key: str = "PlgvMymc7f3tQnJ6",
        bot_name: str = "豆包",
        system_role: str | None = None,
        speaking_style: str | None = None,
        end_smooth_window_ms: int = 1500,
        enable_custom_vad: bool = False,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        初始化豆包 Realtime Model

        Args:
            model: 模型版本，"O" 或 "SC"
            voice: 音色ID
            app_id: 火山引擎APP ID
            access_key: 访问密钥
            resource_id: 资源ID
            app_key: APP Key
            bot_name: 人设名称
            system_role: 背景人设
            speaking_style: 说话风格
            end_smooth_window_ms: VAD静音检测窗口（毫秒）
            enable_custom_vad: 是否启用自定义VAD
            http_session: HTTP会话
            max_session_duration: 最大会话时长（秒）
            conn_options: 连接选项
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
            )
        )

        app_id = app_id or os.environ.get("DOUBAO_APP_ID")
        if app_id is None:
            raise ValueError(
                "app_id must be set either by passing app_id "
                "or by setting the DOUBAO_APP_ID environment variable"
            )

        access_key = access_key or os.environ.get("DOUBAO_ACCESS_KEY")
        if access_key is None:
            raise ValueError(
                "access_key must be set either by passing access_key "
                "or by setting the DOUBAO_ACCESS_KEY environment variable"
            )

        # 初始化配置
        asr_config = events.ASRConfig(
            end_smooth_window_ms=end_smooth_window_ms,
            enable_custom_vad=enable_custom_vad,
        )

        dialog_config = events.DialogConfig(
            bot_name=bot_name,
            system_role=system_role,
            speaking_style=speaking_style,
            model=model,
        )

        tts_config = events.TTSConfig(
            speaker=voice,
            audio_format="pcm",
            sample_rate=OUTPUT_SAMPLE_RATE,
            channel=NUM_CHANNELS,
        )

        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            app_id=app_id,
            access_key=access_key,
            resource_id=resource_id,
            app_key=app_key,
            asr_config=asr_config,
            dialog_config=dialog_config,
            tts_config=tts_config,
            max_session_duration=max_session_duration
            if is_given(max_session_duration)
            else DEFAULT_MAX_SESSION_DURATION,
            conn_options=conn_options,
        )

        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
        system_role: NotGivenOr[str | None] = NOT_GIVEN,
        speaking_style: NotGivenOr[str | None] = NOT_GIVEN,
        end_smooth_window_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """更新配置"""
        if is_given(voice) and self._opts.tts_config:
            self._opts.tts_config.speaker = voice

        if self._opts.dialog_config:
            if is_given(bot_name):
                self._opts.dialog_config.bot_name = bot_name
            if is_given(system_role):
                self._opts.dialog_config.system_role = system_role
            if is_given(speaking_style):
                self._opts.dialog_config.speaking_style = speaking_style

        if is_given(end_smooth_window_ms) and self._opts.asr_config:
            self._opts.asr_config.end_smooth_window_ms = end_smooth_window_ms

        # 通知所有会话更新配置
        for sess in self._sessions:
            sess.update_options(
                voice=voice,
                bot_name=bot_name,
                system_role=system_role,
                speaking_style=speaking_style,
                end_smooth_window_ms=end_smooth_window_ms,
            )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class RealtimeSession(llm.RealtimeSession):
    """豆包 Realtime Session"""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._session_id = str(uuid.uuid4())
        self._connect_id = str(uuid.uuid4())
        self._dialog_id: str | None = None

        self._ws_conn: aiohttp.ClientWebSocketResponse | None = None
        self._msg_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self._input_resampler: rtc.AudioResampler | None = None
        self._output_audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        # 创建聊天上下文来维护对话历史
        self._chat_ctx = llm.ChatContext()

        self._session_ready = False  # 添加会话就绪标志
        self._main_atask = asyncio.create_task(self._main_task(), name="RealtimeSession._main_task")

        # 100ms chunks for input
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._current_generation: _ResponseGeneration | None = None
        self._current_assistant_message: list[str] = []  # 累积当前 assistant 的消息
        self._instructions: str | None = None
        self._is_user_initiated: bool = False  # 跟踪当前响应是否由用户发起
        self._pending_generation_futs: deque[asyncio.Future[llm.GenerationCreatedEvent]] = deque()
        self._pending_user_queries: deque[str] = deque()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        bot_name: NotGivenOr[str] = NOT_GIVEN,
        system_role: NotGivenOr[str | None] = NOT_GIVEN,
        speaking_style: NotGivenOr[str | None] = NOT_GIVEN,
        end_smooth_window_ms: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
    ) -> None:
        """更新会话配置（需要重新建立会话）"""
        # 豆包API需要重新StartSession才能更新配置
        # 这里只更新内存中的配置，下次reconnect时生效
        # tool_choice 暂不支持
        pass

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        while True:
            try:
                # 创建 WebSocket 连接
                self._ws_conn = await self._create_ws_conn()

                # 发送 StartConnection
                await self._send_start_connection()

                # 发送 StartSession
                await self._send_start_session()

                # 运行 WebSocket 收发任务
                await self._run_ws(self._ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries >= max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"Doubao Realtime API connection failed after {num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(
                        num_retries
                    )
                    logger.warning(
                        f"Doubao Realtime API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        """创建 WebSocket 连接"""
        headers = {
            "X-Api-App-ID": self._realtime_model._opts.app_id,
            "X-Api-Access-Key": self._realtime_model._opts.access_key,
            "X-Api-Resource-Id": self._realtime_model._opts.resource_id,
            "X-Api-App-Key": self._realtime_model._opts.app_key,
            "X-Api-Connect-Id": self._connect_id,
        }

        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    url=DOUBAO_BASE_URL, headers=headers
                ),
                self._realtime_model._opts.conn_options.timeout,
            )
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message="Doubao Realtime API connection timed out",
            ) from e

    async def _send_start_connection(self) -> None:
        """发送 StartConnection 事件"""
        event = events.StartConnectionEvent()
        # StartConnection 不需要 session_id，只有 event_id + payload
        data = protocol.encode_client_event(
            event_id=events.DoubaoEventID.START_CONNECTION,
            payload=json.dumps(event.to_dict()).encode("utf-8"),
            session_id=None,  # StartConnection 不需要 session_id
        )
        await self._ws_conn.send_bytes(data)

    async def _send_start_session(self) -> None:
        """发送 StartSession 事件"""
        # 如果有 instructions，使用它覆盖 system_role
        dialog_config = self._realtime_model._opts.dialog_config
        if self._instructions and dialog_config:
            # 创建一个副本并更新 system_role
            dialog_config = deepcopy(dialog_config)
            dialog_config.system_role = self._instructions
            logger.debug("Using custom instructions as system_role")

        event = events.StartSessionEvent(
            asr=self._realtime_model._opts.asr_config,
            dialog=dialog_config,
            tts=self._realtime_model._opts.tts_config,
        )
        # StartSession 需要 session_id
        data = protocol.encode_client_event(
            event_id=events.DoubaoEventID.START_SESSION,
            payload=json.dumps(event.to_dict()).encode("utf-8"),
            session_id=self._session_id,
        )
        await self._ws_conn.send_bytes(data)

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """运行 WebSocket 收发任务"""

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            while True:
                data = await self._msg_queue.get()
                await ws_conn.send_bytes(data)

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIConnectionError(message="Doubao connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_binary_message(msg.data)

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    async def _handle_binary_message(self, data: bytes) -> None:
        """处理二进制消息"""
        try:
            decoded = protocol.DoubaoProtocolCodec.decode_message(data)
            event_id = decoded.get("event_id")

            if event_id == events.DoubaoEventID.SESSION_STARTED:
                payload = json.loads(decoded["payload"])
                self._dialog_id = payload.get("dialog_id")
                logger.info(f"Session started, dialog_id={self._dialog_id}")
                self._session_ready = True  # 标记会话已就绪
                self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

            elif event_id == events.DoubaoEventID.CONNECTION_STARTED:
                logger.info("Connection started")

            elif event_id == events.DoubaoEventID.TTS_RESPONSE:
                # 音频数据
                audio_data = decoded["payload"]
                logger.debug(f"Received TTS_RESPONSE, audio_data length: {len(audio_data)}")

                # 豆包默认返回 32-bit float PCM，需要转换为 16-bit int PCM
                # float32 范围 [-1.0, 1.0]，转换为 int16 范围 [-32768, 32767]

                # 解析为 float32 数组
                num_samples = len(audio_data) // 4  # 每个 float32 占 4 字节
                float_samples = struct.unpack(f'<{num_samples}f', audio_data)  # 小端序 float32

                # 转换为 numpy 数组并缩放到 int16 范围
                np_samples = np.array(float_samples, dtype=np.float32)
                np_samples = np.clip(np_samples, -1.0, 1.0)  # 裁剪到 [-1, 1]
                int16_samples = (np_samples * 32767).astype(np.int16)

                # 创建 AudioFrame
                frame = rtc.AudioFrame(
                    data=int16_samples.tobytes(),
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=num_samples,
                )

                # 发送到当前 generation 的音频流
                if self._current_generation:
                    self._current_generation.audio_ch.send_nowait(frame)
                    logger.debug(
                        "Sent audio frame to generation",
                        extra={"samples": num_samples, "generation_id": id(self._current_generation)},
                    )
                else:
                    logger.warning("Received TTS audio but no current generation exists! Creating one now.")
                    self._create_generation()
                    if self._current_generation:
                        self._current_generation.audio_ch.send_nowait(frame)
                        logger.debug("Created generation and sent audio frame", extra={"samples": num_samples})
                # 也发送到全局音频输出通道（兼容性）
                self._output_audio_ch.send_nowait(frame)

            elif event_id == events.DoubaoEventID.ASR_INFO:
                # 用户说话开始（用于打断）
                self.emit("input_speech_started", llm.InputSpeechStartedEvent())

            elif event_id == events.DoubaoEventID.ASR_ENDED:
                # 用户说话结束
                self.emit("input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True))

            elif event_id == events.DoubaoEventID.ASR_RESPONSE:
                # ASR识别结果
                payload = json.loads(decoded["payload"])
                results = payload.get("results", [])
                for result in results:
                    text = result.get("text", "")
                    is_interim = result.get("is_interim", False)
                    if text:
                        # 发出转录完成事件，聊天历史由 AgentSession 自动维护
                        self.emit(
                            "input_audio_transcription_completed",
                            llm.InputTranscriptionCompleted(
                                item_id="",  # 豆包没有item_id概念
                                transcript=text,
                                is_final=not is_interim,
                            ),
                        )

            elif event_id == events.DoubaoEventID.TTS_SENTENCE_START:
                # 创建新的 generation（如果还没有）
                if not self._current_generation:
                    logger.debug("Creating new generation for TTS_SENTENCE_START")
                    self._create_generation()

            elif event_id == events.DoubaoEventID.TTS_ENDED:
                logger.debug("Received TTS_ENDED event")
                # TTS 结束，关闭当前 generation
                if self._current_generation:
                    self._current_generation.text_ch.close()
                    self._current_generation.audio_ch.close()
                    self._current_generation._done_fut.set_result(None)
                    logger.debug("Closed generation streams")
                    self._current_generation = None

            elif event_id == events.DoubaoEventID.CHAT_RESPONSE:
                payload = json.loads(decoded["payload"])

                # 创建新的 generation（如果还没有）
                if not self._current_generation:
                    logger.debug("Creating new generation for CHAT_RESPONSE")
                    self._create_generation()

                # 发送文本到文本流
                text = payload.get("content", "")
                if text:
                    # 累积 assistant 的消息
                    self._current_assistant_message.append(text)
                    logger.debug(f"Received CHAT_RESPONSE text: {text}")

                    if self._current_generation:
                        self._current_generation.text_ch.send_nowait(text)

            elif event_id == events.DoubaoEventID.CHAT_ENDED:
                logger.debug("Received CHAT_ENDED event")
                # 对话结束，发出 assistant 消息完成事件
                # 注意：不直接修改 chat_ctx，因为它可能是只读的（由 AgentSession 管理）
                # 聊天历史会由上层的 AgentSession 自动维护
                if self._current_assistant_message:
                    full_message = "".join(self._current_assistant_message)

                    # 发出自定义事件，通知上层有新的 assistant 消息
                    self.emit("assistant_message_completed", {"content": full_message})
                    logger.debug(f"Emitted assistant_message_completed: {full_message[:50]}...")

                    self._current_assistant_message.clear()

            elif event_id == events.DoubaoEventID.SESSION_FAILED:
                payload = json.loads(decoded["payload"])
                logger.error(f"Session failed: {payload}")
                self._emit_error(
                    APIError(message=f"Session failed: {payload.get('error', 'unknown')}"),
                    recoverable=False,
                )

            elif event_id == events.DoubaoEventID.SESSION_FINISHED:
                logger.info("Session finished")
                self._session_ready = False  # 会话结束，标记为未就绪

            elif decoded.get('message_type') == protocol.MessageType.ERROR_INFORMATION:
                payload_str = decoded["payload"].decode('utf-8') if decoded["payload"] else "{}"
                try:
                    error_payload = json.loads(payload_str)
                except Exception:
                    error_payload = {"error": payload_str}
                logger.error(f"Received error message: {error_payload}")
                self._emit_error(
                    APIError(message=f"Doubao API error: {error_payload.get('error', 'unknown')}"),
                    recoverable=True,
                )

        except Exception as e:
            logger.exception("Failed to handle binary message", exc_info=e)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """推送音频帧"""
        # 只在会话就绪后才发送音频
        if not self._session_ready:
            return

        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                # CRITICAL: 必须转换为 bytes！直接传 memoryview 给 gzip.compress 会产生无效的 gzip 数据
                audio_bytes = bytes(nf.data)
                binary_data = protocol.encode_client_event(
                    event_id=events.DoubaoEventID.TASK_REQUEST,
                    payload=audio_bytes,
                    session_id=self._session_id,
                    is_audio=True,
                )
                self._msg_queue.put_nowait(binary_data)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def _resample_audio(self, frame: rtc.AudioFrame):
        """重采样音频到16kHz"""
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

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )

        while self._pending_generation_futs:
            fut = self._pending_generation_futs.popleft()
            if not fut.done():
                fut.set_exception(error)

    def _create_generation(self) -> None:
        """创建新的响应生成"""
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        done_fut = asyncio.Future[None]()

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            text_ch=text_ch,
            audio_ch=audio_ch,
            _done_fut=done_fut,
            _created_timestamp=time.time(),
        )

        # 创建 MessageGeneration
        async def _text_stream():
            async for text in text_ch:
                yield text

        async def _audio_stream():
            async for frame in audio_ch:
                yield frame

        # 创建 modalities future（豆包支持文本和音频）
        modalities_fut: asyncio.Future[list[str]] = asyncio.Future()
        modalities_fut.set_result(["text", "audio"])

        message_id = str(uuid.uuid4())
        message_gen = llm.MessageGeneration(
            message_id=message_id,
            text_stream=_text_stream(),
            audio_stream=_audio_stream(),
            modalities=modalities_fut,
        )

        # 发送到 message channel
        self._current_generation.message_ch.send_nowait(message_gen)

        # 发出 generation_created 事件，使用正确的 user_initiated 值
        async def _message_stream():
            async for msg in self._current_generation.message_ch:
                yield msg

        async def _function_stream():
            if False:  # 空流
                yield  # type: ignore

        # 使用标志确定是否是用户发起的
        is_user_initiated = self._is_user_initiated
        # 重置标志以备下次使用
        self._is_user_initiated = False

        logger.debug(
            "Emitting generation_created event",
            extra={
                "user_initiated": is_user_initiated,
                "generation_id": id(self._current_generation),
            },
        )

        event = llm.GenerationCreatedEvent(
            message_stream=_message_stream(),
            function_stream=_function_stream(),
            user_initiated=is_user_initiated,
        )

        self.emit("generation_created", event)

        if self._pending_generation_futs:
            fut = self._pending_generation_futs.popleft()
            if not fut.done():
                fut.set_result(event)
        else:
            logger.warning("No pending generation future to resolve when generation_created emitted")

    async def aclose(self) -> None:
        # 发送 FinishSession
        try:
            finish_data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.FINISH_SESSION,
                payload=json.dumps({}).encode("utf-8"),
                session_id=self._session_id,
            )
            await self._msg_queue.put(finish_data)

            # 等待发送完成并给服务器一点时间处理
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Failed to send FinishSession: {e}")

        # 取消主任务
        if not self._main_atask.done():
            self._main_atask.cancel()
            try:
                await self._main_atask
            except asyncio.CancelledError:
                pass

    # 以下方法为兼容接口，豆包暂不支持
    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """同步聊天上下文，并缓存新增的用户文本以供 generate_reply 使用"""
        try:
            old_len = len(self._chat_ctx.items)
            new_items = chat_ctx.items[old_len:]

            for item in new_items:
                if item.type == "message" and item.role == "user":
                    text = item.text_content
                    if text:
                        self._pending_user_queries.append(text)

            self._chat_ctx = chat_ctx
        except Exception as exc:
            logger.error(f"Failed to update chat context: {exc}")

    async def update_tools(self, tools) -> None:
        pass

    async def update_instructions(self, instructions: str) -> None:
        """更新系统提示（豆包在当前会话无法动态更新，会在重连时生效）"""
        self._instructions = instructions

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def generate_reply(self, *, instructions: NotGivenOr[str] = NOT_GIVEN, with_audio: bool = False) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """
        生成回复

        Args:
            instructions: 指令，会作为用户输入发送给豆包，让 AI 基于此生成回复
            with_audio: False=使用CHAT_TEXT_QUERY(返回文本+音频), True=使用CHAT_TTS_TEXT(指定文本合成)

        Note:
            尽管CHAT_TEXT_QUERY的名字暗示只返回文本，但实际上它会同时返回文本和音频。
            CHAT_TTS_TEXT用于指定文本的TTS合成，不适合用户query场景。
        """
        # 返回一个 Future，等待下一个 generation_created 事件
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        self._pending_generation_futs.append(fut)

        content: str | None = None
        if is_given(instructions) and instructions:
            content = instructions
        elif self._pending_user_queries:
            content = self._pending_user_queries.popleft()

        if content:
            # 标记为用户发起的请求
            self._is_user_initiated = True
            if with_audio:
                # 使用 CHAT_TTS_TEXT，指定文本合成音频（不适合用户query）
                self._send_chat_tts_text(content)
            else:
                # 使用 CHAT_TEXT_QUERY，返回AI生成的文本和音频
                self._send_chat_text_query(content)
        else:
            logger.debug("generate_reply invoked without content to send")

        return fut

    def say(self, content: str, *, allow_interruptions: bool = True, add_to_context: bool = True) -> None:
        """
        让 AI 直接说指定的内容（使用 SAY_HELLO）

        Args:
            content: 要说的内容
            allow_interruptions: 是否允许打断（豆包不支持此参数，保留仅为兼容）
            add_to_context: 是否添加到聊天上下文（参数已废弃，保留仅为兼容）

        Note:
            这个方法让 AI 直接朗读指定的文本，不会触发 AI 生成新内容。
            如果你想让 AI 根据指令生成回复，使用 generate_reply()。
            聊天历史由 AgentSession 自动维护，不需要手动添加到 chat_ctx。
        """
        self._send_say_hello(content)

    def _send_chat_text_query(self, content: str) -> None:
        """发送文本查询，触发 AI 生成回复"""
        try:
            payload = {"content": content}
            data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.CHAT_TEXT_QUERY,
                payload=json.dumps(payload).encode("utf-8"),
                session_id=self._session_id,
            )
            logger.debug(f"Sending CHAT_TEXT_QUERY: {content}")
            self._msg_queue.put_nowait(data)
        except Exception as e:
            logger.error(f"Failed to send CHAT_TEXT_QUERY: {e}")

    def _send_chat_tts_text(self, content: str) -> None:
        """发送文本查询并合成音频（同时返回文本和音频）- 流式发送"""
        try:
            # CHAT_TTS_TEXT 是流式事件，需要发送 start, content, end 三次
            logger.debug(f"Sending CHAT_TTS_TEXT (streaming): {content}")

            # 1. 发送开始标记
            start_payload = {"start": True, "content": "", "end": False}
            start_data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.CHAT_TTS_TEXT,
                payload=json.dumps(start_payload).encode("utf-8"),
                session_id=self._session_id,
            )
            self._msg_queue.put_nowait(start_data)

            # 2. 发送内容
            content_payload = {"start": False, "content": content, "end": False}
            content_data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.CHAT_TTS_TEXT,
                payload=json.dumps(content_payload).encode("utf-8"),
                session_id=self._session_id,
            )
            self._msg_queue.put_nowait(content_data)

            # 3. 发送结束标记
            end_payload = {"start": False, "content": "", "end": True}
            end_data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.CHAT_TTS_TEXT,
                payload=json.dumps(end_payload).encode("utf-8"),
                session_id=self._session_id,
            )
            self._msg_queue.put_nowait(end_data)

            logger.debug("CHAT_TTS_TEXT sent successfully (3 frames)")
        except Exception as e:
            logger.error(f"Failed to send CHAT_TTS_TEXT: {e}")

    def _send_say_hello(self, content: str) -> None:
        """发送 SAY_HELLO 事件让豆包直接朗读文本"""
        try:
            payload = {"content": content}
            data = protocol.encode_client_event(
                event_id=events.DoubaoEventID.SAY_HELLO,
                payload=json.dumps(payload).encode("utf-8"),
                session_id=self._session_id,
            )
            logger.debug(f"Sending SAY_HELLO: {content[:50]}...")
            self._msg_queue.put_nowait(data)
        except Exception as e:
            logger.error(f"Failed to send SAY_HELLO: {e}")

    async def _empty_message_stream(self):
        """空消息流"""
        if False:
            yield  # type: ignore

    async def _empty_function_stream(self):
        """空函数流"""
        if False:
            yield  # type: ignore

    def interrupt(self) -> None:
        pass

    def truncate(self, **kwargs) -> None:
        pass

    def start_user_activity(self) -> None:
        """通知模型用户活动已开始"""
        pass


@dataclass
class _ResponseGeneration:
    """响应生成状态"""
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    _done_fut: asyncio.Future[None]
    _created_timestamp: float
