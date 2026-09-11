# Copyright 2026 LiveKit, Inc.
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

"""Text-to-speech over Model Studio's realtime WebSocket (``qwen3-tts-flash-realtime``).

Model Studio exposes no OpenAI-compatible ``/audio/speech``, so ``openai.TTS(base_url=...)``
would 404. Text goes up incrementally as ``input_text_buffer.append`` and audio comes back
as base64 ``response.audio.delta`` frames. In ``server_commit`` mode the model closes
sentences itself, so audio starts before LiveKit has finished the turn's text.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._realtime import (
    CLOSE_TYPES,
    DEFAULT_FINISH_TIMEOUT,
    RealtimeSocket,
    connect,
    status_error_from,
)
from ._utils import resolve_api_key, resolve_realtime_url
from .models import (
    DEFAULT_REGION,
    DEFAULT_TTS_LANGUAGE_TYPE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    TTS_SAMPLE_RATE,
    QwenRegion,
    TTSLanguageTypes,
    TTSModels,
    TTSVoices,
)

NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: str
    voice: str
    language_type: str
    speech_rate: float | None
    finish_timeout: float

    def session_config(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "voice": self.voice,
            # server_commit lets the model close sentences itself, so audio starts before
            # LiveKit has finished the turn's text.
            "mode": "server_commit",
            "language_type": self.language_type,
            "response_format": "pcm",
            "sample_rate": TTS_SAMPLE_RATE,
        }
        if self.speech_rate is not None:
            session["speech_rate"] = self.speech_rate
        return session


class TTS(tts.TTS):
    """Streaming text-to-speech on Alibaba Cloud Model Studio (Qwen3-TTS realtime)."""

    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_TTS_MODEL,
        voice: TTSVoices | str = DEFAULT_TTS_VOICE,
        language_type: TTSLanguageTypes | str = DEFAULT_TTS_LANGUAGE_TYPE,
        speech_rate: float | None = None,
        finish_timeout: float = DEFAULT_FINISH_TIMEOUT,
        region: QwenRegion = DEFAULT_REGION,
        base_url: str | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Qwen realtime TTS.

        Args:
            model: Model Studio realtime TTS model id.
            voice: Built-in voice name. ``Cherry``, ``Serena``, ``Ethan`` and ``Chelsie``
                speak both Mandarin and English; the dialect voices do not.
            language_type: ``"Auto"`` detects per request; naming the language
                (``"Chinese"``, ``"English"``, ...) improves quality on single-language text.
            speech_rate: Speaking rate multiplier (0.5 to 2.0 per Model Studio). ``None``
                leaves the server default and omits the field.
            finish_timeout: Seconds to wait for ``session.finished`` after asking the server
                to finish; the server drains any audio it still owes in that window.
            region: ``"intl"`` (Singapore) or ``"cn"`` (Beijing). API keys are region-bound.
            base_url: Full realtime WebSocket URL, for example a workspace-dedicated domain.
                Overrides ``region``.
            api_key: Model Studio API key; falls back to ``DASHSCOPE_API_KEY``.
            http_session: Optional aiohttp session to reuse.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = resolve_api_key(api_key)
        self._base_url = resolve_realtime_url(base_url, region)
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            speech_rate=speech_rate,
            finish_timeout=finish_timeout,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Qwen"

    def update_options(
        self,
        *,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        language_type: NotGivenOr[TTSLanguageTypes | str] = NOT_GIVEN,
        speech_rate: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update options for streams opened after this call; the session config is per socket."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language_type):
            self._opts.language_type = language_type
        if is_given(speech_rate):
            self._opts.speech_rate = speech_rate

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        # The transport only streams, so one-shot synthesis rides the same path.
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    """One realtime TTS session: text in, PCM frames out."""

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._qwen = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        socket = RealtimeSocket(
            await connect(
                self._qwen._ensure_session(),
                base_url=self._qwen._base_url,
                model=self._opts.model,
                api_key=self._qwen._api_key,
                timeout=self._conn_options.timeout,
            ),
            finish_timeout=self._opts.finish_timeout,
        )

        async def send() -> None:
            try:
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        # server_commit mode segments on its own; a flush has nothing to add.
                        continue
                    self._mark_started()
                    await socket.send("input_text_buffer.append", text=item)
                # The server flushes any audio it still owes before finishing.
                await socket.finish()
            except ConnectionResetError:
                pass

        async def recv() -> None:
            segment_open = False
            while True:
                msg = await socket.receive()
                if msg.type in CLOSE_TYPES:
                    raise APIConnectionError(
                        "Model Studio closed the TTS socket before session.finished"
                    )
                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Model Studio TTS websocket error")
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                event: dict[str, Any] = json.loads(msg.data)
                kind = event.get("type")

                if kind == "error":
                    raise status_error_from(event)
                elif kind == "response.audio.delta":
                    if not segment_open:
                        # One segment per stream: LiveKit compares its own segment count
                        # with the emitter's and errors on a mismatch.
                        output_emitter.start_segment(segment_id=request_id)
                        segment_open = True
                    output_emitter.push(base64.b64decode(event["delta"]))
                elif kind == "session.finished":
                    if segment_open:
                        output_emitter.end_segment()
                    return

        cancelled = False
        try:
            await socket.send("session.update", session=self._opts.session_config())
            send_task = asyncio.create_task(send(), name="qwen-tts-send")
            recv_task = asyncio.create_task(recv(), name="qwen-tts-recv")
            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                await utils.aio.cancel_and_wait(send_task, recv_task)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if cancelled:
                # A barge-in lands here: LiveKit cancels the synthesis before send() ever
                # reached session.finish, and Model Studio books a socket dropped without
                # it as a failed request. Send the finish but do not wait for the reply:
                # the voice pipeline awaits this stream's aclose() before it clears the
                # playout buffer, so every millisecond spent here is the agent still
                # talking over the user.
                await socket.close_with_finish()
            else:
                await socket.close()
