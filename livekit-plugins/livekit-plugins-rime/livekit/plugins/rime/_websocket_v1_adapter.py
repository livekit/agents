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

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass

from livekit.agents import APIConnectOptions, APIError, http, tts, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _websocket_v1
from ._pool_manager import PoolManager
from ._stream_tts import StreamTTS

_Pool = utils.ConnectionPool[_websocket_v1.Connection]


@dataclass(frozen=True)
class V1SynthesisOptions:
    """Model settings before conversion to the WebSocket v1 wire format."""

    model: str
    speaker: str
    language: NotGivenOr[str] = NOT_GIVEN
    audio_format: _websocket_v1.RimeAudioFormat = _websocket_v1.DEFAULT_AUDIO_FORMAT
    sampling_rate: NotGivenOr[int] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN


class WebSocketV1Adapter:
    """Adapt LiveKit streaming to the Rime WebSocket v1 protocol."""

    def __init__(
        self,
        *,
        websocket_v1_url: str,
        websocket_protocol: _websocket_v1.WebSocketProtocol,
        api_key: str,
        ensure_session: Callable[[], http.HttpSession],
        allow_custom_endpoint: bool = False,
    ) -> None:
        _websocket_v1.validate_websocket_url(
            websocket_v1_url, allow_custom_endpoint=allow_custom_endpoint
        )
        _websocket_v1.codec_for_protocol(websocket_protocol)
        self._websocket_v1_url = websocket_v1_url
        self._websocket_protocol: _websocket_v1.WebSocketProtocol = websocket_protocol
        self._api_key = api_key
        self._ensure_session = ensure_session
        self._allow_custom_endpoint = allow_custom_endpoint
        self._pools = PoolManager(self._new_pool())

    def _new_pool(self) -> _Pool:
        websocket_v1_url = self._websocket_v1_url

        async def _connect(conn_options: APIConnectOptions) -> _websocket_v1.Connection:
            return await _websocket_v1.connect(
                self._ensure_session(),
                websocket_url=websocket_v1_url,
                api_key=self._api_key,
                protocol=self._websocket_protocol,
                conn_options=conn_options,
                allow_custom_endpoint=self._allow_custom_endpoint,
            )

        return utils.ConnectionPool[_websocket_v1.Connection](
            connect_cb=_connect,
            close_cb=_websocket_v1.close,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: V1SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _WebSocketV1SynthesizeStream:
        if not is_given(options.language):
            raise APIError("Rime v1 requires a language", retryable=False)
        protocol_options = _websocket_v1.SynthesisOptions(
            model=options.model,
            speaker=options.speaker,
            language=options.language,
            audio_format=options.audio_format,
            sampling_rate=options.sampling_rate if is_given(options.sampling_rate) else None,
            time_scale_factor=options.time_scale_factor
            if is_given(options.time_scale_factor)
            else None,
            pause_between_brackets=options.pause_between_brackets
            if is_given(options.pause_between_brackets)
            else None,
            phonemize_between_brackets=options.phonemize_between_brackets
            if is_given(options.phonemize_between_brackets)
            else None,
        )
        pool = self._pools.current
        stream = _WebSocketV1SynthesizeStream(
            tts_instance=tts_instance,
            pool=pool,
            pool_manager=self._pools,
            options=protocol_options,
            conn_options=conn_options,
        )
        return stream

    def prewarm(self) -> None:
        self._pools.current.prewarm()

    def update_endpoint(self, websocket_v1_url: str, *, model_changed: bool) -> None:
        """Update the connection URL after validating its model binding."""
        _websocket_v1.validate_websocket_url(
            websocket_v1_url, allow_custom_endpoint=self._allow_custom_endpoint
        )
        model_endpoint_changed = _websocket_v1.model_endpoint_identity(
            websocket_v1_url
        ) != _websocket_v1.model_endpoint_identity(self._websocket_v1_url)
        if model_changed and not model_endpoint_changed:
            raise ValueError("model cannot change without changing websocket_url")
        if websocket_v1_url == self._websocket_v1_url:
            return

        self._websocket_v1_url = websocket_v1_url
        self._pools.replace(self._new_pool())

    async def aclose(self) -> None:
        await self._pools.aclose()


class _WebSocketV1SynthesizeStream(tts.SynthesizeStream):
    """Run one LiveKit stream as one Rime WebSocket v1 synthesis context."""

    def __init__(
        self,
        *,
        tts_instance: tts.TTS,
        pool: _Pool,
        pool_manager: PoolManager[_websocket_v1.Connection],
        options: _websocket_v1.SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        self._sample_rate = tts_instance.sample_rate
        super().__init__(
            tts=StreamTTS(tts_instance, model=options.model), conn_options=conn_options
        )
        self._pool = pool
        self._options = options
        pool_manager.track_stream(pool, self._task)

    def flush(self) -> None:
        """Keep the Rime context open: the v1 protocol has no ``flush`` operation.

        The stream is one Rime context and one LiveKit segment, so no segment boundary
        is recorded and the stream accepts more text after this call.
        """

    def end_input(self) -> None:
        """Finalize Rime input and let the stream wait for the provider ``done`` event."""
        self.input.close()

    async def aclose(self) -> None:
        """Cancel the active Rime context when synthesis has not finished."""
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        async def input_events() -> AsyncIterable[str]:
            async for chunk in self.input:
                if isinstance(chunk, tts.SpeechChunk):
                    yield chunk.text + " "

        connection = await self._pool.get(conn_options=self._conn_options)
        reusable = False
        try:
            result = await _websocket_v1.run_context(
                connection,
                context_id=utils.shortuuid(),
                options=self._options,
                sample_rate=self._sample_rate,
                input_events=input_events(),
                output_emitter=output_emitter,
                read_timeout=self._conn_options.read_timeout,
                write_timeout=self._conn_options.write_timeout,
                mark_started=self.mark_started,
            )
            reusable = result.reusable
        except _websocket_v1.ContextCancelled as error:
            reusable = error.reusable
            raise
        finally:
            if reusable:
                self._pool.put(connection)
            else:
                self._pool.remove(connection)
                await _websocket_v1.close(connection)
