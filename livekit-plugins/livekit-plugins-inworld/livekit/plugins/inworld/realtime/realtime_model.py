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
import os
import time
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import (
    AudioTranscription,
    ConversationItemAdded,
    RealtimeAudioConfig,
    RealtimeAudioConfigOutput,
    RealtimeAudioInputTurnDetection,
)
from openai.types.realtime.session_update_event import SessionUpdateEvent

from livekit.agents import APIConnectionError, llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

from .provider_data import ProviderData

# wss URL; the server assigns a session on connect via the `key` query param
DEFAULT_WS_URL = "wss://api.dev.inworld.ai/api/v1/realtime/session"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
DEFAULT_TTS_MODEL = "inworld-tts-2"
DEFAULT_STT_MODEL = "inworld/inworld-stt-1"
DEFAULT_VOICE = "Ashley"


def _build_ws_url(base_url: str) -> str:
    """Build the Inworld realtime URL, adding the required `key`/`protocol` query params."""
    url = base_url
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query.setdefault("key", [utils.shortuuid("session_")])
    query.setdefault("protocol", ["realtime"])
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query, doseq=True), "")
    )


class RealtimeModel(openai.realtime.RealtimeModel):
    """Inworld Realtime API (speech-to-speech over WebSocket).

    Inworld is wire-compatible with the OpenAI Realtime spec, so this reuses the OpenAI
    plugin's session machinery and only remaps auth, URL, and the LLM/TTS/STT config fields.
    """

    def __init__(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        tts_model: NotGivenOr[str] = NOT_GIVEN,
        stt_model: NotGivenOr[str] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscription | None] = NOT_GIVEN,
        turn_detection: NotGivenOr[
            RealtimeAudioInputTurnDetection | TurnDetection | None
        ] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        provider_data: NotGivenOr[ProviderData] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: LLM provider/model, e.g. ``openai/gpt-4o-mini`` or a router like ``inworld/auto``.
            voice: TTS voice for audio responses. Defaults to ``Ashley``.
            tts_model: TTS model (``inworld-tts-2`` or ``inworld-tts-1.5-mini``).
            stt_model: STT model used to transcribe user audio. Defaults to ``inworld/inworld-stt-1``.
            modalities: Output modalities. Defaults to ``["text", "audio"]``.
            input_audio_transcription: Explicit transcription config; overrides ``stt_model``.
            turn_detection: Server-side turn detection. Defaults to Inworld's ``semantic_vad``.
            tool_choice: Tool selection policy.
            provider_data: Inworld-specific ``providerData`` extensions (stt/tts/memory/
                backchannel/responsiveness/caching/text_generation_config plus user_id/metadata),
                sent verbatim in the session config. See :class:`ProviderData`.
            api_key: Inworld API key. Falls back to ``INWORLD_API_KEY``.
            base_url: Override the realtime WebSocket URL.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before recycling the connection.
            conn_options: Retry/backoff and connection settings.
        """
        key = api_key if is_given(api_key) else os.getenv("INWORLD_API_KEY")
        if not key:
            raise ValueError(
                "Inworld API key is required, either as argument or set"
                " INWORLD_API_KEY environment variable"
            )

        transcription = (
            input_audio_transcription
            if is_given(input_audio_transcription)
            else AudioTranscription(model=stt_model if is_given(stt_model) else DEFAULT_STT_MODEL)
        )

        super().__init__(
            model=model if is_given(model) else DEFAULT_LLM_MODEL,
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            modalities=modalities if is_given(modalities) else ["text", "audio"],
            input_audio_transcription=transcription,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            api_key=key,
            base_url=base_url if is_given(base_url) else DEFAULT_WS_URL,
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )

        self._tts_model = tts_model if is_given(tts_model) else DEFAULT_TTS_MODEL
        self._provider_data = provider_data if is_given(provider_data) else None
        self._provider_label = "Inworld Realtime API"

    @property
    def provider(self) -> str:
        return "Inworld"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(openai.realtime.RealtimeSession):
    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        # Inworld currently reports previous_item_id=None for model-generated function calls
        # even when the conversation is nonempty. Under OpenAI's ordering semantics, None
        # means the item has no predecessor, so LiveKit inserts it at the conversation head.
        # Use the current tail as the predecessor to keep the call after the triggering turn
        # and ensure its subsequent function_call_output is inserted directly after the call.
        if (
            event.item.type == "function_call"
            and event.previous_item_id is None
            and self._remote_chat_ctx._tail is not None
        ):
            event.previous_item_id = self._remote_chat_ctx._tail.item.id

        super()._handle_conversion_item_added(event)

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        # Inworld uses Basic auth (the API key is already base64) and a session URL with
        # `key`/`protocol` query params, unlike OpenAI's Bearer + `?model=` scheme.
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Basic {self._opts.api_key}",
        }
        url = _build_ws_url(self._opts.base_url)

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"{self._realtime_model._provider_label} client connection error"
            ) from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message=f"{self._realtime_model._provider_label} connection timed out",
            ) from e

    def _create_session_update_event(self) -> SessionUpdateEvent | dict[str, Any]:
        event = super()._create_session_update_event()
        model = self._realtime_model
        if not isinstance(event, SessionUpdateEvent) or not isinstance(model, RealtimeModel):
            return event

        session = event.session
        # both fields are Inworld extensions to the OpenAI shape; set them as extra
        # (extra="allow") pydantic fields so they serialize under their exact names
        if isinstance(session.audio, RealtimeAudioConfig) and isinstance(
            session.audio.output, RealtimeAudioConfigOutput
        ):
            output: Any = session.audio.output
            output.model = model._tts_model  # -> audio.output.model (TTS model)

        if model._provider_data:
            session_any: Any = session
            session_any.providerData = model._provider_data

        return event
