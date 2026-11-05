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

"""Streaming speech-to-text for Ringg.

The transport (WebSocket, handshake, event parsing) and the LiveKit adaptation
live in the ``ringglabs`` SDK (``ringglabs.integrations.livekit``), which Ringg
maintains alongside its server protocol. This module exposes that integration
under the standard ``livekit.plugins.<provider>`` names.
"""

from __future__ import annotations

from ringglabs.integrations.livekit import RinggSpeechStream, RinggSTT
from ringglabs.stt import TimeoutConfig

from livekit.agents import vad as lk_vad

from .log import logger

RinggSTTMode = str
"""``"stream"`` (server endpointing) or ``"on_final"`` (client VAD events)."""

SpeechStream = RinggSpeechStream
"""The ``stt.RecognizeStream`` returned by :meth:`STT.stream`."""


class STT(RinggSTT):
    """Ringg streaming speech-to-text.

    Audio is resampled to 16 kHz mono by LiveKit before it reaches the plugin.
    Interim and final transcripts are emitted as ``SpeechEvent``s.

    Endpointing:
        By default the Ringg server decides where each turn ends
        (``mode="stream"``). Pass the same ``VAD`` you give ``AgentSession`` as
        ``vad=`` and the plugin switches to ``mode="on_final"`` with client VAD
        events: it runs its own VAD stream on the pushed audio and tells the
        server when the user starts and stops speaking, so the server returns one
        final transcript per turn.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        language: str = "hi",
        vad: lk_vad.VAD | None = None,
        mode: RinggSTTMode | None = None,
        vad_tail_sil_ms: int = 200,
        vad_confidence: float = 0.55,
        enable_cap_punc: bool = True,
        timeout: TimeoutConfig | None = None,
    ) -> None:
        """Create a new instance of Ringg STT.

        Args:
            api_key: Ringg API key. Falls back to the ``RINGG_API_KEY`` environment
                variable.
            base_url: Ringg STT host (``host[:port][/route]``). ``None`` uses the
                public Ringg endpoint.
            language: Language code, for example ``"hi"``, ``"en"``, ``"ta"``.
            vad: A LiveKit VAD such as ``silero.VAD.load()``. When given, the plugin
                uses client-side endpointing (``mode="on_final"``).
            mode: ``"stream"`` or ``"on_final"``. Defaults to ``"on_final"`` when
                ``vad`` is given, otherwise ``"stream"``.
            vad_tail_sil_ms: Server VAD trailing silence in milliseconds.
            vad_confidence: Server VAD confidence threshold.
            enable_cap_punc: Server-side capitalisation and punctuation.
            timeout: ``ringglabs`` ``TimeoutConfig`` override. By default the
                connect timeout follows LiveKit's ``APIConnectOptions``.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            language=language,
            vad=vad,
            mode=mode,
            vad_tail_sil_ms=vad_tail_sil_ms,
            vad_confidence=vad_confidence,
            enable_cap_punc=enable_cap_punc,
            timeout=timeout,
        )
        logger.debug("ringg stt initialised", extra={"language": language, "mode": mode})
