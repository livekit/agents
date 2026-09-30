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

import openai
from openai.types.beta.realtime.transcription_session_update_param import (
    SessionTurnDetection,
)

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.openai import STT as OpenAISTT

from ._utils import SPRAG_BASE_URL, resolve_api_key, with_attribution
from .models import STTModels


class STT(OpenAISTT):
    def __init__(
        self,
        *,
        model: STTModels | str = "rhythm",
        use_realtime: bool = True,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = SPRAG_BASE_URL,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """
        Create a new instance of Sprag STT.

        Args:
            model: The Sprag model to use for transcription.
            use_realtime: Stream audio over a realtime WebSocket. When False, each turn is
                transcribed with a separate REST request.
            turn_detection: Server-side endpointing for the realtime transport, such as
                ``{"silence_duration_ms": 200}``. Omitted fields keep their defaults.
            api_key: Your Sprag API key. If not provided, will use the SPRAG_API_KEY
                environment variable.
            base_url: The Sprag API base URL.
            client: Optional pre-configured OpenAI AsyncClient instance.
        """
        super().__init__(
            model=model,
            # Sprag models take no language parameter
            language=[],
            use_realtime=use_realtime,
            turn_detection=turn_detection,
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            client=client,
        )
        self._client = with_attribution(self._client)

    @property
    def provider(self) -> str:
        return "Sprag"
