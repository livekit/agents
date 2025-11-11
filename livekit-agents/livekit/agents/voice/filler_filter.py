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

import logging
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import stt
from .agent_activity import AgentActivity
from .agent_session import UserInputTranscribedEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FillerFilterConfig:
    """Configuration for filler word filtering."""

    filler_words: list[str] = field(default_factory=lambda: ["uh", "umm", "hmm", "haan", "huh"])
    confidence_threshold: float = 0.5
    enable_filtering: bool = True
    log_filtered: bool = True
    log_interruptions: bool = True

    def __post_init__(self):
        self.filler_words = [word.lower().strip() for word in self.filler_words]


class FillerFilteredAgentActivity(AgentActivity):
    """AgentActivity that filters filler words during agent speech."""

    def __init__(self, *args, filler_config: FillerFilterConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._filler_config = filler_config or FillerFilterConfig()
        self._filler_words_set = set(self._filler_config.filler_words)

        logger.info(
            f"FillerFilteredAgentActivity initialized with filler words: "
            f"{sorted(self._filler_words_set)}"
        )

    def _is_filler_only(self, text: str) -> bool:
        if not text:
            return False

        normalized = text.lower().strip()
        words = normalized.split()

        if not words:
            return False

        cleaned_words = [word.strip(string.punctuation) for word in words]
        cleaned_words = [w for w in cleaned_words if w]

        if not cleaned_words:
            return False

        return all(word in self._filler_words_set for word in cleaned_words)

    def _is_agent_speaking(self) -> bool:
        return self._current_speech is not None and not self._current_speech.interrupted

    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        if not self._filler_config.enable_filtering:
            super().on_interim_transcript(ev, speaking=speaking)
            return

        transcript = ev.alternatives[0].text if ev.alternatives else ""
        confidence = ev.alternatives[0].confidence if ev.alternatives else 1.0

        if confidence < self._filler_config.confidence_threshold:
            if self._filler_config.log_filtered:
                logger.debug(
                    f"Ignoring low confidence transcript: '{transcript}' "
                    f"(confidence: {confidence:.2f})"
                )
            return

        is_agent_speaking = self._is_agent_speaking()

        if is_agent_speaking and transcript:
            is_filler = self._is_filler_only(transcript)

            if is_filler:
                if self._filler_config.log_filtered:
                    logger.info(f"Ignoring filler words during agent speech: '{transcript}'")

                self._session._user_input_transcribed(
                    UserInputTranscribedEvent(
                        language=ev.alternatives[0].language,
                        transcript=transcript,
                        is_final=False,
                        speaker_id=ev.alternatives[0].speaker_id,
                    )
                )
                return
            else:
                if self._filler_config.log_interruptions:
                    logger.info(f"Processing real interruption during agent speech: '{transcript}'")
        else:
            if transcript and self._filler_config.log_interruptions:
                logger.debug(f"Registering user speech (agent quiet): '{transcript}'")

        super().on_interim_transcript(ev, speaking=speaking)

    def on_final_transcript(self, ev: stt.SpeechEvent) -> None:
        transcript = ev.alternatives[0].text if ev.alternatives else ""

        if transcript and self._filler_config.log_interruptions:
            logger.debug(f"Final transcript: '{transcript}'")

        super().on_final_transcript(ev)
