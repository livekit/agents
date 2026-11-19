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

import asyncio
import time
from typing import Any

from livekit.agents import llm as lk_llm, utils

from .log import logger

DEFAULT_INSTRUCTIONS = (
    "You are a listener giving a backchannel while the user is still speaking. "
    "Reply with ONE or TWO words only, in the user's language and dialect, that show you are "
    "following what they just said (e.g. ممم، أيوه، تمام، صح، طيب). "
    "Never answer, ask a question, or add punctuation or explanation."
)


class BackchannelController:
    """Say a short, context-aware acknowledgment on Nabrah backchannel events.

    On each event the LLM writes one or two words for the user's turn so far; they are
    spoken with `session.say` (interruptible, not added to the chat context). It never
    commits a user turn and never speaks when:

    - interruptions are disabled for the session or current agent (an uninterruptible
      say would replace the user's audio with silence for STT),
    - the agent is not listening, or a backchannel was said less than `cooldown` s ago,
    - the LLM doesn't answer within `timeout` s (a late backchannel is worse than none).

    The user's next words interrupt it, and the user's end of turn cuts it off.

    Args:
        stt: the `nabrah.STTConv` instance used by the session.
        session: the `AgentSession`.
        llm: model for the words; defaults to `session.llm`. A small, fast model is best.
        instructions: system prompt for the words.
        cooldown: minimum seconds between backchannels.
        timeout: seconds to wait for the LLM before skipping the backchannel.
        max_words: words kept from the LLM reply.
    """

    def __init__(
        self,
        stt: Any,
        session: Any,
        *,
        llm: lk_llm.LLM | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        cooldown: float = 3.0,
        timeout: float = 1.0,
        max_words: int = 2,
    ) -> None:
        self.stt, self.session = stt, session
        self.instructions, self.cooldown, self.timeout = instructions, cooldown, timeout
        self.max_words = max_words
        self._llm = llm
        self._last_at = float("-inf")
        self._handle: Any = None
        self._task: asyncio.Task[None] | None = None
        stt.on("backchannel", self._on_backchannel)
        stt.on("end_of_turn", self._stop)

    def _interruptions_allowed(self) -> bool:
        try:
            agent_allows = self.session.current_agent.allow_interruptions
        except RuntimeError:  # session not started
            return False
        if utils.is_given(agent_allows):
            return bool(agent_allows)
        return bool(self.session.options.interruption.get("enabled", True))

    def _busy(self) -> bool:
        return (self._task is not None and not self._task.done()) or (
            self._handle is not None and not self._handle.done()
        )

    def _stop(self, _event: Any = None) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._handle is not None and not self._handle.done():
            self._handle.interrupt()
        self._task = self._handle = None

    def _on_backchannel(self, event: Any) -> None:
        text = (event or {}).get("text", "").strip()
        if (
            not text
            or self._busy()
            or time.monotonic() - self._last_at < self.cooldown
            or self.session.agent_state != "listening"
            or not self._interruptions_allowed()
        ):
            return
        self._last_at = time.monotonic()
        self._task = asyncio.create_task(self._backchannel(text))

    async def _backchannel(self, user_text: str) -> None:
        model = self._llm or self.session.llm
        if not isinstance(model, lk_llm.LLM):
            logger.warning("BackchannelController needs an LLM (pass llm=...); skipped")
            return
        try:
            words = await asyncio.wait_for(self._generate(model, user_text), self.timeout)
        except asyncio.TimeoutError:
            logger.debug("backchannel LLM too slow; skipped")
            return
        except Exception:
            logger.warning("backchannel LLM failed; skipped", exc_info=True)
            return
        # The user may have finished, or the agent started replying, while we waited.
        if not words or self.session.agent_state != "listening":
            return
        self._handle = self.session.say(words, add_to_chat_ctx=False, allow_interruptions=True)

    async def _generate(self, model: lk_llm.LLM, user_text: str) -> str:
        chat_ctx = lk_llm.ChatContext.empty()
        chat_ctx.add_message(role="system", content=self.instructions)
        chat_ctx.add_message(role="user", content=user_text)
        reply = ""
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    reply += chunk.delta.content
                    if len(reply.split()) > self.max_words:
                        break  # enough words: don't wait for the rest
        return " ".join(reply.split()[: self.max_words]).strip(" .,!?؟،\"'")

    def close(self) -> None:
        self.stt.off("backchannel", self._on_backchannel)
        self.stt.off("end_of_turn", self._stop)
        self._stop()
