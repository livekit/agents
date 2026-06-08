# Copyright 2023 LiveKit, Inc.
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

from collections.abc import Callable
from typing import TYPE_CHECKING

from livekit.agents.llm import ChatMessage

from .log import logger
from .stt import _U3_PRO_MODELS, STT

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession
    from livekit.agents.voice.events import ConversationItemAddedEvent

# AssemblyAI caps agent_context at 1500 characters (server-side MAX_PROMPT_CHARS).
# Truncate so a long agent turn can't error the stream's UpdateConfiguration.
_MAX_AGENT_CONTEXT_CHARS = 1500


def enable_agent_context(session: AgentSession) -> Callable[[], None]:
    """Feed the agent's most recent spoken turn into AssemblyAI's ``agent_context``.

    After each agent turn completes, the text the agent said is pushed to the
    AssemblyAI streaming STT via ``STT.update_options(agent_context=...)``,
    giving the model conversational context for transcribing the user's reply.
    The update is sent once per turn (on ``conversation_item_added``) over the
    already-open websocket, and each turn overwrites the previous value.

    Every agent utterance triggers an update — including ``session.say()`` and
    filler speech, not just LLM-generated replies — since each is "what the agent
    said."

    This is opt-in. Call it once after creating the ``AgentSession``::

        session = AgentSession(stt=assemblyai.STT(model="u3-rt-pro"), ...)
        assemblyai.enable_agent_context(session)

    To turn it off again, call the returned function. It detaches the handler so
    no further ``agent_context`` updates are sent; the last value already set on
    the STT stays in effect until you change it::

        disable = assemblyai.enable_agent_context(session)
        ...
        disable()  # stop feeding agent_context (e.g. on agent handoff)

    Cleanup is optional — the handler is released with the session — so for the
    common case you can ignore the return value.

    It is a no-op (logs a warning once) when the session's STT is not an
    AssemblyAI ``STT`` or is not a ``u3-rt-pro`` model — ``agent_context`` is only
    supported there. Turns longer than AssemblyAI's 1500-character ``agent_context``
    limit are truncated. Note that ``session.stt`` resolves to the session-level
    STT; a per-``Agent`` STT override is not observed.

    Args:
        session: The ``AgentSession`` to attach to.

    Returns:
        A callable that unsubscribes the handler (see the example above), e.g.
        for tests or when swapping agents.
    """
    warned = False

    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        nonlocal warned

        item = ev.item
        if not isinstance(item, ChatMessage) or item.role != "assistant":
            return

        text = item.text_content
        if not text:
            return

        agent_stt = session.stt
        if not isinstance(agent_stt, STT):
            if not warned:
                warned = True
                logger.warning(
                    "enable_agent_context requires an AssemblyAI STT on the session; "
                    "got %s. agent_context will not be updated.",
                    type(agent_stt).__name__ if agent_stt is not None else None,
                )
            return

        if agent_stt.model not in _U3_PRO_MODELS:
            if not warned:
                warned = True
                logger.warning(
                    "enable_agent_context requires a 'u3-rt-pro' model; got %r. "
                    "agent_context will not be updated.",
                    agent_stt.model,
                )
            return

        agent_stt.update_options(agent_context=text[:_MAX_AGENT_CONTEXT_CHARS])

    session.on("conversation_item_added", _on_conversation_item_added)

    def _unsubscribe() -> None:
        session.off("conversation_item_added", _on_conversation_item_added)

    return _unsubscribe
