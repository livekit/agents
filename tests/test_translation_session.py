"""
Tests for the beta TranslationSession: a translator listens and speaks but holds no
conversation, so the session supplies the inert agent itself and refuses one that carries
instructions or tools before anything connects.
"""

from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents import llm
from livekit.agents.beta.translation import (
    RetargetableModel,
    TranslationModel,
    TranslationSession,
)
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.voice import Agent, AgentSession

pytestmark = pytest.mark.unit


class _FixedTranslator(llm.DuplexModel):
    """A model whose target language is set for the session's life, like DeepL Voice."""

    def __init__(self, target_language: str = "es") -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True, auto_tool_reply_generation=False
            )
        )
        self._target_language = target_language

    @property
    def target_language(self) -> str:
        return self._target_language

    def session(self) -> llm.DuplexSession:
        raise NotImplementedError

    async def aclose(self) -> None: ...


class _RetargetableTranslator(_FixedTranslator):
    """A model that can be pointed at another language mid-session."""

    def update_options(self, *, target_language: str) -> None:
        self._target_language = target_language


class _PlainDuplexModel(llm.DuplexModel):
    """A duplex model that does not translate and reports no target language."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True, auto_tool_reply_generation=False
            )
        )

    def session(self) -> llm.DuplexSession:
        raise NotImplementedError

    async def aclose(self) -> None: ...


@function_tool()
async def _lookup(query: str) -> str:
    """Look something up.

    Args:
        query: what to look up
    """
    return query


def test_the_protocols_match_a_translating_model_only() -> None:
    assert isinstance(_FixedTranslator(), TranslationModel)
    assert not isinstance(_FixedTranslator(), RetargetableModel)
    assert isinstance(_RetargetableTranslator(), RetargetableModel)
    assert not isinstance(_PlainDuplexModel(), TranslationModel)


def test_the_target_language_comes_from_the_model() -> None:
    model = _FixedTranslator("de")
    session = TranslationSession(model=model)
    assert session.model is model
    assert session.target_language == "de"


def test_a_model_that_reports_no_target_language_leaves_it_unset() -> None:
    assert TranslationSession(model=_PlainDuplexModel()).target_language is None


def test_retargeting_forwards_to_the_model() -> None:
    session = TranslationSession(model=_RetargetableTranslator("es"))
    session.set_target_language("fr")
    assert session.target_language == "fr"


def test_retargeting_a_fixed_model_is_refused() -> None:
    session = TranslationSession(model=_FixedTranslator("es"))
    with pytest.raises(RuntimeError, match="fixed for the session's life"):
        session.set_target_language("fr")
    assert session.target_language == "es"


def test_the_default_agent_carries_nothing_to_serve() -> None:
    agent = TranslationSession(model=_FixedTranslator()).default_agent()
    assert agent.instructions == ""
    assert agent.tools == []


def test_a_translator_runs_without_a_vad() -> None:
    # the model reports the speaker's turns itself
    assert TranslationSession(model=_FixedTranslator()).vad is None


async def test_start_refuses_an_agent_carrying_instructions() -> None:
    session = TranslationSession(model=_FixedTranslator())
    with pytest.raises(ValueError, match="takes no instructions"):
        await session.start(Agent(instructions="be helpful"), room=rtc.Room())


async def test_start_refuses_an_agent_carrying_tools() -> None:
    session = TranslationSession(model=_FixedTranslator())
    with pytest.raises(ValueError, match="cannot call tools"):
        await session.start(Agent(instructions="", tools=[_lookup]), room=rtc.Room())


async def test_blank_instructions_pass_the_instructions_check() -> None:
    session = TranslationSession(model=_FixedTranslator())
    # refused for the tools it cannot reach, not for its blank instructions
    with pytest.raises(ValueError, match="cannot call tools"):
        await session.start(Agent(instructions="  \n ", tools=[_lookup]), room=rtc.Room())


async def test_start_supplies_the_default_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Agent] = []

    async def _capture(self: AgentSession, agent: Agent, **kwargs: object) -> None:
        started.append(agent)

    monkeypatch.setattr(AgentSession, "start", _capture)
    session = TranslationSession(model=_FixedTranslator())
    await session.start(room=rtc.Room())
    assert len(started) == 1
    assert started[0].instructions == ""
    assert started[0].tools == []


async def test_start_keeps_an_agent_the_caller_supplies(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Agent] = []

    async def _capture(self: AgentSession, agent: Agent, **kwargs: object) -> None:
        started.append(agent)

    monkeypatch.setattr(AgentSession, "start", _capture)
    agent = Agent(instructions="")
    session = TranslationSession(model=_FixedTranslator())
    await session.start(agent, room=rtc.Room())
    assert started == [agent]
