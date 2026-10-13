from __future__ import annotations

import asyncio
from typing import Literal, Protocol, overload, runtime_checkable

from livekit import rtc

from ...llm import DuplexModel, RealtimeModel
from ...llm.tool_context import ToolContext
from ...types import NOT_GIVEN, NotGivenOr
from ...voice import room_io
from ...voice.agent import Agent
from ...voice.agent_session import (
    AgentSession,
    RecordingOptions,
    SessionConnectOptions,
    Userdata_T,
)
from ...voice.run_result import RunResult


@runtime_checkable
class TranslationModel(Protocol):
    """A model that interprets speech into another language.

    Implemented alongside a transport base class rather than in place of one: ``DuplexModel``
    for a provider whose output is one continuous stream, ``RealtimeModel`` for one that
    segments its own turns. Translation is a property of what a model says, not of how it
    speaks, so it does not decide which of the two a plugin subclasses.
    """

    @property
    def target_language(self) -> str:
        """The language the model speaks its translation in."""
        ...


@runtime_checkable
class RetargetableModel(TranslationModel, Protocol):
    """A translation model that can be pointed at another language mid-session."""

    def update_options(self, *, target_language: str) -> None: ...


class TranslationSession(AgentSession[Userdata_T]):
    """An ``AgentSession`` that runs a speech-to-speech translation model.

    A translator listens and speaks but holds no conversation: it takes no instructions, calls
    no tools, keeps no chat context and never decides to reply. This session owns that
    difference so the model does not have to answer for an agent's surface it cannot serve —
    there is no ``Agent`` to supply, and one that carries instructions or tools is refused at
    ``start`` rather than part way through a call.

    Everything else is an ordinary session: room I/O, transcripts, metrics, user and agent
    state, and recording all behave as they do anywhere else.

    A plugin tailors this to its provider by subclassing: :meth:`default_agent` to run its own
    agent, :meth:`set_target_language` for a model that switches language its own way.

    Example:
        ```python
        session = TranslationSession(model=VoiceModel(target_language="es"))
        await session.start(room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        model: DuplexModel | RealtimeModel,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        user_away_timeout: float | None = None,
        conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """
        Args:
            model: The translation model. It occupies the session's engine slot, the same one
                an LLM or realtime model would.
            userdata: Arbitrary state carried on the session.
            user_away_timeout: Seconds of mutual silence before the speaker is marked away.
                Off by default: a translator cannot check whether they are still there.
            conn_options: Retry and backoff settings.
            loop: Event loop to run on.
        """
        super().__init__(
            llm=model,
            # the model reports the speaker's turns itself; a VAD would only duplicate them
            vad=None,
            userdata=userdata,
            user_away_timeout=user_away_timeout,
            conn_options=conn_options,
            loop=loop,
        )
        self._model = model

    @property
    def model(self) -> DuplexModel | RealtimeModel:
        """The translation model this session runs."""
        return self._model

    @property
    def target_language(self) -> str | None:
        """The language the model translates into, or ``None`` when it reports none."""
        if isinstance(self._model, TranslationModel):
            return self._model.target_language
        return None

    def set_target_language(self, language: str) -> None:
        """Point the model at another language.

        Forwards to the model's ``update_options`` when it can be retargeted mid-session.
        Override for a provider that switches another way, such as by reconnecting.

        Raises:
            RuntimeError: If the model's target language is fixed for the session's life.
        """
        if not isinstance(self._model, RetargetableModel):
            raise RuntimeError(
                f"{type(self._model).__name__} translates into a language fixed for the "
                "session's life; start a new session to translate into another"
            )
        self._model.update_options(target_language=language)

    def default_agent(self) -> Agent:
        """The agent this session runs when ``start`` is given none.

        Override to run an agent of your own; it must still carry no instructions and no tools.
        """
        return Agent(instructions="")

    @overload
    async def start(
        self,
        agent: Agent | None = None,
        *,
        capture_run: Literal[True],
        room: NotGivenOr[rtc.Room] = NOT_GIVEN,
        room_options: NotGivenOr[room_io.RoomOptions] = NOT_GIVEN,
        session_host: NotGivenOr[bool] = NOT_GIVEN,
        record: bool | RecordingOptions = True,
        # deprecated
        room_input_options: NotGivenOr[room_io.RoomInputOptions] = NOT_GIVEN,
        room_output_options: NotGivenOr[room_io.RoomOutputOptions] = NOT_GIVEN,
    ) -> RunResult: ...

    @overload
    async def start(
        self,
        agent: Agent | None = None,
        *,
        capture_run: Literal[False] = False,
        room: NotGivenOr[rtc.Room] = NOT_GIVEN,
        room_options: NotGivenOr[room_io.RoomOptions] = NOT_GIVEN,
        session_host: NotGivenOr[bool] = NOT_GIVEN,
        record: bool | RecordingOptions = True,
        # deprecated
        room_input_options: NotGivenOr[room_io.RoomInputOptions] = NOT_GIVEN,
        room_output_options: NotGivenOr[room_io.RoomOutputOptions] = NOT_GIVEN,
    ) -> None: ...

    async def start(
        self,
        agent: Agent | None = None,
        *,
        capture_run: bool = False,
        room: NotGivenOr[rtc.Room] = NOT_GIVEN,
        room_options: NotGivenOr[room_io.RoomOptions] = NOT_GIVEN,
        session_host: NotGivenOr[bool] = NOT_GIVEN,
        record: NotGivenOr[bool | RecordingOptions] = NOT_GIVEN,
        # deprecated
        room_input_options: NotGivenOr[room_io.RoomInputOptions] = NOT_GIVEN,
        room_output_options: NotGivenOr[room_io.RoomOutputOptions] = NOT_GIVEN,
    ) -> RunResult | None:
        """Start the translation session.

        Args:
            agent: Optional. Defaults to :meth:`default_agent`. An agent carrying instructions
                or tools is refused: the model can serve neither.

        See :meth:`AgentSession.start` for the remaining arguments.
        """
        agent = agent if agent is not None else self.default_agent()
        _reject_conversational_agent(agent, type(self._model).__name__)
        return await super().start(  # type: ignore[call-overload, no-any-return, misc]
            agent,
            capture_run=capture_run,
            room=room,
            room_options=room_options,
            session_host=session_host,
            record=record,
            room_input_options=room_input_options,
            room_output_options=room_output_options,
        )


def _reject_conversational_agent(agent: Agent, model_name: str) -> None:
    instructions = agent.instructions
    if not isinstance(instructions, str) or instructions.strip():
        raise ValueError(
            f"{model_name} translates what it hears and takes no instructions; "
            f"{type(agent).__name__} carries some. Steer the translation through the model's "
            "own options instead, such as its formality or glossary."
        )

    if tools := agent.tools:
        raise ValueError(
            f"{model_name} cannot call tools; {type(agent).__name__} carries "
            f"{sorted(tool.id for tool in ToolContext(tools).flatten())}. "
            "Run tools in a separate session."
        )
