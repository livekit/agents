"""Decision-model checks that can block or steer the agent's replies."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

from .. import inference, llm
from ..llm.decisions import Decision, DecisionModel, DecisionResponse, _format_transcript
from ..log import logger
from ..tokenize.blingfire import SentenceTokenizer
from ..types import NOT_GIVEN, FlushSentinel, NotGivenOr
from ..utils import aio
from ..utils.misc import is_given

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .events import ConversationItemAddedEvent, UserInputTranscribedEvent

GuardrailHook = Literal["user_turn", "agent_sentence", "agent_turn"]
GuardrailInput = Literal["turn", "history"]
GuardrailOnError = Literal["allow", "block"]


@dataclass(frozen=True)
class Block:
    """Do not let the reply, or the rest of it, be spoken.

    Args:
        say: Optional replacement spoken in place of the blocked reply.
    """

    say: str | None = None


@dataclass(frozen=True)
class Steer:
    """Let the reply go ahead with extra instructions for this turn only."""

    instructions: str


class DecisionEvaluated(BaseModel):
    type: Literal["decision_evaluated"] = "decision_evaluated"
    hook: GuardrailHook
    response: DecisionResponse
    input_text: str
    """The text the decisions were evaluated on: the user turn, the reply so far, or the
    complete reply. With ``input="history"``, the conversation was sent before it."""
    speech_id: str | None = None
    """The reply this check applies to, for ``agent_sentence`` and ``agent_turn``."""

    def value(self, name: str, default: Any = None) -> Any:
        """The result for ``name``, or ``default`` if that decision has no result."""
        result = self.response.results.get(name)
        return default if result is None else result.value


_Verdict = Block | Steer | None
UserTurnHandler = Callable[[DecisionEvaluated], _Verdict | Awaitable[_Verdict]]
AgentSentenceHandler = Callable[[DecisionEvaluated], Block | None | Awaitable[Block | None]]
AgentTurnHandler = Callable[[DecisionEvaluated], None | Awaitable[None]]

_UserTurnHandlerT = TypeVar("_UserTurnHandlerT", bound=UserTurnHandler)
_AgentSentenceHandlerT = TypeVar("_AgentSentenceHandlerT", bound=AgentSentenceHandler)
_AgentTurnHandlerT = TypeVar("_AgentTurnHandlerT", bound=AgentTurnHandler)


@dataclass
class _Check:
    hook: GuardrailHook
    handler: Callable[[DecisionEvaluated], Any]
    decisions: Mapping[str, Decision]
    input: GuardrailInput
    timeout: float
    on_error: GuardrailOnError
    eager: bool = False

    @property
    def name(self) -> str:
        return getattr(self.handler, "__qualname__", repr(self.handler))


@dataclass
class _EagerEvaluation:
    input: str  # the text the model received, so reuse can check it is still the same
    task: asyncio.Task[DecisionResponse]


class DecisionGuardrail:
    """Runs decision-model checks on a session and lets them block or steer replies.

    A session has at most one guardrail. Register all its checks on it with the
    decorators; each check can evaluate any mix of decisions:

    - :meth:`on_user_turn` runs before the agent replies to a user turn. Its handlers can
      return :class:`Block` (no reply for this turn) or :class:`Steer` (extra instructions
      for this reply).
    - :meth:`on_agent_sentence` runs on each sentence of the agent's reply before it is
      spoken. Its handlers can return :class:`Block` to stop the reply at that sentence.
    - :meth:`on_agent_turn` runs on each complete agent message. It can't change anything.

    Example::

        guardrail = DecisionGuardrail(session, model="openai/gpt-6-luna-decisions")

        @guardrail.on_user_turn(decisions={
            "abusive": Probability("The user is abusive or threatening."),
        })
        def before_reply(ev: DecisionEvaluated) -> Block | None:
            if ev.value("abusive", 0.0) > 0.8:
                return Block(say="I'm going to end the call here.")
            return None

    Args:
        session: The session to check.
        model: The decision model, or an inference model string.
        timeout: Default time a check may hold up the reply, in seconds.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        model: str | DecisionModel,
        timeout: float = 1.0,
    ) -> None:
        if session._guardrail is not None:
            # a second guardrail would run unnoticed next to the first: add its checks there
            raise RuntimeError(
                "this session already has a DecisionGuardrail, register the checks on it"
            )
        self._session = session
        self._owns_model = isinstance(model, str)
        self._model: DecisionModel = (
            inference.DecisionModel(model) if isinstance(model, str) else model
        )
        self._timeout = timeout
        self._checks: list[_Check] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

        # the user turn in progress, accumulated from final transcripts (eager checks)
        self._turn_text = ""
        self._eager: dict[int, _EagerEvaluation] = {}

        session.on("user_input_transcribed", self._on_user_input_transcribed)
        session.on("conversation_item_added", self._on_conversation_item_added)
        session._guardrail = self

    @property
    def model(self) -> DecisionModel:
        return self._model

    def on_user_turn(
        self,
        *,
        decisions: Mapping[str, Decision],
        input: GuardrailInput = "turn",
        timeout: NotGivenOr[float] = NOT_GIVEN,
        on_error: GuardrailOnError = "allow",
        eager: bool = True,
    ) -> Callable[[_UserTurnHandlerT], _UserTurnHandlerT]:
        """Check each user turn before the agent replies.

        Args:
            decisions: Evaluated together in one request.
            input: ``"turn"`` sends the user's turn; ``"history"`` also sends the
                conversation before it.
            timeout: Longest the check may delay the reply. Defaults to the guardrail's.
            on_error: ``"block"`` treats a failed or timed-out check as :class:`Block`;
                ``"allow"`` lets the reply go ahead.
            eager: Re-evaluate on every final transcript so the result is usually ready
                by the end of the turn.
        """

        def decorator(handler: _UserTurnHandlerT) -> _UserTurnHandlerT:
            self._add_check("user_turn", handler, decisions, input, timeout, on_error, eager=eager)
            return handler

        return decorator

    def on_agent_sentence(
        self,
        *,
        decisions: Mapping[str, Decision],
        input: GuardrailInput = "turn",
        timeout: NotGivenOr[float] = NOT_GIVEN,
        on_error: GuardrailOnError = "allow",
    ) -> Callable[[_AgentSentenceHandlerT], _AgentSentenceHandlerT]:
        """Check each sentence of the agent's reply before it is spoken.

        The input is the reply so far, ending with the sentence being checked. Write the
        decisions about its last sentence.

        Args:
            decisions: Evaluated together in one request per sentence.
            input: ``"turn"`` sends the reply so far; ``"history"`` also sends the
                conversation before it.
            timeout: Longest one sentence may wait. Defaults to the guardrail's.
            on_error: ``"block"`` treats a failed or timed-out check as :class:`Block`;
                ``"allow"`` lets the sentence through.
        """

        def decorator(handler: _AgentSentenceHandlerT) -> _AgentSentenceHandlerT:
            self._add_check("agent_sentence", handler, decisions, input, timeout, on_error)
            return handler

        return decorator

    def on_agent_turn(
        self,
        *,
        decisions: Mapping[str, Decision],
        input: GuardrailInput = "turn",
        timeout: NotGivenOr[float] = NOT_GIVEN,
    ) -> Callable[[_AgentTurnHandlerT], _AgentTurnHandlerT]:
        """Check each complete agent message, after it was spoken.

        Args:
            decisions: Evaluated together in one request per message.
            input: ``"turn"`` sends the message; ``"history"`` sends the conversation
                ending with it.
            timeout: Longest the evaluation may take. Defaults to the guardrail's.
        """

        def decorator(handler: _AgentTurnHandlerT) -> _AgentTurnHandlerT:
            self._add_check("agent_turn", handler, decisions, input, timeout, "allow")
            return handler

        return decorator

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.off("user_input_transcribed", self._on_user_input_transcribed)
        self._session.off("conversation_item_added", self._on_conversation_item_added)
        if self._session._guardrail is self:
            self._session._guardrail = None
        self._reset_turn()
        await aio.cancel_and_wait(*self._tasks)
        if self._owns_model:
            await self._model.aclose()

    # region: internals

    def _add_check(
        self,
        hook: GuardrailHook,
        handler: Callable[[DecisionEvaluated], Any],
        decisions: Mapping[str, Decision],
        input: GuardrailInput,
        timeout: NotGivenOr[float],
        on_error: GuardrailOnError,
        *,
        eager: bool = False,
    ) -> None:
        if not decisions:
            raise ValueError("a guardrail check needs at least one decision")
        self._checks.append(
            _Check(
                hook=hook,
                handler=handler,
                decisions=dict(decisions),
                input=input,
                timeout=timeout if is_given(timeout) else self._timeout,
                on_error=on_error,
                eager=eager,
            )
        )

    def _checks_for(self, hook: GuardrailHook) -> list[_Check]:
        return [check for check in self._checks if check.hook == hook]

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _reset_turn(self) -> None:
        self._turn_text = ""
        for eager in self._eager.values():
            eager.task.cancel()
        self._eager.clear()

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if self._closed or not ev.is_final or not ev.transcript:
            return
        eager_checks = [c for c in self._checks_for("user_turn") if c.eager]
        if not eager_checks:
            return

        self._turn_text = f"{self._turn_text} {ev.transcript}".strip()
        for check in eager_checks:
            # only the newest transcript matters: drop the evaluation still running
            if (previous := self._eager.get(id(check))) is not None:
                previous.task.cancel()
            evaluate_input = self._user_turn_input(check, self._turn_text, self._agent_history())
            task = asyncio.ensure_future(
                self._model.evaluate(input=evaluate_input, decisions=check.decisions)
            )
            task.add_done_callback(_retrieve_exception)
            self._eager[id(check)] = _EagerEvaluation(input=_as_text(evaluate_input), task=task)

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if self._closed or not isinstance(ev.item, llm.ChatMessage):
            return
        if ev.item.role == "assistant" and (text := ev.item.text_content):
            for check in self._checks_for("agent_turn"):
                evaluate_input: str | llm.ChatContext = text
                if check.input == "history":
                    evaluate_input = self._session.history.copy()
                self._spawn(
                    self._run_check(check, evaluate_input, input_text=text, speech_id=ev.item.id)
                )

    def _agent_history(self) -> llm.ChatContext:
        # the context the reply is generated from, before on_user_turn_completed edits it
        try:
            return self._session.current_agent.chat_ctx
        except RuntimeError:
            return self._session.history

    @staticmethod
    def _user_turn_input(
        check: _Check, text: str, history: llm.ChatContext
    ) -> str | llm.ChatContext:
        if check.input == "turn":
            return text
        chat_ctx = history.copy()
        chat_ctx.add_message(role="user", content=text)
        return chat_ctx

    async def _user_turn_verdicts(
        self, *, chat_ctx: llm.ChatContext, user_message: llm.ChatMessage
    ) -> list[_Verdict]:
        checks = self._checks_for("user_turn")
        transcript = user_message.text_content or ""
        eager = self._eager
        self._eager = {}
        self._turn_text = ""

        pending: list[Awaitable[_Verdict]] = []
        for check in checks:
            evaluate_input = self._user_turn_input(check, transcript, chat_ctx)
            evaluation: asyncio.Task[DecisionResponse] | None = None
            if (prior := eager.pop(id(check), None)) is not None:
                # reuse only if the model saw the same input, including any changes the
                # agent's on_user_turn_completed made to the context
                if _normalize(prior.input) == _normalize(_as_text(evaluate_input)):
                    evaluation = prior.task
                else:
                    prior.task.cancel()
            pending.append(
                self._run_check(check, evaluate_input, input_text=transcript, evaluation=evaluation)
            )
        for leftover in eager.values():
            leftover.task.cancel()
        return list(await asyncio.gather(*pending))

    async def _sentence_verdicts(
        self, *, reply: str, chat_ctx: llm.ChatContext, speech_id: str
    ) -> list[_Verdict]:
        pending: list[Awaitable[_Verdict]] = []
        for check in self._checks_for("agent_sentence"):
            evaluate_input: str | llm.ChatContext = reply
            if check.input == "history":
                evaluate_input = chat_ctx.copy()
                evaluate_input.add_message(role="assistant", content=reply)
            pending.append(
                self._run_check(check, evaluate_input, input_text=reply, speech_id=speech_id)
            )
        return list(await asyncio.gather(*pending))

    async def _run_check(
        self,
        check: _Check,
        evaluate_input: str | llm.ChatContext,
        *,
        input_text: str,
        speech_id: str | None = None,
        evaluation: asyncio.Task[DecisionResponse] | None = None,
    ) -> _Verdict:
        failed: _Verdict = Block() if check.on_error == "block" else None
        if evaluation is None:
            evaluation = asyncio.ensure_future(
                self._model.evaluate(input=evaluate_input, decisions=check.decisions)
            )
        # the timeout covers the evaluation and the handler together
        deadline = asyncio.get_running_loop().time() + check.timeout
        try:
            response = await asyncio.wait_for(evaluation, timeout=check.timeout)
        except Exception as e:
            logger.warning(
                "guardrail check failed",
                extra={"hook": check.hook, "handler": check.name, "error": repr(e)},
            )
            return failed

        if response.errors:
            # a decision without a result fails the whole check: a handler reading its
            # default could otherwise let through what on_error="block" should stop
            logger.warning(
                "guardrail check failed",
                extra={
                    "hook": check.hook,
                    "handler": check.name,
                    "failed_decisions": sorted(response.errors),
                },
            )
            return failed

        ev = DecisionEvaluated(
            hook=check.hook, response=response, input_text=input_text, speech_id=speech_id
        )
        try:
            verdict = check.handler(ev)
            if inspect.isawaitable(verdict):
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                verdict = await asyncio.wait_for(asyncio.ensure_future(verdict), remaining)
        except asyncio.TimeoutError:
            logger.warning(
                "guardrail handler timed out", extra={"hook": check.hook, "handler": check.name}
            )
            return failed
        except Exception:
            logger.exception(
                "error in guardrail handler", extra={"hook": check.hook, "handler": check.name}
            )
            return failed

        if verdict is None:
            return None
        if isinstance(verdict, Block) and check.hook in ("user_turn", "agent_sentence"):
            return verdict
        if isinstance(verdict, Steer) and check.hook == "user_turn":
            return verdict
        # only the type: Block/Steer carry text that may quote the conversation
        logger.warning(
            "guardrail handler returned a result its hook doesn't support, ignoring it",
            extra={
                "hook": check.hook,
                "handler": check.name,
                "result_type": type(verdict).__name__,
            },
        )
        return None

    # endregion


def _combine(verdicts: list[_Verdict]) -> Block | Steer | None:
    """The first Block wins; otherwise every Steer applies, in registration order."""
    for verdict in verdicts:
        if isinstance(verdict, Block):
            return verdict
    steers = [v.instructions for v in verdicts if isinstance(v, Steer)]
    return Steer(instructions="\n".join(steers)) if steers else None


async def run_user_turn_checks(
    guardrail: DecisionGuardrail,
    *,
    chat_ctx: llm.ChatContext,
    user_message: llm.ChatMessage,
) -> Block | Steer | None:
    """Run every ``on_user_turn`` check and combine their results."""
    if not guardrail._checks_for("user_turn"):
        return None
    verdicts = await guardrail._user_turn_verdicts(chat_ctx=chat_ctx, user_message=user_message)
    return _combine(verdicts)


def has_sentence_checks(guardrail: DecisionGuardrail) -> bool:
    return bool(guardrail._checks_for("agent_sentence"))


# a sentence with its pending check (None for whitespace, which isn't checked), or a
# segment boundary passed through unchanged
_SentenceItem = tuple[str, "asyncio.Task[Block | None] | None"] | FlushSentinel


async def gate_agent_sentences(
    guardrail: DecisionGuardrail,
    text: AsyncIterable[str | FlushSentinel],
    *,
    chat_ctx: llm.ChatContext,
    speech_id: str,
    on_block: Callable[[Block], Awaitable[None]],
) -> AsyncGenerator[str | FlushSentinel, None]:
    """Release the reply text one checked sentence at a time.

    Every sentence is checked as soon as it is complete, so later sentences are checked
    while earlier ones are spoken. Sentences are released in order. When one is blocked,
    nothing from it onwards is released and ``on_block`` is awaited.
    """
    tokenizer = SentenceTokenizer(retain_format=True)
    queue = aio.Chan[_SentenceItem]()
    checks: set[asyncio.Task[Block | None]] = set()
    reply = ""

    async def _check(reply_so_far: str) -> Block | None:
        verdicts = await guardrail._sentence_verdicts(
            reply=reply_so_far, chat_ctx=chat_ctx, speech_id=speech_id
        )
        verdict = _combine(verdicts)
        return verdict if isinstance(verdict, Block) else None

    def _submit(text: str) -> None:
        nonlocal reply
        reply += text
        if not text.strip():
            queue.send_nowait((text, None))
            return
        task = asyncio.create_task(_check(reply))
        checks.add(task)
        queue.send_nowait((text, task))

    async def _split() -> None:
        pending = ""
        try:
            async for chunk in text:
                if isinstance(chunk, FlushSentinel):
                    if pending:
                        _submit(pending)
                    pending = ""
                    queue.send_nowait(chunk)
                    continue
                pending += chunk
                sentences = tokenizer.tokenize(pending)
                # every sentence but the last is complete; the last may still grow
                if len(sentences) > 1 and "".join(sentences) == pending:
                    for sentence in sentences[:-1]:
                        _submit(sentence)
                    pending = sentences[-1]
            if pending:
                _submit(pending)
        finally:
            queue.close()

    split_task = asyncio.create_task(_split())
    try:
        async for item in queue:
            if isinstance(item, FlushSentinel):
                yield item
                continue
            sentence, task = item
            if task is not None and (block := await task) is not None:
                logger.debug(
                    "guardrail blocked the agent's reply",
                    extra={"speech_id": speech_id, "lk.pii.text": sentence},
                )
                await on_block(block)
                return
            yield sentence
    finally:
        await aio.cancel_and_wait(split_task, *checks)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _as_text(evaluate_input: str | llm.ChatContext) -> str:
    """The text a decision model receives for this input."""
    if isinstance(evaluate_input, str):
        return evaluate_input
    return _format_transcript(evaluate_input)


def _retrieve_exception(task: asyncio.Task[Any]) -> None:
    # eager evaluations are often cancelled or never awaited; failures surface when used
    if not task.cancelled():
        task.exception()


__all__ = [
    "Block",
    "DecisionEvaluated",
    "DecisionGuardrail",
    "Steer",
]
