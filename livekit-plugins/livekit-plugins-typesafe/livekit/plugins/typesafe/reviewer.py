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
import json
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import aiohttp

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.events import ConversationItemAddedEvent, FunctionToolsExecutedEvent

from ._client import DEFAULT_BASE_URL, DEFAULT_MODEL, SystemOneClient
from .checks import CALIBRATED_FOR, Check, TurnState, default_checks
from .log import logger

_Chunk = llm.ChatChunk | str

NUDGE_PREFIX = (
    "Internal feedback from an automated reviewer, not from the user. Do not mention "
    "this feedback to the user, and do not apologize. The response you just produced "
    "has the following issues:"
)
NUDGE_SUFFIX = "Address these issues in your next response."

_RESULTS_LIMIT = 200
_PENDING_GATE_LIMIT = 8


@dataclass
class Verdict:
    """The outcome of one review."""

    triggered_checks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)
    reviewed_reply: str = ""
    """The reply this verdict judged, so a recorded verdict stands on its own."""
    created_at: float = field(default_factory=time.time)
    duration: float = 0.0
    """Wall-clock seconds the evaluation took, including the round trigger."""
    model: str | None = None
    """Versioned model that answered, e.g. ``jev-1.13.0``.

    An alias like ``jev-latest`` moves under you, so the version behind a
    threshold you tuned is worth keeping with the numbers you tuned it on.
    """
    usage: dict[str, int] = field(default_factory=dict)
    """Token counts for this check. Only input tokens are billed."""
    evaluated: bool = True
    """False when the check could not run and the turn went unjudged.

    Keeps "nothing was wrong" distinguishable from "nobody looked", which ``ok``
    cannot do on its own because a failed check fails open.
    """

    @property
    def needs_correction(self) -> bool:
        """Whether a check triggered and the agent should be nudged.

        Deliberately not an ``ok`` flag: a review that could not run fails open
        and would read as approval. Pair with :attr:`evaluated` to tell "nothing
        was wrong" from "nobody looked".
        """
        return bool(self.triggered_checks)

    def summary(self) -> str:
        """One line: every check's value, whether it triggered, and what it cost."""
        checks = " ".join(
            f"{'!' if check_id in self.triggered_checks else ''}{check_id}={_answer_value(answer)}"
            for check_id, answer in self.answers.items()
        )
        if not self.evaluated:
            checks = "unjudged"
        return (
            f"{checks} [{self.duration * 1000:.0f}ms "
            f"{self.usage.get('input_tokens', '?')}tok {self.model or '-'}]"
        )


class Reviewer:
    """Watches an agent against its own prompt and nudges it back on course.

    The agent's ``instructions`` and tool catalog are sent to TypeSafe as state,
    so the checks are written once and apply to any agent without being
    rewritten. Every check rides in a single request.

    Two placements, and they compose:

    * observe (:meth:`attach`), the default. The review starts once the assistant's
      message is committed and runs off the reply path entirely, so it adds nothing
      to time-to-first-token. A trigger appends a correction to the chat context, which
      the next generation picks up.
    * gate (:meth:`gate`), opted into per check via ``default_checks(gated_check_ids=...)``.
      The draft is held, evaluated, and redrafted once if a gated check triggers, all
      before any audio. Nothing bad is spoken, but the reply waits for the full
      generation plus one evaluation.

    Gating needs the agent to delegate, since a realtime model has no
    ``llm_node`` to hold::

        class MyAgent(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                return await self.reviewer.gate(self, chat_ctx, tools, model_settings)

    A failed or slow check is logged and treated as a pass. This is a nudge
    layer, not an enforcement boundary: blocking every reply whenever TypeSafe
    is unreachable would take the agent down with it. Do not use it as the only
    thing standing between a caller and a harmful answer.

    Args:
        checks: Judgments to run. Defaults to :func:`default_checks`.
        api_key: TypeSafe key. Falls back to ``TYPESAFE_API_KEY``.
        base_url: API root, for staging or a proxy.
        model: Model or alias to evaluate with.
        timeout: Hard ceiling on one evaluation. Past it the turn goes unchecked.
        history_turns: Conversation items of context sent with each check.
        http_session: Session to use instead of the agent's shared one.
        on_verdict: Called with every :class:`Verdict`, whether a check triggered or
            not. The place to hang metrics or transcript annotations.
    """

    def __init__(
        self,
        *,
        checks: list[Check] | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 2.0,
        history_turns: int = 6,
        http_session: aiohttp.ClientSession | None = None,
        on_verdict: Callable[[Verdict], None] | None = None,
        _client: SystemOneClient | None = None,
    ) -> None:
        self._checks = checks if checks is not None else default_checks()
        # Only the shipped thresholds are tied to a measured version; once the
        # caller supplies their own, drift is theirs to track.
        self._thresholds_are_default = checks is None
        self._warned_model_drift = False
        self._history_turns = history_turns
        self._on_verdict = on_verdict
        self._client = _client or SystemOneClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            http_session=http_session,
        )

        # A plain bounded deque. How a call ended, above all whether the caller
        # hung up, is the cheapest label available for tuning the thresholds
        # above, but only if the probabilities behind each turn survive to the
        # end of the call. Use on_verdict instead to stream every turn of a very
        # long call out as it happens.
        self._results: deque[Verdict] = deque(maxlen=_RESULTS_LIMIT)
        self._session: AgentSession | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._tools_used: list[str] = []
        self._tool_results: list[dict[str, str]] = []
        # One string rather than a set of ids: the gate holds the reply, so at most
        # one draft is ever in flight between gate() clearing it and
        # conversation_item_added firing. Key by message id if that stops holding.
        # Verdicts a gate has already reached, queued per draft text and drained
        # when the matching assistant item commits. A queue rather than a set
        # because two generations can be in flight with identical short text
        # ("Okay."), and a set would let the second be judged and billed twice.
        # Deferring to commit also keeps a gate nudge behind the reply it is
        # about, instead of in front of a message that does not exist yet.
        # ponytail: bounded, and abandoned drafts are evicted oldest-first; a
        # generation id would key this exactly if the framework exposed one at
        # gate time.
        self._pending_gate: OrderedDict[str, deque[Verdict]] = OrderedDict()

    @property
    def checks(self) -> list[Check]:
        return self._checks

    @property
    def results(self) -> list[Verdict]:
        """Every verdict reached on this session, oldest first.

        Read it when the call ends to pair how the call went with what the
        checks saw on the way. A caller who hangs up early is the outcome label
        these probabilities are worth calibrating against::

            @session.on("close")
            def _on_close(ev: CloseEvent) -> None:
                if ev.reason == CloseReason.PARTICIPANT_DISCONNECTED:
                    log_for_tuning(reviewer.results)
        """
        return list(self._results)

    def attach(self, session: AgentSession) -> None:
        """Start observing ``session``. Adds no latency to the reply path."""
        if self._session is not None:
            raise RuntimeError("this Reviewer is already attached to a session")
        self._session = session
        session.on("conversation_item_added", self._on_item)
        session.on("function_tools_executed", self._on_tools_executed)
        gated = [r.id for r in self._checks if r.gated]
        logger.debug(
            "reviewer attached",
            extra={
                "checks": [r.id for r in self._checks],
                "gated": gated or None,
                "model": self._client.model,
            },
        )

    def detach(self) -> None:
        """Stop observing and cancel any check still in flight."""
        if self._session is None:
            return
        self._session.off("conversation_item_added", self._on_item)
        self._session.off("function_tools_executed", self._on_tools_executed)
        self._session = None
        in_flight = [t for t in self._tasks if not t.done()]
        for task in in_flight:
            task.cancel()
        logger.debug(
            "reviewer detached",
            extra={"checks": len(self._results), "cancelled": len(in_flight)},
        )

    def _on_tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        for call, out in ev.zipped():
            self._tools_used.append(call.name)
            self._tool_results.append({"tool": call.name, "result": str(out.output)[:2000]})

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message":
            return

        if item.role == "user":  # type: ignore[union-attr]
            self._tools_used.clear()
            self._tool_results.clear()
            return

        if item.role != "assistant":  # type: ignore[union-attr]
            return

        text = item.text_content  # type: ignore[union-attr]
        if not text:
            return

        session = self._session
        if session is None:
            return
        # The agent that produced this reply, captured now rather than when the
        # detached review resumes: a handoff can make session.current_agent
        # somebody else while the request is in flight, and judging A's reply
        # against B's instructions would be meaningless.
        #
        # Not airtight. conversation_item_added names the message but not its
        # author, so a handoff that lands before this handler runs is still
        # attributed to the successor. Closing that needs agent ownership on
        # the event itself, which the framework does not expose today.
        author = session.current_agent

        queued = self._pending_gate.get(text)
        if queued:
            # Gate already ran every check on this draft. Apply its correction
            # here, now that the reply it criticises is actually in the context.
            verdict = queued.popleft()
            if not queued:
                self._pending_gate.pop(text, None)
            if verdict.triggered_checks:
                self._spawn(self._apply_nudge_to(author, verdict))
            return

        self._spawn(self._observe(text, item.id, author))  # type: ignore[union-attr]

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _observe(self, reply: str, item_id: str, agent: Agent) -> None:
        if self._session is None:
            return
        try:
            verdict = await self.review(agent, reply, exclude_item_id=item_id)
            if verdict.triggered_checks:
                await self._apply_nudge_to(agent, verdict)
        except asyncio.CancelledError:
            raise
        except Exception:
            # This runs detached from the reply path, so an unhandled error here
            # would otherwise vanish into a dropped task with the call none the wiser.
            logger.exception("reviewer check crashed, the call continues unsteered")

    async def review(
        self,
        agent: Agent,
        reviewed_reply: str,
        *,
        checks: list[Check] | None = None,
        exclude_item_id: str | None = None,
    ) -> Verdict:
        """Evaluate one drafted reply and return the verdict without acting on it.

        Args:
            exclude_item_id: Chat item to leave out of the transcript, normally
                the committed assistant message this reply came from. It already
                travels to the model as ``reviewed_reply``.
        """
        state = self._build_state(agent, reviewed_reply, exclude_item_id=exclude_item_id)
        return await self._evaluate(state, checks if checks is not None else self._checks)

    async def gate(
        self,
        agent: Agent,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
        *,
        max_redrafts: int = 1,
        draft: Callable[[llm.ChatContext], Any] | None = None,
    ) -> AsyncIterable[_Chunk]:
        """Hold the draft back until it passes, instead of correcting the next reply.

        Nothing is spoken until the gated checks clear, so a bad reply is never
        heard, at the cost of the full generation plus one evaluation before any
        audio starts.

        Falls straight through when no check is gated, or when the draft carries
        tool calls. Nothing has been spoken yet in that case, so there is nothing
        to hold back.

        Args:
            draft: Produces a reply for a chat context. Defaults to the agent's
                stock LLM node; pass one to gate a custom ``llm_node``.
        """
        produce = draft or (lambda ctx: Agent.default.llm_node(agent, ctx, tools, model_settings))

        gated_ids = {c.id for c in self._checks if c.gated}
        if not gated_ids:
            return await _as_stream(produce(chat_ctx))

        ctx = chat_ctx
        verdict = Verdict()
        for attempt in range(max_redrafts + 1):
            chunks = await _drain(produce(ctx))
            text = _text_of(chunks)

            if _has_tool_calls(chunks) or not text:
                return _replay(chunks)

            # #1: the pending user message is committed to agent.chat_ctx only
            # after the speech is scheduled, so the generation context is the
            # only place it exists right now. #4: `tools` is the flattened
            # catalog for this turn, which includes session and MCP tools.
            # #2: evaluate everything, but let only the gated checks hold the
            # draft; the rest still produce answers, metrics and a nudge.
            state = self._build_state(agent, text, chat_ctx=ctx, tools=tools)
            verdict = await self._evaluate(state, self._checks)
            gated_triggered = [c for c in verdict.triggered_checks if c in gated_ids]
            if not gated_triggered:
                logger.debug(
                    "gate cleared the draft",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": round(verdict.duration * 1000),
                        "triggered_checks": verdict.triggered_checks or None,
                    },
                )
                # Queued, not applied: the assistant message does not exist
                # yet, so nudging here would put the correction in front of the
                # reply it criticises. _on_item applies it once that commits.
                self._queue_gate_verdict(text, verdict)
                return _replay(chunks)

            if attempt == max_redrafts:
                logger.warning(
                    "gated checks still triggered after redrafting, releasing the reply "
                    "and correcting the next turn instead",
                    extra={
                        "triggered_checks": gated_triggered,
                        "redrafts": max_redrafts,
                        "lk.pii.reviewed_reply": text[:500],
                    },
                )
                self._queue_gate_verdict(text, verdict)
                return _replay(chunks)

            logger.debug(
                "gate triggered, redrafting",
                extra={"triggered_checks": gated_triggered, "attempt": attempt},
            )
            ctx = ctx.copy()
            ctx.add_message(role="system", content=self._nudge_text(verdict))

        raise AssertionError("unreachable")  # pragma: no cover

    def _build_state(
        self,
        agent: Agent,
        reviewed_reply: str,
        *,
        chat_ctx: llm.ChatContext | None = None,
        tools: list[llm.Tool] | None = None,
        exclude_item_id: str | None = None,
    ) -> TurnState:
        """Assemble what the reviewer is shown about this turn.

        Args:
            chat_ctx: Context the reply was generated from. Gate must pass this:
                the pending user message reaches ``agent.chat_ctx`` only once the
                speech is scheduled, so during generation it lives nowhere else.
            tools: Flattened catalog for this turn. Falls back to the session's
                effective catalog, which unlike ``agent.tools`` also carries
                tools registered on the ``AgentSession`` and over MCP.
        """
        raw_instructions = agent.instructions
        # Voice turns resolve the audio variant, and an Instructions built by
        # resolve_template can hold the whole prompt there with an empty common
        # part. Rendering bare would hide rules the model was actually given.
        instructions = (
            raw_instructions
            if isinstance(raw_instructions, str)
            else raw_instructions.render(modality="audio")
        )

        catalog: list[llm.Tool] | list[llm.Tool | llm.Toolset]
        if tools is not None:
            catalog = tools
        else:
            catalog = self._effective_tools(agent)

        tool_infos: list[dict[str, Any]] = []
        for name, tool in llm.ToolContext(catalog).function_tools.items():
            description: str | None
            if llm.is_function_tool(tool):
                description = tool.info.description
            elif llm.is_raw_function_tool(tool):
                description = tool.info.raw_schema.get("description")
            else:
                description = None
            tool_infos.append({"name": name, "description": description})

        # Only what was actually said. The agent's instructions also sit in
        # chat_ctx as a system message, and every nudge this plugin injects
        # lands there too. Copying either into `transcript` would repeat the whole
        # prompt in a field that already holds it as `instructions`, and would show
        # the reviewer its own past notes as if the caller had said them.
        source = chat_ctx if chat_ctx is not None else agent.chat_ctx
        transcript: list[dict[str, str]] = []
        for item in source.items:
            if getattr(item, "type", None) != "message":
                continue
            if item.role not in ("user", "assistant"):  # type: ignore[union-attr]
                continue
            # Exclude by identity, never by text: an earlier turn can repeat the
            # reply word for word ("Yes." answered with "Yes.") and would vanish.
            if exclude_item_id is not None and item.id == exclude_item_id:
                continue
            text = item.text_content  # type: ignore[union-attr]
            if text:
                transcript.append({"role": item.role, "text": text})  # type: ignore[union-attr]
        transcript = transcript[-self._history_turns :]

        return TurnState(
            instructions=instructions,
            available_tools=tool_infos,
            transcript=transcript,
            reviewed_reply=reviewed_reply,
            tools_used_this_turn=list(self._tools_used),
            tool_results=list(self._tool_results),
        )

    def _effective_tools(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        """Every tool the LLM could call this turn, not just the agent's own."""
        session = self._session
        if session is None:
            return list(agent.tools)
        try:
            return list(session.tools) + list(agent.tools)
        except Exception:  # pragma: no cover - session not started
            return list(agent.tools)

    async def _evaluate(self, state: TurnState, checks: list[Check]) -> Verdict:
        questions: dict[str, Any] = {}
        active: list[Check] = []
        for check in checks:
            question = check.build(state)
            if question is not None:
                questions[check.id] = question
                active.append(check)

        if not questions:
            logger.debug("no checks apply to this turn, nothing to review")
            verdict = Verdict(reviewed_reply=state.reviewed_reply, evaluated=False)
            self._record(verdict)
            return verdict

        payload = state.as_payload()
        started = time.monotonic()
        try:
            response = await self._client.evaluate(payload, questions)
        except Exception as e:
            # Fail open: a reviewer check that cannot run must not silence the agent.
            verdict = Verdict(
                reviewed_reply=state.reviewed_reply,
                duration=time.monotonic() - started,
                evaluated=False,
            )
            logger.warning(
                "reviewer check failed, this turn went unjudged",
                extra={
                    # APIStatusError stringifies the response body, which can
                    # echo back the prompt, transcript or reply we sent it.
                    "lk.pii.error": str(e),
                    "error_type": type(e).__name__,
                    "checks": [r.id for r in active],
                    "state_chars": _payload_chars(payload),
                    "elapsed_ms": round(verdict.duration * 1000),
                },
            )
            self._record(verdict)
            return verdict

        answers = response.get("answers", {})
        self._warn_on_model_drift(response.get("model"))
        verdict = Verdict(
            answers=answers,
            reviewed_reply=state.reviewed_reply,
            duration=time.monotonic() - started,
            model=response.get("model"),
            usage=response.get("usage") or {},
        )
        readable = 0
        for check in active:
            answer = answers.get(check.id)
            if answer is None:
                logger.warning(
                    "check got no answer back", extra={"check": check.id, "model": verdict.model}
                )
                continue
            try:
                triggered = check.triggers_when(answer, state)
            except (KeyError, TypeError) as e:
                logger.warning(
                    "check could not read its answer",
                    extra={
                        "check": check.id,
                        "error": str(e),
                        "lk.pii.answer": answer,
                    },
                )
                continue
            readable += 1
            if triggered:
                verdict.triggered_checks.append(check.id)
                verdict.reasons.append(check.reason)

        # "Nobody looked" covers a partial look too. A 200 carrying {"answers": {}}
        # would otherwise land as a clean, fully evaluated turn.
        verdict.evaluated = readable == len(active)
        if not verdict.evaluated:
            logger.warning(
                "not every check came back, treating the turn as unjudged",
                extra={"answered": readable, "expected": len(active), "model": verdict.model},
            )

        # Every check's value, every turn. Without this you cannot tell a check
        # that sat at 0.51 from one that sat at 0.99, which is the whole of
        # knowing whether a threshold is in the right place.
        logger.debug("check %s", verdict.summary())
        self._record(verdict)
        return verdict

    def _queue_gate_verdict(self, text: str, verdict: Verdict) -> None:
        """Hand a gate's verdict to the commit path, keyed by the draft text."""
        queue = self._pending_gate.get(text)
        if queue is None:
            queue = deque()
            self._pending_gate[text] = queue
        queue.append(verdict)
        self._pending_gate.move_to_end(text)
        # A replayed stream that never commits would otherwise leave its entry
        # behind for a later identical reply to consume.
        while len(self._pending_gate) > _PENDING_GATE_LIMIT:
            stale, _ = self._pending_gate.popitem(last=False)
            logger.debug("evicting an uncommitted gate verdict", extra={"chars": len(stale)})

    def _warn_on_model_drift(self, answered_by: str | None) -> None:
        """Say so once if the shipped thresholds are judging an unmeasured version."""
        if (
            self._warned_model_drift
            or not self._thresholds_are_default
            or answered_by is None
            or answered_by == CALIBRATED_FOR
        ):
            return
        self._warned_model_drift = True
        logger.warning(
            "default thresholds were measured on a different model version, so they "
            "may no longer separate compliant replies from violating ones; re-measure "
            "them or pin the model",
            extra={"answered_by": answered_by, "calibrated_for": CALIBRATED_FOR},
        )

    def _record(self, verdict: Verdict) -> None:
        self._results.append(verdict)
        if self._on_verdict is None:
            return
        try:
            self._on_verdict(verdict)
        except Exception:
            # A broken metrics callback must not take the call down with it.
            logger.exception("on_verdict callback raised")

    def _nudge_text(self, verdict: Verdict) -> str:
        reasons = "\n".join(f"- {reason}" for reason in verdict.reasons)
        return f"{NUDGE_PREFIX}\n{reasons}\n\n{NUDGE_SUFFIX}"

    async def _apply_nudge_to(self, agent: Agent, verdict: Verdict) -> None:
        """Nudge the agent that produced the reply, if it is still the one talking.

        A handoff can retire that agent mid-review. Writing the correction into
        an inactive agent would go unread, and writing it into its successor
        would tell a different agent, under different instructions, to fix
        behaviour it never produced.
        """
        session = self._session
        if session is not None and session.current_agent is not agent:
            logger.debug(
                "dropping a correction for an agent that has since handed off",
                extra={"triggered_checks": verdict.triggered_checks},
            )
            return
        await self._apply_nudge(agent, verdict)

    async def _apply_nudge(self, agent: Agent, verdict: Verdict) -> None:
        logger.info(
            "nudging the agent back on course",
            extra={
                "triggered_checks": verdict.triggered_checks,
                "reasons": verdict.reasons,
                "elapsed_ms": round(verdict.duration * 1000),
                "model": verdict.model,
                "lk.pii.reviewed_reply": verdict.reviewed_reply[:500],
            },
        )
        ctx = agent.chat_ctx.copy()
        ctx.add_message(role="system", content=self._nudge_text(verdict))
        await agent.update_chat_ctx(ctx)


def _answer_value(answer: dict[str, Any]) -> str:
    """The one number that matters for a check, for a log line."""
    kind = answer.get("type")
    if kind == "noul":
        return f"{answer.get('noul', float('nan')):.2f}"
    if kind == "choice":
        return f"{answer.get('choice')}@{answer.get('confidence', float('nan')):.2f}"
    if kind == "score":
        return (
            f"{answer.get('score', float('nan')):.2f}@{answer.get('confidence', float('nan')):.2f}"
        )
    return "?"


def _payload_chars(payload: Any) -> int:
    """Rough size of what was sent, for diagnosing a 422 against the 32k state cap."""
    try:
        return len(json.dumps(payload, default=str))
    except Exception:
        return -1


async def _as_stream(node: Any) -> AsyncIterable[_Chunk]:
    """Normalize every shape ``llm_node`` is allowed to return into a stream.

    The framework accepts a coroutine resolving to a whole string, a single
    ``ChatChunk`` or ``None`` as well as an async iterable, so a custom node
    delegating to :meth:`Reviewer.gate` can hand us any of them.
    """
    stream = await node if asyncio.iscoroutine(node) else node
    if stream is None:
        return _replay([])
    if isinstance(stream, (str, llm.ChatChunk)):
        return _replay([stream])
    return cast(AsyncIterable[_Chunk], stream)


async def _drain(node: Any) -> list[_Chunk]:
    return [chunk async for chunk in await _as_stream(node)]


def _text_of(chunks: list[_Chunk]) -> str:
    parts = []
    for chunk in chunks:
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
            parts.append(chunk.delta.content)
    return "".join(parts)


def _has_tool_calls(chunks: list[_Chunk]) -> bool:
    return any(
        isinstance(c, llm.ChatChunk) and c.delta is not None and bool(c.delta.tool_calls)
        for c in chunks
    )


def _replay(chunks: list[_Chunk]) -> AsyncIterable[_Chunk]:
    async def gen() -> AsyncIterable[_Chunk]:
        for chunk in chunks:
            yield chunk

    return gen()
