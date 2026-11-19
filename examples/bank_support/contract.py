"""A flow contract enforced at the tool boundary.

The customer's flow document, reduced to a table of rules about tools, and a
hook that enforces the table before and after every tool call. The system
prompt is never edited: when a rule is broken the tool is not called and the
model receives a short instruction as the tool's reply; when a reply holds a
value a later tool needs, the reply says so.

Three kinds of rule cover every flow document seen so far:

- dependency: an argument must come from an earlier tool's reply (``needs``);
- prerequisite: a tool may only run after another has succeeded (``after``);
- terminal: the call may only end once the work is done and the caller has
  declined "anything else" (``after_any`` + ``requires_closing_question``);

plus one hygiene rule, ``formats``: a value must have the right shape, and a
wrong shape is read back to the caller instead of guessed.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import RunContext, ToolError

logger = logging.getLogger("bank-support.contract")

# Filler speech during a tool call: the first one 2 s in, then every 3.5 s, at
# most three times. Callers treat silence as absence; a caller who has heard
# nothing for a few seconds asks "are you still there?" and the reply in
# progress is lost.
FILLER_TEXT = "One moment."
FILLER_DELAY_S = 2.0
FILLER_INTERVAL_S = 3.5
FILLER_MAX = 3

DECLINE_WORDS = (
    "no",
    "nope",
    "nothing",
    "that's all",
    "that's it",
    "that is all",
    "all set",
    "i'm good",
    "i am good",
    "we're done",
    "goodbye",
    "bye",
)
ACCEPT_WORDS = ("yes", "yeah", "yep", "actually", "one more", "also", "another", "and ")


@dataclass(frozen=True)
class Rule:
    """What must hold before a tool runs, and what its reply should say.

    Attributes:
        write: the tool changes state; an identical successful call is not repeated.
        needs: argument name -> "producer_tool.path" the value must come from;
            path is a key, "key.sub", or "key[].sub" for a list of records.
        after: tools that must have succeeded earlier in the call.
        after_any: at least one of these must have succeeded (``declined`` means
            the caller declined the work the agent offered).
        formats: argument name -> (regex, description read to the caller).
        undo: the tool that takes this write back.
        requires_closing_question: the agent must have asked whether there is
            anything else and the caller must have declined.
    """

    write: bool = False
    needs: dict[str, str] = field(default_factory=dict)
    after: list[str] = field(default_factory=list)
    after_any: list[str] = field(default_factory=list)
    formats: dict[str, tuple[str, str]] = field(default_factory=dict)
    undo: str | None = None
    requires_closing_question: bool = False


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool = True


@dataclass
class CallState:
    """Everything the contract reads about one call: the tool replies so far, the
    caller's turns, and the completion hooks of the task in progress."""

    records: list[ToolRecord] = field(default_factory=list)
    user_turns: list[str] = field(default_factory=list)
    completions: dict[str, Callable[[str, dict[str, Any], str], None]] = field(default_factory=dict)
    declined: bool = False

    def record(self, name: str, arguments: dict[str, Any], result: str, ok: bool = True) -> None:
        self.records.append(ToolRecord(name, dict(arguments), result, ok))

    def succeeded(self, name: str) -> list[ToolRecord]:
        return [r for r in self.records if r.name == name and r.ok]

    def values_from(self, path: str) -> set[str]:
        """Every value at ``producer.path`` across that producer's successful replies."""
        producer, _, keypath = path.partition(".")
        out: set[str] = set()
        for r in self.succeeded(producer):
            try:
                data = json.loads(r.result)
            except (TypeError, ValueError):
                continue
            for v in _walk(data, keypath):
                out.add(str(v))
        return out


def _walk(data: Any, keypath: str) -> list[Any]:
    if not keypath:
        return [data]
    key, _, rest = keypath.partition(".")
    many = key.endswith("[]")
    key = key[:-2] if many else key
    if not isinstance(data, dict) or key not in data:
        return []
    value = data[key]
    if many:
        if not isinstance(value, list):
            return []
        return [x for item in value for x in _walk(item, rest)]
    return _walk(value, rest)


def closing_question_declined(history: list[tuple[str, str]]) -> bool:
    """True when the agent's last "anything else?" was answered with a decline."""
    asked_at = None
    for i, (role, text) in enumerate(history):
        if role == "assistant" and "anything else" in text.lower():
            asked_at = i
    if asked_at is None:
        return False
    answers = [t.lower() for role, t in history[asked_at + 1 :] if role == "user"]
    if not answers:
        return False
    answer = answers[0]
    if any(w in answer for w in ACCEPT_WORDS):
        return False
    return any(w in answer for w in DECLINE_WORDS)


_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def digits_heard(text: str) -> list[str]:
    """Digit strings a caller spoke, as the lookup needs them ("four one five" -> "415").

    Runs of digit words, with "double"/"triple" expanded, pauses and commas
    ignored; runs shorter than four digits are dropped (ages, counts).
    """
    tokens = re.findall(r"[a-z]+|\d", text.lower())
    runs: list[str] = []
    current = ""
    repeat = 1
    for tok in tokens:
        if tok in ("double", "triple"):
            repeat = 2 if tok == "double" else 3
            continue
        if tok in _DIGIT_WORDS or tok.isdigit():
            current += (_DIGIT_WORDS.get(tok, tok)) * repeat
            repeat = 1
            continue
        if tok in ("and", "then", "uh", "um"):
            continue
        if current:
            runs.append(current)
            current = ""
        repeat = 1
    if current:
        runs.append(current)
    return [r for r in runs if len(r) >= 4]


class Contract:
    """The table and the two hooks it generates."""

    def __init__(self, rules: dict[str, Rule]) -> None:
        self.rules = rules

    # --- before the call ------------------------------------------------------------------

    def check(
        self, name: str, arguments: dict[str, Any], state: CallState, history: list[tuple[str, str]]
    ) -> str | None:
        """None when the call may run; otherwise the reply the model gets instead."""
        rule = self.rules.get(name)
        if rule is None:
            return None
        for arg, (pattern, description) in rule.formats.items():
            value = arguments.get(arg)
            if value is not None and not re.fullmatch(pattern, str(value)):
                return (
                    f'NOT DONE: {arg} must be {description} and "{value}" is not. Read it back '
                    "to the caller digit by digit and ask them to repeat the part that is missing."
                )
        missing = [t for t in rule.after if not state.succeeded(t)]
        if missing:
            return f"NOT DONE: {name} requires {missing[0]} first in this call. Call {missing[0]}, then try again."
        if rule.after_any:
            done = [
                t
                for t in rule.after_any
                if (t == "declined" and state.declined) or state.succeeded(t)
            ]
            if not done:
                options = ", ".join(t for t in rule.after_any if t != "declined")
                return (
                    f"NOT ENDED: the call is not finished. None of the caller's requests has been completed "
                    f"({options}) and the caller has not declined them. Finish the work or confirm that the "
                    "caller wants nothing done, then ask whether there is anything else."
                )
        for arg, path in rule.needs.items():
            producer = path.partition(".")[0]
            given = arguments.get(arg)
            values = [
                str(v) for v in (given if isinstance(given, list) else [given]) if v is not None
            ]
            seen = state.values_from(path)
            if not state.succeeded(producer):
                return f"NOT DONE: {arg} must come from the reply of {producer}, which has not been called in this call. Call {producer} first."
            unknown = [v for v in values if v not in seen]
            if unknown:
                return (
                    f'NOT DONE: {arg} "{unknown[0]}" is not one of the values {producer} returned '
                    f"({', '.join(sorted(seen)) or 'none'}). Use one of those exactly, or call {producer} again."
                )
        if rule.write:
            for r in state.succeeded(name):
                if r.arguments == arguments:
                    return (
                        f"ALREADY DONE: {name} with these exact arguments succeeded earlier in this call. "
                        "Do not repeat it; tell the caller it is done."
                    )
        if rule.requires_closing_question and not closing_question_declined(history):
            return (
                "NOT ENDED: before ending, ask the caller whether there is anything else you can help with, "
                "and end only after they say there is nothing else."
            )
        return None

    # --- after the call -------------------------------------------------------------------

    def narrate(self, name: str, arguments: dict[str, Any], result: str) -> str:
        """The reply plus one sentence saying where its values go, or how to take a write back."""
        notes: list[str] = []
        try:
            data = json.loads(result)
        except (TypeError, ValueError):
            data = None
        if data is not None:
            for consumer, rule in self.rules.items():
                for arg, path in rule.needs.items():
                    producer, _, keypath = path.partition(".")
                    if producer != name:
                        continue
                    values = [str(v) for v in _walk(data, keypath)]
                    if values:
                        shown = ", ".join(values[:6]) + (" ..." if len(values) > 6 else "")
                        notes.append(f"{arg} for {consumer}: {shown}")
        rule = self.rules.get(name)
        if rule is not None and rule.write and rule.undo:
            notes.append(f"done; to take this back, call {rule.undo}")
        if not notes:
            return result
        return result + "\n\n(" + "; ".join(notes) + ")"

    # --- the hook ------------------------------------------------------------------------

    def guarded(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool function (before ``@function_tool``): check, fillers, record, narrate.

        The wrapped function keeps its signature, so ``function_tool`` builds the
        same schema the model would have seen without the hook.
        """
        name = fn.__name__
        contract = self

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = next((a for a in args if isinstance(a, RunContext)), None)
            if ctx is None:
                ctx = kwargs.get("ctx")
            state: CallState = ctx.userdata.state
            history = _history(ctx)
            refusal = contract.check(name, kwargs, state, history)
            if refusal is not None:
                logger.info("%s refused: %s", name, refusal[:120])
                return refusal
            fillers = asyncio.create_task(_say_fillers(ctx))
            try:
                result = await fn(*args, **kwargs)
            except ToolError as exc:
                state.record(name, kwargs, str(exc), ok=False)
                raise
            finally:
                fillers.cancel()
            text = result if isinstance(result, str) else json.dumps(result)
            state.record(name, kwargs, text)
            hook = state.completions.get(name)
            if hook is not None:
                hook(name, kwargs, text)
            return contract.narrate(name, kwargs, text)

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper


def _history(ctx: RunContext) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        for item in ctx.session.history.items:
            role = getattr(item, "role", None)
            text = getattr(item, "text_content", None)
            if role in ("user", "assistant") and text:
                out.append((role, text))
    except Exception:  # noqa: BLE001 - history is best effort
        pass
    return out


async def _say_fillers(ctx: RunContext) -> None:
    try:
        await asyncio.sleep(FILLER_DELAY_S)
        for _ in range(FILLER_MAX):
            if getattr(ctx.session, "user_state", None) != "speaking":
                ctx.session.say(FILLER_TEXT, add_to_chat_ctx=True)
            await asyncio.sleep(FILLER_INTERVAL_S)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a filler must never break the turn
        logger.exception("filler speech failed")
