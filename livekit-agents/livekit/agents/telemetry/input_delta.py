"""Input deltas for LLM spans (``RecordingOptions.input_delta``).

Purpose
-------
Each LLM span records the input of the model. In a long session, the input of each turn
contains all of the previous turns. Thus, the spans become larger at each turn. When the
session records with ``input_delta``, a span records only the part of the input that
changed. The model always receives all of the input. This module changes only the
telemetry.

Terms
-----
- **Record**: the list of entries that a span records. On an ``llm_node`` span, the record
  is ``lk.pii.chat_ctx``, with one entry for each chat item. On an ``llm_request`` span,
  the record is ``gen_ai.input.messages``, with one entry for each GenAI message.
- **Instructions**: the instructions message of the agent. On an ``llm_request`` span,
  the instructions are in ``gen_ai.system_instructions``, not in the record. All other
  system messages are entries of the record, at their position.
- **Generation**: one LLM inference of the agent. A turn with tool calls has more than one
  generation.
- **Commit**: a generation is committed when its speech is scheduled. A discarded
  preemptive generation is not committed.
- **Parent**: the span of the last committed generation that has the same span name.

Procedure
---------
For each span, the tracker does these steps:

1. It calculates a key for each item of the record. The key is the item ID and a
   fingerprint of the content of the item (``ChatItem._fingerprint``).
2. It compares the keys with the keys of the parent, in order. It finds the longest
   prefix that the two records share.
3. The span records only the entries after this prefix.
4. ``lk.input.dropped_from_base`` gives the number of parent entries after this prefix.
5. On an ``llm_request`` span, the tracker compares the instructions with the
   instructions of the parent. If the text is the same, the span does not record
   ``gen_ai.system_instructions``.
6. If the span shares no entries and no instructions with the parent, the span records
   all of the input, without ``lk.input.*`` attributes.
7. The tracker keeps the keys of the span. When the generation is committed, these keys
   replace the keys of the parent.

Rebuild
-------
To make the full record of a span again:

1. Make the full record of the parent (``lk.input.base_span_id``). Use this procedure
   again if the parent also has a parent.
2. Remove the last ``lk.input.dropped_from_base`` entries.
3. Add the entries of this span.

If an ``llm_request`` span with a parent has no ``gen_ai.system_instructions``, use the
instructions of the parent.

Rules
-----
- The comparison does not use the role of an item. A system message that stays at its
  position is part of the shared prefix. The framework adds the expressive guide again at
  the end of each reply. Thus, the old guide is in the dropped entries, and the new guide
  is in the recorded entries.
- An edited, removed or moved item stops the shared prefix at that item. The span records
  the entries from that item. It does not record all of the input.
- In ``gen_ai.input.messages``, consecutive tool calls of an assistant turn are one
  message, also when a skipped item (for example, a config update) is between them. The
  prefix never stops inside such a message.
- On an ``llm_node`` span, the instructions are the first entry of ``lk.pii.chat_ctx``.
  Thus, a change of the instructions causes a full record of ``lk.pii.chat_ctx``.
- The fingerprint does not include media data. It uses the image ID and the audio
  transcript.
- Each ``AgentActivity`` has its own tracker. After a handoff, the first span records all
  of the input.

Example
-------
In expressive mode, ``G`` is the expressive guide, and ``I`` is the instructions message.

====================  =============================  =========================  ========
Generation            Input sent to the model        Record of ``llm_node``     Dropped
====================  =============================  =========================  ========
Turn 1                ``[I, u1, G]``                 ``[I, u1, G]`` (full)      none
Turn 2                ``[I, u1, a1, u2, G]``         ``[a1, u2, G]``            1
Turn 3, tool call     ``[I, …, u2, a2, u3, G]``      ``[a2, u3, G]``            1
Turn 3, tool reply    ``[I, …, u3, G, fc, fo]``      ``[fc, fo]``               0
====================  =============================  =========================  ========
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from . import trace_types
from .gen_ai import (
    _conversation_messages,
    _instruction_parts,
    _json,
    _message_layout,
    _split_instructions,
)

if TYPE_CHECKING:
    from ..llm import ChatContext, ChatItem


@dataclass(frozen=True)
class InputDeltaSite:
    """Where an LLM input is recorded, which decides what the record counts."""

    name: str
    chat_ctx: bool
    """The span records ``lk.pii.chat_ctx`` (every chat item, in order) rather than
    ``gen_ai.input.messages`` and ``gen_ai.system_instructions``."""


LLM_NODE = InputDeltaSite("llm_node", chat_ctx=True)
LLM_REQUEST = InputDeltaSite("llm_request", chat_ctx=False)


@dataclass
class _InputBaseline:
    keys: list[tuple[str, bytes]]
    """(id, content fingerprint) of each item the span records, in order"""
    layout: list[str]
    """per item, how it lands in gen_ai.input.messages: "new", "merged" into the previous
    message (consecutive tool calls) or "skipped" (non-conversational)"""
    instructions: str
    span_context: trace.SpanContext


@dataclass
class InputDelta:
    """How much of an LLM input a span records; the model always receives all of it.

    With ``base`` set, the record continues the one on that span, its parent: remove the
    parent's last ``dropped_from_base`` entries, then append this span's (``chat_ctx`` on
    ``llm_node``, ``conversation`` on ``llm_request``). Empty ``instructions`` then mean
    the parent's apply.
    """

    chat_ctx: ChatContext
    instructions: list[ChatItem]
    conversation: list[ChatItem]
    base: trace.SpanContext | None = None
    dropped_from_base: int | None = None

    @classmethod
    def full(cls, chat_ctx: ChatContext) -> InputDelta:
        instructions, conversation = _split_instructions(chat_ctx.items)
        return cls(chat_ctx=chat_ctx, instructions=instructions, conversation=conversation)

    def system_instructions(self) -> list[dict[str, Any]]:
        return _instruction_parts(self.instructions)

    def input_messages(self) -> list[dict[str, Any]]:
        return _conversation_messages(self.conversation)


class InputDeltaTracker:
    """Per-agent memory of the LLM input recorded for the last committed generation, which
    the next generation's spans continue (see :func:`compute`)."""

    def __init__(self) -> None:
        self._baselines: dict[InputDeltaSite, _InputBaseline] = {}

    def begin(self) -> InputDeltaScope:
        return InputDeltaScope(self)


class InputDeltaScope:
    """One generation's view of an :class:`InputDeltaTracker`: the inputs its spans
    recorded, which become the tracker's baseline once the generation is committed.

    A generation that is never committed (a discarded preemptive generation, an aborted
    reply) is compared against the baseline but never replaces it.
    """

    def __init__(self, tracker: InputDeltaTracker) -> None:
        self._tracker = tracker
        self._pending: dict[InputDeltaSite, _InputBaseline] = {}
        self._committed = False

    def commit(self) -> None:
        self._committed = True
        self._tracker._baselines.update(self._pending)

    def delta(self, site: InputDeltaSite, chat_ctx: ChatContext, span: trace.Span) -> InputDelta:
        from ..llm import ChatContext

        full = InputDelta.full(chat_ctx)
        records = list(chat_ctx.items) if site.chat_ctx else full.conversation
        current = _InputBaseline(
            keys=_item_keys(records),
            layout=_message_layout(records),
            instructions=_json(full.system_instructions()),
            span_context=span.get_span_context(),
        )
        parent = self._tracker._baselines.get(site)

        if span.is_recording():
            # the content recorded on the span: a preemptive generation's message may still
            # change before it is committed, and the next span must then record that edit
            self._pending[site] = current
            if self._committed:
                self._tracker._baselines[site] = current

        if parent is None:
            return full
        shared = _shared_prefix(current, parent, gen_ai_messages=not site.chat_ctx)
        if site.chat_ctx:
            if shared == 0:
                return full
            return InputDelta(
                chat_ctx=ChatContext(records[shared:]),
                instructions=full.instructions,
                conversation=full.conversation,
                base=parent.span_context,
                dropped_from_base=len(parent.keys) - shared,
            )

        same_instructions = current.instructions == parent.instructions
        if shared == 0 and not same_instructions:
            return full
        return InputDelta(
            chat_ctx=chat_ctx,
            instructions=[] if same_instructions else full.instructions,
            conversation=records[shared:],
            base=parent.span_context,
            dropped_from_base=parent.layout[shared:].count("new"),
        )


def _shared_prefix(
    current: _InputBaseline, parent: _InputBaseline, *, gen_ai_messages: bool
) -> int:
    n = 0
    for a, b in zip(current.keys, parent.keys, strict=False):
        if a != b:
            break
        n += 1
    if gen_ai_messages:
        # never cut inside a message, on either side
        while n > 0 and (_inside_message(current.layout, n) or _inside_message(parent.layout, n)):
            n -= 1
    return n


def _inside_message(layout: list[str], n: int) -> bool:
    """Whether a cut before item ``n`` splits a message: the next item that lands in a
    message is a tool call merged into the one before the cut (also across skipped items,
    such as a config update)."""
    while n < len(layout) and layout[n] == "skipped":
        n += 1
    return n < len(layout) and layout[n] == "merged"


def _item_keys(items: Sequence[ChatItem]) -> list[tuple[str, bytes]]:
    return [(item.id, item._fingerprint()) for item in items]


_scope: contextvars.ContextVar[InputDeltaScope | None] = contextvars.ContextVar(
    "lk_input_delta_scope", default=None
)


def set_scope(scope: InputDeltaScope | None) -> contextvars.Token[InputDeltaScope | None]:
    """Make ``scope`` the generation the spans created from here record against."""
    return _scope.set(scope)


def reset_scope(token: contextvars.Token[InputDeltaScope | None]) -> None:
    _scope.reset(token)


def active() -> bool:
    return _scope.get() is not None


def compute(site: InputDeltaSite, chat_ctx: ChatContext, span: trace.Span) -> InputDelta:
    """Decide how much of ``chat_ctx`` the span records (see the module docstring). The
    model always receives all of it; this only affects telemetry.

    The span's own input is held as pending, and becomes the parent of the next
    generation's only once this one is committed. Without ``input_delta``, return all
    of ``chat_ctx``.
    """
    if (scope := _scope.get()) is None:
        return InputDelta.full(chat_ctx)
    return scope.delta(site, chat_ctx, span)


def set_attributes(span: trace.Span, delta: InputDelta) -> None:
    """Point a span whose record continues another span's at that parent."""
    if delta.base is None or not span.is_recording():
        return
    attrs: dict[str, AttributeValue] = {
        trace_types.ATTR_INPUT_DELTA: True,
        trace_types.ATTR_INPUT_BASE_SPAN_ID: trace.format_span_id(delta.base.span_id),
    }
    if delta.dropped_from_base is not None:
        attrs[trace_types.ATTR_INPUT_DROPPED_FROM_BASE] = delta.dropped_from_base
    span.set_attributes(attrs)
    span.add_link(delta.base)
