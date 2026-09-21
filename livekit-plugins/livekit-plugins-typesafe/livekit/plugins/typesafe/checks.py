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

"""Checks: the judgments Jev makes about a drafted reply, and when they trigger.

A check owns three things: the typed question sent to TypeSafe, the predicate
that turns the answer into a pass/fail, and the sentence handed to the agent
when it fails. Every check in a review rides in one request. Jev ingests the state
once and answers all of them in parallel, so adding a check costs tokens but not a
round trigger.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .log import logger

# A Choice may define at most 255 options; we reserve one for "none".
_MAX_TOOL_OPTIONS = 254


@dataclass
class TurnState:
    """Everything Jev is shown about the current turn.

    ``instructions`` is the agent's own system prompt and ``available_tools`` its tool
    catalog, so the checks below are written against any agent without being
    rewritten per agent.
    """

    instructions: str
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    reviewed_reply: str = ""
    tools_used_this_turn: list[str] = field(default_factory=list)
    tool_results: list[dict[str, str]] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            "available_tools": self.available_tools,
            "transcript": self.transcript,
            "reviewed_reply": self.reviewed_reply,
            "tools_used_this_turn": self.tools_used_this_turn,
            "tool_results": self.tool_results,
        }


@dataclass(frozen=True)
class Check:
    """One judgment about the drafted reply.

    Args:
        id: Question id; answers come back under this key.
        build: Returns the TypeSafe question payload, or ``None`` to skip this
            check for the current turn (e.g. the agent has no tools).
        triggers_when: Given the answer and the state, whether the check failed.
        reason: Appended to the nudge when the check triggers.
        gated: When True and the agent delegates to :meth:`Reviewer.gate`, a trigger
            redrafts the reply before any audio is produced. Observe-only checks
            correct on the following turn instead.
    """

    id: str
    build: Callable[[TurnState], dict[str, Any] | None]
    triggers_when: Callable[[dict[str, Any], TurnState], bool]
    reason: str
    gated: bool = False


def _tool_options(state: TurnState) -> dict[str, Any] | None:
    if not state.available_tools:
        return None
    if len(state.available_tools) > _MAX_TOOL_OPTIONS:
        logger.warning(
            "skipping the expected_tool check, the agent has more tools than a Choice "
            "question can hold",
            extra={"tools": len(state.available_tools), "limit": _MAX_TOOL_OPTIONS},
        )
        return None
    options: dict[str, Any] = {
        t["name"]: t.get("description") or None for t in state.available_tools
    }
    options["none"] = "No tool call is needed; the assistant can answer from the conversation."
    return options


def default_checks(
    *,
    follows_instructions: float = 0.5,
    unsupported_claim: float = 0.6,
    advances_task: float = 0.4,
    expected_tool_confidence: float = 0.7,
    severity: float = 2.0,
    gated_check_ids: Sequence[str] = (),
) -> list[Check]:
    """The generic check set, derived entirely from the agent's own prompt and tools.

    Every threshold is a starting point, not a calibrated default. TypeSafe's
    own guidance is emphatic that thresholds must be evaluated against your data
    and the cost of acting on them; treat these as something to tune on
    recordings of real calls.

    Args:
        follows_instructions: Trip when the probability the reply obeys the prompt
            falls below this.
        unsupported_claim: Trip when the probability the reply invented a fact
            rises above this.
        advances_task: Trip when the probability the reply advances the task falls
            below this.
        expected_tool_confidence: Only trust a tool mismatch when the Choice is
            at least this confident.
        severity: Trip when the weighted severity level reaches this. Levels run
            0 (on track) to 3 (must not continue).
        gated_check_ids: Ids of the checks that should block a reply rather than correct the
            next one. Gating costs the full draft plus one evaluation before any
            audio starts; leave it empty unless a check is worth that.
    """
    gated = set(gated_check_ids)

    def _noul(instructions: Any, criteria: dict[str, str]) -> Callable[[TurnState], dict]:
        def build(state: TurnState) -> dict[str, Any]:
            return {"type": "noul", "instructions": instructions, "criteria": criteria}

        return build

    checks = [
        Check(
            id="follows_instructions",
            build=_noul(
                "The assistant operates under the rules in `instructions`. Does "
                "`reviewed_reply` comply with every rule in `instructions`?",
                {
                    "true": "The reply obeys every instruction, constraint, and prohibition "
                    "stated in `instructions`.",
                    "false": "The reply breaks, ignores, or contradicts at least one rule in "
                    "`instructions`.",
                },
            ),
            triggers_when=lambda a, _s: a["noul"] < follows_instructions,
            reason="it broke a rule in your instructions",
            gated="follows_instructions" in gated,
        ),
        Check(
            id="unsupported_claim",
            build=_noul(
                "Does `reviewed_reply` state a specific fact that is not supported by "
                "`instructions`, by `transcript`, or by `tool_results`?",
                {
                    "true": "The reply asserts a concrete detail (a price, date, name, "
                    "identifier, availability, or policy term) that appears in none of "
                    "`instructions`, `transcript`, or `tool_results`.",
                    "false": "Every concrete detail in the reply traces back to `instructions`, "
                    "`transcript`, or `tool_results`, or the reply asserts no concrete "
                    "details at all.",
                },
            ),
            triggers_when=lambda a, _s: a["noul"] > unsupported_claim,
            reason="it stated something none of your instructions or tool results support",
            gated="unsupported_claim" in gated,
        ),
        Check(
            id="advances_task",
            build=_noul(
                "Is `reviewed_reply` advancing the task that `instructions` describes?",
                {
                    "true": "The reply moves the conversation toward the goal `instructions` sets, "
                    "including reasonable small talk and clarifying questions in service "
                    "of it.",
                    "false": "The reply has drifted onto a subject `instructions` does not cover, "
                    "or stalls without moving the task forward.",
                },
            ),
            triggers_when=lambda a, _s: a["noul"] < advances_task,
            reason="it drifted away from the task",
            gated="advances_task" in gated,
        ),
        Check(
            id="expected_tool",
            build=lambda state: (
                None
                if (options := _tool_options(state)) is None
                else {
                    "type": "choice",
                    "instructions": "Given the user's most recent message in `transcript`, "
                    "which of `available_tools` should the assistant have called before "
                    "replying?",
                    "criteria": options,
                }
            ),
            triggers_when=lambda a, s: (
                a["confidence"] >= expected_tool_confidence
                and a["choice"] != "none"
                and a["choice"] not in s.tools_used_this_turn
            ),
            reason="it answered without calling the tool that question needed",
            gated="expected_tool" in gated,
        ),
        Check(
            id="severity",
            build=lambda _state: {
                "type": "score",
                "instructions": "Judged against `instructions`, how far off course is `reviewed_reply`?",
                "criteria": [
                    "On track. The reply is what `instructions` asks for in this situation.",
                    "Minor drift. Slightly off tone, length, or focus, but harmless.",
                    "Clearly wrong but recoverable. The next reply can correct it without damage.",
                    "The assistant must not continue on this path. Continuing would mislead "
                    "the user or break `instructions` in a way the next reply cannot undo.",
                ],
            },
            triggers_when=lambda a, _s: a["score"] >= severity,
            reason="the conversation is going off course",
            gated="severity" in gated,
        ),
    ]

    unknown = gated - {r.id for r in checks}
    if unknown:
        raise ValueError(f"unknown check id(s) in gated_check_ids: {sorted(unknown)}")

    return checks
