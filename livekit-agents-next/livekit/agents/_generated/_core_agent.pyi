# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from livekit.agents.llm import LLM
from livekit.agents.voice.agent_host import AgentHost

from .agent_run import AgentDefinition, EventExpectation, ObservedEvent, RunRequest, RunUpdate

class VoiceAgent:
    """`_core.VoiceAgent(definition, host, host_context)`: an [`AgentFrame`], built where the
    definition converts.
    """
    def __init__(
        self,
        definition: AgentDefinition,
        host: AgentHost,
        host_context: object,
    ) -> None: ...

class RunEvents:
    """The observed events of one invocation and the position `next_event` reads from. Each call
    returns the index of the event it matched.
    """
    def __init__(self, events: Sequence[ObservedEvent]) -> None: ...
    def at(self, index: int, expected: EventExpectation | None) -> int:
        """The event at `index` (from the end when negative), meeting `expected` when given."""
    def next_event(self, expected: EventExpectation | None) -> int:
        """Advances to the next event of `expected`'s kind, or the next event when `None`."""
    def skip_next_event_if(self, expected: EventExpectation) -> int | None:
        """Advances past the next event only if it meets `expected`."""
    def skip_next(self, count: int) -> None: ...
    def no_more_events(self) -> None: ...
    def contains(self, start: int, stop: int, expected: EventExpectation) -> int:
        """The first event of `start..stop` meeting `expected`."""
    def failure(self, message: str, index: int | None) -> str:
        """The message of a failure about the event at `index`, or the current one: `message` and
        the events with that one marked.
        """

class AgentRunner:
    """Runs invocations of one agent definition: each `run` streams its updates, then its settled
    result; closing the stream cancels the invocation, whose work still settles.
    """
    def __init__(self, definition: AgentDefinition, llm: LLM[Any], host: AgentHost) -> None:
        """The runner of `definition` on `llm`, its callbacks run by `host`."""
    def run(self, request: RunRequest) -> AgentRunUpdates:
        """Starts one invocation; nothing runs until the first pull."""
    def progress(self, invocation_id: str, operation: int, message: str) -> None:
        """`ctx.update(message)` from the tool body running as host `operation` of invocation
        `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
        end is dropped. Refused for an invocation this runner is not running.
        """

class AgentRunUpdates:
    async def recv(self) -> RunUpdate | None: ...
    async def aclose(self) -> None: ...
