# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from typing import Any

from livekit.agents.llm import LLM
from livekit.agents.voice.agent_host import AgentHost

from .agent_run import AgentDefinition, RunRequest, RunUpdate

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
