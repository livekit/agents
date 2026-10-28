# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` star-imports it.

from __future__ import annotations

from typing import Any

from livekit.agents.llm import LLM
from livekit.agents.voice.agent_host import AgentHost

class AgentRunner:
    """Runs invocations of one agent definition: each `run` streams its updates, then its settled
    result; closing the stream cancels the invocation, whose work still settles.
    """
    def __init__(self, definition: dict[str, object], llm: LLM[Any], host: AgentHost) -> None:
        """The runner of `definition` on `llm`, its callbacks run by `host`."""
    def run(self, request: dict[str, object]) -> AgentRunUpdates:
        """Starts one invocation; nothing runs until the first pull."""

class AgentRunUpdates:
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...
