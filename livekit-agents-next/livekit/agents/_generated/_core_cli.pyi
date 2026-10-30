# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from livekit.agents.cli import _DevServer

from .cli import CliParseResult, LogRecord

def parse_cli(argv: Sequence[str]) -> CliParseResult: ...

def format_log_record(record: LogRecord) -> str:
    """One structured JSON line, including host-formatted exceptions and stack traces."""

async def dev_channel(
    cli_addr: str,
    agent_name: str,
    url: str,
    api_key: str,
    api_secret: str,
    server: _DevServer,
) -> None:
    """The development launcher connection."""

class Shutdown:
    """Native enforcement of the process's one shutdown deadline."""
    def __new__(cls, timeout: float, drain_timeout: float) -> Self: ...
    def listen(self) -> None:
        """Own process termination signals. Fails if another launcher is already listening."""
    async def wait(self) -> None:
        """Wait until shutdown starts, whether requested by the host or a native signal."""
    def begin(self) -> bool:
        """Begin shutdown once. Repeated requests never extend the deadline."""
    def remaining(self) -> float:
        """Seconds left until the deadline, or the configured budget before shutdown begins."""
    def drain_remaining(self) -> float:
        """Drain time remaining while reserving the configured cleanup budget."""
    def closed(self) -> None:
        """Disarm only after host cleanup and event-loop closure finish."""
