# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/options.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, kw_only=True, slots=True)
class TurnOptions:
    """A text door's turn options; a field left out is the default."""

    policy: PolicySpec = None
    """`inbox`: one policy for every channel, or a map by channel over the
    default map; `None` is the default map.
    """
    debounce: DebounceSpec = None
    """The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    `None` is the defaults.
    """
    wait_timeout: float | None = None
    """How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    `None` is a minute.
    """


Policy: TypeAlias = Literal["batch", "interrupt", "queue", "concurrent", "reject"]
"""A message during a run: one of the five policies."""


PolicySpec: TypeAlias = Policy | dict[str, Policy] | None
"""`inbox` as a door declares it: nothing (the default map), one policy for every
channel, or a map that overrides the default map channel by channel.
"""


DebounceSpec: TypeAlias = float | dict[str, float] | None
"""The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
map per channel.
"""
