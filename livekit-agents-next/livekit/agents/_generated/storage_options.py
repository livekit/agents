# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, TypeAlias

from typing_extensions import TypedDict


@dataclass(kw_only=True, slots=True)
class TurnOptions:
    """A text door's turn options (`a2a_session`'s); a field left out is the default."""

    inbox: PolicySpec = None
    """The turn policy for a message that arrives while one runs: one for every channel, or a
    map by channel over the default map; `None` is the default map.
    """
    debounce: DebounceSpec = None
    """How long to wait for more of a burst before a turn starts: one wait for every channel,
    or a map by channel; `None` is the defaults.
    """
    wait_timeout: timedelta | None = None
    """How long a waiter polls for the lane before its task fails `lease-timeout`; `None` is a
    minute.
    """


class TurnOptionsKwargs(TypedDict, total=False):
    """`TurnOptions`'s fields as keyword arguments."""

    inbox: PolicySpec
    """The turn policy for a message that arrives while one runs: one for every channel, or a
    map by channel over the default map; `None` is the default map.
    """
    debounce: DebounceSpec
    """How long to wait for more of a burst before a turn starts: one wait for every channel,
    or a map by channel; `None` is the defaults.
    """
    wait_timeout: timedelta | None
    """How long a waiter polls for the lane before its task fails `lease-timeout`; `None` is a
    minute.
    """


Policy: TypeAlias = Literal["batch", "interrupt", "queue", "concurrent", "reject"]
"""A message during a run: one of the five policies."""


PolicySpec: TypeAlias = Policy | dict[str, Policy] | None
"""`inbox` as a door declares it: nothing (the default map), one policy for every
channel, or a map that overrides the default map channel by channel.
"""


DebounceSpec: TypeAlias = timedelta | dict[str, timedelta] | None
"""The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
map per channel.
"""
