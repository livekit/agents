# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/a2a/options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TurnOptions(BaseModel):
    """A text door's turn options; a field left out is the default."""

    policy: Policy | dict[str, Policy] | None = None
    """`inbox`: one policy for every channel, or a map by channel over the
    default map; `None` is the default map.
    """
    debounce: float | dict[str, float] | None = None
    """The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    `None` is the defaults.
    """
    wait_timeout: float | None = None
    """How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    `None` is a minute.
    """


Policy = Literal["batch", "interrupt", "queue", "concurrent", "reject"]
"""A message during a run: one of the five policies."""
