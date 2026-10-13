# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/run_result.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing_extensions import TypedDict


class RunOutputOptions(TypedDict, total=False):
    """Structured-output behavior for :meth:`AgentSession.run`.

    Can be passed as a plain dict::

        sess.run(
            user_input=...,
            output_type=MyOutput,
            output_options={"max_retries": 2, "retry_instructions": "Call submit_result."},
        )

    Pass ``output_options=None`` to disable the retry behavior.
    """

    max_retries: int
    """Re-prompts when a run ends without its ``output_type``, before raising
    UnexpectedModelBehavior. Defaults to ``2``.
    """
    retry_instructions: str
    """Override the built-in retry prompt."""


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
RunOutputOptions.__module__ = "livekit.agents.voice.run_result"
