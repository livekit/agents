# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TypedDict


class LemonSliceOptions(TypedDict, total=False):
    """Provider-specific options for a LemonSlice inference avatar.

    Mirrors the per-provider option dicts in :mod:`inference.stt` /
    :mod:`inference.tts` (e.g. ``CartesiaOptions``); passed via ``extra_kwargs``.
    All keys are optional.
    """

    image_url: str
    prompt: str
    idle_prompt: str
    idle_timeout: int


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
LemonSliceOptions.__module__ = "livekit.agents.inference.avatar"
