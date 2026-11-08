# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tts/stream_adapter.rs
#     rust/crates/lk-agents-abi/src/tts/stream_pacer.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TypeAlias

from livekit.agents.tts import SentenceStreamPacer as _HostSentenceStreamPacer

SentenceStreamPacer = _HostSentenceStreamPacer


TextPacing: TypeAlias = SentenceStreamPacer | bool
"""`StreamAdapter(text_pacing=)`: a pacer, or `True` for the default one."""
