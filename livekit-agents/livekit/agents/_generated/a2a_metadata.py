# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/a2a/metadata.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Final

EXTENSION_URI: Final = "https://livekit.io/a2a/ext/agents/v1"

EXTENSIONS_HEADER: Final = "A2A-Extensions"

CHANNEL: Final = "https://livekit.io/a2a/ext/agents/v1/channel"

KIND: Final = "https://livekit.io/a2a/ext/agents/v1/kind"

INTERRUPTED: Final = "https://livekit.io/a2a/ext/agents/v1/interrupted"

INTERRUPTION_REASON: Final = "https://livekit.io/a2a/ext/agents/v1/interruptionReason"

MERGED_INTO: Final = "https://livekit.io/a2a/ext/agents/v1/mergedInto"

REASON: Final = "https://livekit.io/a2a/ext/agents/v1/reason"

VERBATIM: Final = "https://livekit.io/a2a/ext/agents/v1/verbatim"

KIND_DELEGATION: Final = "delegation"

KIND_CHAT_CTX: Final = "chat_ctx"

ANSWER_ARTIFACT_NAME: Final = "answer"
