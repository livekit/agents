# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/workflows/prompts.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Final

END_CALL_DESCRIPTION: Final = "\nEnds the current call and disconnects immediately.\n\nCall when:\n- The user clearly indicates they are done (e.g., “that’s all, bye”).\n\nDo not call when:\n- The user asks to pause, hold, or transfer.\n- Intent is unclear.\n\nThis is the final action the agent can take.\nOnce called, no further interaction is possible with the user.\nDon't generate any other text or response when the tool is called.\n"
"""`EndCallTool`'s description, before its `extra_description`."""
