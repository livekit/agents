# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/worker/remote_session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Final

STATE_OPTION_FIELDS: Final = ("endpointing", "interruption", "max_tool_steps", "user_away_timeout", "transcription_timeout", "preemptive_generation", "min_consecutive_speech_delay", "use_tts_aligned_transcript", "ivr_detection")
"""The option keys a [`SessionState`] row's `options` carries, in the reference's order. Only the
key set is pinned: values are the host's `str()` of each option.
"""

STATE_OPTION_FIELDS: Final = ("endpointing", "interruption", "max_tool_steps", "user_away_timeout", "transcription_timeout", "preemptive_generation", "min_consecutive_speech_delay", "use_tts_aligned_transcript", "ivr_detection")
"""The option keys a [`SessionState`] row's `options` carries, in the reference's order. Only the
key set is pinned: values are the host's `str()` of each option.
"""
