# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/worker/remote_session.rs

from __future__ import annotations

from typing import Final

STATE_OPTION_FIELDS: Final = ("endpointing", "interruption", "max_tool_steps", "user_away_timeout", "transcription_timeout", "preemptive_generation", "min_consecutive_speech_delay", "use_tts_aligned_transcript", "ivr_detection")
"""The option keys a [`SessionState`] row's `options` carries, in the reference's order. Only the
key set is pinned: values are the host's `str()` of each option.
"""
