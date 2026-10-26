//! The fuzz crate's regression tests and suites, one module per file, built as one binary:
//! each integration test file is its own binary otherwise, and linking the voice core into
//! each of them made a core edit cost a minute of release builds.
mod closing_on_own_error;
mod determinism;
mod dtmf_close;
mod dtmf_interrupt_has_cause;
mod filler_source_no_host;
mod filler_source_raises;
mod filler_source;
mod filler_while_draining;
mod hold_at_close;
mod hold_backlog_settles;
mod idle_admitted_before_hold;
mod idle_after_untranscribed_turn;
mod idle_while_closing;
mod left_at_close;
mod live_options;
mod path_needs_both_switches;
mod realtime_transcript_forwarded;
mod realtime_waiter_journalled;
mod refused_speech_announced;
mod replaced_inputs;
mod replaced_output_detached;
mod replaced_output_resets;
mod silent_audio_say;
mod silent_segment;
mod tool_started_then_interrupted;
mod transcript_path_reopened;
mod vad_interruption_has_cause;
mod wordless_audio_say;
mod late_mic_released;
