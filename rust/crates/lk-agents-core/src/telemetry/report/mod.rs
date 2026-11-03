//! The session report's pure projections (docs/internals/telemetry.md "Session report"): one
//! `SessionReport` and its recording options become the Cloud records ([`records()`]) and the
//! recording upload ([`upload()`]), bounds decided here, before any I/O. [`Tagger`] holds the
//! session's tags. `lk-agents-net` exports and uploads.

mod chat_item;
mod records;
mod tagger;
mod upload;

pub use records::records;
pub use tagger::{DISABLED_MARKERS, Tagger, is_disabled_response};
pub use upload::{RecordingUpload, TRANSCRIPT_BYTES, TRANSCRIPT_ITEMS, upload};

/// Whether the session's event retention may keep an event of `kind`: state changes, errors and
/// the close always, content-bearing events with content consent, nothing else.
pub fn event_allowed(kind: &str, capture_content: bool, redaction: bool) -> bool {
    match kind {
        // The summary's error counts and close reason read these; they carry no content.
        "user_state_changed"
        | "agent_state_changed"
        | "user_transcription_timeout"
        | "error"
        | "close" => true,
        "user_input_transcribed"
        | "conversation_item_added"
        | "function_tools_executed"
        | "tool_execution_updated"
        | "agent_false_interruption" => super::policy::content_allowed(capture_content, redaction),
        // Live handles, media arrays and unknown schemas are never report snapshots.
        _ => false,
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
