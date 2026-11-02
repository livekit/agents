//! Synchronous recognition-stream policy; the transcript accumulator is a private child.
//! Audio EOF, provider finalization, transcript delivery, and stream completion stay distinct.

mod stream;
mod transcript;

pub use stream::{FINAL_TRANSCRIPT_QUIET, SttStream, SttStreamEvent};
