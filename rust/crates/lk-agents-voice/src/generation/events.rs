//! Events shared directly by generation planning and its runtime forwarding channel.

use crate::transcription::Visible;

/// A segment is one or more chunks followed by `End`. Raw text retains synthesis markup;
/// visible text is suitable for the conversation item. The same event crosses the runtime
/// channel without a second enum or an adapter conversion.
#[derive(Debug)]
pub enum SpeechSegmentEvent {
    Chunk(Visible),
    End,
}
