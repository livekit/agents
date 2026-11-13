//! A socket TTS stream's segments: the framework cuts the input at each flush and names each
//! segment; the plugin encodes each segment's input and reports what the provider answered.

use std::sync::Arc;

use crate::TranscriptChunk;
use crate::errors::ProviderError;
use crate::tts::stream_pacer::PushedAudio;
use crate::tts::types::SpeechChunk;
use crate::types::provider::ReadDeadline;

/// A segment opens; its text follows. The id is the framework's, unique to the segment.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentStart {
    pub segment_id: String,
}

/// Text of a segment: `chunk.text` in the provider's syntax, `chunk.said` its words.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentText {
    pub segment_id: String,
    pub chunk: SpeechChunk,
}

/// A segment's text ended: the provider owes its audio, then its `SegmentDone`.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentEnd {
    pub segment_id: String,
}

/// What a socket TTS stream's `encode` receives for its segments.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", emit, transparent))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentInput {
    Start(SegmentStart),
    Text(SegmentText),
    End(SegmentEnd),
}

/// Audio of a segment, encoded as the stream's `mime_type`.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentAudio {
    pub segment_id: String,
    pub data: Vec<u8>,
}

/// Word timings of a segment, from its start.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentWords {
    pub segment_id: String,
    pub words: Vec<TranscriptChunk>,
}

/// The provider finished a segment: its audio is complete.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", dataclass, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentDone {
    pub segment_id: String,
}

/// What a socket TTS stream's `on_message` returns for a segment.
#[crate::host(record(module = "tts_segments", path = "livekit.agents.tts", extract, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentOutput {
    Audio(SegmentAudio),
    Words(SegmentWords),
    Done(SegmentDone),
}

/// `_core.TtsSegments`: one attempt's segments. The input as `encode` takes it (a chunk opens a
/// segment when none is open, a flush ends it), each segment's audio played in order into the
/// emitter (a later segment's waits for the ones before it) and completed once, and the read
/// deadline, run while a segment whose text ended is unanswered.
#[crate::host(class(name = "TtsSegments"))]
pub trait TtsSegments: Send + Sync + Sized + 'static {
    /// `label` names the provider in a protocol failure.
    #[host(new)]
    fn new(emitter: Arc<dyn PushedAudio>, deadline: Arc<dyn ReadDeadline>, label: String) -> Self;
    /// A chunk of the input.
    #[host(call)]
    fn push(&self, chunk: SpeechChunk) -> Result<(), SegmentError>;
    /// A flush: the open segment's text ended.
    #[host(call)]
    fn flush(&self);
    /// The next input `encode` takes; `None` while none is queued.
    #[host(call)]
    fn next(&self) -> Result<Option<SegmentInput>, SegmentError>;
    /// What `on_message` returned for a segment.
    #[host(call)]
    fn deliver(&self, output: SegmentOutput) -> Result<(), SegmentError>;
    /// Whether `segment_id` is a segment not yet done: a multiplexed stream's context.
    #[host(call)]
    fn holds(&self, segment_id: String) -> bool;
    /// The segments not yet done, in order.
    #[host(getter)]
    fn open(&self) -> Vec<String>;
    /// Waits until every segment is done.
    #[host(await)]
    async fn settled(&self);
}

/// Why a segment call failed: a report that breaks the protocol, or the emitter's failure.
#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum SegmentError {
    #[error(transparent)]
    #[host(delegate)]
    Protocol(#[from] ProviderError),
    #[error(transparent)]
    #[host(delegate)]
    Emitter(#[from] crate::media::emitter::EmitterError),
}
