use crate::generation::SpeechSegmentEvent;
use crate::transcription::{SegmentText, SpeechViews};
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::metrics::base::Metadata;
use lk_agents_abi::{Extra, LlmChunk};
use lk_agents_core::metrics::provider::LlmMetricState;

pub enum LlmRead {
    End,
    Provider,
    CancellableProvider,
}

/// An unsafe host generator is never raced with cancellation.
pub fn before_read(interrupted: bool, output_closed: bool, cancel_safe: bool) -> LlmRead {
    if interrupted || output_closed {
        LlmRead::End
    } else if cancel_safe {
        LlmRead::CancellableProvider
    } else {
        LlmRead::Provider
    }
}

/// One request's words and the metric fold shared with the provider boundary.
pub struct LlmGenerationData {
    pub tps: Option<f64>,
    pub started_at: f64,
    pub metrics: LlmMetricState,
    pub extra: Extra,
    pub produced: bool,
    called: bool,
    open: bool,
    output_tokens: Option<u64>,
    window: Option<(f64, f64)>,
    views: SpeechViews,
    segment: SegmentText,
}

impl LlmGenerationData {
    /// `label` and `metadata`: the LLM's, stamped on the record. `views`: how the voice reads it.
    pub fn new(
        started_at: f64,
        label: String,
        metadata: Option<Metadata>,
        views: SpeechViews,
    ) -> Self {
        Self {
            tps: None,
            started_at,
            metrics: LlmMetricState::new(started_at, label, metadata),
            extra: Extra::new(),
            produced: false,
            called: false,
            open: false,
            output_tokens: None,
            window: None,
            segment: SegmentText::new(views.clone()),
            views,
        }
    }

    /// Calls have already been dispatched; the same chunk's words precede its flush sentinel or
    /// its calls, either of which ends the segment.
    /// `at` (seconds on the session's clock) times the streaming window; `wall` is read only for
    /// the first response.
    pub fn chunk(
        &mut self,
        chunk: &LlmChunk,
        at: f64,
        wall: impl FnOnce() -> f64,
    ) -> Vec<SpeechSegmentEvent> {
        let mut effects = Vec::new();
        self.produced |= chunk.has_response();
        if let Some(delta) = &chunk.delta {
            if let Some(extra) = &delta.extra {
                self.extra.extend(extra.clone());
            }
            if let Some(text) = delta.content.as_ref().filter(|text| !text.is_empty()) {
                self.window = Some((self.window.map_or(at, |(first, _)| first), at));
                self.open = true;
                effects.push(SpeechSegmentEvent::Chunk(self.segment.push(text)));
            }
        }
        if let Some(usage) = &chunk.usage {
            self.output_tokens = Some(usage.output_tokens);
        }
        self.called |= !chunk.node_text;
        self.metrics.observe(chunk, wall);
        // A tool call ends the sentence before it: the words after it are a segment of their own.
        let tool_call = chunk.delta.as_ref().is_some_and(|delta| !delta.tool_calls.is_empty());
        if chunk.flush || tool_call {
            self.end_segment(&mut effects);
        }
        effects
    }

    /// End an open segment on every exit path; a one-chunk reply has no streaming rate.
    pub fn finish_stream(&mut self) -> Vec<SpeechSegmentEvent> {
        let mut effects = Vec::new();
        self.end_segment(&mut effects);
        self.tps = match (self.output_tokens, self.window) {
            (Some(tokens), Some((first, last))) if last > first => {
                Some(tokens as f64 / (last - first))
            }
            _ => None,
        };
        effects
    }

    fn end_segment(&mut self, effects: &mut Vec<SpeechSegmentEvent>) {
        if !std::mem::replace(&mut self.open, false) {
            return;
        }
        let fresh = SegmentText::new(self.views.clone());
        let tail = std::mem::replace(&mut self.segment, fresh).finish();
        if !tail.is_blank() {
            effects.push(SpeechSegmentEvent::Chunk(tail));
        }
        effects.push(SpeechSegmentEvent::End);
    }

    /// A clean request emits metrics only when the model, rather than just a node, yielded.
    pub fn collect_metrics(
        &self,
        failed: bool,
        now: f64,
        cancelled: bool,
        speech_id: &str,
    ) -> Option<LlmMetrics> {
        if failed || !self.called {
            return None;
        }
        let mut record = self.metrics.finish(false, now, now, cancelled)?;
        record.speech_id = Some(speech_id.to_owned());
        Some(record)
    }
}
