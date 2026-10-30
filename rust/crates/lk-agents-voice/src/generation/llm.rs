//! Generated words, streaming segments and per-request model observations.

use crate::transcription::Visible;
use crate::transcription::VisibleStream;
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::{Extra, LlmChunk};
use lk_agents_core::telemetry::provider::{Context, LlmMetricState};

pub enum WordEffect {
    Chunk(Visible),
    End,
}

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
    visible: VisibleStream,
}

impl LlmGenerationData {
    pub fn new(started_at: f64) -> Self {
        Self {
            tps: None,
            started_at,
            metrics: LlmMetricState::new(started_at),
            extra: Extra::new(),
            produced: false,
            called: false,
            open: false,
            output_tokens: None,
            window: None,
            visible: VisibleStream::new(),
        }
    }

    /// Calls have already been dispatched; the same chunk's words precede its flush sentinel.
    pub fn chunk(&mut self, chunk: &LlmChunk, now: f64) -> Vec<WordEffect> {
        let mut effects = Vec::new();
        self.produced |= chunk.has_response();
        if let Some(delta) = &chunk.delta {
            if let Some(extra) = &delta.extra {
                self.extra.extend(extra.clone());
            }
            if let Some(text) = delta.content.as_ref().filter(|text| !text.is_empty()) {
                self.window = Some((self.window.map_or(now, |(first, _)| first), now));
                self.open = true;
                effects.push(WordEffect::Chunk(self.visible.push(text)));
            }
        }
        if let Some(usage) = &chunk.usage {
            self.output_tokens = Some(usage.output_tokens);
        }
        self.called |= !chunk.node_text;
        self.metrics.observe(chunk, || now);
        if chunk.flush {
            self.end_segment(&mut effects);
        }
        effects
    }

    /// End an open segment on every exit path; a one-chunk reply has no streaming rate.
    pub fn finish_stream(&mut self) -> Vec<WordEffect> {
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

    fn end_segment(&mut self, effects: &mut Vec<WordEffect>) {
        if !std::mem::replace(&mut self.open, false) {
            return;
        }
        let tail = std::mem::replace(&mut self.visible, VisibleStream::new()).flush();
        if !tail.text.is_empty() {
            effects.push(WordEffect::Chunk(tail));
        }
        effects.push(WordEffect::End);
    }

    /// A clean request emits metrics only when the model, rather than just a node, yielded.
    pub fn collect_metrics(
        &self,
        failed: bool,
        now: f64,
        context: Context,
        speech_id: &str,
    ) -> Option<LlmMetrics> {
        if failed || !self.called {
            return None;
        }
        let mut record = self.metrics.finish(false, || (now, context))?;
        record.speech_id = Some(speech_id.to_owned());
        Some(record)
    }
}
