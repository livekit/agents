//! Generated words, streaming segments and per-request model observations.

use super::{Attempt, Retry};
use crate::trace::ProviderKind;
use crate::transcription::Visible;
use crate::transcription::VisibleStream;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::telemetry::provider::{Context, LlmMetricState};
use lk_agents_abi::{Extra, LlmChunk, Ts};

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
    completion_tokens: Option<u32>,
    window: Option<(f64, f64)>,
    visible: VisibleStream,
}

impl LlmGenerationData {
    pub fn new(started_at: f64) -> Self {
        Self {
            tps: None,
            started_at,
            metrics: LlmMetricState::for_host(started_at),
            extra: Extra::new(),
            produced: false,
            called: false,
            open: false,
            completion_tokens: None,
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
            self.completion_tokens = Some(usage.completion_tokens);
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
        self.tps = match (self.completion_tokens, self.window) {
            (Some(tokens), Some((first, last))) if last > first => {
                Some(f64::from(tokens) / (last - first))
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

pub struct RetryWait {
    pub failed_attempt: u32,
    pub at: Ts,
}

/// A failed attempt's retry eligibility and absolute wake time.
pub fn retry_after(
    attempt: &mut Attempt,
    leg: ProviderKind,
    error: &ProviderError,
    now: Ts,
) -> Option<RetryWait> {
    let kind = match leg {
        ProviderKind::Llm => lk_agents_abi::types::retry::RequestKind::Llm,
        ProviderKind::Tts => lk_agents_abi::types::retry::RequestKind::TtsStream,
    };
    match attempt.failed(error, kind) {
        Retry::After { attempt, after } => {
            Some(RetryWait { failed_attempt: attempt, at: now.saturating_add(after) })
        }
        Retry::Never => None,
    }
}
