//! Model request spans (docs/internals/telemetry-reference.md "Span trees"): one `llm_request` or
//! `tts_request` per request wherever it is made, an `llm_request_run` or `tts_request_run` per
//! attempt, and the request's usage, timing and 1.8.10 metrics summary recorded as it ends. The
//! provider's `request_span` names what its request records as: an LLM fallback adapter records its
//! own spans and a registered node makes its own request, so the caller opens none and calls it
//! under its own context instead. A request that failed ends failed, also when it is dropped.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::{LlmMetrics, Metadata, TtsMetrics};
use lk_agents_abi::telemetry::decl::{Content, Json, Label, Measured, Seconds};
use lk_agents_abi::telemetry::instruments::{
    GenAiInputTokens, GenAiLabels, GenAiOperationDuration, GenAiOutputTokens, ModelLabels,
    TokenModality,
};
use lk_agents_abi::telemetry::spans::agent_turn::ResponseFunctionCall;
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
pub(crate) use lk_agents_abi::telemetry::spans::common::OutputType;
use lk_agents_abi::telemetry::spans::inference::{
    AttemptEnd, GenAiUsage, LlmMetricsSummary, LlmRequest, LlmRequestEnd, TtsFallbackAdapter,
    TtsMetricsSummary, TtsRequest, TtsRequestEnd, TtsStreamAdapter, TtsTarget, UtcInstant,
};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{ChatRequest, Llm, LlmChunk, ProviderId, SynthesizedAudio, Tts};
use lk_agents_core::metrics::provider::LlmMetricState;
use lk_agents_core::telemetry::gen_ai;

// What the crate's logic names, spelled `telemetry::` there (`telemetry.md`, rule 4).
pub(crate) use lk_agents_abi::telemetry::decl::GenAiContent;
pub(crate) use lk_agents_abi::telemetry::instruments::DuplexOutputLag;
pub use lk_agents_abi::telemetry::spans::inference::RequestSpan;
pub(crate) use lk_agents_abi::telemetry::spans::inference::{
    Attempt, LlmFallbackAdapter, LlmFallbackChoice, LlmRequestRun, TtsRequestRun,
};
pub(crate) use lk_agents_abi::telemetry::spans::text::{
    JudgeEvaluation, JudgeEvaluationEnd, JudgeVerdict,
};
pub(crate) use lk_agents_net::telemetry::{
    Cx, ErrorType, Failed, Failure, ScopedSpan, end, measure, start,
};

/// The GenAI registry's spelling of a provider, `unknown` when it has none.
pub fn provider_name(provider: &str) -> String {
    gen_ai::provider_name(provider).unwrap_or_else(|| "unknown".to_owned())
}

/// A model's `M` labels: its provider as the GenAI registry spells it, and its name.
pub fn model_labels(id: &ProviderId) -> ModelLabels {
    ModelLabels::of(Some(&provider_name(&id.provider)), Some(&id.model))
}

fn seconds(from: u64, to: u64) -> f64 {
    to.saturating_sub(from) as f64 / 1e9
}

fn captured<T>(values: Vec<T>) -> Option<GenAiContent<Json<Vec<T>>>> {
    (!values.is_empty()).then(|| GenAiContent(Json(values)))
}

/// One LLM request, retries included: its span and what its chunks reported, folded once into
/// its metrics.
pub struct LlmTrace {
    /// `None` for an LLM that records its own spans (a fallback adapter) or makes no request (a
    /// registered `llm_node`).
    span: Option<ScopedSpan<LlmRequest>>,
    cx: Cx,
    id: ProviderId,
    started: u64,
    fold: LlmMetricState,
    first_chunk: Option<u64>,
    response_id: Option<String>,
    usage: Option<TokenUsage>,
    /// Message capture: the response messages on the span.
    capture: bool,
    /// The response text and calls (id, name, arguments).
    text: String,
    calls: Vec<ResponseFunctionCall>,
    attempts: u32,
    /// Provider request ids of the attempt in flight.
    request_ids: Vec<String>,
    /// The request failed: its span ends with this error.
    failure: Option<ProviderError>,
}

impl LlmTrace {
    /// Opens the request's span under `cx`'s current span, with the request's messages when
    /// message capture is on. A fallback adapter or a registered `llm_node` opens none: its call
    /// runs under `cx` ([`LlmTrace::cx`]).
    pub fn start(cx: &Cx, llm: &dyn Llm, req: &ChatRequest) -> Self {
        LlmTrace::open(cx, llm, req, None)
    }

    /// A fallback adapter's recovery probe of `llm`.
    pub fn recovery(cx: &Cx, llm: &dyn Llm, req: &ChatRequest) -> Self {
        LlmTrace::open(cx, llm, req, Some(true))
    }

    fn open(cx: &Cx, llm: &dyn Llm, req: &ChatRequest, recovery: Option<bool>) -> Self {
        let id = llm.id();
        let capture = cx.capture_content();
        let span = (llm.request_span() == RequestSpan::Request).then(|| {
            let items = &req.chat_ctx.items;
            lk_agents_net::telemetry::start(
                cx,
                LlmRequest {
                    operation: GenAiOperation::Chat,
                    provider: provider_name(&id.provider),
                    model: id.model.clone(),
                    stream: true,
                    output_type: OutputType::Text,
                    system_instructions: captured(gen_ai::system_instructions(items))
                        .filter(|_| capture),
                    input_messages: captured(gen_ai::input_messages(items)).filter(|_| capture),
                    tool_definitions: captured(gen_ai::tool_definitions(&req.tools))
                        .filter(|_| capture),
                    recovery,
                },
            )
        });
        let cx = span.as_ref().map_or_else(|| cx.clone(), |span| span.cx().clone());
        let started = cx.now_nanos();
        LlmTrace {
            span,
            fold: LlmMetricState::new(
                started as f64 / 1e9,
                id.label.clone(),
                Some(Metadata {
                    model_name: Some(id.model.clone()),
                    model_provider: Some(id.provider.clone()),
                }),
            ),
            cx,
            id,
            started,
            first_chunk: None,
            response_id: None,
            usage: None,
            capture,
            text: String::new(),
            calls: Vec::new(),
            attempts: 0,
            request_ids: Vec::new(),
            failure: None,
        }
    }

    /// The request's context: what its provider call runs under.
    pub fn cx(&self) -> &Cx {
        &self.cx
    }

    /// Opens the next attempt's span; `None` when the LLM records its own.
    pub fn attempt(&mut self) -> Option<ScopedSpan<LlmRequestRun>> {
        let retry_count = self.attempts;
        self.attempts += 1;
        self.request_ids.clear();
        let span = self.span.as_ref()?;
        Some(lk_agents_net::telemetry::start(
            span.cx(),
            LlmRequestRun {
                attempt: Attempt {
                    provider: provider_name(&self.id.provider),
                    model: self.id.model.clone(),
                    retry_count,
                },
            },
        ))
    }

    /// Ends an attempt's span: its provider request ids, or the failure that ended it.
    pub fn end_attempt(
        &mut self,
        span: Option<ScopedSpan<LlmRequestRun>>,
        failed: Option<&ProviderError>,
    ) {
        let Some(span) = span else { return };
        let end = AttemptEnd::new(std::mem::take(&mut self.request_ids));
        if let Some(error) = failed {
            lk_agents_net::telemetry::end(span, Failed { end, error });
        } else {
            lk_agents_net::telemetry::end(span, end);
        }
    }

    /// Records what a chunk reports: the first chunk's time, ids, usage and the response.
    pub fn observe(&mut self, chunk: &LlmChunk) {
        let now = self.cx.now_nanos();
        if self.fold.observe(chunk, || now as f64 / 1e9) {
            self.first_chunk = Some(now);
        }
        if !chunk.id.is_empty() {
            if self.response_id.is_none() {
                self.response_id = Some(chunk.id.clone());
            }
            if !self.request_ids.contains(&chunk.id) {
                self.request_ids.push(chunk.id.clone());
            }
        }
        if let Some(usage) = &chunk.usage {
            self.usage = Some((**usage).clone());
        }
        if let Some(delta) = &chunk.delta {
            if let Some(text) = &delta.content {
                self.text.push_str(text);
            }
            self.calls.extend(delta.tool_calls.iter().map(|c| ResponseFunctionCall {
                call_id: c.call_id.clone(),
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            }));
        }
    }

    /// The request failed: it ends with `error`.
    pub fn fail(&mut self, error: &ProviderError) {
        self.failure = Some(error.clone());
    }

    /// When the first chunk with a response arrived (unix nanoseconds).
    pub fn first_chunk_at(&self) -> Option<u64> {
        self.first_chunk
    }

    /// The response's text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The response's function calls so far.
    pub fn calls(&self) -> &[ResponseFunctionCall] {
        &self.calls
    }

    /// Ends the request with its usage, timing and metrics, records its measurements, and returns
    /// its metrics. `interrupted`: the reader stopped before the response ended.
    pub fn end(mut self, interrupted: bool, speech_id: Option<&str>) -> Option<LlmMetrics> {
        let now = self.cx.now_nanos();
        let at = now as f64 / 1e9;
        let failed = self.failure.is_some();
        let metrics = self.fold.finish(failed, at, at, interrupted).map(|mut m| {
            m.speech_id = speech_id.map(str::to_owned);
            m
        });
        let Some(span) = self.span.take() else { return metrics };
        let cx = span.cx();
        let usage = self.usage.clone().unwrap_or_default();
        let provider = provider_name(&self.id.provider);
        let labels = || GenAiLabels {
            operation: GenAiOperation::Chat,
            provider: Label(provider.clone()),
            model: Label(self.id.model.clone()),
            response_model: Some(Label(self.id.model.clone())),
        };
        lk_agents_net::telemetry::measure(
            cx,
            &GenAiOperationDuration { operation: labels() },
            seconds(self.started, now),
        );
        let input = [usage.input_text_tokens, usage.input_image_tokens, usage.input_audio_tokens];
        for (modality, count) in by_modality(usage.input_tokens, input) {
            let labels = GenAiInputTokens { operation: labels(), modality };
            lk_agents_net::telemetry::measure(cx, &labels, count);
        }
        let output =
            [usage.output_text_tokens, usage.output_image_tokens, usage.output_audio_tokens];
        for (modality, count) in by_modality(usage.output_tokens, output) {
            let labels = GenAiOutputTokens { operation: labels(), modality };
            lk_agents_net::telemetry::measure(cx, &labels, count);
        }
        let summary = metrics.as_ref().map(|m| Json(LlmMetricsSummary::from(m)));
        let finish = gen_ai::finish_reason(!self.calls.is_empty(), interrupted);
        let output = self
            .capture
            .then(|| captured(gen_ai::output_messages(&self.text, &self.calls, finish)))
            .flatten();
        let positive = |n: Option<u64>| n.filter(|n| *n > 0);
        let end = LlmRequestEnd {
            metrics: summary,
            usage: GenAiUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cache_read_input_tokens: positive(usage.cache_read_tokens),
                cache_write_input_tokens: positive(usage.cache_write_tokens),
                reasoning_output_tokens: positive(usage.output_reasoning_tokens),
            },
            response_id: self.response_id.take(),
            response_model: Some(self.id.model.clone()),
            finish_reasons: vec![finish],
            time_to_first_chunk: Measured(
                self.first_chunk.map(|at| Seconds(seconds(self.started, at))),
            ),
            output_messages: output,
            completion_start_time: self
                .first_chunk
                .map(|at| Json(UtcInstant { unix_seconds: at as f64 / 1e9 })),
        };
        if let Some(error) = &self.failure {
            lk_agents_net::telemetry::end(span, Failed { end, error });
        } else {
            lk_agents_net::telemetry::end(span, end);
        }
        metrics
    }
}

/// A request dropped after it failed ends failed; otherwise its span ends as cancelled.
impl Drop for LlmTrace {
    fn drop(&mut self) {
        if let (Some(span), Some(error)) = (self.span.take(), &self.failure) {
            lk_agents_net::telemetry::end(span, error);
        }
    }
}

/// One TTS request (a segment's synthesis), retries included. A request to an adapter is named
/// for it (`tts_fallback_adapter`, `tts_stream_adapter`), as 1.8.10 named it; the adapter's own
/// requests open under its attempts. A registered node makes its own request: none opens.
pub struct TtsTrace {
    /// `None` once ended, or for a registered node.
    span: Option<TtsRequestSpan>,
    /// The `cx` of its span (the caller's for a node), kept past its end for calls made after it.
    cx: Cx,
    id: ProviderId,
    attempts: u32,
    request_ids: Vec<String>,
    /// The request failed: its span ends with this error.
    failure: Option<ProviderError>,
}

enum TtsRequestSpan {
    Request(ScopedSpan<TtsRequest>),
    Fallback(ScopedSpan<TtsFallbackAdapter>),
    Stream(ScopedSpan<TtsStreamAdapter>),
}

impl TtsRequestSpan {
    fn cx(&self) -> &Cx {
        match self {
            TtsRequestSpan::Request(span) => span.cx(),
            TtsRequestSpan::Fallback(span) => span.cx(),
            TtsRequestSpan::Stream(span) => span.cx(),
        }
    }

    /// Ends the span: with `end` unless it failed, with `failure` when it did.
    fn end(self, end: Option<TtsRequestEnd>, failure: Option<&ProviderError>) {
        fn finish<D>(
            span: ScopedSpan<D>,
            end: Option<TtsRequestEnd>,
            failure: Option<&ProviderError>,
        ) where
            D: lk_agents_abi::telemetry::decl::SpanDecl<End = TtsRequestEnd>,
            TtsRequestEnd: lk_agents_net::telemetry::SpanEnd<D>,
        {
            match (end, failure) {
                (Some(end), Some(error)) => {
                    lk_agents_net::telemetry::end(span, Failed { end, error });
                }
                (Some(end), None) => {
                    lk_agents_net::telemetry::end(span, end);
                }
                (None, Some(error)) => {
                    lk_agents_net::telemetry::end(span, error);
                }
                (None, None) => drop(span),
            }
        }
        match self {
            TtsRequestSpan::Request(span) => finish(span, end, failure),
            TtsRequestSpan::Fallback(span) => finish(span, end, failure),
            TtsRequestSpan::Stream(span) => finish(span, end, failure),
        }
    }
}

impl TtsTrace {
    /// Opens the request's span under `cx`'s current span.
    pub fn start(cx: &Cx, tts: &dyn Tts, streaming: bool) -> Self {
        let id = tts.id();
        let target = TtsTarget {
            streaming,
            label: tts.label(),
            provider: provider_name(&id.provider),
            model: id.model.clone(),
        };
        use lk_agents_net::telemetry::start;
        let span = match tts.request_span() {
            RequestSpan::Request => {
                Some(TtsRequestSpan::Request(start(cx, TtsRequest { target, recovery: None })))
            }
            RequestSpan::FallbackAdapter => {
                Some(TtsRequestSpan::Fallback(start(cx, TtsFallbackAdapter { target })))
            }
            RequestSpan::StreamAdapter => {
                Some(TtsRequestSpan::Stream(start(cx, TtsStreamAdapter { target })))
            }
            RequestSpan::Node => None,
        };
        let cx = span.as_ref().map_or(cx, TtsRequestSpan::cx).clone();
        TtsTrace { span, cx, id, attempts: 0, request_ids: Vec::new(), failure: None }
    }

    pub fn cx(&self) -> &Cx {
        &self.cx
    }

    /// Opens the next attempt's span, ended with [`TtsTrace::end_attempt`]; `None` for a node.
    pub fn attempt(&mut self) -> Option<ScopedSpan<TtsRequestRun>> {
        let retry_count = self.attempts;
        self.attempts += 1;
        self.request_ids.clear();
        self.span.as_ref()?;
        Some(lk_agents_net::telemetry::start(
            &self.cx,
            TtsRequestRun {
                attempt: Attempt {
                    provider: provider_name(&self.id.provider),
                    model: self.id.model.clone(),
                    retry_count,
                },
            },
        ))
    }

    /// A frame's provider request id, for the attempt in flight.
    pub fn observe(&mut self, audio: &SynthesizedAudio) {
        self.request_id(audio.request_id.clone());
        audio.provider_request_ids.iter().for_each(|id| self.request_id(id.clone()));
    }

    /// A provider request id of the attempt in flight.
    pub fn request_id(&mut self, id: String) {
        if !id.is_empty() && !self.request_ids.contains(&id) {
            self.request_ids.push(id);
        }
    }

    pub fn end_attempt(
        &mut self,
        span: Option<ScopedSpan<TtsRequestRun>>,
        failed: Option<&ProviderError>,
    ) {
        let Some(span) = span else { return };
        let end = AttemptEnd::new(std::mem::take(&mut self.request_ids));
        if let Some(error) = failed {
            lk_agents_net::telemetry::end(span, Failed { end, error });
        } else {
            lk_agents_net::telemetry::end(span, end);
        }
    }

    /// The request failed: it ends with `error`.
    pub fn fail(&mut self, error: &ProviderError) {
        self.failure = Some(error.clone());
    }

    /// Ends the request with the text synthesized and its metrics.
    pub fn end(mut self, text: String, metrics: Option<&TtsMetrics>) {
        let end = TtsRequestEnd {
            input_text: Some(Content(text)),
            metrics: metrics.map(|m| Json(TtsMetricsSummary::from(m))),
        };
        if let Some(span) = self.span.take() {
            span.end(Some(end), self.failure.as_ref());
        }
    }
}

/// A request dropped after it failed ends failed; otherwise its span ends as cancelled.
impl Drop for TtsTrace {
    fn drop(&mut self) {
        if let Some(span) = self.span.take() {
            span.end(None, self.failure.as_ref());
        }
    }
}

/// `total` tokens split as the GenAI registry asks: each reported modality, the rest (or a `0`
/// when no usage arrived) under `unknown`.
fn by_modality(
    total: u64,
    [text, image, audio]: [Option<u64>; 3],
) -> impl Iterator<Item = (TokenModality, u64)> {
    let known =
        [(TokenModality::Text, text), (TokenModality::Image, image), (TokenModality::Audio, audio)];
    let sum: u64 = known.iter().filter_map(|(_, n)| *n).sum();
    let rest =
        (sum == 0 || sum < total).then(|| (TokenModality::Unknown, total.saturating_sub(sum)));
    known.into_iter().filter_map(|(m, n)| n.filter(|n| *n > 0).map(|n| (m, n))).chain(rest)
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod tests;
