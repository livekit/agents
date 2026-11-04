//! Model request spans (docs/internals/telemetry-reference.md "Span trees"): one `llm_request` or
//! `tts_request` per request wherever it is made, an `llm_request_run` or `tts_request_run` per
//! attempt, and the request's usage, timing and 1.8.10 metrics summary recorded as it ends. A fallback
//! adapter records its own spans, and a registered `llm_node` makes its own request: the caller
//! opens none and hands it the context instead.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::{Metadata, TtsMetrics};
use lk_agents_abi::telemetry::decl::{Content, Json, Label, Measured, Seconds};
use lk_agents_abi::telemetry::instruments::{
    GenAiLabels, GenAiOperationDuration, GenAiTokenUsage, ModelLabels, TokenType,
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
pub(crate) use lk_agents_abi::telemetry::spans::inference::{
    Attempt, LlmFallbackAdapter, LlmFallbackChoice, LlmRequestRun, TtsRequestRun,
};
pub(crate) use lk_agents_abi::telemetry::spans::text::{
    JudgeEvaluation, JudgeEvaluationEnd, JudgeVerdict,
};
pub(crate) use lk_agents_net::telemetry::{
    Cx, ErrorType, Failure, ScopedSpan, Tracer, end_with, fail, start,
};

/// `llm.FallbackAdapter`'s label: an LLM that records its own `llm_fallback_adapter` spans.
pub const FALLBACK_LLM_LABEL: &str = "llm.FallbackAdapter";

/// `tts.FallbackAdapter`'s label: its request is a `tts_fallback_adapter`.
pub const FALLBACK_TTS_LABEL: &str = "tts.FallbackAdapter";

/// What a stream adapter's label starts with: its request is a `tts_stream_adapter`.
pub const STREAM_ADAPTER_LABEL: &str = "StreamAdapter[";

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

/// One LLM request, retries included: its span and what its chunks reported.
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
}

impl LlmTrace {
    /// Opens the request's span under `cx`'s current span, with the request's messages when
    /// message capture is on. A fallback adapter or a registered `llm_node` gets the context in
    /// `req` instead.
    pub fn start(cx: &Cx, llm: &dyn Llm, req: &mut ChatRequest) -> Self {
        LlmTrace::open(cx, llm, req, None)
    }

    /// A fallback adapter's recovery probe of `llm`.
    pub fn recovery(cx: &Cx, llm: &dyn Llm, req: &mut ChatRequest) -> Self {
        LlmTrace::open(cx, llm, req, Some(true))
    }

    fn open(cx: &Cx, llm: &dyn Llm, req: &mut ChatRequest, recovery: Option<bool>) -> Self {
        let id = llm.id();
        let capture = cx.capture_content();
        let span = if llm.is_node() || llm.label() == FALLBACK_LLM_LABEL {
            req.trace = Some(cx.trace_context());
            None
        } else {
            let items = &req.chat_ctx.items;
            Some(lk_agents_net::telemetry::start(
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
                    input: None,
                    recovery,
                },
            ))
        };
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
        if let Some(error) = failed {
            lk_agents_net::telemetry::fail(&span, error);
        }
        lk_agents_net::telemetry::end_with(
            span,
            AttemptEnd::new(std::mem::take(&mut self.request_ids)),
        );
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

    pub fn fail(&self, error: &ProviderError) {
        if let Some(span) = &self.span {
            lk_agents_net::telemetry::fail(span, error);
        }
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

    /// Ends the request with its usage, timing and metrics, and records its measurements.
    /// `interrupted`: the reader stopped before the response ended.
    pub fn end(self, interrupted: bool, speech_id: Option<&str>) {
        let Some(span) = self.span else { return };
        let cx = span.cx();
        let usage = self.usage.clone().unwrap_or_default();
        let provider = provider_name(&self.id.provider);
        let labels = || GenAiLabels {
            operation: GenAiOperation::Chat,
            provider: Label(provider.clone()),
            model: Label(self.id.model.clone()),
            response_model: Some(Label(self.id.model.clone())),
        };
        let now = cx.now_nanos();
        lk_agents_net::telemetry::measure(
            cx,
            &GenAiOperationDuration { operation: labels() },
            seconds(self.started, now),
        );
        for (token_type, count) in
            [(TokenType::Input, usage.input_tokens), (TokenType::Output, usage.output_tokens)]
        {
            lk_agents_net::telemetry::measure(
                cx,
                &GenAiTokenUsage { operation: labels(), token_type },
                count,
            );
        }
        let now = now as f64 / 1e9;
        let metrics = self.fold.finish(false, now, now, interrupted);
        let summary = metrics.map(|mut m| {
            m.speech_id = speech_id.map(str::to_owned);
            Json(LlmMetricsSummary::from(&m))
        });
        let finish = gen_ai::finish_reason(!self.calls.is_empty(), interrupted);
        let output = self
            .capture
            .then(|| captured(gen_ai::output_messages(&self.text, &self.calls, finish)))
            .flatten();
        let positive = |n: Option<u64>| n.filter(|n| *n > 0);
        lk_agents_net::telemetry::end_with(
            span,
            LlmRequestEnd {
                metrics: summary,
                usage: GenAiUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_read_input_tokens: positive(usage.cache_read_tokens),
                    cache_write_input_tokens: positive(usage.cache_write_tokens),
                    reasoning_output_tokens: positive(usage.output_reasoning_tokens),
                },
                response_id: self.response_id,
                response_model: Some(self.id.model.clone()),
                finish_reasons: vec![finish],
                time_to_first_chunk: Measured(
                    self.first_chunk.map(|at| Seconds(seconds(self.started, at))),
                ),
                output_messages: output,
                completion_start_time: self
                    .first_chunk
                    .map(|at| Json(UtcInstant { unix_seconds: at as f64 / 1e9 })),
            },
        );
    }
}

/// One TTS request (a segment's synthesis), retries included. A request to an adapter is named
/// for it (`tts_fallback_adapter`, `tts_stream_adapter`), as 1.8.10 named it; the adapter's own
/// requests open under its attempts.
pub struct TtsTrace {
    span: TtsRequestSpan,
    id: ProviderId,
    attempts: u32,
    request_ids: Vec<String>,
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
}

impl TtsTrace {
    /// Opens the request's span under `cx`'s current span.
    pub fn start(cx: &Cx, tts: &dyn Tts, streaming: bool) -> Self {
        let id = tts.id();
        let label = tts.label();
        let target = TtsTarget {
            streaming,
            label: label.clone(),
            provider: provider_name(&id.provider),
            model: id.model.clone(),
        };
        let span = if label == FALLBACK_TTS_LABEL {
            TtsRequestSpan::Fallback(lk_agents_net::telemetry::start(
                cx,
                TtsFallbackAdapter { target },
            ))
        } else if label.starts_with(STREAM_ADAPTER_LABEL) {
            TtsRequestSpan::Stream(lk_agents_net::telemetry::start(cx, TtsStreamAdapter { target }))
        } else {
            TtsRequestSpan::Request(lk_agents_net::telemetry::start(
                cx,
                TtsRequest { target, recovery: None },
            ))
        };
        TtsTrace { span, id, attempts: 0, request_ids: Vec::new() }
    }

    pub fn cx(&self) -> &Cx {
        self.span.cx()
    }

    /// Opens the next attempt's span; the caller ends it with [`TtsTrace::end_attempt`].
    pub fn attempt(&mut self) -> ScopedSpan<TtsRequestRun> {
        let retry_count = self.attempts;
        self.attempts += 1;
        self.request_ids.clear();
        lk_agents_net::telemetry::start(
            self.cx(),
            TtsRequestRun {
                attempt: Attempt {
                    provider: provider_name(&self.id.provider),
                    model: self.id.model.clone(),
                    retry_count,
                },
            },
        )
    }

    /// A frame's provider request id, for the attempt in flight.
    pub fn observe(&mut self, audio: &SynthesizedAudio) {
        self.request_id(audio.request_id.clone());
    }

    /// A provider request id of the attempt in flight.
    pub fn request_id(&mut self, id: String) {
        if !id.is_empty() && !self.request_ids.contains(&id) {
            self.request_ids.push(id);
        }
    }

    pub fn end_attempt(&mut self, span: ScopedSpan<TtsRequestRun>, failed: Option<&ProviderError>) {
        if let Some(error) = failed {
            lk_agents_net::telemetry::fail(&span, error);
        }
        lk_agents_net::telemetry::end_with(
            span,
            AttemptEnd::new(std::mem::take(&mut self.request_ids)),
        );
    }

    pub fn fail(&self, error: &ProviderError) {
        match &self.span {
            TtsRequestSpan::Request(span) => lk_agents_net::telemetry::fail(span, error),
            TtsRequestSpan::Fallback(span) => lk_agents_net::telemetry::fail(span, error),
            TtsRequestSpan::Stream(span) => lk_agents_net::telemetry::fail(span, error),
        }
    }

    /// Ends the request with the text synthesized and its metrics.
    pub fn end(self, text: String, metrics: Option<&TtsMetrics>) {
        let end = TtsRequestEnd {
            input_text: Some(Content(text)),
            metrics: metrics.map(|m| Json(TtsMetricsSummary::from(m))),
        };
        match self.span {
            TtsRequestSpan::Request(span) => lk_agents_net::telemetry::end_with(span, end),
            TtsRequestSpan::Fallback(span) => lk_agents_net::telemetry::end_with(span, end),
            TtsRequestSpan::Stream(span) => lk_agents_net::telemetry::end_with(span, end),
        }
    }
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod tests;
