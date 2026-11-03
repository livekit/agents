//! Model request spans (docs/internals/telemetry-reference.md "Span trees"): one `llm_request` per
//! request wherever it is made, an `llm_request_run` per attempt, and the request's usage and
//! timing recorded as it ends.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::telemetry::decl::{Label, Measured, Seconds};
use lk_agents_abi::telemetry::instruments::{
    GenAiLabels, GenAiOperationDuration, GenAiTokenUsage, LlmInputCachedTokens, LlmInputTokens,
    LlmOutputTokens, ModelLabels, TokenType,
};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::inference::{
    Attempt, AttemptEnd, GenAiUsage, LlmRequest, LlmRequestEnd, LlmRequestRun,
};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{LlmChunk, ProviderId};
use lk_agents_net::telemetry::{Cx, ScopedSpan};

/// One LLM request, retries included: its span and what its chunks reported.
pub struct LlmTrace {
    span: ScopedSpan<LlmRequest>,
    id: ProviderId,
    started: u64,
    /// Seconds to the first chunk with a response.
    ttft: Option<f64>,
    usage: Option<TokenUsage>,
    response_id: Option<String>,
    attempts: u32,
}

impl LlmTrace {
    /// Opens the request's span under `cx`'s current span.
    pub fn start(cx: &Cx, id: ProviderId) -> Self {
        let span = cx.start_span(LlmRequest {
            operation: GenAiOperation::Chat,
            provider: id.provider.clone(),
            model: id.model.clone(),
            stream: true,
            output_type: "text".to_owned(),
            system_instructions: None,
            input_messages: None,
            tool_definitions: None,
            input: None,
            recovery: None,
        });
        let started = cx.tracer().now_nanos();
        LlmTrace { span, id, started, ttft: None, usage: None, response_id: None, attempts: 0 }
    }

    /// The request's context: what its provider call runs under.
    pub fn cx(&self) -> &Cx {
        self.span.cx()
    }

    /// Opens the next attempt's span; the caller ends it with the attempt.
    pub fn attempt(&mut self) -> ScopedSpan<LlmRequestRun> {
        let retry_count = self.attempts;
        self.attempts += 1;
        self.span.cx().start_span(LlmRequestRun {
            attempt: Attempt {
                provider: self.id.provider.clone(),
                model: self.id.model.clone(),
                retry_count,
            },
        })
    }

    /// Records what a chunk reports: the first chunk's time, the response id and the usage.
    pub fn observe(&mut self, chunk: &LlmChunk) {
        if self.ttft.is_none() && chunk.has_response() {
            let now = self.span.cx().tracer().now_nanos();
            self.ttft = Some(now.saturating_sub(self.started) as f64 / 1e9);
        }
        if self.response_id.is_none() && !chunk.id.is_empty() {
            self.response_id = Some(chunk.id.clone());
        }
        if let Some(usage) = &chunk.usage {
            self.usage = Some((**usage).clone());
        }
    }

    /// Takes what a reader's metrics fold computed from the chunks, in place of [`observe`].
    ///
    /// [`observe`]: Self::observe
    pub fn metrics(&mut self, metrics: &LlmMetrics) {
        self.ttft = metrics.ttft;
        self.usage = Some(metrics.tokens.clone());
        self.response_id = Some(metrics.request_id.clone()).filter(|id| !id.is_empty());
    }

    pub fn fail(&self, error: &ProviderError) {
        self.span.fail(error);
    }

    /// Ends the request with its usage and timing, and records its measurements.
    pub fn end(self) {
        let cx = self.span.cx();
        let usage = self.usage.unwrap_or_default();
        let model = ModelLabels {
            provider: Label(self.id.provider.clone()),
            model: Label(self.id.model.clone()),
        };
        let labels = || GenAiLabels {
            operation: GenAiOperation::Chat,
            provider: Label(self.id.provider.clone()),
            model: Label(self.id.model.clone()),
            response_model: None,
        };
        let now = cx.tracer().now_nanos();
        let seconds = |from: u64, to: u64| to.saturating_sub(from) as f64 / 1e9;
        cx.measure(&GenAiOperationDuration { operation: labels() }, seconds(self.started, now));
        for (token_type, count) in
            [(TokenType::Input, usage.input_tokens), (TokenType::Output, usage.output_tokens)]
        {
            cx.measure(&GenAiTokenUsage { operation: labels(), token_type }, count);
        }
        cx.measure(&LlmInputTokens { model: clone(&model) }, usage.input_tokens);
        cx.measure(&LlmOutputTokens { model: clone(&model) }, usage.output_tokens);
        if let Some(cached) = usage.cache_read_tokens {
            cx.measure(&LlmInputCachedTokens { model }, cached);
        }
        let ttft = self.ttft.map(Seconds);
        self.span.end(LlmRequestEnd {
            legacy: None,
            usage: GenAiUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cache_read_input_tokens: usage.cache_read_tokens,
                input_cached_tokens: usage.cache_read_tokens,
                cache_write_input_tokens: usage.cache_write_tokens,
                reasoning_output_tokens: usage.output_reasoning_tokens,
                reasoning_tokens: usage.output_reasoning_tokens,
            },
            response_id: self.response_id,
            response_model: None,
            finish_reasons: Vec::new(),
            time_to_first_chunk: Measured(ttft),
            output_messages: None,
            completion_start_time: None,
        });
    }
}

fn clone(labels: &ModelLabels) -> ModelLabels {
    ModelLabels { provider: labels.provider.clone(), model: labels.model.clone() }
}

/// Ends an attempt's span: its provider request ids, or the failure that ended it.
pub fn end_attempt(span: ScopedSpan<LlmRequestRun>, failed: Option<&ProviderError>) {
    match failed {
        Some(error) => span.fail(error),
        None => span.end(AttemptEnd { provider_request_ids: Vec::new() }),
    }
}
