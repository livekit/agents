//! usage.md's accounting: request starts, provider measurements and tool-body starts in, the
//! immutable [`Usage`] snapshot out. A run, a text-session turn and a voice session each own one
//! [`Ledger`]; the runtime only feeds it.

use std::collections::BTreeMap;

use lk_agents_abi::CompletionUsage;
use lk_agents_abi::metrics::base::{AgentMetrics, Metadata};
use lk_agents_abi::usage::{
    EotUsage, InterruptionUsage, LlmUsage, ModelUsage, SttUsage, TokenUsage, TtsUsage, Usage,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llm,
    Stt,
    Tts,
    Eot,
    Interruption,
}

/// What a provider reported for one request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Measurement {
    pub tokens: TokenUsage,
    pub connection_duration: Option<f64>,
    pub audio_duration: Option<f64>,
    pub characters_count: Option<u64>,
}

#[derive(Debug, Clone)]
struct Bucket {
    kind: ModelKind,
    provider: Option<String>,
    model: Option<String>,
    requests: u64,
    /// Requests closed without a complete measurement.
    unmeasured: u64,
    /// Every contribution so far; `None` before the first.
    measured: Option<Measurement>,
}

/// A started request awaiting its measurement; `partial` is the latest cumulative report.
#[derive(Debug, Clone)]
struct Pending {
    bucket: usize,
    partial: Option<Measurement>,
}

#[derive(Debug, Clone, Default)]
pub struct Ledger {
    buckets: Vec<Bucket>,
    pending: BTreeMap<u64, Pending>,
    tool_calls: u64,
    settled: bool,
    revision: u64,
}

impl Ledger {
    /// Changes on every update: a driver republishes the snapshot only when it moved.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// A framework-visible provider attempt started; it is unmeasured until [`finish`](Self::finish).
    pub fn start(&mut self, request: u64, kind: ModelKind, provider: &str, model: &str) {
        let bucket = self.bucket(kind, provider, model);
        self.buckets[bucket].requests += 1;
        self.pending.insert(request, Pending { bucket, partial: None });
        self.revision += 1;
    }

    /// A cumulative report for `request`: it replaces the previous one, never adds to it.
    pub fn report(&mut self, request: u64, measurement: Measurement) {
        if let Some(pending) = self.pending.get_mut(&request) {
            pending.partial = Some(measurement);
            self.revision += 1;
        }
    }

    /// `request` ended. Its last report counts as consumption either way; the request stays
    /// unmeasured unless it `completed` with a report.
    pub fn finish(&mut self, request: u64, completed: bool) {
        let Some(Pending { bucket, partial }) = self.pending.remove(&request) else { return };
        let bucket = &mut self.buckets[bucket];
        if !(completed && partial.is_some()) {
            bucket.unmeasured += 1;
        }
        if let Some(partial) = partial {
            fold(&mut bucket.measured, &partial);
        }
        self.revision += 1;
    }

    /// `requests` that started and were measured at once (a provider metrics record).
    pub fn measured(
        &mut self,
        kind: ModelKind,
        metadata: Option<&Metadata>,
        requests: u64,
        measurement: Measurement,
    ) {
        let (provider, model) = metadata
            .map(|m| (m.model_provider.as_deref(), m.model_name.as_deref()))
            .unwrap_or_default();
        let bucket = self.bucket(kind, provider.unwrap_or_default(), model.unwrap_or_default());
        let bucket = &mut self.buckets[bucket];
        bucket.requests += requests;
        fold(&mut bucket.measured, &measurement);
        self.revision += 1;
    }

    /// A provider metrics record: one measured request (EOT and interruption records count their
    /// own). VAD, end-of-utterance and avatar records are performance metrics, not usage.
    pub fn observe(&mut self, metrics: &AgentMetrics) {
        let some = |n: u32| (n != 0).then_some(u64::from(n));
        match metrics {
            AgentMetrics::Llm(m) => {
                let tokens = TokenUsage {
                    input_tokens: m.prompt_tokens.into(),
                    output_tokens: m.completion_tokens.into(),
                    cache_read_tokens: some(m.prompt_cached_tokens),
                    cache_write_tokens: some(m.cache_creation_tokens),
                    output_reasoning_tokens: some(m.reasoning_tokens),
                    ..TokenUsage::default()
                };
                self.measured(
                    ModelKind::Llm,
                    m.metadata.as_ref(),
                    1,
                    Measurement { tokens, ..Default::default() },
                );
            }
            AgentMetrics::RealtimeModel(m) => {
                let (input, output) = (&m.input_token_details, &m.output_token_details);
                let cached = input.cached_tokens_details;
                let tokens = TokenUsage {
                    input_tokens: m.input_tokens.into(),
                    output_tokens: m.output_tokens.into(),
                    cache_read_tokens: Some(input.cached_tokens.into()),
                    input_text_tokens: Some(input.text_tokens.into()),
                    input_audio_tokens: Some(input.audio_tokens.into()),
                    input_image_tokens: Some(input.image_tokens.into()),
                    output_text_tokens: Some(output.text_tokens.into()),
                    output_audio_tokens: Some(output.audio_tokens.into()),
                    output_image_tokens: some(output.image_tokens),
                    cache_read_text_tokens: cached.map(|c| c.text_tokens.into()),
                    cache_read_audio_tokens: cached.map(|c| c.audio_tokens.into()),
                    cache_read_image_tokens: cached.map(|c| c.image_tokens.into()),
                    ..TokenUsage::default()
                };
                let connection_duration = (m.session_duration > 0.0).then_some(m.session_duration);
                let measurement = Measurement { tokens, connection_duration, ..Default::default() };
                self.measured(ModelKind::Llm, m.metadata.as_ref(), 1, measurement);
            }
            AgentMetrics::Stt(m) => {
                let tokens = TokenUsage {
                    input_tokens: m.input_tokens.into(),
                    output_tokens: m.output_tokens.into(),
                    input_audio_tokens: some(m.input_audio_tokens),
                    ..TokenUsage::default()
                };
                let audio_duration = Some(m.audio_duration);
                let measurement = Measurement { tokens, audio_duration, ..Default::default() };
                self.measured(ModelKind::Stt, m.metadata.as_ref(), 1, measurement);
            }
            AgentMetrics::Tts(m) => {
                let tokens = TokenUsage {
                    input_tokens: m.input_tokens.into(),
                    output_tokens: m.output_tokens.into(),
                    ..TokenUsage::default()
                };
                let measurement = Measurement {
                    tokens,
                    audio_duration: Some(m.audio_duration),
                    characters_count: Some(m.characters_count.into()),
                    ..Default::default()
                };
                self.measured(ModelKind::Tts, m.metadata.as_ref(), 1, measurement);
            }
            AgentMetrics::EotInference(m) => {
                let n = m.num_requests.into();
                self.measured(ModelKind::Eot, m.metadata.as_ref(), n, Measurement::default());
            }
            AgentMetrics::Interruption(m) => {
                let n = m.num_requests.into();
                self.measured(
                    ModelKind::Interruption,
                    m.metadata.as_ref(),
                    n,
                    Measurement::default(),
                );
            }
            AgentMetrics::Vad(_) | AgentMetrics::Eou(_) | AgentMetrics::Avatar(_) => {}
        }
    }

    /// The operation finished local execution and accounting cleanup.
    pub fn settle(&mut self) {
        self.settled = true;
        self.revision += 1;
    }

    /// A tool body started, whatever its outcome.
    pub fn tool_started(&mut self) {
        self.tool_calls += 1;
        self.revision += 1;
    }

    /// Takes in an ended scope's ledger (a text-session invocation): its requests still awaiting a
    /// measurement stay unmeasured.
    pub fn absorb(&mut self, other: &Ledger) {
        let mut moved = BTreeMap::new();
        for (i, b) in other.buckets.iter().enumerate() {
            let at = self.bucket_by(b.kind, b.provider.clone(), b.model.clone());
            let bucket = &mut self.buckets[at];
            bucket.requests += b.requests;
            bucket.unmeasured += b.unmeasured;
            if let Some(m) = &b.measured {
                fold(&mut bucket.measured, m);
            }
            moved.insert(i, at);
        }
        for pending in other.pending.values() {
            let bucket = &mut self.buckets[moved[&pending.bucket]];
            bucket.unmeasured += 1;
            if let Some(partial) = &pending.partial {
                fold(&mut bucket.measured, partial);
            }
        }
        self.tool_calls += other.tool_calls;
        self.revision += 1;
    }

    /// Takes in an independent operation's snapshot.
    pub fn add(&mut self, usage: &Usage) {
        for m in &usage.models {
            let (kind, provider, model, requests, unmeasured, measurement) = parts(m);
            let at = self.bucket_by(kind, provider, model);
            let bucket = &mut self.buckets[at];
            bucket.requests += requests;
            bucket.unmeasured += unmeasured;
            // Only a request that reported anything contributes its breakdowns.
            if requests > unmeasured || measurement.tokens != TokenUsage::default() {
                fold(&mut bucket.measured, &measurement);
            }
        }
        self.tool_calls += usage.tool_calls;
        self.revision += 1;
    }

    pub fn snapshot(&self) -> Usage {
        let settled = self.settled;
        let mut usage = Usage { tool_calls: self.tool_calls, settled, ..Usage::default() };
        let mut aggregate: Option<Measurement> = None;
        for (i, b) in self.buckets.iter().enumerate() {
            let mut measured = b.measured.clone();
            let mut unmeasured = b.unmeasured;
            for pending in self.pending.values().filter(|p| p.bucket == i) {
                unmeasured += 1;
                if let Some(partial) = &pending.partial {
                    fold(&mut measured, partial);
                }
            }
            if let Some(m) = &measured {
                fold(
                    &mut aggregate,
                    &Measurement { tokens: m.tokens.clone(), ..Default::default() },
                );
            }
            usage.requests += b.requests;
            usage.unmeasured_requests += unmeasured;
            usage.models.push(model_usage(b, unmeasured, measured.unwrap_or_default()));
        }
        usage.tokens = aggregate.map(|m| m.tokens).unwrap_or_default();
        usage.complete = settled && usage.unmeasured_requests == 0;
        usage
    }

    fn bucket(&mut self, kind: ModelKind, provider: &str, model: &str) -> usize {
        // `"unknown"` is `ProviderId`'s stand-in for a provider that names nothing.
        let name = |s: &str| (!s.is_empty() && s != "unknown").then(|| s.to_owned());
        self.bucket_by(kind, name(provider), name(model))
    }

    fn bucket_by(
        &mut self,
        kind: ModelKind,
        provider: Option<String>,
        model: Option<String>,
    ) -> usize {
        if let Some(i) = self
            .buckets
            .iter()
            .position(|b| b.kind == kind && b.provider == provider && b.model == model)
        {
            return i;
        }
        self.buckets.push(Bucket {
            kind,
            provider,
            model,
            requests: 0,
            unmeasured: 0,
            measured: None,
        });
        self.buckets.len() - 1
    }
}

/// Independent operations' snapshots as one: settled once every one is.
pub fn combined(usages: &[Usage]) -> Usage {
    let mut ledger = Ledger::default();
    usages.iter().for_each(|u| ledger.add(u));
    ledger.settled = usages.iter().all(|u| u.settled);
    ledger.snapshot()
}

/// A chat response's reported usage. The reported zeros of this record are indistinguishable from
/// absent breakdowns, so they stay absent.
pub fn completion_tokens(usage: &CompletionUsage) -> Measurement {
    let some = |n: u32| (n != 0).then_some(u64::from(n));
    let tokens = TokenUsage {
        input_tokens: usage.prompt_tokens.into(),
        output_tokens: usage.completion_tokens.into(),
        cache_read_tokens: some(usage.prompt_cached_tokens.max(usage.cache_read_tokens)),
        cache_write_tokens: some(usage.cache_creation_tokens),
        output_reasoning_tokens: some(usage.reasoning_tokens),
        ..TokenUsage::default()
    };
    Measurement { tokens, ..Default::default() }
}

fn parts(m: &ModelUsage) -> (ModelKind, Option<String>, Option<String>, u64, u64, Measurement) {
    let tokens = |t: &TokenUsage| Measurement { tokens: t.clone(), ..Measurement::default() };
    match m {
        ModelUsage::Llm(u) => (
            ModelKind::Llm,
            u.provider.clone(),
            u.model.clone(),
            u.requests,
            u.unmeasured_requests,
            Measurement { connection_duration: u.connection_duration, ..tokens(&u.tokens) },
        ),
        ModelUsage::Stt(u) => (
            ModelKind::Stt,
            u.provider.clone(),
            u.model.clone(),
            u.requests,
            u.unmeasured_requests,
            Measurement { audio_duration: u.audio_duration, ..tokens(&u.tokens) },
        ),
        ModelUsage::Tts(u) => (
            ModelKind::Tts,
            u.provider.clone(),
            u.model.clone(),
            u.requests,
            u.unmeasured_requests,
            Measurement {
                audio_duration: u.audio_duration,
                characters_count: u.characters_count,
                ..tokens(&u.tokens)
            },
        ),
        ModelUsage::Eot(u) => (
            ModelKind::Eot,
            u.provider.clone(),
            u.model.clone(),
            u.requests,
            u.unmeasured_requests,
            tokens(&u.tokens),
        ),
        ModelUsage::Interruption(u) => (
            ModelKind::Interruption,
            u.provider.clone(),
            u.model.clone(),
            u.requests,
            u.unmeasured_requests,
            tokens(&u.tokens),
        ),
    }
}

fn model_usage(b: &Bucket, unmeasured_requests: u64, m: Measurement) -> ModelUsage {
    let (provider, model, requests, tokens) =
        (b.provider.clone(), b.model.clone(), b.requests, m.tokens);
    match b.kind {
        ModelKind::Llm => ModelUsage::Llm(LlmUsage {
            provider,
            model,
            requests,
            unmeasured_requests,
            tokens,
            connection_duration: m.connection_duration,
        }),
        ModelKind::Stt => ModelUsage::Stt(SttUsage {
            provider,
            model,
            requests,
            unmeasured_requests,
            tokens,
            audio_duration: m.audio_duration,
        }),
        ModelKind::Tts => ModelUsage::Tts(TtsUsage {
            provider,
            model,
            requests,
            unmeasured_requests,
            tokens,
            audio_duration: m.audio_duration,
            characters_count: m.characters_count,
        }),
        ModelKind::Eot => {
            ModelUsage::Eot(EotUsage { provider, model, requests, unmeasured_requests, tokens })
        }
        ModelKind::Interruption => ModelUsage::Interruption(InterruptionUsage {
            provider,
            model,
            requests,
            unmeasured_requests,
            tokens,
        }),
    }
}

/// Adds `m` to `acc`. A breakdown or billing quantity stays known only while every contribution
/// supplies it: partial breakdowns are never presented as totals.
fn fold(acc: &mut Option<Measurement>, m: &Measurement) {
    let Some(a) = acc else {
        *acc = Some(m.clone());
        return;
    };
    fn both<T: std::ops::Add<Output = T>>(a: Option<T>, b: Option<T>) -> Option<T> {
        Some(a? + b?)
    }
    let (t, u) = (&mut a.tokens, &m.tokens);
    t.input_tokens += u.input_tokens;
    t.output_tokens += u.output_tokens;
    for (x, y) in [
        (&mut t.cache_read_tokens, u.cache_read_tokens),
        (&mut t.cache_write_tokens, u.cache_write_tokens),
        (&mut t.output_reasoning_tokens, u.output_reasoning_tokens),
        (&mut t.input_text_tokens, u.input_text_tokens),
        (&mut t.input_audio_tokens, u.input_audio_tokens),
        (&mut t.input_image_tokens, u.input_image_tokens),
        (&mut t.output_text_tokens, u.output_text_tokens),
        (&mut t.output_audio_tokens, u.output_audio_tokens),
        (&mut t.output_image_tokens, u.output_image_tokens),
        (&mut t.cache_read_text_tokens, u.cache_read_text_tokens),
        (&mut t.cache_read_audio_tokens, u.cache_read_audio_tokens),
        (&mut t.cache_read_image_tokens, u.cache_read_image_tokens),
    ] {
        *x = both(*x, y);
    }
    a.connection_duration = both(a.connection_duration, m.connection_duration);
    a.audio_duration = both(a.audio_duration, m.audio_duration);
    a.characters_count = both(a.characters_count, m.characters_count);
}

#[cfg(test)]
mod tests;
