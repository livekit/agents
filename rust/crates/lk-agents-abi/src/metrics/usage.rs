//! `metrics/usage.py`'s rollup.

pub mod records;

pub use records::{
    AgentSessionUsage, BaseModelUsage, EotModelUsage, InterruptionModelUsage, LlmModelUsage,
    ModelUsage, SttModelUsage, TtsModelUsage,
};

use lk_agents_abi::metrics::base::{
    AgentMetrics, EotInferenceMetrics, InterruptionMetrics, LlmMetrics, Metadata,
    RealtimeModelMetrics, SttMetrics, TtsMetrics,
};

/// `usage.py:152`: `Some("")` and `None` share the `""` bucket.
fn provider_model(metadata: Option<&Metadata>) -> (String, String) {
    match metadata {
        Some(m) => {
            (m.model_provider.clone().unwrap_or_default(), m.model_name.clone().unwrap_or_default())
        }
        None => (String::new(), String::new()),
    }
}

/// Insertion-ordered: `flatten`'s order is what the session report carries.
#[derive(Debug, Clone, Default)]
struct OrderedByModel<V> {
    entries: Vec<((String, String), V)>,
}

impl<V: Default> OrderedByModel<V> {
    fn entry(&mut self, provider: &str, model: &str) -> &mut V {
        let key = (provider.to_owned(), model.to_owned());
        // `position` then index: the borrow checker rejects find-or-push with `iter_mut().find()`.
        let idx = match self.entries.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.entries.push((key, V::default()));
                self.entries.len() - 1
            }
        };
        &mut self.entries[idx].1
    }

    fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }
}

/// `usage.py:134`.
#[derive(Debug, Clone, Default)]
pub struct ModelUsageCollector {
    llm: OrderedByModel<LlmModelUsage>,
    tts: OrderedByModel<TtsModelUsage>,
    stt: OrderedByModel<SttModelUsage>,
    interruption: OrderedByModel<InterruptionModelUsage>,
    eot: OrderedByModel<EotModelUsage>,
}

/// `metrics.ModelUsageCollector` -- the live per-model rollup.
#[crate::host(object(
    pid = false,
    cell = "mutex",
    stub = "def collect(self, metrics: Any) -> None: ..."
))]
impl ModelUsageCollector {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one metric in.
    ///
    /// VAD, EOU and avatar metrics are not billed; the explicit no-op arms make a new metric type a
    /// compile error. `Interruption` has no producer yet: no `AgentSession` builds that detector.
    pub fn collect(&mut self, metrics: &AgentMetrics) {
        match metrics {
            AgentMetrics::Llm(m) => self.collect_llm(m),
            AgentMetrics::RealtimeModel(m) => self.collect_realtime(m),
            AgentMetrics::Tts(m) => self.collect_tts(m),
            AgentMetrics::Stt(m) => self.collect_stt(m),
            AgentMetrics::Interruption(m) => self.collect_interruption(m),
            AgentMetrics::EotInference(m) => self.collect_eot(m),
            AgentMetrics::Vad(_) | AgentMetrics::Eou(_) | AgentMetrics::Avatar(_) => {}
        }
    }

    fn collect_llm(&mut self, m: &LlmMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        let u = self.llm.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        u.input_tokens += u64::from(m.prompt_tokens);
        u.input_cached_tokens += u64::from(m.prompt_cached_tokens);
        u.input_cache_creation_tokens += u64::from(m.cache_creation_tokens);
        u.output_tokens += u64::from(m.completion_tokens);
        // No per-modality breakdown: inventing one would double count against a realtime model.
    }

    fn collect_realtime(&mut self, m: &RealtimeModelMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        // The same map as `collect_llm`: a text LLM and a realtime model under one key merge.
        let u = self.llm.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        let input = &m.input_token_details;
        // Absent details: per-modality cached counters get 0, `cached_tokens` still lands.
        let cached = input.cached_tokens_details.unwrap_or_default();

        u.input_tokens += u64::from(m.input_tokens);
        u.input_cached_tokens += u64::from(input.cached_tokens);
        u.input_text_tokens += u64::from(input.text_tokens);
        u.input_cached_text_tokens += u64::from(cached.text_tokens);
        u.input_image_tokens += u64::from(input.image_tokens);
        u.input_cached_image_tokens += u64::from(cached.image_tokens);
        u.input_audio_tokens += u64::from(input.audio_tokens);
        u.input_cached_audio_tokens += u64::from(cached.audio_tokens);

        u.output_text_tokens += u64::from(m.output_token_details.text_tokens);
        u.output_audio_tokens += u64::from(m.output_token_details.audio_tokens);
        // `image_tokens` is not summed (no field); `UsageSummary` differs on purpose.
        u.output_tokens += u64::from(m.output_tokens);
        u.session_duration += m.session_duration;
    }

    fn collect_tts(&mut self, m: &TtsMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        let u = self.tts.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        u.input_tokens += u64::from(m.input_tokens);
        u.output_tokens += u64::from(m.output_tokens);
        u.characters_count += u64::from(m.characters_count);
        u.audio_duration += m.audio_duration;
    }

    fn collect_stt(&mut self, m: &SttMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        let u = self.stt.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        u.input_tokens += u64::from(m.input_tokens);
        u.output_tokens += u64::from(m.output_tokens);
        u.audio_duration += m.audio_duration;
    }

    fn collect_interruption(&mut self, m: &InterruptionMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        let u = self.interruption.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        // `num_requests` is per event, not cumulative.
        u.total_requests += u64::from(m.num_requests);
    }

    fn collect_eot(&mut self, m: &EotInferenceMetrics) {
        let (provider, model) = provider_model(m.metadata.as_ref());
        let u = self.eot.entry(&provider, &model);
        u.provider = provider;
        u.model = model;
        u.total_requests += u64::from(m.num_requests);
    }

    /// `usage.py:262`. One entry per provider/model pair, grouped LLM, TTS, STT, interruption,
    /// EOT, insertion-ordered within each group; cloned, as the reference deep-copies.
    #[host(method)]
    pub fn flatten(&self) -> Vec<ModelUsage> {
        let mut out = Vec::new();
        out.extend(self.llm.values().cloned().map(ModelUsage::Llm));
        out.extend(self.tts.values().cloned().map(ModelUsage::Tts));
        out.extend(self.stt.values().cloned().map(ModelUsage::Stt));
        out.extend(self.interruption.values().cloned().map(ModelUsage::Interruption));
        out.extend(self.eot.values().cloned().map(ModelUsage::Eot));
        out
    }

    /// [`flatten`](Self::flatten) wrapped in the record the session event and telemetry carry.
    pub fn session_usage(&self) -> AgentSessionUsage {
        AgentSessionUsage { model_usage: self.flatten() }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyModelUsageCollector {
    /// Fold one metric record in. A record of a type the rollup ignores is a no-op, not an error.
    fn collect(&self, metrics: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<()> {
        let Some(record) = AgentMetrics::extract_known(metrics)? else { return Ok(()) };
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).collect(&record);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::base::{CachedTokenDetails, InputTokenDetails, OutputTokenDetails};
    use crate::metrics::testkit as tk;

    fn llm_of(c: &ModelUsageCollector) -> Vec<LlmModelUsage> {
        c.llm.values().cloned().collect()
    }

    // Catches: routing realtime metrics into a map of their own.
    #[test]
    fn realtime_and_llm_merge_into_one_entry_under_the_same_key() {
        let mut c = ModelUsageCollector::new();
        let mut llm = tk::llm();
        llm.metadata = tk::meta("openai", "gpt-4o");
        llm.prompt_tokens = 10;
        llm.completion_tokens = 3;
        c.collect(&AgentMetrics::Llm(llm));

        let mut rt = tk::realtime();
        rt.metadata = tk::meta("openai", "gpt-4o");
        rt.input_tokens = 100;
        rt.output_tokens = 40;
        c.collect(&AgentMetrics::RealtimeModel(rt));

        let usage = llm_of(&c);
        assert_eq!(usage.len(), 1, "one provider/model key, one record");
        assert_eq!(usage[0].input_tokens, 110);
        assert_eq!(usage[0].output_tokens, 43);
    }

    // Catches: reading the cached breakdown unconditionally.
    #[test]
    fn absent_cached_breakdown_still_records_the_cached_total() {
        let mut c = ModelUsageCollector::new();
        let mut rt = tk::realtime();
        rt.metadata = tk::meta("openai", "gpt-realtime");
        rt.input_token_details = InputTokenDetails {
            audio_tokens: 30,
            text_tokens: 20,
            image_tokens: 5,
            cached_tokens: 40,
            cached_tokens_details: None,
        };
        c.collect(&AgentMetrics::RealtimeModel(rt));

        let u = &llm_of(&c)[0];
        assert_eq!(u.input_cached_tokens, 40);
        assert_eq!(u.input_cached_audio_tokens, 0);
        assert_eq!(u.input_cached_text_tokens, 0);
        assert_eq!(u.input_cached_image_tokens, 0);
        assert_eq!((u.input_audio_tokens, u.input_text_tokens, u.input_image_tokens), (30, 20, 5));
    }

    // Catches: crossing the three cached modalities.
    #[test]
    fn cached_breakdown_lands_on_the_matching_modality() {
        let mut c = ModelUsageCollector::new();
        let mut rt = tk::realtime();
        rt.input_token_details = InputTokenDetails {
            cached_tokens: 6,
            cached_tokens_details: Some(CachedTokenDetails {
                audio_tokens: 1,
                text_tokens: 2,
                image_tokens: 3,
            }),
            ..Default::default()
        };
        c.collect(&AgentMetrics::RealtimeModel(rt));

        let u = &llm_of(&c)[0];
        assert_eq!(u.input_cached_audio_tokens, 1);
        assert_eq!(u.input_cached_text_tokens, 2);
        assert_eq!(u.input_cached_image_tokens, 3);
    }

    // Catches: summing `output_token_details.image_tokens` into the LLM usage.
    #[test]
    fn realtime_output_image_tokens_are_dropped_not_folded_into_output_tokens() {
        let mut c = ModelUsageCollector::new();
        let mut rt = tk::realtime();
        rt.output_tokens = 7;
        rt.output_token_details =
            OutputTokenDetails { text_tokens: 4, audio_tokens: 3, image_tokens: 99 };
        c.collect(&AgentMetrics::RealtimeModel(rt));

        let u = &llm_of(&c)[0];
        assert_eq!(u.output_tokens, 7, "the total is the provider's, not a recomputed sum");
        assert_eq!(u.output_text_tokens + u.output_audio_tokens, 7);
    }

    // Catches: `cache_creation_tokens` fed from the realtime branch, or dropped from the LLM one.
    #[test]
    fn cache_creation_tokens_come_only_from_the_llm_branch() {
        let mut c = ModelUsageCollector::new();
        let mut llm = tk::llm();
        llm.prompt_tokens = 100;
        llm.prompt_cached_tokens = 60;
        llm.cache_creation_tokens = 25;
        c.collect(&AgentMetrics::Llm(llm));

        let u = &llm_of(&c)[0];
        assert_eq!(u.input_cache_creation_tokens, 25);
        assert_eq!(u.input_cached_tokens, 60);
        assert_eq!(u.input_tokens, 100, "cache writes are not added to the input total");
    }

    // Catches: dropping metrics that carry no metadata instead of billing `("", "")`.
    #[test]
    fn missing_and_empty_metadata_share_the_empty_key() {
        let mut c = ModelUsageCollector::new();
        let mut a = tk::stt();
        a.audio_duration = 1.5;
        c.collect(&AgentMetrics::Stt(a));

        let mut b = tk::stt();
        b.metadata = Some(Metadata { model_name: Some(String::new()), model_provider: None });
        b.audio_duration = 2.5;
        c.collect(&AgentMetrics::Stt(b));

        let flat = c.flatten();
        assert_eq!(flat.len(), 1);
        assert_eq!(
            flat[0],
            ModelUsage::Stt(SttModelUsage { audio_duration: 4.0, ..Default::default() })
        );
    }

    // Catches: sorting the flattened list, or reordering the five kinds.
    #[test]
    fn flatten_is_grouped_by_kind_then_insertion_ordered() {
        let mut c = ModelUsageCollector::new();
        let mut tts = tk::tts();
        tts.metadata = tk::meta("cartesia", "sonic");
        c.collect(&AgentMetrics::Tts(tts));

        for provider in ["zeta", "alpha"] {
            let mut llm = tk::llm();
            llm.metadata = tk::meta(provider, "m");
            c.collect(&AgentMetrics::Llm(llm));
        }

        let flat = c.flatten();
        let kinds: Vec<_> = flat.iter().map(ModelUsage::type_literal).collect();
        assert_eq!(kinds, ["llm_usage", "llm_usage", "tts_usage"], "LLM group precedes TTS");
        match (&flat[0], &flat[1]) {
            (ModelUsage::Llm(a), ModelUsage::Llm(b)) => {
                assert_eq!((a.provider.as_str(), b.provider.as_str()), ("zeta", "alpha"));
            }
            _ => unreachable!(),
        }
    }

    // Catches: billing VAD, EOU or avatar metrics.
    #[test]
    fn unbilled_metrics_create_no_entries() {
        let mut c = ModelUsageCollector::new();
        c.collect(&AgentMetrics::Vad(tk::vad()));
        c.collect(&AgentMetrics::Eou(tk::eou()));
        c.collect(&AgentMetrics::Avatar(tk::avatar()));
        assert!(c.flatten().is_empty());
    }

    // Catches: counting one request per event instead of `num_requests`.
    #[test]
    fn request_counters_add_num_requests_not_one() {
        let mut c = ModelUsageCollector::new();
        let mut i = tk::interruption();
        i.metadata = tk::meta("livekit", "adaptive");
        i.num_requests = 4;
        c.collect(&AgentMetrics::Interruption(i));

        let mut e = tk::eot();
        e.metadata = tk::meta("livekit", "turn-detector-v1");
        e.num_requests = 3;
        c.collect(&AgentMetrics::EotInference(e));

        assert_eq!(
            c.flatten(),
            vec![
                ModelUsage::Interruption(InterruptionModelUsage {
                    provider: "livekit".into(),
                    model: "adaptive".into(),
                    total_requests: 4,
                }),
                ModelUsage::Eot(EotModelUsage {
                    provider: "livekit".into(),
                    model: "turn-detector-v1".into(),
                    total_requests: 3,
                }),
            ]
        );
    }

    // Catches: a dropped or cross-wired counter in `collect_tts` / `collect_stt`, or assigning
    // instead of accumulating.
    #[test]
    fn every_tts_and_stt_counter_is_fed_from_its_own_field_and_accumulates() {
        let mut c = ModelUsageCollector::new();
        for (input, output, chars, audio) in [(3, 7, 13, 0.25), (5, 11, 17, 0.5)] {
            let mut t = tk::tts();
            t.metadata = tk::meta("cartesia", "sonic-3");
            t.input_tokens = input;
            t.output_tokens = output;
            t.characters_count = chars;
            t.audio_duration = audio;
            c.collect(&AgentMetrics::Tts(t));
        }
        for (input, output, audio) in [(19, 29, 1.5), (23, 31, 2.5)] {
            let mut m = tk::stt();
            m.metadata = tk::meta("deepgram", "nova-3");
            m.input_tokens = input;
            m.output_tokens = output;
            m.audio_duration = audio;
            c.collect(&AgentMetrics::Stt(m));
        }

        assert_eq!(
            c.flatten(),
            vec![
                ModelUsage::Tts(TtsModelUsage {
                    provider: "cartesia".into(),
                    model: "sonic-3".into(),
                    input_tokens: 8,
                    output_tokens: 18,
                    characters_count: 30,
                    audio_duration: 0.75,
                }),
                ModelUsage::Stt(SttModelUsage {
                    provider: "deepgram".into(),
                    model: "nova-3".into(),
                    input_tokens: 42,
                    output_tokens: 60,
                    audio_duration: 4.0,
                }),
            ]
        );
    }

    // Catches: dropping `session_duration` (xAI bills by it), or not summing it.
    #[test]
    fn realtime_session_duration_accumulates_and_is_the_only_llm_counter_it_touches() {
        let mut c = ModelUsageCollector::new();
        for seconds in [12.5, 30.25] {
            let mut rt = tk::realtime();
            rt.metadata = tk::meta("xai", "grok-realtime");
            rt.session_duration = seconds;
            c.collect(&AgentMetrics::RealtimeModel(rt));
        }
        assert_eq!(
            llm_of(&c)[0],
            LlmModelUsage {
                provider: "xai".into(),
                model: "grok-realtime".into(),
                session_duration: 42.75,
                ..Default::default()
            }
        );
    }


}
