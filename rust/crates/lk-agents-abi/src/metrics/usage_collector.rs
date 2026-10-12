//! `metrics/usage_collector.py`: the deprecated flat rollup, still public API. Its differences from
//! `ModelUsageCollector` are the reference's and pinned by tests.

pub mod records;

use crate::metrics::base::{AgentMetrics, LlmMetrics, RealtimeModelMetrics, SttMetrics, TtsMetrics};

pub use records::UsageSummary;

/// `usage_collector.py:59`. Deprecated.
#[derive(Debug, Clone, Default)]
pub struct UsageCollector {
    summary: UsageSummary,
}

/// `metrics.UsageCollector` -- the deprecated session-wide totals, with no per-model breakdown.
#[crate::host(object(
    pid = false,
    cell = "mutex",
    stub = "def collect(self, metrics: Any) -> None: ..."
))]
impl UsageCollector {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one metric in. Only four of the nine metric types contribute.
    pub fn collect(&mut self, metrics: &AgentMetrics) {
        match metrics {
            AgentMetrics::Llm(m) => self.collect_llm(m),
            AgentMetrics::RealtimeModel(m) => self.collect_realtime(m),
            AgentMetrics::Tts(m) => self.collect_tts(m),
            AgentMetrics::Stt(m) => self.collect_stt(m),
            // No arm in the reference.
            AgentMetrics::Vad(_)
            | AgentMetrics::Eou(_)
            | AgentMetrics::EotInference(_)
            | AgentMetrics::Interruption(_)
            | AgentMetrics::Avatar(_) => {}
        }
    }

    fn collect_llm(&mut self, m: &LlmMetrics) {
        let s = &mut self.summary;
        s.llm_prompt_tokens += u64::from(m.prompt_tokens);
        s.llm_prompt_cached_tokens += u64::from(m.prompt_cached_tokens);
        s.llm_completion_tokens += u64::from(m.completion_tokens);
        // `cache_creation_tokens` is deliberately absent -- the reference never reads it here.
    }

    fn collect_realtime(&mut self, m: &RealtimeModelMetrics) {
        let s = &mut self.summary;
        let input = &m.input_token_details;
        let cached = input.cached_tokens_details.unwrap_or_default();

        // Realtime input lands on `llm_prompt_tokens`, the same counter a text LLM's prompt uses.
        s.llm_prompt_tokens += u64::from(m.input_tokens);
        s.llm_prompt_cached_tokens += u64::from(input.cached_tokens);

        s.llm_input_text_tokens += u64::from(input.text_tokens);
        s.llm_input_cached_text_tokens += u64::from(cached.text_tokens);
        s.llm_input_image_tokens += u64::from(input.image_tokens);
        s.llm_input_cached_image_tokens += u64::from(cached.image_tokens);
        s.llm_input_audio_tokens += u64::from(input.audio_tokens);
        s.llm_input_cached_audio_tokens += u64::from(cached.audio_tokens);

        s.llm_output_text_tokens += u64::from(m.output_token_details.text_tokens);
        s.llm_output_image_tokens += u64::from(m.output_token_details.image_tokens);
        s.llm_output_audio_tokens += u64::from(m.output_token_details.audio_tokens);
        s.llm_completion_tokens += u64::from(m.output_tokens);
        // No `session_duration`: the field postdates this collector.
    }

    fn collect_tts(&mut self, m: &TtsMetrics) {
        self.summary.tts_characters_count += u64::from(m.characters_count);
        self.summary.tts_audio_duration += m.audio_duration;
    }

    fn collect_stt(&mut self, m: &SttMetrics) {
        self.summary.stt_audio_duration += m.audio_duration;
    }

    /// `get_summary`, which `deepcopy`s so a caller cannot mutate the collector's state.
    #[host(method)]
    pub fn summary(&self) -> UsageSummary {
        self.summary.clone()
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyUsageCollector {
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

    // Catches: adding `cache_creation_tokens` to the LLM branch.
    #[test]
    fn llm_cache_creation_tokens_are_ignored_entirely() {
        let mut c = UsageCollector::new();
        let mut m = tk::llm();
        m.prompt_tokens = 10;
        m.prompt_cached_tokens = 4;
        m.cache_creation_tokens = 90;
        c.collect(&AgentMetrics::Llm(m));

        let s = c.summary();
        assert_eq!(s.llm_prompt_tokens, 10);
        assert_eq!(s.llm_prompt_cached_tokens, 4);
    }

    // Catches: dropping `output_token_details.image_tokens` (the new collector drops it; this one does not).
    #[test]
    fn realtime_output_image_tokens_are_summed_here_unlike_the_new_collector() {
        let mut c = UsageCollector::new();
        let mut m = tk::realtime();
        m.output_token_details =
            OutputTokenDetails { text_tokens: 1, audio_tokens: 2, image_tokens: 3 };
        c.collect(&AgentMetrics::RealtimeModel(m));
        assert_eq!(c.summary().llm_output_image_tokens, 3);
    }

    // Catches: any of the twelve `+=` lines in `collect_realtime` dropped or cross-wired. Every
    // value below is distinct.
    #[test]
    fn every_realtime_counter_is_fed_from_its_own_field() {
        let mut c = UsageCollector::new();
        let mut m = tk::realtime();
        m.input_tokens = 101;
        m.output_tokens = 201;
        m.session_duration = 7.5;
        m.input_token_details = InputTokenDetails {
            audio_tokens: 11,
            text_tokens: 12,
            image_tokens: 13,
            cached_tokens: 14,
            cached_tokens_details: Some(CachedTokenDetails {
                audio_tokens: 21,
                text_tokens: 22,
                image_tokens: 23,
            }),
        };
        m.output_token_details =
            OutputTokenDetails { text_tokens: 31, audio_tokens: 32, image_tokens: 33 };
        c.collect(&AgentMetrics::RealtimeModel(m));

        // A whole-struct comparison, so fields that must stay zero are pinned too.
        assert_eq!(
            c.summary(),
            UsageSummary {
                llm_prompt_tokens: 101,
                llm_prompt_cached_tokens: 14,
                llm_input_audio_tokens: 11,
                llm_input_cached_audio_tokens: 21,
                llm_input_text_tokens: 12,
                llm_input_cached_text_tokens: 22,
                llm_input_image_tokens: 13,
                llm_input_cached_image_tokens: 23,
                llm_completion_tokens: 201,
                llm_output_audio_tokens: 32,
                llm_output_image_tokens: 33,
                llm_output_text_tokens: 31,
                tts_characters_count: 0,
                tts_audio_duration: 0.0,
                stt_audio_duration: 0.0,
            }
        );
    }

    // Catches: routing realtime input to a counter separate from `llm_prompt_tokens`.
    #[test]
    fn realtime_input_shares_the_prompt_counter_with_text_llms() {
        let mut c = UsageCollector::new();
        let mut llm = tk::llm();
        llm.prompt_tokens = 7;
        c.collect(&AgentMetrics::Llm(llm));

        let mut rt = tk::realtime();
        rt.input_tokens = 13;
        rt.input_token_details = InputTokenDetails {
            cached_tokens: 5,
            cached_tokens_details: Some(CachedTokenDetails {
                audio_tokens: 5,
                ..Default::default()
            }),
            ..Default::default()
        };
        c.collect(&AgentMetrics::RealtimeModel(rt));

        let s = c.summary();
        assert_eq!(s.llm_prompt_tokens, 20);
        assert_eq!(s.llm_prompt_cached_tokens, 5);
        assert_eq!(s.llm_input_cached_audio_tokens, 5);
    }

    // Catches: adding the STT and TTS token counters.
    #[test]
    fn stt_and_tts_token_counters_are_not_summarised() {
        let mut c = UsageCollector::new();
        let mut stt = tk::stt();
        stt.audio_duration = 3.0;
        stt.input_tokens = 100;
        stt.output_tokens = 50;
        c.collect(&AgentMetrics::Stt(stt));

        let mut tts = tk::tts();
        tts.audio_duration = 1.5;
        tts.characters_count = 42;
        tts.input_tokens = 9;
        c.collect(&AgentMetrics::Tts(tts));

        assert_eq!(
            c.summary(),
            UsageSummary {
                stt_audio_duration: 3.0,
                tts_audio_duration: 1.5,
                tts_characters_count: 42,
                ..Default::default()
            }
        );
    }

    // Catches: extending this collector to the interruption and EOT models.
    #[test]
    fn interruption_and_eot_metrics_leave_the_summary_untouched() {
        let mut c = UsageCollector::new();
        let mut i = tk::interruption();
        i.num_requests = 5;
        c.collect(&AgentMetrics::Interruption(i));
        let mut e = tk::eot();
        e.num_requests = 5;
        c.collect(&AgentMetrics::EotInference(e));
        assert_eq!(c.summary(), UsageSummary::default());
    }
}
