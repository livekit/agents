use crate::agent_session::VoiceDriver;
use crate::audio_recognition::EndOfTurnInfo;
use lk_agents_abi::RecognitionUsage;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::productstats::ProductStat;
use lk_agents_abi::productstats::models::{
    ModelKind, ModelLabels, ModelRequests, ModelTranscriptionDelay, ModelTtft, ModelUsage,
};

// The product statistics a voice session measures itself: what only it knows of its models.
// They leave with the session's telemetry records; the session never records them.

impl VoiceDriver {
    /// The session STT's labels: a `FallbackAdapter`'s are the instance serving.
    fn stt_stats(&self) -> Option<ModelLabels> {
        self.stt_identity().map(|id| ModelLabels::of(ModelKind::Stt, &id))
    }

    /// What the session's STT reported hearing.
    pub(crate) fn count_stt_usage(&mut self, usage: &RecognitionUsage) {
        let Some(model) = self.stt_stats() else { return };
        let heard = ModelUsage {
            audio_duration: usage.audio_duration.unwrap_or_default(),
            ..ModelUsage::tokens(&usage.tokens)
        };
        self.telemetry.product_stats(heard.stats(&model));
    }

    /// A committed turn's transcription delay, once per turn: nothing when the final came before
    /// the end of speech, or when neither the VAD nor the STT detected that end.
    pub(crate) fn count_transcription_delay(&mut self, info: &EndOfTurnInfo) {
        let Some((delay, endpointing)) = info.transcription_after_endpoint() else { return };
        let Some(model) = self.stt_stats() else { return };
        let labels = ModelTranscriptionDelay { model, endpointing };
        self.telemetry.product_stats([ProductStat::of(&labels, delay.as_secs_f64())]);
    }

    /// One response of the session's realtime or duplex model: the request, its tokens and its
    /// time to the first token. The labels are the installed model's, never the record's.
    pub(crate) fn count_realtime_response(&mut self, metrics: &AgentMetrics) {
        let AgentMetrics::RealtimeModel(response) = metrics else { return };
        let Some(realtime) = self.activity.realtime() else { return };
        let kind = match realtime.capabilities.supports_overlapping_speech {
            true => ModelKind::Duplex,
            false => ModelKind::Realtime,
        };
        let model = ModelLabels::of(kind, &realtime.provider);
        let mut stats = vec![ProductStat::of(&ModelRequests { model: model.clone() }, 1u64)];
        stats.extend(ModelUsage::tokens(&response.tokens).stats(&model));
        if let Some(ttft) = response.ttft.filter(|ttft| !response.cancelled && *ttft >= 0.0) {
            stats.push(ProductStat::of(&ModelTtft { model }, ttft));
        }
        self.telemetry.product_stats(stats);
    }
}

#[cfg(test)]
#[path = "product_stats_tests.rs"]
mod tests;
