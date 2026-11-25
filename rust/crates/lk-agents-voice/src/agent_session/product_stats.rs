use crate::agent_session::VoiceDriver;
use crate::audio_recognition::EndOfTurnInfo;
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::RecognitionUsage;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::productstats::ProductStat;
use lk_agents_abi::productstats::models::{
    ModelKind, ModelLabels, ModelRequests, ModelTranscriptionDelay, ModelTtft, ModelUsage,
};
use lk_agents_abi::productstats::sessions::{
    Entrypoint, FalseInterruption, Interruption, Pipeline, TurnDetection, VoiceSessions,
};

// The product statistics a voice session measures itself: its own count, and what only it knows
// of its models. They leave with the session's telemetry records; the session never records them.

impl VoiceDriver {
    /// The session starts: its one count, with the configuration it starts with.
    pub(crate) fn count_voice_session(&mut self) {
        let config = &self.session.config;
        let pipeline = match self.activity.realtime().map(|realtime| &realtime.capabilities) {
            None => Pipeline::SttLlmTts,
            Some(model) if model.supports_overlapping_speech => Pipeline::Duplex,
            Some(model) if !model.audio_output && self.session.tts.is_some() => {
                Pipeline::RealtimeTts
            }
            Some(_) => Pipeline::Realtime,
        };
        // A turn detection model decides on top of the VAD's or the STT's end of speech.
        let model = config.legs.detector.is_some();
        let turn_detection = match config.effective().turn.owner() {
            TurnDetectionMode::Vad | TurnDetectionMode::Stt if model => TurnDetection::Model,
            TurnDetectionMode::Vad if config.legs.vad => TurnDetection::Vad,
            // The VAD-based path without a VAD ends turns on the STT's finals alone.
            TurnDetectionMode::Vad | TurnDetectionMode::Stt => TurnDetection::Stt,
            TurnDetectionMode::Manual => TurnDetection::Manual,
            TurnDetectionMode::RealtimeLlm => TurnDetection::RealtimeLlm,
        };
        // The detector the session was given, after the gate that decides whether one runs.
        let interruption = match config.effective().turn.interruption.enabled() {
            false => Interruption::Off,
            true if config.legs.adaptive_detector => Interruption::Adaptive,
            true => Interruption::Vad,
        };
        let false_interruption = match self.activity.resumes_false_interruptions() {
            true => FalseInterruption::On,
            false => FalseInterruption::Off,
        };
        let started = VoiceSessions { pipeline, turn_detection, interruption, false_interruption };
        self.telemetry.product_stats([ProductStat::of(&started, 1u64)]);
    }

    /// The session ends: its one count, with what its agents did.
    pub(crate) fn count_session_end(&mut self) {
        let ended = self.session.facts.ended(Entrypoint::RtcSession);
        self.telemetry.product_stats([ended]);
    }

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
