use crate::agent_session::VoiceDriver;
use crate::audio_recognition::EndOfTurnInfo;
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::productstats::features::{Feature, SwitchedModel};
use lk_agents_abi::productstats::models::{
    ModelKind, ModelLabels, ModelRequests, ModelTranscriptionDelay, ModelTtft, ModelUsage,
};
use lk_agents_abi::productstats::sessions::{
    Entrypoint, FalseInterruption, Interruption, Pipeline, TurnDetection, VoiceSessions,
};
use lk_agents_abi::resources::{LlmRef, SttRef, TtsRef};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{Maybe, ProviderId, RecognitionUsage};
use std::collections::BTreeMap;

// The product statistics a voice session measures itself: its own count, and what only it knows
// of its models. They leave with the session's telemetry records; the session never records them.

/// The model of each kind the session started with, which a switch is told against; `None`
/// when it had none.
#[derive(Default)]
pub(crate) struct StartedWith(BTreeMap<SwitchedModel, Option<ProviderId>>);

impl StartedWith {
    pub(crate) fn models(stt: &Option<SttRef>, tts: &Option<TtsRef>, llm: &Option<LlmRef>) -> Self {
        StartedWith(BTreeMap::from([
            (SwitchedModel::Stt, stt.as_ref().map(|stt| stt.provider.clone())),
            (SwitchedModel::Tts, tts.as_ref().map(|tts| tts.provider.clone())),
            (SwitchedModel::Llm, llm.as_ref().map(|llm| llm.provider.clone())),
        ]))
    }

    /// The session opened on this realtime model.
    pub(crate) fn realtime(&mut self, model: Option<ProviderId>) {
        self.0.insert(SwitchedModel::Realtime, model);
    }
}

impl VoiceDriver {
    /// A model of `kind` is installed, however it came (`update_options`, `override_options`, an
    /// agent that brings its own): a switch when it is another than the one the session started
    /// with.
    pub(crate) fn count_model_installed(&mut self, kind: SwitchedModel, model: &ProviderId) {
        let started = self.session.started_with.0.get(&kind).cloned().flatten();
        if started.as_ref() != Some(model) {
            self.session.stats.used(Feature::ModelSwitch(kind));
        }
    }

    /// The models a frame installs, each told against the one the session started with.
    pub(crate) fn count_models_installed(
        &mut self,
        stt: &Maybe<Option<SttRef>>,
        tts: &Maybe<Option<TtsRef>>,
        llm: &Maybe<Option<LlmRef>>,
    ) {
        if let Maybe::Given(Some(stt)) = stt {
            self.count_model_installed(SwitchedModel::Stt, &stt.provider.clone());
        }
        if let Maybe::Given(Some(tts)) = tts {
            self.count_model_installed(SwitchedModel::Tts, &tts.provider.clone());
        }
        if let Maybe::Given(Some(llm)) = llm {
            self.count_model_installed(SwitchedModel::Llm, &llm.provider.clone());
        }
    }

    /// The STT's language after an event, against the one before it: the person changed language
    /// when both are known and differ. An STT that reports none never counts.
    pub(crate) fn count_language(&mut self, before: Option<LanguageCode>) {
        let now = self.activity.heard_language();
        if let (Some(before), Some(now)) = (before, now)
            && before.language() != now.language()
        {
            self.session.stats.used(Feature::LanguageSwitch);
        }
    }

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
        self.telemetry.measure_product(&started, 1u64.into());
    }

    /// The session ends: its one count with what its agents did, and each feature it used.
    pub(crate) fn count_session_end(&mut self) {
        let stats = std::mem::take(&mut self.session.stats);
        stats.emit(Entrypoint::RtcSession, &mut self.telemetry);
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
        heard.measure(&model, &mut self.telemetry);
    }

    /// A committed turn's transcription delay, once per turn: nothing when the final came before
    /// the end of speech, or when neither the VAD nor the STT detected that end.
    pub(crate) fn count_transcription_delay(&mut self, info: &EndOfTurnInfo) {
        let Some((delay, endpointing)) = info.transcription_after_endpoint() else { return };
        let Some(model) = self.stt_stats() else { return };
        let labels = ModelTranscriptionDelay { model, endpointing };
        self.telemetry.measure_product(&labels, delay.as_secs_f64().into());
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
        self.telemetry.measure_product(&ModelRequests { model: model.clone() }, 1u64.into());
        ModelUsage::tokens(&response.tokens).measure(&model, &mut self.telemetry);
        if let Some(ttft) = response.ttft.filter(|ttft| !response.cancelled && *ttft >= 0.0) {
            self.telemetry.measure_product(&ModelTtft { model }, ttft.into());
        }
    }
}

#[cfg(test)]
#[path = "product_stats_tests.rs"]
mod tests;
