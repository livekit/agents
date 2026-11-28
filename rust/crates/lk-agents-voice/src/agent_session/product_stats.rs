use crate::agent_session::VoiceDriver;
use crate::audio_recognition::EndOfTurnInfo;
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::productstats::features::{FeatureLanguageSwitch, FeatureModelSwitch};
use lk_agents_abi::productstats::models::{
    ModelKind, ModelLabels, ModelRequests, ModelTranscriptionDelay, ModelTtft, ModelUsage,
};
use lk_agents_abi::productstats::sessions::{
    Entrypoint, FalseInterruption, Interruption, Pipeline, Sessions, TurnDetection, VoiceSessions,
};
use lk_agents_abi::resources::{LlmRef, RealtimeRef, SttRef, TtsRef};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{Maybe, ProviderId, RecognitionUsage};
use std::collections::BTreeMap;

// The product statistics a voice session measures itself: its own count, and what only it knows
// of its models. They leave with the session's telemetry records; the session never records them.

/// A model installed in one of the session's slots: its kind, whether it is an adapter over
/// other models (a fallback chain, one model whatever member serves), and its identity.
type InstalledModel = (ModelKind, bool, ProviderId);

/// The model installed now in each slot (STT, TTS, LLM, realtime), which the next one is
/// switched from; `None` when the session has none there.
#[derive(Default)]
pub(crate) struct Installed(BTreeMap<ModelKind, Option<InstalledModel>>);

impl Installed {
    pub(crate) fn models(stt: &Option<SttRef>, tts: &Option<TtsRef>, llm: &Option<LlmRef>) -> Self {
        let model = |kind, adapter, id: &ProviderId| (kind, adapter, id.clone());
        Installed(BTreeMap::from([
            (ModelKind::Stt, stt.as_ref().map(|m| model(ModelKind::Stt, m.adapter, &m.provider))),
            (ModelKind::Tts, tts.as_ref().map(|m| model(ModelKind::Tts, m.adapter, &m.provider))),
            (ModelKind::Llm, llm.as_ref().map(|m| model(ModelKind::Llm, m.adapter, &m.provider))),
        ]))
    }

    /// The session opened on this realtime or duplex model.
    pub(crate) fn realtime(&mut self, model: Option<&RealtimeRef>) {
        self.0.insert(ModelKind::Realtime, model.map(realtime_model));
    }
}

/// A realtime session's model: a duplex one says so by speaking over the caller.
pub(crate) fn realtime_model(session: &RealtimeRef) -> InstalledModel {
    let kind = match session.capabilities.supports_overlapping_speech {
        true => ModelKind::Duplex,
        false => ModelKind::Realtime,
    };
    (kind, false, session.provider.clone())
}

impl VoiceDriver {
    /// A model is installed in `slot`, however it came (`update_options`, `override_options`,
    /// an agent that brings its own), or removed (`None`): a switch when it is another than the
    /// one installed there just before. Installing where there was none switches from nothing; a
    /// model of the same identity installed again is the same model; and an adapter installed
    /// over an adapter is the same model, whichever member each reports.
    pub(crate) fn count_model_installed(&mut self, slot: ModelKind, model: Option<InstalledModel>) {
        let before = self.session.installed.0.insert(slot, model.clone()).flatten();
        let (Some((from, was_adapter, old)), Some((to, adapter, new))) = (before, model) else {
            return;
        };
        if !(was_adapter && adapter) && (from, &old) != (to, &new) {
            self.telemetry.measure(&FeatureModelSwitch { from, to }, 1u64.into());
        }
    }

    /// The models a frame installs or removes.
    pub(crate) fn count_models_installed(
        &mut self,
        stt: &Maybe<Option<SttRef>>,
        tts: &Maybe<Option<TtsRef>>,
        llm: &Maybe<Option<LlmRef>>,
    ) {
        let model = |kind, adapter, id: &ProviderId| (kind, adapter, id.clone());
        if let Maybe::Given(stt) = stt {
            let stt = stt.as_ref().map(|m| model(ModelKind::Stt, m.adapter, &m.provider));
            self.count_model_installed(ModelKind::Stt, stt);
        }
        if let Maybe::Given(tts) = tts {
            let tts = tts.as_ref().map(|m| model(ModelKind::Tts, m.adapter, &m.provider));
            self.count_model_installed(ModelKind::Tts, tts);
        }
        if let Maybe::Given(llm) = llm {
            let llm = llm.as_ref().map(|m| model(ModelKind::Llm, m.adapter, &m.provider));
            self.count_model_installed(ModelKind::Llm, llm);
        }
    }

    /// The STT's language after an event, against the one before it: the person changed language
    /// when both are known and differ. An STT that reports none never counts.
    pub(crate) fn count_language(&mut self, before: Option<LanguageCode>) {
        let now = self.activity.heard_language();
        if let (Some(before), Some(now)) = (before, now)
            && before.language() != now.language()
        {
            self.telemetry.measure(&FeatureLanguageSwitch, 1u64.into());
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
        self.telemetry.measure(&started, 1u64.into());
    }

    /// The call ends: its one count.
    pub(crate) fn count_session_end(&mut self) {
        let ended = Sessions { entrypoint: Entrypoint::RtcSession };
        self.telemetry.measure(&ended, 1u64.into());
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
        self.telemetry.measure(&labels, delay.as_secs_f64().into());
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
        self.telemetry.measure(&ModelRequests { model: model.clone() }, 1u64.into());
        ModelUsage::tokens(&response.tokens).measure(&model, &mut self.telemetry);
        if let Some(ttft) = response.ttft.filter(|ttft| !response.cancelled && *ttft >= 0.0) {
            self.telemetry.measure(&ModelTtft { model }, ttft.into());
        }
    }
}

#[cfg(test)]
#[path = "product_stats_tests.rs"]
mod tests;
