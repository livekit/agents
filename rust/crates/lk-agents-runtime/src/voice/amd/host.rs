//! AMD's host boundary: `AMD(session, ...)`'s options checked and converted, a run handed to the
//! host as `_core.AmdRun`, its events as the host's records, and a host decision model behind the
//! runtime's [`DecisionModel`] port.

use super::{AmdConfig, AmdEnterError, AmdModel, AmdRun, ChoiceRequest, Classifier, DecisionModel};
use crate::voice::VoiceSession;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::{Invalid, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::amd as abi;
use lk_agents_abi::voice::session::SessionRefused;
use lk_agents_voice::amd::AmdAttachRefused;
use lk_agents_voice::amd::events::{
    AmdEvent, AmdReason, Category, MenuObserved, Outcome, Prediction, Stage,
};
use lk_agents_voice::amd::options::AmdOptions;
use std::sync::Arc;
use std::time::Duration;

/// `AMD(...)`'s options as the machine takes them, checked in the constructor's order: the
/// `ValueError` it raises otherwise.
pub fn options(host: &abi::AmdOptions) -> Result<AmdOptions, Invalid> {
    const NOT_POSITIVE: &str = "amd timeouts and max_uncertain_turns must be positive";
    let positive = |seconds: f64| {
        Duration::try_from_secs_f64(seconds)
            .ok()
            .filter(|duration| !duration.is_zero())
            .ok_or_else(|| Invalid(NOT_POSITIVE.into()))
    };
    let timeouts =
        [host.idle_timeout, host.voicemail_idle_timeout, host.timeout, host.inference_timeout];
    let [idle_timeout, voicemail_idle_timeout, timeout, inference_timeout] =
        match timeouts.map(positive) {
            [Ok(a), Ok(b), Ok(c), Ok(d)] if host.max_uncertain_turns >= 1 => [a, b, c, d],
            _ => return Err(Invalid(NOT_POSITIVE.into())),
        };
    let max_uncertain_turns = u32::try_from(host.max_uncertain_turns).unwrap_or(u32::MAX);
    if host.max_inference_timeouts < 1 {
        return Err(Invalid("max_inference_timeouts must be positive".into()));
    }
    let max_inference_timeouts = u32::try_from(host.max_inference_timeouts).unwrap_or(u32::MAX);
    let machine_silence_threshold = Duration::try_from_secs_f64(host.machine_silence_threshold)
        .map_err(|_| Invalid("machine_silence_threshold must be non-negative".into()))?;
    let options = AmdOptions {
        screening_instructions: host.screening_instructions.clone(),
        voicemail_instructions: host.voicemail_instructions.clone(),
        ivr_instructions: host.ivr_instructions.clone(),
        human_instructions: host.human_instructions.clone(),
        opening_text: host.opening_text.clone(),
        opening_instructions: host.opening_instructions.clone(),
        wait_until_answered: host.wait_until_answered,
        idle_timeout,
        voicemail_idle_timeout,
        timeout,
        inference_timeout,
        machine_silence_threshold,
        max_uncertain_turns,
        max_inference_timeouts,
    };
    options.validate().map_err(|error| Invalid(error.to_string()))?;
    Ok(options)
}

/// What one run needs beyond its session, from what the host resolved. The host closes the
/// models it created once the run completed, so none is closed here.
pub fn config(entry: abi::AmdEntry) -> Result<AmdConfig, abi::AmdEnterRefused> {
    let options =
        options(&entry.options).map_err(|Invalid(why)| abi::AmdEnterRefused::Invalid(why))?;
    let classifier = match entry.classifier {
        abi::AmdClassifier::Llm(llm) => {
            Classifier::Llm(AmdModel::supplied(llm.model, llm.inference))
        }
        abi::AmdClassifier::Decision(decision) => {
            let model: Arc<dyn DecisionModel> = Arc::new(HostDecision(decision.model));
            Classifier::Decision(AmdModel::supplied(model, decision.inference))
        }
    };
    Ok(AmdConfig {
        options,
        classifier,
        stt: entry.stt.map(|stt| AmdModel::supplied(stt.model, stt.inference)),
        participant: entry.participant,
        dtmf_tool: entry.dtmf_tool,
        llm_conn_options: entry.llm_conn_options,
        stt_conn_options: entry.stt_conn_options,
    })
}

/// A refused entry as `AMD.__aenter__` raises it: `RuntimeError` for the session's state,
/// `ValueError` for options and models that do not fit it.
pub fn refused(error: AmdEnterError) -> abi::AmdEnterRefused {
    let message = error.to_string();
    match error {
        AmdEnterError::Options(_) | AmdEnterError::NotStreaming => {
            abi::AmdEnterRefused::Invalid(message)
        }
        AmdEnterError::Refused(why) => match why {
            AmdAttachRefused::AlreadyAttached
            | AmdAttachRefused::HandoffInFlight
            | AmdAttachRefused::TaskRunning => abi::AmdEnterRefused::Refused(message),
            AmdAttachRefused::RealtimeServerTurnDetection
            | AmdAttachRefused::RealtimeTools
            | AmdAttachRefused::RealtimeNoStt => abi::AmdEnterRefused::Invalid(message),
            AmdAttachRefused::Closing => SessionRefused::Closed.into(),
        },
        AmdEnterError::Session(refused) => refused.into(),
    }
}

/// `async with AMD(session, ...)` from the host's entry.
pub async fn enter(
    session: &VoiceSession,
    entry: abi::AmdEntry,
) -> Result<AmdRun, abi::AmdEnterRefused> {
    let config = config(entry)?;
    super::enter(session, config).await.map_err(refused)
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn category(category: Category) -> abi::AmdCategory {
    match category {
        Category::Human => abi::AmdCategory::Human,
        Category::MachineIvr => abi::AmdCategory::MachineIvr,
        Category::MachineScreening => abi::AmdCategory::MachineScreening,
        Category::MachineVm => abi::AmdCategory::MachineVm,
        Category::MachineUnavailable => abi::AmdCategory::MachineUnavailable,
        Category::Wait => abi::AmdCategory::Wait,
        Category::Uncertain => abi::AmdCategory::Uncertain,
    }
}

fn stage(stage: Stage) -> abi::AmdCategory {
    category(stage.category())
}

fn reason(reason: AmdReason) -> abi::AmdReason {
    match reason {
        AmdReason::Prediction => abi::AmdReason::Prediction,
        AmdReason::Reused => abi::AmdReason::Reused,
        AmdReason::InferenceTimeout => abi::AmdReason::InferenceTimeout,
        AmdReason::InferenceError => abi::AmdReason::InferenceError,
        AmdReason::InternalError => abi::AmdReason::InternalError,
        AmdReason::Finished => abi::AmdReason::Finished,
        AmdReason::MaxUncertainTurns => abi::AmdReason::MaxUncertainTurns,
        AmdReason::Timeout => abi::AmdReason::Timeout,
        AmdReason::IdleTimeout => abi::AmdReason::IdleTimeout,
        AmdReason::Cancelled => abi::AmdReason::Cancelled,
        AmdReason::ParticipantMissing => abi::AmdReason::ParticipantMissing,
        AmdReason::ParticipantDisconnected => abi::AmdReason::ParticipantDisconnected,
        AmdReason::AgentChanged => abi::AmdReason::AgentChanged,
    }
}

/// `amd_prediction`'s record.
pub fn prediction(prediction: &Prediction) -> abi::AmdPredictionEvent {
    abi::AmdPredictionEvent {
        speech_duration: seconds(prediction.speech_duration),
        category: category(prediction.category),
        reason: reason(prediction.reason),
        transcript: prediction.transcript.clone(),
        delay: seconds(prediction.delay),
        turn_id: prediction.turn_id,
        stage: stage(prediction.stage),
        inference_duration: prediction.inference_duration.map(seconds),
    }
}

/// `amd_completed`'s record.
pub fn completed(outcome: &Outcome) -> abi::AmdCompletedEvent {
    abi::AmdCompletedEvent {
        category: stage(outcome.category),
        reason: reason(outcome.reason),
        voicemail_message_played: outcome.voicemail_message_played,
        last_prediction: outcome.last_prediction.as_ref().map(prediction),
    }
}

fn menu(menu: &MenuObserved) -> abi::AmdMenuObservedEvent {
    abi::AmdMenuObservedEvent {
        session_id: menu.session_id.clone(),
        turn_id: menu.turn_id,
        menu: menu.menu.clone(),
        options: (menu.options.iter())
            .map(|option| abi::IvrMenuOption {
                label: option.label.clone(),
                dtmf: option.dtmf.clone(),
                spoken_response: option.spoken_response.clone(),
            })
            .collect(),
        extraction_duration: seconds(menu.extraction_duration),
    }
}

/// One of the run's events as the host's record.
pub fn event(event: &AmdEvent) -> abi::AmdEvent {
    match event {
        AmdEvent::Prediction(p) => abi::AmdEvent::Prediction(prediction(p)),
        AmdEvent::MenuObserved(m) => abi::AmdEvent::MenuObserved(menu(m)),
        AmdEvent::Completed(outcome) => abi::AmdEvent::Completed(completed(outcome)),
    }
}

impl abi::AmdRun for AmdRun {
    fn lifecycle(&self) -> abi::AmdLifecycle {
        AmdRun::lifecycle(self)
    }

    fn next_event(&self) -> BoxFuture<'static, Option<abi::AmdEvent>> {
        let next = AmdRun::next_event(self);
        Box::pin(async move { next.await.as_ref().map(event) })
    }

    fn execute(&self) -> BoxFuture<'static, abi::AmdCompletedEvent> {
        let outcome = AmdRun::execute(self);
        Box::pin(async move { completed(&outcome.await) })
    }

    fn aclose(&self) -> BoxFuture<'static, abi::AmdCompletedEvent> {
        let outcome = AmdRun::close(self);
        Box::pin(async move { completed(&outcome.await) })
    }

    fn listener_failed(&self) {
        self.fail();
    }
}

/// AMD's pure rules, as the host's types read them.
pub struct Rules;

impl abi::AmdRules for Rules {
    fn check_options(options: abi::AmdOptions) -> Result<(), Invalid> {
        self::options(&options).map(drop)
    }

    fn prediction_is_human(event: abi::AmdPredictionEvent) -> bool {
        event.category == abi::AmdCategory::Human
    }

    fn prediction_is_machine(event: abi::AmdPredictionEvent) -> bool {
        matches!(
            event.category,
            abi::AmdCategory::MachineScreening
                | abi::AmdCategory::MachineIvr
                | abi::AmdCategory::MachineVm
                | abi::AmdCategory::MachineUnavailable
        )
    }

    fn completed_is_error(event: abi::AmdCompletedEvent) -> bool {
        matches!(
            event.reason,
            abi::AmdReason::InternalError
                | abi::AmdReason::ParticipantMissing
                | abi::AmdReason::AgentChanged
                | abi::AmdReason::Cancelled
        )
    }

    fn auto_models() -> bool {
        use lk_agents_abi::inference::InferenceCredentials as _;
        let url = lk_agents_net::env::server_setting("LIVEKIT_URL").unwrap_or_default();
        let key = lk_agents_inference::credentials::Credentials::inference_api_key(None);
        let secret = lk_agents_inference::credentials::Credentials::inference_api_secret(None);
        lk_agents_net::strings::is_cloud_url(&url) && !key.is_empty() && !secret.is_empty()
    }
}

/// A host decision model behind the runtime's port: `evaluate` with one `Choice`, through the
/// host module's glue.
struct HostDecision(Arc<dyn abi::AmdDecisionModel>);

impl DecisionModel for HostDecision {
    fn label(&self) -> String {
        self.0.label().unwrap_or_else(|_| "decision model".to_owned())
    }

    fn choose<'a>(
        &'a self,
        cx: &Cx,
        request: &'a ChoiceRequest,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            let request = abi::AmdChoiceRequest {
                input: request.input.clone(),
                name: request.name.clone(),
                prompt: request.prompt.clone(),
                options: request.options.clone(),
                conn_options: request.conn_options,
                extra_headers: request.extra_headers.clone(),
            };
            self.0.choose(&cx, &request).await
        })
    }
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;
