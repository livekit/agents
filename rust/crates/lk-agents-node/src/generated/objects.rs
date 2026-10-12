//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]

use napi::Env;
use napi_derive::napi;

/// Stateful, provider-agnostic markup stripper for one transcript segment.
#[napi(js_name = "TranscriptMarkupStripper")]
pub struct TranscriptMarkupStripper {
    pub(crate) inner: lk_agents_abi::tts::markup::TranscriptMarkupStripper,
}

#[napi]
impl TranscriptMarkupStripper {
    #[napi(constructor)]
    pub fn new() -> Self {
        TranscriptMarkupStripper { inner: lk_agents_abi::tts::markup::TranscriptMarkupStripper::new() }
    }

    #[napi(js_name = "push")]
    pub fn push(&mut self, text: String) -> String {
        self.inner.push(&text)
    }

    #[napi(js_name = "flush")]
    pub fn flush(&mut self) -> String {
        self.inner.flush()
    }

    #[napi(getter, js_name = "tags")]
    pub fn tags(&self) -> Vec<crate::generated::records::ExpressiveTag> {
        self.inner.tags().iter().map(|v| crate::generated::records::ExpressiveTag::from(v)).collect::<Vec<_>>()
    }

    #[napi(js_name = "expressionAttribute")]
    pub fn expression_attribute(&self) -> Option<std::collections::HashMap<String, String>> {
        self.inner.expression_attribute_map().map(|v| v.into_iter().map(|(k, v)| (k, v)).collect::<std::collections::HashMap<_, _>>())
    }
}

/// `RecordingOptions`, as five booleans.
#[napi(js_name = "RecordingOptions")]
pub struct RecordingOptions {
    pub(crate) inner: lk_agents_abi::telemetry::report::RecordingOptions,
}

#[napi]
impl RecordingOptions {
    #[napi(constructor)]
    pub fn new(audio: Option<bool>, traces: Option<bool>, logs: Option<bool>, transcript: Option<bool>, redaction: Option<bool>) -> Self {
        let audio = audio.unwrap_or(false);
        let traces = traces.unwrap_or(false);
        let logs = logs.unwrap_or(false);
        let transcript = transcript.unwrap_or(false);
        let redaction = redaction.unwrap_or(false);
        RecordingOptions { inner: lk_agents_abi::telemetry::report::RecordingOptions::new(audio, traces, logs, transcript, redaction) }
    }

    #[napi(getter, js_name = "audio")]
    pub fn audio(&self) -> bool {
        self.inner.audio()
    }

    #[napi(getter, js_name = "traces")]
    pub fn traces(&self) -> bool {
        self.inner.traces()
    }

    #[napi(getter, js_name = "logs")]
    pub fn logs(&self) -> bool {
        self.inner.logs()
    }

    #[napi(getter, js_name = "transcript")]
    pub fn transcript(&self) -> bool {
        self.inner.transcript()
    }

    #[napi(getter, js_name = "redaction")]
    pub fn redaction(&self) -> bool {
        self.inner.redaction()
    }

    #[napi(js_name = "recordingEnabled")]
    pub fn recording_enabled(&self) -> bool {
        self.inner.recording_enabled()
    }
}

/// The "data recording is disabled" latch (`upload_gate.py`): one transition, one warning.
#[napi(js_name = "UploadGate")]
pub struct UploadGate {
    pub(crate) inner: lk_agents_abi::telemetry::report::UploadGate,
}

#[napi]
impl UploadGate {
    #[napi(constructor)]
    pub fn new() -> Self {
        UploadGate { inner: lk_agents_abi::telemetry::report::UploadGate::new() }
    }

    #[napi(js_name = "reset")]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[napi(getter, js_name = "disabled")]
    pub fn disabled(&self) -> bool {
        self.inner.disabled()
    }

    #[napi(js_name = "disable")]
    pub fn disable(&mut self) -> bool {
        self.inner.disable()
    }

    #[napi(js_name = "isDisabledResponse")]
    pub fn is_disabled_response(status_code: u16, body: napi::bindgen_prelude::Buffer) -> bool {
        lk_agents_abi::telemetry::report::UploadGate::is_disabled_response(status_code, &body)
    }
}

/// Session tags and outcome used in job reports.
#[napi(js_name = "Tagger")]
pub struct Tagger {
    pub(crate) inner: lk_agents_abi::telemetry::report::Tagger,
}

#[napi]
impl Tagger {
    #[napi(constructor)]
    pub fn new() -> Self {
        Tagger { inner: lk_agents_abi::telemetry::report::Tagger::with_real_clock() }
    }

    #[napi(js_name = "success")]
    pub fn success(&mut self, reason: Option<String>) {
        self.inner.success(reason);
    }

    #[napi(js_name = "fail")]
    pub fn fail(&mut self, reason: Option<String>) {
        self.inner.fail(reason);
    }

    #[napi(js_name = "add")]
    pub fn add(&mut self, tag: String, metadata: Option<serde_json::Value>) {
        self.inner.add(&tag, metadata);
    }

    #[napi(js_name = "remove")]
    pub fn remove(&mut self, tag: String) {
        self.inner.remove(&tag);
    }

    #[napi(js_name = "tags")]
    pub fn tags(&self) -> Vec<String> {
        self.inner.tags().map(|v| v.to_string()).collect::<Vec<_>>()
    }

    #[napi(js_name = "sortedTags")]
    pub fn sorted_tags(&self) -> Vec<String> {
        self.inner.sorted_tags().into_iter().map(|v| v.to_string()).collect::<Vec<_>>()
    }

    #[napi(js_name = "entries")]
    pub fn entries(&self) -> Vec<(String, Option<serde_json::Value>, f64)> {
        self.inner.entries_wire().into_iter().map(|v| { let (t0, t1, t2,) = v; (t0, t1, lk_agents_abi::epoch_secs(t2)) }).collect::<Vec<_>>()
    }

    #[napi(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[napi(js_name = "evaluations")]
    pub fn evaluations(&self) -> Vec<crate::generated::records::EvaluationRecord> {
        self.inner.evaluations().iter().map(|v| crate::generated::records::EvaluationRecord::from(v)).collect::<Vec<_>>()
    }

    #[napi(js_name = "outcome", ts_return_type = "'success' | 'fail' | null")]
    pub fn outcome(&self) -> Option<String> {
        self.inner.outcome().map(|v| v.as_str().to_owned())
    }

    #[napi(js_name = "outcomeReason")]
    pub fn outcome_reason(&self) -> Option<String> {
        self.inner.outcome_reason().map(|v| v.to_string())
    }

    #[napi(js_name = "recordEvaluation")]
    pub fn record_evaluation(&mut self, result: &crate::generated::objects::EvaluationResult) {
        self.inner.record_evaluation(&result.inner);
    }

    #[napi(js_name = "hasEvals")]
    pub fn has_evals(&self) -> bool {
        self.inner.has_evals()
    }
}

/// TF-IDF loop detection over a sliding window of final transcripts.
#[napi(js_name = "IvrLoopDetector")]
pub struct IvrLoopDetector {
    pub(crate) inner: std::sync::Mutex<lk_agents_voice::ivr::ivr_activity::TfidfLoopDetector>,
}

#[napi]
impl IvrLoopDetector {
    #[napi(constructor)]
    pub fn new(env: Env, window_size: Option<i64>, similarity_threshold: Option<f64>, consecutive_threshold: Option<u32>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let window_size = match window_size { Some(v) => <usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("window_size: {e}")))?, None => 20 };
            let similarity_threshold = similarity_threshold.unwrap_or(0.85);
            let consecutive_threshold = consecutive_threshold.unwrap_or(3);
            Ok(IvrLoopDetector { inner: std::sync::Mutex::new(lk_agents_voice::ivr::ivr_activity::TfidfLoopDetector::new(window_size, similarity_threshold, consecutive_threshold).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?) })
        })
    }

    #[napi(js_name = "reset")]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.reset();
    }

    #[napi(js_name = "addChunk")]
    pub fn add_chunk(&self, chunk: String) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.add_chunk(&chunk);
    }

    #[napi(js_name = "checkLoopDetection")]
    pub fn check_loop_detection(&self) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.check_loop_detection()
    }
}

/// `AgentSession(**kwargs)`'s options: `kwargs` crosses once as a raw dict (deprecated kwargs are
/// migrated here), and the resolved configuration comes back as the dict `session.options` views.
#[napi(js_name = "SessionOptions")]
pub struct SessionOptions {
    pub(crate) inner: lk_agents_voice::agent_activity::options::HostOptions,
}

#[napi]
impl SessionOptions {
    #[napi(js_name = "setAecWarmup")]
    pub fn set_aec_warmup(&self, duration: Option<f64>) {
        self.inner.set_aec_warmup(duration);
    }

    #[napi(getter, js_name = "sttContextSource")]
    pub fn stt_context_source(&self) -> String {
        self.inner.stt_context_source().to_string()
    }

    #[napi(getter, js_name = "turnDetectionExplicit")]
    pub fn turn_detection_explicit(&self) -> bool {
        self.inner.turn_detection_explicit()
    }
}

/// A cancelled Python awaiter drops its future and wakes nothing here, so the host calls
/// `abandon`(TaskHandle::abandon) in its `finally` for the driver to hand the parent frame back.
#[napi(js_name = "TaskHandle")]
pub struct TaskHandle {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::workflows::HostTaskHandle>,
}

#[napi]
impl TaskHandle {
    #[napi(constructor)]
    pub fn new(id: Option<String>) -> Self {
        TaskHandle { inner: std::sync::Arc::new(lk_agents_voice::workflows::HostTaskHandle::for_host(id)) }
    }

    #[napi(getter, js_name = "id")]
    pub fn id(&self) -> String {
        self.inner.host_id()
    }

    #[napi(getter, js_name = "done")]
    pub fn done(&self) -> bool {
        self.inner.host_done()
    }

    #[napi(getter, js_name = "phase")]
    pub fn phase(&self) -> String {
        self.inner.host_phase().to_string()
    }

    #[napi(js_name = "fail")]
    pub fn fail(&self, env: Env, message: String, class_name: Option<String>, reason: Option<String>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.host_fail(message, class_name, reason).map_err(crate::generated::errors::task_error_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "cancel")]
    pub fn cancel(&self, name: Option<String>) {
        self.inner.host_cancel(name);
    }

    #[napi(js_name = "requestCancel")]
    pub fn request_cancel(&self) {
        self.inner.host_request_cancel();
    }

    #[napi(js_name = "abandon")]
    pub fn abandon(&self) -> bool {
        self.inner.host_abandon()
    }

    #[napi(js_name = "waitForInactive", ts_return_type = "Promise<void>")]
    pub fn wait_for_inactive<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.host_wait_for_inactive().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "SpeechHandle")]
pub struct SpeechHandle {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::speech_handle::HostSpeech>,
}

#[napi]
impl SpeechHandle {
    #[napi(constructor)]
    pub fn new(speech_id: Option<String>, allow_interruptions: Option<bool>) -> Self {
        let speech_id = speech_id.unwrap_or(String::new());
        let allow_interruptions = allow_interruptions.unwrap_or(true);
        SpeechHandle { inner: std::sync::Arc::new(lk_agents_voice::speech_handle::HostSpeech::detached(speech_id, allow_interruptions)) }
    }

    #[napi(getter, js_name = "id")]
    pub fn id(&self) -> String {
        self.inner.id()
    }

    #[napi(getter, js_name = "numSteps")]
    pub fn num_steps(&self) -> u32 {
        self.inner.num_steps()
    }

    #[napi(getter, js_name = "scheduled")]
    pub fn scheduled(&self) -> bool {
        self.inner.scheduled()
    }

    #[napi(getter, js_name = "interrupted")]
    pub fn interrupted(&self) -> bool {
        self.inner.interrupted()
    }

    #[napi(js_name = "done")]
    pub fn done(&self) -> bool {
        self.inner.done()
    }

    #[napi(getter, js_name = "allowInterruptions")]
    pub fn allow_interruptions(&self) -> bool {
        self.inner.allow_interruptions()
    }

    #[napi(setter, js_name = "allowInterruptions")]
    pub fn set_allow_interruptions(&self, env: Env, value: bool) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.set_allow_interruptions(value).map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "holdInterruptions")]
    pub fn hold_interruptions(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.hold_interruptions().map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "releaseInterruptions")]
    pub fn release_interruptions(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.release_interruptions().map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "waitForPlayout", ts_return_type = "Promise<void>")]
    pub fn wait_for_playout<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_playout().await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitForGeneration", ts_return_type = "Promise<void>")]
    pub fn wait_for_generation<'env>(&self, env: &'env Env, step: Option<i64>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (step,) = crate::host_error::run(env, || {
            let step = step.map(|v| Ok::<_, crate::host_error::JsError>(<usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("step: {e}")))?)).transpose()?;
            Ok((step,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_generation(step).await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitForInterrupted", ts_return_type = "Promise<void>")]
    pub fn wait_for_interrupted<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_interrupted().await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitUntilCreated", ts_return_type = "Promise<void>")]
    pub fn wait_until_created<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_until_created().await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "Workflow")]
pub struct Workflow {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::workflows::builtins::HostWorkflow>,
}

#[napi]
impl Workflow {
    #[napi(getter, js_name = "instructions")]
    pub fn instructions(&self) -> (String, Option<String>, Option<String>) {
        self.inner.host_instructions()
    }

    #[napi(getter, js_name = "onEnterPrompt")]
    pub fn on_enter_prompt(&self) -> String {
        self.inner.on_enter_prompt()
    }

    #[napi(getter, js_name = "warnings")]
    pub fn warnings(&self) -> Vec<String> {
        self.inner.warnings()
    }
}

#[napi(js_name = "WorkflowReply")]
pub struct WorkflowReply {
    pub(crate) inner: lk_agents_voice::workflows::builtins::WorkflowReply,
}

#[napi]
impl WorkflowReply {
    #[napi(getter, js_name = "reply")]
    pub fn reply(&self) -> Option<String> {
        self.inner.reply()
    }

    #[napi(getter, js_name = "error")]
    pub fn error(&self) -> Option<String> {
        self.inner.error()
    }

    #[napi(getter, js_name = "errorClass")]
    pub fn error_class(&self) -> Option<String> {
        self.inner.error_class()
    }

    #[napi(getter, js_name = "errorReason")]
    pub fn error_reason(&self) -> Option<String> {
        self.inner.error_reason()
    }

    #[napi(getter, js_name = "completed")]
    pub fn completed(&self) -> bool {
        self.inner.completed()
    }

    #[napi(getter, js_name = "failure")]
    pub fn failure(&self) -> Option<String> {
        self.inner.failure()
    }

    #[napi(getter, js_name = "failureClass")]
    pub fn failure_class(&self) -> Option<String> {
        self.inner.failure_class()
    }

    #[napi(getter, js_name = "failureReason")]
    pub fn failure_reason(&self) -> Option<String> {
        self.inner.failure_reason()
    }
}

/// GetDtmfResult(user_input: 'str')
#[napi(js_name = "GetDtmfResult")]
pub struct GetDtmfResult {
    pub(crate) inner: lk_agents_voice::workflows::builtins::dtmf::DtmfResult,
}

#[napi]
impl GetDtmfResult {
    #[napi(constructor)]
    pub fn new(user_input: String) -> Self {
        GetDtmfResult { inner: lk_agents_voice::workflows::builtins::dtmf::DtmfResult::new(user_input) }
    }

    #[napi(getter, js_name = "userInput")]
    pub fn user_input(&self) -> String {
        self.inner.user_input()
    }
}

#[napi(js_name = "WarmTransferPlan")]
pub struct WarmTransferPlan {
    pub(crate) inner: lk_agents_voice::workflows::builtins::warm_transfer::HostWarmTransfer,
}

#[napi]
impl WarmTransferPlan {
    #[napi(getter, js_name = "instructions")]
    pub fn instructions(&self) -> (String, Option<String>, Option<String>) {
        self.inner.instructions()
    }

    #[napi(getter, js_name = "warnings")]
    pub fn warnings(&self) -> Vec<String> {
        self.inner.warnings()
    }

    #[napi(js_name = "humanAgentRoom")]
    pub fn human_agent_room(caller_room: String) -> String {
        lk_agents_voice::workflows::builtins::warm_transfer::HostWarmTransfer::human_agent_room(&caller_room)
    }

    #[napi(js_name = "humanAgentIdentity")]
    pub fn human_agent_identity() -> String {
        lk_agents_voice::workflows::builtins::warm_transfer::HostWarmTransfer::human_agent_identity()
    }

    #[napi(js_name = "sipRequest")]
    pub fn sip_request(&self, human_agent_room: String) -> crate::generated::records::SipParticipantRequest {
        crate::generated::records::SipParticipantRequest::from(&self.inner.sip_request(&human_agent_room))
    }

    #[napi(js_name = "ioEnabled")]
    pub fn io_enabled(&mut self, env: Env, enabled: bool, current: crate::generated::records::IoEnableState) -> napi::Result<crate::generated::records::IoEnableState> {
        crate::host_error::run(&env, || {
            let current = current.try_into()?;
            Ok(crate::generated::records::IoEnableState::from(&self.inner.io_enabled(enabled, current)))
        })
    }
}

/// GetDOBResult(date_of_birth: 'date', time_of_birth: 'time | None' = None)
#[napi(js_name = "GetDOBResult")]
pub struct GetDOBResult {
    pub(crate) inner: lk_agents_abi::voice::workflows::GetDOBResult,
}

#[napi]
impl GetDOBResult {
    #[napi(js_name = "formatLong")]
    pub fn format_long(&self) -> String {
        self.inner.format_long()
    }

    #[napi(js_name = "formatTime")]
    pub fn format_time(&self) -> Option<String> {
        self.inner.format_time()
    }
}

#[napi(js_name = "LlmResponseState")]
pub struct LlmResponseState {
    pub(crate) inner: lk_agents_inference::llm::ResponseState,
}

#[napi]
impl LlmResponseState {
    #[napi(constructor)]
    pub fn new() -> Self {
        LlmResponseState { inner: lk_agents_inference::llm::ResponseState::new() }
    }

    #[napi(js_name = "observe")]
    pub fn observe(&mut self, env: Env, chunk: crate::generated::records::LlmChunk) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let chunk = chunk.try_into()?;
            self.inner.observe(&chunk);
            Ok(())
        })
    }

    #[napi(getter, js_name = "retryable")]
    pub fn retryable(&self) -> bool {
        self.inner.retryable()
    }
}

/// `beta/workflows/task_group.py`'s scheduler as the host drives it: the loop body (factories,
/// awaits, callbacks, summarisation) stays in Python; this decides the stack order, the visited set
/// (written before the await), each outcome's arm, the `out_of_scope` tool and regress validation.
/// Result values never enter Rust. Each step comes back as its `kind`-tagged dict.
#[napi(js_name = "TaskSchedule")]
pub struct TaskSchedule {
    pub(crate) inner: std::sync::Mutex<lk_agents_voice::workflows::builtins::task_group::TaskGroupSchedule>,
}

#[napi]
impl TaskSchedule {
    #[napi(constructor)]
    pub fn new(summarize_chat_ctx: Option<bool>, return_exceptions: Option<bool>) -> Self {
        let summarize_chat_ctx = summarize_chat_ctx.unwrap_or(true);
        let return_exceptions = return_exceptions.unwrap_or(false);
        TaskSchedule { inner: std::sync::Mutex::new(lk_agents_voice::workflows::builtins::task_group::TaskGroupSchedule::new(summarize_chat_ctx, return_exceptions)) }
    }

    #[napi(js_name = "visited")]
    pub fn visited(&self, task_id: String) -> bool {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.visited(&task_id)
    }

    #[napi(js_name = "add")]
    pub fn add(&self, id: String, description: String) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.host_add(id, description);
    }

    #[napi(js_name = "validateRegress")]
    pub fn validate_regress(&self, task_ids: Vec<String>) -> Option<String> {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.host_validate_regress(task_ids)
    }
}

#[napi(js_name = "SessionClock")]
pub struct SessionClock {
    pub(crate) inner: lk_agents_voice::agent_session::Timebase,
}

#[napi]
impl SessionClock {
}

#[napi(js_name = "TextPusher")]
pub struct TextPusher {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::agent_session::TextPusher>,
}

#[napi]
impl TextPusher {
    #[napi(js_name = "push", ts_return_type = "Promise<boolean>")]
    pub fn push<'env>(&self, env: &'env Env, text: String) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, bool>> {
        let (text,) = crate::host_error::run(env, || {
            Ok((text,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.push(text).await)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "end")]
    pub fn end(&self) {
        self.inner.end();
    }
}

#[napi(js_name = "FramePusher")]
pub struct FramePusher {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::agent_session::FramePusher>,
}

#[napi]
impl FramePusher {
    #[napi(js_name = "end")]
    pub fn end(&self) {
        self.inner.end();
    }
}

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
use crate::session::AgentSession;

#[napi]
impl AgentSession {
    #[napi(js_name = "start", ts_return_type = "Promise<void>")]
    pub fn start<'env>(&self, env: &'env Env, capture_run: Option<bool>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            let capture_run = capture_run.unwrap_or(false);
            self.inner.start(capture_run).map_err(crate::generated::errors::open_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::generated::errors::start_failed_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitUntilReady", ts_return_type = "Promise<void>")]
    pub fn wait_until_ready<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            Ok(self.inner.wait_until_ready())
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "setAecWarmup")]
    pub fn set_aec_warmup(&self, env: Env, duration: Option<f64>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.set_aec_warmup(duration).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "resolvedOptions")]
    pub fn resolved_options(&self) -> Option<crate::generated::records::ResolvedSessionOptions> {
        self.inner.resolved_options().map(|v| crate::generated::records::ResolvedSessionOptions::from(&v))
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env, reason: Option<String>, drain: Option<bool>, has_error: Option<bool>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            let reason = reason.as_deref().unwrap_or("user_initiated");
            let drain = drain.unwrap_or(false);
            let has_error = has_error.unwrap_or(false);
            self.inner.aclose(&reason, drain, has_error).map_err(crate::generated::errors::unknown_close_reason_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "closeIo", ts_return_type = "Promise<void>")]
    pub fn close_io<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            Ok(self.inner.host_close_io())
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(getter, js_name = "linkedParticipant")]
    pub fn linked_participant(&self) -> Option<String> {
        self.inner.linked_participant()
    }

    #[napi(js_name = "setParticipant", ts_return_type = "Promise<void>")]
    pub fn set_participant<'env>(&self, env: &'env Env, identity: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            Ok(self.inner.set_participant(identity))
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "unsetParticipant", ts_return_type = "Promise<void>")]
    pub fn unset_participant<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            Ok(self.inner.unset_participant())
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "steps")]
    pub fn steps(&self) -> Vec<String> {
        self.inner.host_steps()
    }

    #[napi(js_name = "hookReturned")]
    pub fn hook_returned(&self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.hook_returned(id).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "hookFailed")]
    pub fn hook_failed(&self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.hook_failed(id).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "hookRendered")]
    pub fn hook_rendered(&self, env: Env, id: i64, text: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.hook_rendered(id, text).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "hookToolsExecuted")]
    pub fn hook_tools_executed(&self, env: Env, id: i64, reply_required: Vec<bool>, handoff_required: bool) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.hook_tools_executed(id, reply_required, handoff_required).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "hookFiller")]
    pub fn hook_filler(&self, env: Env, id: i64, text: Option<String>, spoke: Option<bool>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            let spoke = spoke.unwrap_or(false);
            self.inner.hook_filler(id, text, spoke).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "fillerStart")]
    pub fn filler_start(&self, env: Env, id: i64, call_id: String, speech_id: String, delay: f64, interval: Option<f64>, max_steps: Option<u32>, text: Option<String>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.filler_start(id, call_id, &speech_id, delay, interval, max_steps, text).map_err(crate::generated::errors::command_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "fillerStop")]
    pub fn filler_stop(&self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.filler_stop(id).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "clearUserTurn")]
    pub fn clear_user_turn(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.clear_user_turn().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "setIoEnabled")]
    pub fn set_io_enabled(&self, env: Env, leg: String, enabled: bool) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.set_io_enabled(&leg, enabled).map_err(crate::generated::errors::command_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "interrupt", ts_return_type = "Promise<void>")]
    pub fn interrupt<'env>(&self, env: &'env Env, force: Option<bool>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            let force = force.unwrap_or(false);
            self.inner.interrupt(force).map_err(crate::generated::errors::session_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::generated::errors::not_interruptible_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "drain", ts_return_type = "Promise<string | null>")]
    pub fn drain<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Option<String>>> {
        let reply = crate::host_error::run(env, || {
            self.inner.drain().map_err(crate::generated::errors::session_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "dtmfStart", ts_return_type = "Promise<DtmfComplete | DtmfIncomplete | DtmfStopped>")]
    pub fn dtmf_start<'env>(&self, env: &'env Env, num_digits: i64, ask_for_confirmation: Option<bool>, dtmf_input_timeout: Option<f64>, dtmf_stop_event: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, crate::generated::records::DtmfOutcomeWire>> {
        let reply = crate::host_error::run(env, || {
            let num_digits = <usize>::try_from(num_digits).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("num_digits: {e}")))?;
            let ask_for_confirmation = ask_for_confirmation.unwrap_or(false);
            let dtmf_input_timeout = dtmf_input_timeout.unwrap_or(4.0);
            let dtmf_stop_event = dtmf_stop_event.unwrap_or("#".to_owned());
            self.inner.dtmf_start(num_digits, ask_for_confirmation, dtmf_input_timeout, dtmf_stop_event).map_err(crate::generated::errors::command_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(reply.await)) }, |env, out| match out {
            Ok(out) => Ok(crate::generated::records::DtmfOutcomeWire::from(&out)),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "dtmfReplyRunning")]
    pub fn dtmf_reply_running(&self) -> bool {
        self.inner.dtmf_reply_running()
    }

    #[napi(js_name = "dtmfDigit")]
    pub fn dtmf_digit(&self, env: Env, digit: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.dtmf_digit(digit).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "dtmfStop")]
    pub fn dtmf_stop(&self) {
        self.inner.dtmf_stop();
    }

    #[napi(js_name = "waitForIdle", ts_return_type = "Promise<void>")]
    pub fn wait_for_idle<'env>(&self, env: &'env Env, hold: Option<i64>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let reply = crate::host_error::run(env, || {
            let hold = hold.map(|v| Ok::<_, crate::host_error::JsError>(<u64>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("hold: {e}")))?)).transpose()?;
            self.inner.wait_for_idle(hold).map_err(crate::generated::errors::session_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::host_error::idle_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "holdFloor", ts_return_type = "Promise<number>")]
    pub fn hold_floor<'env>(&self, env: &'env Env, origin: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, i64>> {
        let reply = crate::host_error::run(env, || {
            self.inner.hold_floor(origin.as_deref()).map_err(crate::generated::errors::command_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::host_error::idle_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(i64::try_from(out).unwrap_or(i64::MAX)),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "releaseFloor")]
    pub fn release_floor(&self, env: Env, hold: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let hold = <u64>::try_from(hold).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("hold: {e}")))?;
            self.inner.release_floor(hold);
            Ok(())
        })
    }

    #[napi(js_name = "claimUserTurn")]
    pub fn claim_user_turn(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.claim_user_turn().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "releaseUserTurn")]
    pub fn release_user_turn(&self) {
        self.inner.release_user_turn();
    }

    #[napi(js_name = "run")]
    pub fn run(&self, env: Env, user_input: Option<String>, expects_output: bool, max_retries: u32, retry_instructions: String, origin: Option<String>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.run(user_input, expects_output, max_retries, retry_instructions, origin.as_deref()).map_err(crate::generated::errors::command_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "ivrFinal")]
    pub fn ivr_final(&self, env: Env, text: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.ivr_final(&text).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "ivrClose")]
    pub fn ivr_close(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.ivr_close().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdStartDetectionTimer")]
    pub fn amd_start_detection_timer(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_start_detection_timer().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdStartListening")]
    pub fn amd_start_listening(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_start_listening().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdSettle")]
    pub fn amd_settle(&self, env: Env, category: String, reason: String) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_settle(&category, &reason).map_err(crate::generated::errors::command_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdEndInput")]
    pub fn amd_end_input(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_end_input().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdExecute")]
    pub fn amd_execute(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_execute().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "amdClose")]
    pub fn amd_close(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.amd_close().map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "keyterms")]
    pub fn keyterms(&self) -> Option<Vec<String>> {
        self.inner.keyterms()
    }

    #[napi(js_name = "setStaticKeyterms")]
    pub fn set_static_keyterms(&self, env: Env, terms: Vec<String>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.set_static_keyterms(terms).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "commitUserTurn", ts_return_type = "Promise<string>")]
    pub fn commit_user_turn<'env>(&self, env: &'env Env, transcript_timeout: Option<f64>, stt_flush_duration: Option<f64>, skip_reply: Option<bool>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, String>> {
        let reply = crate::host_error::run(env, || {
            let transcript_timeout = transcript_timeout.unwrap_or(2.0);
            let stt_flush_duration = stt_flush_duration.unwrap_or(2.0);
            let skip_reply = skip_reply.unwrap_or(false);
            self.inner.commit_user_turn(transcript_timeout, stt_flush_duration, skip_reply).map_err(crate::generated::errors::session_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::host_error::commit_superseded_to_node)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// The LLM a judge calls, plus the host's live `submit_verdict` tool list. Built by `judge_llm`.
#[napi(js_name = "JudgeLlm")]
pub struct JudgeLlm {
    pub(crate) inner: std::sync::Arc<lk_agents_abi::evals::JudgeLlm>,
}

#[napi]
impl JudgeLlm {
    #[napi(getter, js_name = "model")]
    pub fn model(&self) -> String {
        self.inner.model().to_string()
    }

    #[napi(getter, js_name = "hostBacked")]
    pub fn host_backed(&self) -> bool {
        self.inner.is_host_backed()
    }

    #[napi(getter, js_name = "supportsTemperature")]
    pub fn supports_temperature(&self) -> bool {
        self.inner.supports_temperature()
    }
}

/// `metrics.ModelUsageCollector` -- the live per-model rollup.
#[napi(js_name = "ModelUsageCollector")]
pub struct ModelUsageCollector {
    pub(crate) inner: std::sync::Mutex<lk_agents_abi::metrics::ModelUsageCollector>,
}

#[napi]
impl ModelUsageCollector {
    #[napi(constructor)]
    pub fn new() -> Self {
        ModelUsageCollector { inner: std::sync::Mutex::new(lk_agents_abi::metrics::ModelUsageCollector::new()) }
    }

    #[napi(js_name = "flatten", ts_return_type = "Array<LlmModelUsage | TtsModelUsage | SttModelUsage | InterruptionModelUsage | EotModelUsage>")]
    pub fn flatten(&self) -> Vec<crate::generated::records::ModelUsage> {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.flatten().into_iter().map(|v| crate::generated::records::ModelUsage::from(&v)).collect::<Vec<_>>()
    }
}

/// `metrics.UsageCollector` -- the deprecated session-wide totals, with no per-model breakdown.
#[napi(js_name = "UsageCollector")]
pub struct UsageCollector {
    pub(crate) inner: std::sync::Mutex<lk_agents_abi::metrics::UsageCollector>,
}

#[napi]
impl UsageCollector {
    #[napi(constructor)]
    pub fn new() -> Self {
        UsageCollector { inner: std::sync::Mutex::new(lk_agents_abi::metrics::UsageCollector::new()) }
    }

    #[napi(js_name = "summary")]
    pub fn summary(&self) -> crate::generated::records::UsageSummary {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::generated::records::UsageSummary::from(&inner.summary())
    }
}

#[napi(js_name = "ExpFilter")]
pub struct ExpFilter {
    pub(crate) inner: lk_agents_abi::ExpFilter,
}

#[napi]
impl ExpFilter {
    #[napi(constructor)]
    pub fn new(env: Env, alpha: f64, max_val: Option<f64>, min_val: Option<f64>, initial: Option<f64>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let max_val = match max_val { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            let min_val = match min_val { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            let initial = match initial { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            Ok(ExpFilter { inner: lk_agents_abi::ExpFilter::new(alpha, max_val, min_val, initial).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))? })
        })
    }

    #[napi(js_name = "reset")]
    pub fn reset(&mut self, env: Env, alpha: Option<f64>, initial: Option<f64>, min_val: Option<f64>, max_val: Option<f64>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let alpha = match alpha { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            let initial = match initial { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            let min_val = match min_val { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            let max_val = match max_val { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            self.inner.reset(alpha, initial, min_val, max_val).map_err(|e| crate::host_error::JsError::new("AssertionError", e.to_string()))?;
            Ok(())
        })
    }

    #[napi(js_name = "apply")]
    pub fn apply(&mut self, env: Env, exp: f64, sample: Option<f64>) -> napi::Result<f64> {
        crate::host_error::run(&env, || {
            let sample = match sample { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };
            self.inner.apply(exp, sample).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
        })
    }

    #[napi(getter, js_name = "value")]
    pub fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    #[napi(js_name = "updateBase")]
    pub fn update_base(&mut self, alpha: f64) {
        self.inner.update_base(alpha);
    }
}

#[napi(js_name = "MovingAverage")]
pub struct MovingAverage {
    pub(crate) inner: lk_agents_abi::MovingAverage,
}

#[napi]
impl MovingAverage {
    #[napi(constructor)]
    pub fn new(env: Env, window_size: i64) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let window_size = <usize>::try_from(window_size).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("window_size: {e}")))?;
            Ok(MovingAverage { inner: lk_agents_abi::MovingAverage::new(window_size) })
        })
    }

    #[napi(js_name = "addSample")]
    pub fn add_sample(&mut self, sample: f64) {
        self.inner.add_sample(sample);
    }

    #[napi(js_name = "getAvg")]
    pub fn get_avg(&self) -> f64 {
        self.inner.get_avg()
    }

    #[napi(js_name = "reset")]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[napi(js_name = "size")]
    pub fn size(&self) -> i64 {
        i64::try_from(self.inner.size()).unwrap_or(i64::MAX)
    }
}

#[napi(js_name = "AudioByteStream")]
pub struct AudioByteStream {
    pub(crate) inner: lk_agents_abi::AudioByteStream,
}

#[napi]
impl AudioByteStream {
    #[napi(constructor)]
    pub fn new(sample_rate: u32, num_channels: u16, samples_per_channel: Option<u32>, progressive: Option<bool>) -> Self {
        let progressive = progressive.unwrap_or(false);
        AudioByteStream { inner: lk_agents_abi::AudioByteStream::new(sample_rate, num_channels, samples_per_channel, progressive) }
    }

    #[napi(js_name = "clear")]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[napi(js_name = "SentencePacer")]
pub struct SentencePacer {
    pub(crate) inner: lk_agents_abi::tts::stream_pacer::SentencePacer,
}

#[napi]
impl SentencePacer {
    #[napi(constructor)]
    pub fn new(env: Env, minimum_audio: f64, maximum_text: i64) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let maximum_text = <usize>::try_from(maximum_text).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("maximum_text: {e}")))?;
            Ok(SentencePacer { inner: lk_agents_abi::tts::stream_pacer::SentencePacer::new(minimum_audio, maximum_text) })
        })
    }

    #[napi(js_name = "push")]
    pub fn push(&mut self, text: String) {
        self.inner.push(text);
    }

    #[napi(js_name = "end")]
    pub fn end(&mut self) {
        self.inner.end();
    }

    #[napi(js_name = "poll")]
    pub fn poll(&mut self, now: f64, audio: f64) -> (Option<String>, f64, Option<f64>) {
        self.inner.poll(now, audio)
    }
}

/// A thread-safe buffer that behaves like an IO stream.
#[napi(js_name = "StreamBuffer")]
pub struct StreamBuffer {
    pub(crate) inner: lk_agents_media::StreamBuffer,
}

#[napi]
impl StreamBuffer {
    #[napi(constructor)]
    pub fn new() -> Self {
        StreamBuffer { inner: lk_agents_media::StreamBuffer::new() }
    }

    #[napi(js_name = "write")]
    pub fn write(&self, data: napi::bindgen_prelude::Buffer) {
        self.inner.write(&data);
    }

    #[napi(js_name = "endInput")]
    pub fn end_input(&self) {
        self.inner.end_input();
    }

    #[napi(js_name = "close")]
    pub fn close(&self) {
        self.inner.close();
    }

    #[napi(js_name = "read")]
    pub fn read(&self, env: Env, size: Option<i64>) -> napi::Result<napi::bindgen_prelude::Buffer> {
        crate::host_error::run(&env, || {
            let size = match size { Some(v) => <isize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("size: {e}")))?, None => -1 };
            Ok(napi::bindgen_prelude::Buffer::from(self.inner.read_up_to(size)))
        })
    }
}

/// `AudioStreamDecoder`'s engine: bytes in, interleaved 16-bit PCM out, one decoder per stream.
#[napi(js_name = "AudioDecoder")]
pub struct AudioDecoder {
    pub(crate) inner: std::sync::Mutex<lk_agents_media::Decoder>,
}

#[napi]
impl AudioDecoder {
    #[napi(constructor)]
    pub fn new(env: Env, mime: String, sample_rate: Option<u32>, num_channels: Option<u16>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let sample_rate = sample_rate.unwrap_or(48000);
            let num_channels = num_channels.unwrap_or(1);
            Ok(AudioDecoder { inner: std::sync::Mutex::new(lk_agents_media::Decoder::new(&mime, sample_rate, num_channels).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?) })
        })
    }

    #[napi(getter, js_name = "outputFormat")]
    pub fn output_format(&self) -> Option<(u32, u16)> {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.output_format_pair()
    }

    #[napi(js_name = "push")]
    pub fn push(&self, env: Env, data: napi::bindgen_prelude::Buffer) -> napi::Result<napi::bindgen_prelude::Buffer> {
        crate::host_error::run(&env, || {
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(napi::bindgen_prelude::Buffer::from(inner.push_bytes(&data).map_err(|e| crate::host_error::JsError::new("RuntimeError", e.to_string()))?))
        })
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self, env: Env) -> napi::Result<napi::bindgen_prelude::Buffer> {
        crate::host_error::run(&env, || {
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(napi::bindgen_prelude::Buffer::from(inner.flush_bytes().map_err(|e| crate::host_error::JsError::new("RuntimeError", e.to_string()))?))
        })
    }
}

#[napi(js_name = "FrameDecoder")]
pub struct FrameDecoder {
    pub(crate) inner: lk_agents_voice::remote_session::FrameDecoder,
}

#[napi]
impl FrameDecoder {
    #[napi(constructor)]
    pub fn new() -> Self {
        FrameDecoder { inner: lk_agents_voice::remote_session::FrameDecoder::new() }
    }

    #[napi(getter, js_name = "pending")]
    pub fn pending(&self) -> i64 {
        i64::try_from(self.inner.pending()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "decodeFrames")]
    pub fn decode_frames(&mut self, env: Env, chunk: napi::bindgen_prelude::Buffer) -> napi::Result<Vec<napi::bindgen_prelude::Buffer>> {
        crate::host_error::run(&env, || {
            Ok(self.inner.decode_frames(&chunk).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?.into_iter().map(|v| napi::bindgen_prelude::Buffer::from(v)).collect::<Vec<_>>())
        })
    }
}

/// `voice/remote_session.py`'s `SessionHost`, running in Rust. Events queue from construction.
#[napi(js_name = "RemoteSessionHost")]
pub struct RemoteSessionHost {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::remote_session::driver::HostRuntime>,
}

#[napi]
impl RemoteSessionHost {
    #[napi(getter, js_name = "started")]
    pub fn started(&self) -> bool {
        self.inner.started()
    }

    #[napi(js_name = "inFlight")]
    pub fn in_flight(&self) -> Vec<String> {
        self.inner.in_flight()
    }

    #[napi(js_name = "sendEncodedEvent")]
    pub fn send_encoded_event(&self, env: Env, event: napi::bindgen_prelude::Buffer, created_at: Option<f64>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.send_encoded(&event, created_at).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?;
            Ok(())
        })
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<unknown>")]
    pub fn aclose<'env>(&self, env: &'env Env, drain: f64) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Vec<String>>> {
        let (drain,) = crate::host_error::run(env, || {
            let drain = std::time::Duration::try_from_secs_f64(drain).map_err(|e| crate::host_error::JsError::new("ValueError", format!("drain: {e}")))?;
            Ok((drain,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose(drain).await)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// `voice/remote_session.py`'s `RemoteSession`, running in Rust: ids, correlation, timeouts and the
/// ready poll.
#[napi(js_name = "RemoteSessionClient")]
pub struct RemoteSessionClient {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::remote_session::driver::ClientRuntime>,
}

#[napi]
impl RemoteSessionClient {
    #[napi(getter, js_name = "started")]
    pub fn started(&self) -> bool {
        self.inner.started()
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "request", ts_return_type = "Promise<unknown>")]
    pub fn request<'env>(&self, env: &'env Env, request: napi::bindgen_prelude::Buffer, timeout: f64) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, napi::bindgen_prelude::Buffer>> {
        let (request, timeout,) = crate::host_error::run(env, || {
            let request = request.to_vec();
            Ok((request, timeout,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.request_encoded(request, timeout).await.map_err(crate::generated::errors::request_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(napi::bindgen_prelude::Buffer::from(out)),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitForReady", ts_return_type = "Promise<void>")]
    pub fn wait_for_ready<'env>(&self, env: &'env Env, timeout: f64, retry_interval: f64) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (timeout, retry_interval,) = crate::host_error::run(env, || {
            Ok((timeout, retry_interval,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_ready_secs(timeout, retry_interval).await.map_err(crate::generated::errors::ready_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// `TTS.Markup`, keyed by `_provider_key()`. An unknown key is the no-dialect markup.
#[napi(js_name = "Markup")]
pub struct Markup {
    pub(crate) inner: lk_agents_abi::tts::markup::Markup,
}

#[napi]
impl Markup {
    #[napi(constructor)]
    pub fn new(provider_key: String) -> Self {
        Markup { inner: lk_agents_abi::tts::markup::Markup::from_provider_key(&provider_key) }
    }

    #[napi(js_name = "_providerKey")]
    pub fn _provider_key(&self) -> String {
        self.inner.provider_key().to_string()
    }

    #[napi(js_name = "normalize")]
    pub fn normalize(&self, text: String) -> String {
        self.inner.normalize(&text)
    }

    #[napi(js_name = "convert")]
    pub fn convert(&self, text: String) -> String {
        self.inner.convert(&text)
    }
}

/// One element of `ChatMessage.content`: `str | ImageContent | AudioContent`, built by the host one
/// constructor per arm so Rust never reads attributes or dispatches on `.type`.
#[napi(js_name = "ChatContent")]
pub struct ChatContent {
    pub(crate) inner: lk_agents_abi::ChatContent,
}

#[napi]
impl ChatContent {
    #[napi(getter, js_name = "kind")]
    pub fn kind(&self) -> String {
        self.inner.kind().to_string()
    }
}

/// One row of conversation history: the discriminated union of `chat_context.py:407`, built by the
/// host one constructor per variant. `created_at` is a wall clock; ids are always host-supplied.
#[napi(js_name = "ChatItem")]
pub struct ChatItem {
    pub(crate) inner: lk_agents_abi::ChatItem,
}

#[napi]
impl ChatItem {
    #[napi(getter, js_name = "type")]
    pub fn type_literal(&self) -> String {
        self.inner.type_literal().to_string()
    }

    #[napi(getter, js_name = "id")]
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    #[napi(getter, js_name = "createdAt")]
    pub fn created_at(&self) -> f64 {
        lk_agents_abi::epoch_secs(self.inner.created_at())
    }
}

/// One tool as the search sees it; identity stays with the host.
#[napi(js_name = "SearchDocument")]
pub struct SearchDocument {
    pub(crate) inner: lk_agents_abi::tokenize::search::Document,
}

#[napi]
impl SearchDocument {
    #[napi(constructor)]
    pub fn new(name: String, description: String, parameters: Vec<(String, String)>) -> Self {
        SearchDocument { inner: lk_agents_abi::tokenize::search::Document::new(&name, &description, &parameters) }
    }

    #[napi(js_name = "keywordScore")]
    pub fn keyword_score(&self, env: Env, terms: Vec<String>) -> napi::Result<f64> {
        crate::host_error::run(&env, || {
            self.inner.keyword_score(&terms).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))
        })
    }
}

#[napi(js_name = "BM25Index")]
pub struct BM25Index {
    pub(crate) inner: lk_agents_abi::tokenize::search::Bm25,
}

#[napi]
impl BM25Index {
    #[napi(constructor)]
    pub fn new(k1: f64, b: f64) -> Self {
        BM25Index { inner: lk_agents_abi::tokenize::search::Bm25::new(k1, b) }
    }

    #[napi(js_name = "clear")]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[napi(js_name = "score")]
    pub fn score(&self, doc: &crate::generated::objects::SearchDocument, terms: Vec<String>) -> f64 {
        self.inner.score(&doc.inner, &terms)
    }
}

/// A host sink's side (`io.AudioOutput._core_tracker()`): it reports its playback events here, in
/// its own `time.time()` stamps and float seconds.
#[napi(js_name = "PlaybackTracker")]
pub struct PlaybackTracker {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::io::PlaybackTracker>,
}

#[napi]
impl PlaybackTracker {
    #[napi(constructor)]
    pub fn new() -> Self {
        PlaybackTracker { inner: std::sync::Arc::new(lk_agents_voice::io::PlaybackTracker::new()) }
    }

    #[napi(js_name = "playbackStarted")]
    pub fn playback_started(&self, created_at: f64) {
        self.inner.report_started(created_at);
    }

    #[napi(js_name = "playbackProgressed")]
    pub fn playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        self.inner.report_progressed(started_at, offset, duration);
    }

    #[napi(js_name = "playbackFinished")]
    pub fn playback_finished(&self, playback_position: f64, interrupted: bool, synchronized_transcript: Option<String>) -> bool {
        self.inner.report_finished(playback_position, interrupted, synchronized_transcript)
    }
}

#[napi(js_name = "CoreTextOutput")]
pub struct CoreTextOutput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::io::CoreTextOutput>,
}

#[napi]
impl CoreTextOutput {
    #[napi(js_name = "captureText", ts_return_type = "Promise<void>")]
    pub fn capture_text<'env>(&self, env: &'env Env, text: String) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (text,) = crate::host_error::run(env, || {
            Ok((text,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.capture_text(text).await.map_err(crate::generated::errors::io_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "captureTimed", ts_return_type = "Promise<void>")]
    pub fn capture_timed<'env>(&self, env: &'env Env, text: String, start_time: Option<f64>, end_time: Option<f64>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (text, start_time, end_time,) = crate::host_error::run(env, || {
            Ok((text, start_time, end_time,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.capture_timed(text, start_time, end_time).await.map_err(crate::generated::errors::io_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self) {
        self.inner.flush();
    }
}

#[napi(js_name = "CoreAudioSink")]
pub struct CoreAudioSink {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::io::CoreAudioSink>,
}

#[napi]
impl CoreAudioSink {
    #[napi(js_name = "flush")]
    pub fn flush(&self) {
        self.inner.flush();
    }

    #[napi(js_name = "clearBuffer")]
    pub fn clear_buffer(&self) {
        self.inner.clear_buffer();
    }
}

/// The host's view: what `_core_rtc.LocalParticipant._core_publish_audio_track` builds its
/// `rtc.AudioSource`, `LocalAudioTrack` and `rtc.TrackPublishOptions` from.
#[napi(js_name = "TrackPublishOptions")]
pub struct TrackPublishOptions {
    pub(crate) inner: lk_agents_abi::voice::rtc::TrackPublishOptions,
}

#[napi]
impl TrackPublishOptions {
    #[napi(getter, js_name = "name")]
    pub fn name(&self) -> String {
        self.inner.name()
    }

    #[napi(getter, js_name = "source", ts_return_type = "'unknown' | 'camera' | 'microphone' | 'screenshare' | 'screenshare_audio'")]
    pub fn source(&self) -> String {
        self.inner.source().as_str().to_owned()
    }

    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    #[napi(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> u16 {
        self.inner.num_channels()
    }

    #[napi(getter, js_name = "queueSizeMs")]
    pub fn queue_size_ms(&self) -> u32 {
        self.inner.queue_size_ms()
    }
}

/// The host's view: what `_core_rtc.TrackPublication._core_open_audio_stream` opens its
/// `rtc.AudioStream` (and, with `auto_gain_control`, its `AudioProcessingModule`) with.
#[napi(js_name = "AudioStreamOptions")]
pub struct AudioStreamOptions {
    pub(crate) inner: lk_agents_abi::voice::rtc::AudioStreamOptions,
}

#[napi]
impl AudioStreamOptions {
    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate()
    }

    #[napi(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> Option<u16> {
        self.inner.num_channels()
    }

    #[napi(getter, js_name = "frameSizeMs")]
    pub fn frame_size_ms(&self) -> Option<u32> {
        self.inner.frame_size_ms()
    }

    #[napi(getter, js_name = "autoGainControl")]
    pub fn auto_gain_control(&self) -> bool {
        self.inner.auto_gain_control()
    }
}

/// What a host `io.VideoInput` answers `_core_next_frame()` with: its frame (interned; only the key
/// crosses back) and the size it read off it.
#[napi(js_name = "VideoFrameRef")]
pub struct VideoFrameRef {
    pub(crate) inner: lk_agents_abi::voice::rtc::VideoFrameRef,
}

#[napi]
impl VideoFrameRef {
}

/// The host's view: `_core_rtc.LocalParticipant._core_stream_text` spreads it into
/// `stream_text(..)`, leaving `sender_identity` out when it is `None` (an explicit `None` would
/// publish the user's transcript as the agent).
#[napi(js_name = "StreamTextOptions")]
pub struct StreamTextOptions {
    pub(crate) inner: lk_agents_abi::voice::rtc::StreamTextOptions,
}

#[napi]
impl StreamTextOptions {
    #[napi(getter, js_name = "topic")]
    pub fn topic(&self) -> String {
        self.inner.topic()
    }

    #[napi(getter, js_name = "attributes")]
    pub fn attributes(&self) -> std::collections::HashMap<String, String> {
        self.inner.attributes().into_iter().map(|(k, v)| (k, v)).collect::<std::collections::HashMap<_, _>>()
    }
}

/// The host's view: `_core_rtc.LocalParticipant._core_publish_transcription` builds the
/// `rtc.Transcription` and its `rtc.TranscriptionSegment`s (keyword dicts, `final` spelled plainly).
#[napi(js_name = "Transcription")]
pub struct Transcription {
    pub(crate) inner: lk_agents_abi::voice::rtc::Transcription,
}

#[napi]
impl Transcription {
    #[napi(getter, js_name = "segments")]
    pub fn segments(&self) -> Vec<crate::generated::records::TranscriptionSegment> {
        self.inner.segments().into_iter().map(|v| crate::generated::records::TranscriptionSegment::from(&v)).collect::<Vec<_>>()
    }
}

/// What a host audio source's `PlayoutLedger` hands back, and the core reads back unchanged.
#[napi(js_name = "PlaybackInfo")]
pub struct PlaybackInfo {
    pub(crate) inner: lk_agents_abi::voice::rtc::PlaybackInfo,
}

#[napi]
impl PlaybackInfo {
    #[napi(getter, js_name = "played")]
    pub fn played(&self) -> f64 {
        self.inner.played().as_secs_f64()
    }

    #[napi(getter, js_name = "discarded")]
    pub fn discarded(&self) -> f64 {
        self.inner.discarded().as_secs_f64()
    }
}

/// What a host `io.AudioOutput` answers its `_core_capabilities()` with.
#[napi(js_name = "AudioOutputCapabilities")]
pub struct AudioOutputCapabilities {
    pub(crate) inner: lk_agents_abi::voice::rtc::AudioOutputCapabilities,
}

#[napi]
impl AudioOutputCapabilities {
    #[napi(constructor)]
    pub fn new(pause: bool) -> Self {
        AudioOutputCapabilities { inner: lk_agents_abi::voice::rtc::AudioOutputCapabilities::new(pause) }
    }

    #[napi(getter, js_name = "pause")]
    pub fn pause(&self) -> bool {
        self.inner.pause()
    }
}

#[napi(js_name = "RoomAudioOutput")]
pub struct RoomAudioOutput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::room_io::_output::RoomAudioLeg>,
}

#[napi]
impl RoomAudioOutput {
    #[napi(js_name = "start", ts_return_type = "Promise<void>")]
    pub fn start<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.start().await.map_err(crate::generated::errors::io_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate()
    }

    #[napi(getter, js_name = "canPause")]
    pub fn can_pause(&self) -> bool {
        self.inner.can_pause()
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self) {
        self.inner.flush();
    }

    #[napi(js_name = "clearBuffer")]
    pub fn clear_buffer(&self) {
        self.inner.clear_buffer();
    }

    #[napi(js_name = "pause")]
    pub fn pause(&self) {
        self.inner.pause();
    }

    #[napi(js_name = "resume")]
    pub fn resume(&self) {
        self.inner.resume();
    }
}

#[napi(js_name = "RoomTranscriptionOutput")]
pub struct RoomTranscriptionOutput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::room_io::_output::transcription::RoomTranscriptionLeg>,
}

#[napi]
impl RoomTranscriptionOutput {
    #[napi(js_name = "captureText", ts_return_type = "Promise<void>")]
    pub fn capture_text<'env>(&self, env: &'env Env, text: String) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (text,) = crate::host_error::run(env, || {
            Ok((text,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.capture_text(text).await.map_err(crate::generated::errors::io_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self) {
        self.inner.flush();
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "FakeRoom")]
pub struct FakeRoom {
    pub(crate) inner: lk_agents_voice::room_io::rooms::ScriptedRoom,
}

#[napi]
impl FakeRoom {
    #[napi(constructor)]
    pub fn new(env: Env, name: Option<String>, sample_rate: Option<u32>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let name = name.unwrap_or("fake".to_owned());
            let sample_rate = sample_rate.unwrap_or(48_000);
            Ok(FakeRoom { inner: lk_agents_voice::room_io::rooms::ScriptedRoom::new(name, sample_rate).map_err(crate::generated::errors::room_refused_to_node)? })
        })
    }

    #[napi(getter, js_name = "name")]
    pub fn name(&self) -> String {
        self.inner.name()
    }

    #[napi(js_name = "join")]
    pub fn join(&self, identity: Option<String>, track_sid: Option<String>) {
        let identity = identity.unwrap_or("user".to_owned());
        self.inner.join(identity, track_sid);
    }

    #[napi(js_name = "speak")]
    pub fn speak(&self, env: Env, frames: Option<i64>) -> napi::Result<i64> {
        crate::host_error::run(&env, || {
            let frames = match frames { Some(v) => <usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("frames: {e}")))?, None => 1 };
            Ok(i64::try_from(self.inner.speak(frames).map_err(crate::generated::errors::room_refused_to_node)?).unwrap_or(i64::MAX))
        })
    }

    #[napi(js_name = "advance")]
    pub fn advance(&self, env: Env, seconds: f64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.advance(seconds).map_err(crate::generated::errors::room_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "playedBytes")]
    pub fn played_bytes(&self) -> i64 {
        i64::try_from(self.inner.played_bytes()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "discardedBytes")]
    pub fn discarded_bytes(&self) -> i64 {
        i64::try_from(self.inner.discarded_bytes()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "textStreams")]
    pub fn text_streams(&self) -> Vec<(String, String)> {
        self.inner.text_streams()
    }

    #[napi(js_name = "sendChat")]
    pub fn send_chat(&self, identity: String, text: String, stream_id: String) -> bool {
        self.inner.send_chat(identity, text, stream_id)
    }

    #[napi(js_name = "leave")]
    pub fn leave(&self, env: Env, identity: String, reason: Option<String>) -> napi::Result<bool> {
        crate::host_error::run(&env, || {
            let reason = reason.unwrap_or("client_initiated".to_owned());
            self.inner.leave(identity, reason).map_err(crate::generated::errors::room_refused_to_node)
        })
    }

    #[napi(js_name = "close")]
    pub fn close(&self, env: Env, reason: Option<String>) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let reason = reason.unwrap_or("client_initiated".to_owned());
            self.inner.close(reason).map_err(crate::generated::errors::room_refused_to_node)?;
            Ok(())
        })
    }
}

#[napi(js_name = "OfflineRoom")]
pub struct OfflineRoom {
    pub(crate) inner: lk_agents_voice::room_io::rooms::OfflineRoom,
}

#[napi]
impl OfflineRoom {
    #[napi(constructor)]
    pub fn new(env: Env, name: Option<String>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let name = name.unwrap_or("console".to_owned());
            Ok(OfflineRoom { inner: lk_agents_voice::room_io::rooms::OfflineRoom::new(name).map_err(crate::generated::errors::room_refused_to_node)? })
        })
    }

    #[napi(getter, js_name = "name")]
    pub fn name(&self) -> String {
        self.inner.name()
    }
}

/// `_core.ReplaceTransform`: the one pure transform that crosses as a value. `push(chunk)` /
/// `flush()` answer a list, not a joined string, because the synchroniser paces per piece; empty
/// means "held back". Not frozen: a transform is a buffer, and two threads pushing into one should
/// get `Already borrowed`, not an interleaved transcript.
#[napi(js_name = "ReplaceTransform")]
pub struct ReplaceTransform {
    pub(crate) inner: lk_agents_voice::transcription::text_transforms::Replace,
}

#[napi]
impl ReplaceTransform {
    #[napi(constructor)]
    pub fn new(replacements: Vec<(String, String)>, case_sensitive: Option<bool>) -> Self {
        let case_sensitive = case_sensitive.unwrap_or(false);
        ReplaceTransform { inner: lk_agents_voice::transcription::text_transforms::Replace::for_host(replacements, case_sensitive) }
    }

    #[napi(js_name = "push")]
    pub fn push(&mut self, chunk: String) -> Vec<String> {
        self.inner.push_chunk(&chunk)
    }

    #[napi(js_name = "flush")]
    pub fn flush(&mut self) -> Vec<String> {
        self.inner.flush_held()
    }
}

#[napi(js_name = "Sound")]
pub struct Sound {
    pub(crate) inner: lk_agents_voice::background_audio::Sound,
}

#[napi]
impl Sound {
}

/// `_core.PlayState`: what `PlayHandle` wraps.
#[napi(js_name = "PlayState")]
pub struct PlayState {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::background_audio::PlaybackHandle>,
}

#[napi]
impl PlayState {
    #[napi(constructor)]
    pub fn new(fade_out: Option<f64>) -> Self {
        let fade_out = fade_out.unwrap_or(0.0);
        PlayState { inner: std::sync::Arc::new(lk_agents_voice::background_audio::PlaybackHandle::for_host(fade_out)) }
    }

    #[napi(getter, js_name = "done")]
    pub fn done(&self) -> bool {
        self.inner.done()
    }

    #[napi(getter, js_name = "stopRequested")]
    pub fn stop_requested(&self) -> bool {
        self.inner.stop_requested()
    }

    #[napi(js_name = "stop")]
    pub fn stop(&self) -> bool {
        self.inner.stop()
    }

    #[napi(js_name = "markPlayoutDone")]
    pub fn mark_playout_done(&self) -> bool {
        self.inner.finish()
    }

    #[napi(js_name = "waitForPlayout", ts_return_type = "Promise<void>")]
    pub fn wait_for_playout<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.wait().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// `_core.BackgroundAudioPlayer`: the mixer and the published track, under `background_audio.py`.
#[napi(js_name = "BackgroundAudioPlayer")]
pub struct BackgroundAudioPlayer {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::background_audio::BackgroundAudioPlayer>,
}

#[napi]
impl BackgroundAudioPlayer {
    #[napi(constructor)]
    pub fn new(env: Env, sample_rate: Option<u32>, num_channels: Option<u16>, blocksize: Option<i64>, stream_timeout_ms: Option<i64>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let sample_rate = sample_rate.unwrap_or(48_000);
            let num_channels = num_channels.unwrap_or(1);
            let blocksize = match blocksize { Some(v) => <usize>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("blocksize: {e}")))?, None => 4800 };
            let stream_timeout_ms = match stream_timeout_ms { Some(v) => <u64>::try_from(v).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("stream_timeout_ms: {e}")))?, None => 200 };
            Ok(BackgroundAudioPlayer { inner: std::sync::Arc::new(lk_agents_voice::background_audio::BackgroundAudioPlayer::for_host(sample_rate, num_channels, blocksize, stream_timeout_ms)) })
        })
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.close().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// The request as a host `LLM._core_chat` reads it: a `_core.ChatRequest` handle over a clone. The
/// host objects cross as their side-table keys (`_core.resolve_host_ref`), and the context as its
/// `ChatContext.to_dict()` form, read only when the request carries no host context of its own.
#[napi(js_name = "ChatRequest")]
pub struct ChatRequest {
    pub(crate) inner: lk_agents_abi::ChatRequest,
}

#[napi]
impl ChatRequest {
    #[napi(getter, js_name = "chatCtxHost")]
    pub fn chat_ctx_host(&self) -> Option<i64> {
        self.inner.chat_ctx_host().map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }

    #[napi(getter, js_name = "chatCtx")]
    pub fn chat_ctx(&self) -> serde_json::Value {
        self.inner.chat_ctx()
    }

    #[napi(getter, js_name = "toolNames")]
    pub fn tool_names(&self) -> Vec<String> {
        self.inner.tool_names()
    }

    #[napi(getter, js_name = "toolsHost")]
    pub fn tools_host(&self) -> Option<i64> {
        self.inner.tools_host().map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }

    #[napi(getter, js_name = "toolsHostExtra")]
    pub fn tools_host_extra(&self) -> Vec<i64> {
        self.inner.tools_host_extra().into_iter().map(|v| i64::try_from(v).unwrap_or(i64::MAX)).collect::<Vec<_>>()
    }

    #[napi(getter, js_name = "parallelToolCalls")]
    pub fn parallel_tool_calls(&self) -> Option<bool> {
        self.inner.parallel_tool_calls()
    }

    #[napi(getter, js_name = "toolChoice")]
    pub fn tool_choice(&self) -> Option<serde_json::Value> {
        self.inner.tool_choice()
    }

    #[napi(getter, js_name = "extra")]
    pub fn extra(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        self.inner.extra()
    }

    #[napi(getter, js_name = "inferenceClass")]
    pub fn inference_class(&self) -> Option<String> {
        self.inner.inference_class()
    }
}

/// The conversation a host turn detector scores, as `_core.TurnContext`: its
/// `ChatContext.to_dict()` form (timestamps kept; images and audio are not rendered), which the
/// host rebuilds with `ChatContext.from_dict`.
#[napi(js_name = "TurnContext")]
pub struct TurnContext {
    pub(crate) inner: lk_agents_abi::ChatContext,
}

#[napi]
impl TurnContext {
    #[napi(js_name = "toHostDict")]
    pub fn to_host_dict(&self) -> serde_json::Value {
        self.inner.to_host_dict()
    }
}

/// `LLMStream`'s metric fold as the host drives it (`_core.LlmMetricState`).
#[napi(js_name = "LlmMetricState")]
pub struct LlmMetricState {
    pub(crate) inner: std::sync::Mutex<lk_agents_abi::telemetry::provider::LlmMetricState>,
}

#[napi]
impl LlmMetricState {
    #[napi(constructor)]
    pub fn new(start: f64) -> Self {
        LlmMetricState { inner: std::sync::Mutex::new(lk_agents_abi::telemetry::provider::LlmMetricState::for_host(start)) }
    }

    #[napi(js_name = "observe")]
    pub fn observe(&self, env: Env, now: f64, chunk: crate::generated::records::LlmChunk) -> napi::Result<bool> {
        crate::host_error::run(&env, || {
            let chunk = chunk.try_into()?;
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(inner.observe_host(now, chunk))
        })
    }

    #[napi(js_name = "finish")]
    pub fn finish(&self, now: f64, failed: bool, host: (f64, String, bool, String, String)) -> Option<crate::generated::records::LlmMetrics> {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.finish_host(now, failed, host).map(|v| crate::generated::records::LlmMetrics::from(&v))
    }
}

/// The TTS streams' metric fold as the host drives it (`_core.TtsMetricState`).
#[napi(js_name = "TtsMetricState")]
pub struct TtsMetricState {
    pub(crate) inner: std::sync::Mutex<lk_agents_abi::telemetry::provider::TtsMetricState>,
}

#[napi]
impl TtsMetricState {
    #[napi(constructor)]
    pub fn new(streamed: bool) -> Self {
        TtsMetricState { inner: std::sync::Mutex::new(lk_agents_abi::telemetry::provider::TtsMetricState::for_host(streamed)) }
    }

    #[napi(js_name = "queueText")]
    pub fn queue_text(&self, text: String) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.queue_text_host(text);
    }

    #[napi(js_name = "observe")]
    pub fn observe(&self, env: Env, now: f64, start: f64, audio: crate::generated::records::SynthesizedAudio) -> napi::Result<bool> {
        crate::host_error::run(&env, || {
            let audio = audio.try_into()?;
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(inner.observe_host(now, start, audio))
        })
    }

    #[napi(js_name = "finish")]
    pub fn finish(&self, now: f64, start: f64, failed: bool, batch_text: String, input_tokens: u32, output_tokens: u32, acquire_time: f64, connection_reused: bool, host: (f64, String, bool, String, String)) -> Option<crate::generated::records::TtsMetrics> {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.finish_host(now, start, failed, &batch_text, input_tokens, output_tokens, acquire_time, connection_reused, host).map(|v| crate::generated::records::TtsMetrics::from(&v))
    }
}

/// `llm.FallbackAdapter`'s engine as the host builds it. The public class stays Python (users
/// construct, subclass and call it standalone); its `chat()` calls `chat_for_host`(Self::chat_for_host).
#[napi(js_name = "FallbackLlm")]
pub struct FallbackLlm {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::fallback::FallbackLlm>,
}

#[napi]
impl FallbackLlm {
    #[napi(getter, js_name = "activeIndex")]
    pub fn active_index(&self) -> i64 {
        i64::try_from(self.inner.active_index()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose_for_host().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

/// `stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[napi(js_name = "FallbackStt")]
pub struct FallbackStt {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::fallback::FallbackStt>,
}

#[napi]
impl FallbackStt {
    #[napi(getter, js_name = "activeIndex")]
    pub fn active_index(&self) -> i64 {
        i64::try_from(self.inner.active_index()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose_for_host().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "TtsTokenLog")]
pub struct TtsTokenLog {
    pub(crate) inner: lk_agents_inference::fallback::TokenLog,
}

#[napi]
impl TtsTokenLog {
    #[napi(constructor)]
    pub fn new() -> Self {
        TtsTokenLog { inner: lk_agents_inference::fallback::TokenLog::for_host() }
    }

    #[napi(js_name = "push")]
    pub fn push(&self, text: String) {
        self.inner.push_text(text);
    }

    #[napi(js_name = "flush")]
    pub fn flush(&self) {
        self.inner.push_flush();
    }

    #[napi(js_name = "end")]
    pub fn end(&self) {
        self.inner.end_input();
    }

    #[napi(js_name = "snapshot")]
    pub fn snapshot(&self) -> (Vec<Option<String>>, bool) {
        self.inner.replay_tokens()
    }
}

/// `tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[napi(js_name = "FallbackTts")]
pub struct FallbackTts {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::fallback::FallbackTts>,
}

#[napi]
impl FallbackTts {
    #[napi(getter, js_name = "activeIndex")]
    pub fn active_index(&self) -> i64 {
        i64::try_from(self.inner.active_index()).unwrap_or(i64::MAX)
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose_for_host().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(getter, js_name = "capabilities")]
    pub fn capabilities(&self) -> crate::generated::records::TtsCapabilities {
        crate::generated::records::TtsCapabilities::from(&self.inner.merged_capabilities())
    }

    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> u32 {
        self.inner.output_sample_rate()
    }

    #[napi(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> u16 {
        self.inner.output_num_channels()
    }
}

#[napi(js_name = "RealtimeFallback")]
pub struct RealtimeFallback {
    pub(crate) inner: std::sync::Mutex<lk_agents_inference::fallback::RealtimePolicy>,
}

#[napi]
impl RealtimeFallback {
    #[napi(constructor)]
    pub fn new(env: Env, model_count: i64, cooldown: f64) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let model_count = <usize>::try_from(model_count).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("model_count: {e}")))?;
            Ok(RealtimeFallback { inner: std::sync::Mutex::new(lk_agents_inference::fallback::RealtimePolicy::for_host(model_count, cooldown).map_err(|e| crate::host_error::JsError::new("ValueError", e.to_string()))?) })
        })
    }

    #[napi(getter, js_name = "activeIndex")]
    pub fn active_index(&self) -> i64 {
        let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        i64::try_from(inner.active_index()).unwrap_or(i64::MAX)
    }

    #[napi(setter, js_name = "activeIndex")]
    pub fn set_active_index(&self, env: Env, index: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let index = <usize>::try_from(index).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("index: {e}")))?;
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            inner.set_active_index(index);
            Ok(())
        })
    }

    #[napi(js_name = "isAvailable")]
    pub fn is_available(&self, env: Env, index: i64) -> napi::Result<bool> {
        crate::host_error::run(&env, || {
            let index = <usize>::try_from(index).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("index: {e}")))?;
            let inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(inner.is_available(index))
        })
    }

    #[napi(js_name = "setAvailable")]
    pub fn set_available(&self, env: Env, index: i64, available: bool, now: f64) -> napi::Result<Option<crate::generated::records::AvailabilityChange>> {
        crate::host_error::run(&env, || {
            let index = <usize>::try_from(index).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("index: {e}")))?;
            let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(inner.set_available(index, available, now).map(|v| crate::generated::records::AvailabilityChange::from(&v)))
        })
    }

    #[napi(js_name = "nextAvailableIndex")]
    pub fn next_available_index(&self, now: f64, exclude_current: Option<bool>) -> (Option<i64>, Vec<crate::generated::records::AvailabilityChange>) {
        let exclude_current = exclude_current.unwrap_or(false);
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        { let (t0, t1,) = inner.next_available_index(now, exclude_current); (t0.map(|v| i64::try_from(v).unwrap_or(i64::MAX)), t1.into_iter().map(|v| crate::generated::records::AvailabilityChange::from(&v)).collect::<Vec<_>>()) }
    }

    #[napi(js_name = "onChildError")]
    pub fn on_child_error(&self, recoverable: bool, swap_busy: bool, now: f64) -> crate::generated::records::ErrorPlan {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::generated::records::ErrorPlan::from(&inner.on_child_error(recoverable, swap_busy, now))
    }

    #[napi(js_name = "restartTarget")]
    pub fn restart_target(&self, switch_model: bool, now: f64) -> (i64, Vec<crate::generated::records::AvailabilityChange>) {
        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        { let (t0, t1,) = inner.restart_target(switch_model, now); (i64::try_from(t0).unwrap_or(i64::MAX), t1.into_iter().map(|v| crate::generated::records::AvailabilityChange::from(&v)).collect::<Vec<_>>()) }
    }
}

/// `inference/stt.py`'s WebSocket, over `lk-agents-inference::stt::NativeStt`.
#[napi(js_name = "InferenceStt")]
pub struct InferenceStt {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::stt::NativeStt>,
}

#[napi]
impl InferenceStt {
    #[napi(js_name = "updateSessionKeyterms")]
    pub fn update_session_keyterms(&self, keyterms: Vec<String>) -> Option<serde_json::Map<String, serde_json::Value>> {
        self.inner.update_session_keyterms(&keyterms)
    }
}

/// `inference/vad.py`: Silero VAD with the ONNX model in the host and the machine in Rust.
#[napi(js_name = "InferenceVad")]
pub struct InferenceVad {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::vad::NativeVad>,
}

#[napi]
impl InferenceVad {
    #[napi(js_name = "updateOptions")]
    pub fn update_options(&self, min_speech_duration: f64, min_silence_duration: f64, prefix_padding_duration: f64, max_buffered_speech: f64, activation_threshold: f64, deactivation_threshold: f64) {
        self.inner.update_options(min_speech_duration, min_silence_duration, prefix_padding_duration, max_buffered_speech, activation_threshold, deactivation_threshold);
    }
}

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::NativeTts`.
#[napi(js_name = "InferenceTts")]
pub struct InferenceTts {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::tts::NativeTts>,
}

#[napi]
impl InferenceTts {
    #[napi(js_name = "setExpressive")]
    pub fn set_expressive(&self, expressive: bool) {
        self.inner.set_expressive(expressive);
    }

    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    #[napi(getter, js_name = "alignedTranscript")]
    pub fn aligned_transcript(&self) -> bool {
        self.inner.aligned_transcript()
    }
}

/// `_core.NativeRealtimeModel`: a realtime model whose connection, reconnect and conversation
/// mirror are in Rust. Not a `RealtimeModel` subclass (the ABC stays Python); `AgentSession`
/// recognises it by identity, and nothing per-turn crosses.
#[napi(js_name = "NativeRealtimeModel")]
pub struct NativeRealtimeModel {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::realtime::NativeRealtime>,
}

#[napi]
impl NativeRealtimeModel {
    #[napi(getter, js_name = "provider")]
    pub fn provider(&self) -> String {
        self.inner.provider().to_string()
    }

    #[napi(getter, js_name = "model")]
    pub fn model(&self) -> String {
        self.inner.model().to_string()
    }

    #[napi(getter, js_name = "label")]
    pub fn label(&self) -> String {
        self.inner.label()
    }

    #[napi(js_name = "capabilities")]
    pub fn capabilities(&self) -> crate::generated::records::RealtimeCapabilities {
        crate::generated::records::RealtimeCapabilities::from(&self.inner.capabilities())
    }
}

#[napi(js_name = "ConnectionPoolState")]
pub struct ConnectionPoolState {
    pub(crate) inner: lk_agents_abi::utils::connection_pool::ConnectionPool,
}

#[napi]
impl ConnectionPoolState {
    #[napi(constructor)]
    pub fn new(maximum_age: Option<f64>, refresh_on_get: bool) -> Self {
        ConnectionPoolState { inner: lk_agents_abi::utils::connection_pool::ConnectionPool::new(maximum_age, refresh_on_get) }
    }

    #[napi(js_name = "connected")]
    pub fn connected(&mut self, env: Env, id: i64, now: f64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.connected(id, now);
            Ok(())
        })
    }

    #[napi(js_name = "empty")]
    pub fn empty(&self) -> bool {
        self.inner.empty()
    }

    #[napi(js_name = "get")]
    pub fn get(&mut self, now: f64) -> Option<i64> {
        self.inner.get(now).map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }

    #[napi(js_name = "put")]
    pub fn put(&mut self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.put(id);
            Ok(())
        })
    }

    #[napi(js_name = "remove")]
    pub fn remove(&mut self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.remove(id);
            Ok(())
        })
    }

    #[napi(js_name = "invalidate")]
    pub fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    #[napi(js_name = "nextClosing")]
    pub fn next_closing(&mut self) -> Option<i64> {
        self.inner.next_closing().map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }
}

/// `session.input.video` when the room is the source, as `_core.RoomVideoInput`: built only by
/// the session (the leg shares the audio leg's participant link), so there is no constructor.
#[napi(js_name = "RoomVideoInput")]
pub struct RoomVideoInput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::room_io::_input::ParticipantVideoInput>,
}

#[napi]
impl RoomVideoInput {
    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "VideoSampler")]
pub struct VideoSampler {
    pub(crate) inner: lk_agents_voice::room_io::_input::VideoSampler,
}

#[napi]
impl VideoSampler {
    #[napi(constructor)]
    pub fn new(speaking_fps: f64, silent_fps: f64) -> Self {
        VideoSampler { inner: lk_agents_voice::room_io::_input::VideoSampler::new(speaking_fps, silent_fps) }
    }

    #[napi(getter, js_name = "speakingFps")]
    pub fn speaking_fps(&self) -> f64 {
        self.inner.speaking_fps()
    }

    #[napi(getter, js_name = "silentFps")]
    pub fn silent_fps(&self) -> f64 {
        self.inner.silent_fps()
    }

    #[napi(setter, js_name = "speakingFps")]
    pub fn set_speaking_fps(&self, fps: f64) {
        self.inner.set_speaking_fps(fps);
    }

    #[napi(setter, js_name = "silentFps")]
    pub fn set_silent_fps(&self, fps: f64) {
        self.inner.set_silent_fps(fps);
    }

    #[napi(js_name = "shouldSample")]
    pub fn should_sample(&self, speaking: bool, now_secs: f64) -> bool {
        self.inner.should_sample(speaking, now_secs)
    }
}

#[napi(js_name = "EvaluationResult")]
pub struct EvaluationResult {
    pub(crate) inner: lk_agents_abi::evals::EvaluationResult,
}

#[napi]
impl EvaluationResult {
    #[napi(constructor)]
    pub fn new() -> Self {
        EvaluationResult { inner: lk_agents_abi::evals::EvaluationResult::new() }
    }

    #[napi(js_name = "score")]
    pub fn score(&self) -> f64 {
        self.inner.score()
    }

    #[napi(js_name = "allPassed")]
    pub fn all_passed(&self) -> bool {
        self.inner.all_passed()
    }

    #[napi(js_name = "anyPassed")]
    pub fn any_passed(&self) -> bool {
        self.inner.any_passed()
    }

    #[napi(js_name = "majorityPassed")]
    pub fn majority_passed(&self) -> bool {
        self.inner.majority_passed()
    }

    #[napi(js_name = "noneFailed")]
    pub fn none_failed(&self) -> bool {
        self.inner.none_failed()
    }
}

/// `utils/hw/cpu.py`'s three monitors as one. Starts no threads, so it is safe before core init.
#[napi(js_name = "CpuMonitor")]
pub struct CpuMonitor {
    pub(crate) inner: lk_agents_worker::CpuMonitor,
}

#[napi]
impl CpuMonitor {
    #[napi(constructor)]
    pub fn new() -> Self {
        CpuMonitor { inner: lk_agents_worker::CpuMonitor::new() }
    }

    #[napi(getter, js_name = "kind", ts_return_type = "'cgroup_v2' | 'cgroup_v1' | 'default'")]
    pub fn kind(&self) -> String {
        self.inner.kind().as_str().to_owned()
    }

    #[napi(js_name = "cpuCount")]
    pub fn cpu_count(&self) -> f64 {
        f64::from(self.inner.cpu_count())
    }

    #[napi(js_name = "cpuPercent")]
    pub fn cpu_percent(&mut self, interval: Option<f64>) -> Option<f64> {
        let interval = interval.unwrap_or(0.5);
        self.inner.cpu_percent_over(interval).map(|v| f64::from(v))
    }
}

#[napi(js_name = "MarkdownFilter")]
pub struct MarkdownFilter {
    pub(crate) inner: lk_agents_voice::transcription::filters::MarkdownFilter,
}

#[napi]
impl MarkdownFilter {
    #[napi(constructor)]
    pub fn new() -> Self {
        MarkdownFilter { inner: lk_agents_voice::transcription::filters::MarkdownFilter::new() }
    }

    #[napi(js_name = "push")]
    pub fn push(&mut self, chunk: String) -> Vec<String> {
        self.inner.push(&chunk)
    }

    #[napi(js_name = "flush")]
    pub fn flush(&mut self) -> Vec<String> {
        self.inner.flush()
    }
}

/// `_core.TranscriptSynchronizer`: the node every native session builds, handed to a host that
/// assembles a synchroniser by hand (the console, a custom chain), so pacing is one implementation
/// on one clock.
#[napi(js_name = "TranscriptSynchronizer")]
pub struct TranscriptSynchronizer {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::transcription::synchronizer::TranscriptSynchronizer>,
}

#[napi]
impl TranscriptSynchronizer {
    #[napi(js_name = "warnings")]
    pub fn warnings(&self) -> Vec<String> {
        self.inner.warning_names().into_iter().map(|v| v.to_string()).collect::<Vec<_>>()
    }

    #[napi(getter, js_name = "enabled")]
    pub fn enabled(&self) -> bool {
        self.inner.enabled()
    }

    #[napi(js_name = "setEnabled")]
    pub fn set_enabled(&self, enabled: bool) {
        self.inner.set_enabled(enabled);
    }

    #[napi(js_name = "rotateSegment")]
    pub fn rotate_segment(&self) {
        self.inner.rotate_segment();
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}

#[napi(js_name = "McpSession")]
pub struct McpSession {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::mcp::session::McpSession>,
}

#[napi]
impl McpSession {
    #[napi(constructor)]
    pub fn new(client_session_timeout_seconds: f64) -> Self {
        McpSession { inner: std::sync::Arc::new(lk_agents_inference::mcp::session::McpSession::new(client_session_timeout_seconds)) }
    }

    #[napi(getter, js_name = "connected")]
    pub fn connected(&self) -> bool {
        self.inner.connected()
    }

    #[napi(setter, js_name = "allowedTools")]
    pub fn set_allowed_tools(&self, allowed: Vec<String>) {
        self.inner.set_allowed_tools(allowed);
    }

    #[napi(js_name = "invalidateCache")]
    pub fn invalidate_cache(&self) {
        self.inner.invalidate_cache();
    }

    #[napi(js_name = "detach")]
    pub fn detach(&self) {
        self.inner.detach();
    }

    #[napi(js_name = "connectHttp", ts_return_type = "Promise<void>")]
    pub fn connect_http<'env>(&self, env: &'env Env, url: String, headers: std::collections::HashMap<String, String>, #[napi(ts_arg_type = "'sse' | 'streamable_http'")] transport: String, timeout: f64, sse_read_timeout: f64) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (url, headers, transport, timeout, sse_read_timeout,) = crate::host_error::run(env, || {
            let headers = headers.into_iter().map(|(k, v)| (k, v)).collect();
            let transport = transport.parse().map_err(|e| crate::host_error::JsError::new("ValueError", format!("transport: {e}")))?;
            Ok((url, headers, transport, timeout, sse_read_timeout,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.connect_http(url, headers, transport, timeout, sse_read_timeout).await.map_err(crate::generated::errors::mcp_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "connectStdio", ts_return_type = "Promise<void>")]
    pub fn connect_stdio<'env>(&self, js_env: &'env Env, command: String, args: Vec<String>, env: std::collections::HashMap<String, String>, cwd: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (command, args, env, cwd,) = crate::host_error::run(js_env, || {
            let env = env.into_iter().map(|(k, v)| (k, v)).collect();
            Ok((command, args, env, cwd,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        js_env.spawn_future_with_callback(async move { Ok(inner.connect_stdio(command, args, env, cwd).await.map_err(crate::generated::errors::mcp_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "listTools", ts_return_type = "Promise<unknown>")]
    pub fn list_tools<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, Vec<serde_json::Value>>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.list_tools().await.map_err(crate::generated::errors::mcp_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "close", ts_return_type = "Promise<void>")]
    pub fn close<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.close().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }
}
