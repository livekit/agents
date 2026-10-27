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

    #[napi(getter, js_name = "closed")]
    pub fn closed(&self) -> bool {
        self.inner.closed()
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

#[napi(js_name = "ExitSignals")]
pub struct ExitSignals {
    pub(crate) inner: std::sync::Arc<lk_agents_worker::cli::ExitSignals>,
}

#[napi]
impl ExitSignals {
    #[napi(constructor)]
    pub fn new(escalate: Option<bool>) -> Self {
        let escalate = escalate.unwrap_or(true);
        ExitSignals { inner: std::sync::Arc::new(lk_agents_worker::cli::ExitSignals::new(escalate)) }
    }

    #[napi(js_name = "signal")]
    pub fn signal(&self, exit_served: bool) -> String {
        self.inner.signal(exit_served).to_string()
    }

    #[napi(js_name = "served")]
    pub fn served(&self) {
        self.inner.served();
    }

    #[napi(js_name = "draining")]
    pub fn draining(&self) {
        self.inner.draining();
    }
}

/// `AgentSession(**kwargs)`'s options: `kwargs` crosses once as a raw dict, and the resolved
/// configuration comes back as the dict `session.options` views.
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

    #[napi(getter, js_name = "turnDetectionExplicit")]
    pub fn turn_detection_explicit(&self) -> bool {
        self.inner.turn_detection_explicit()
    }
}

#[napi(js_name = "SpeechHandle")]
pub struct SpeechHandle {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::speech_handle::SpeechHandle>,
}

#[napi]
impl SpeechHandle {
    #[napi(constructor)]
    pub fn new(speech_id: Option<String>, allow_interruptions: Option<bool>) -> Self {
        let speech_id = speech_id.unwrap_or(String::new());
        let allow_interruptions = allow_interruptions.unwrap_or(true);
        SpeechHandle { inner: std::sync::Arc::new(lk_agents_voice::speech_handle::SpeechHandle::detached(speech_id, allow_interruptions)) }
    }

    #[napi(getter, js_name = "id")]
    pub fn id(&self) -> String {
        self.inner.id_for_host()
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
        self.inner.interrupted_for_host()
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
            self.inner.set_allow_interruptions_for_host(value).map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "holdInterruptions")]
    pub fn hold_interruptions(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.hold_interruptions_for_host().map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "releaseInterruptions")]
    pub fn release_interruptions(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.release_interruptions_for_host().map_err(crate::generated::errors::speech_refused_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "waitForPlayout", ts_return_type = "Promise<void>")]
    pub fn wait_for_playout<'env>(&self, env: &'env Env, origin: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (origin,) = crate::host_error::run(env, || {
            Ok((origin,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_playout_for_host(origin).await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
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
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_generation_for_host(step).await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitForInterrupted", ts_return_type = "Promise<void>")]
    pub fn wait_for_interrupted<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_interrupted_for_host().await.map_err(crate::generated::errors::speech_refused_to_node)) }, |env, out| match out {
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

/// GetDtmfResult(user_input: 'str')
#[napi(js_name = "GetDtmfResult")]
pub struct GetDtmfResult {
    pub(crate) inner: lk_agents_voice::workflows::dtmf::DtmfResult,
}

#[napi]
impl GetDtmfResult {
    #[napi(constructor)]
    pub fn new(user_input: String) -> Self {
        GetDtmfResult { inner: lk_agents_voice::workflows::dtmf::DtmfResult::new(user_input) }
    }

    #[napi(getter, js_name = "userInput")]
    pub fn user_input(&self) -> String {
        self.inner.user_input()
    }
}

#[napi(js_name = "WarmTransfer")]
pub struct WarmTransfer {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::workflows::warm_transfer::HostWarmTransfer>,
}

#[napi]
impl WarmTransfer {
    #[napi(getter, js_name = "instructions")]
    pub fn instructions(&self) -> (String, Option<String>, Option<String>) {
        self.inner.instructions()
    }

    #[napi(js_name = "tools")]
    pub fn tools(&self) -> Vec<serde_json::Value> {
        self.inner.tools()
    }

    #[napi(js_name = "callTool", ts_return_type = "Promise<void>")]
    pub fn call_tool<'env>(&self, env: &'env Env, name: String, arguments: serde_json::Value) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (name, arguments,) = crate::host_error::run(env, || {
            Ok((name, arguments,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.call_tool(name, arguments).await.map_err(crate::generated::errors::transfer_failed_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
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

#[napi(js_name = "InferenceAvatar")]
pub struct InferenceAvatar {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::avatar::InferenceAvatar>,
}

#[napi]
impl InferenceAvatar {
    #[napi(constructor)]
    pub fn new(env: Env, model: String, avatar_identity: Option<String>, avatar_name: Option<String>, extra_kwargs: serde_json::Map<String, serde_json::Value>, base_url: String, api_key: String, api_secret: String, timeout: f64, max_retry: u32, retry_interval: f64) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            Ok(InferenceAvatar { inner: std::sync::Arc::new(lk_agents_inference::avatar::InferenceAvatar::new(&model, avatar_identity, avatar_name, extra_kwargs, base_url, api_key, api_secret, timeout, max_retry, retry_interval).map_err(crate::generated::errors::avatar_error_to_node)?) })
        })
    }

    #[napi(getter, js_name = "provider")]
    pub fn provider(&self) -> String {
        self.inner.provider()
    }

    #[napi(getter, js_name = "avatarIdentity")]
    pub fn avatar_identity(&self) -> String {
        self.inner.avatar_identity()
    }

    #[napi(getter, js_name = "sessionId")]
    pub fn session_id(&self) -> Option<String> {
        self.inner.session_id()
    }

    #[napi(getter, js_name = "providerSessionId")]
    pub fn provider_session_id(&self) -> Option<String> {
        self.inner.provider_session_id()
    }

    #[napi(js_name = "checkUnstarted")]
    pub fn check_unstarted(&self, env: Env) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.check_unstarted().map_err(crate::generated::errors::avatar_error_to_node)?;
            Ok(())
        })
    }

    #[napi(js_name = "create", ts_return_type = "Promise<number>")]
    pub fn create<'env>(&self, env: &'env Env, room_name: String, room_sid: String, livekit_url: Option<String>, agent_identity: String, headers: Vec<(String, String)>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, u32>> {
        let (room_name, room_sid, livekit_url, agent_identity, headers,) = crate::host_error::run(env, || {
            Ok((room_name, room_sid, livekit_url, agent_identity, headers,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.create(room_name, room_sid, livekit_url, agent_identity, headers).await.map_err(crate::generated::errors::avatar_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "terminate", ts_return_type = "Promise<void>")]
    pub fn terminate<'env>(&self, env: &'env Env, headers: Vec<(String, String)>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let (headers,) = crate::host_error::run(env, || {
            Ok((headers,))
        })?;
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.terminate(headers).await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
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

#[napi(js_name = "SessionClock")]
pub struct SessionClock {
    pub(crate) inner: lk_agents_voice::agent_session::Timebase,
}

#[napi]
impl SessionClock {
}

/// `voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
/// `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
/// borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
use crate::session::VoiceSession;

#[napi]
impl VoiceSession {
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

    #[napi(getter, js_name = "usage", ts_return_type = "Array<LlmModelUsage | TtsModelUsage | SttModelUsage | InterruptionModelUsage | EotModelUsage>")]
    pub fn usage(&self) -> Vec<crate::generated::records::ModelUsage> {
        self.inner.usage().into_iter().map(|v| crate::generated::records::ModelUsage::from(&v)).collect::<Vec<_>>()
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

    #[napi(js_name = "shutdown")]
    pub fn shutdown(&self, drain: Option<bool>) {
        let drain = drain.unwrap_or(true);
        self.inner.shutdown(drain);
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

    #[napi(js_name = "endCall")]
    pub fn end_call(&self, env: Env, speech: &crate::generated::objects::SpeechHandle) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            self.inner.end_call(&speech.inner).map_err(crate::generated::errors::session_refused_to_node)?;
            Ok(())
        })
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

    #[napi(js_name = "waitForIdleAndHold", ts_return_type = "Promise<number>")]
    pub fn wait_for_idle_and_hold<'env>(&self, env: &'env Env, origin: Option<String>) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, i64>> {
        let reply = crate::host_error::run(env, || {
            self.inner.wait_for_idle_and_hold(origin.as_deref()).map_err(crate::generated::errors::command_refused_to_node)
        })?;
        env.spawn_future_with_callback(async move { Ok(reply.await.map_err(crate::host_error::idle_refused_to_node)) }, |env, out| match out {
            Ok(out) => Ok(i64::try_from(out).unwrap_or(i64::MAX)),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "releaseIdleHold")]
    pub fn release_idle_hold(&self, env: Env, hold: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let hold = <u64>::try_from(hold).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("hold: {e}")))?;
            self.inner.release_idle_hold(hold);
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

    #[napi(js_name = "amdListening")]
    pub fn amd_listening(&self) -> bool {
        self.inner.amd_listening()
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

    #[napi(getter, js_name = "bufferedDuration")]
    pub fn buffered_duration(&self) -> f64 {
        self.inner.buffered_duration()
    }

    #[napi(js_name = "resetProgressive")]
    pub fn reset_progressive(&mut self) {
        self.inner.reset_progressive();
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
    pub fn new(env: Env, mime: String, sample_rate: Option<Option<u32>>, num_channels: Option<u16>) -> napi::Result<Self> {
        crate::host_error::run(&env, || {
            let sample_rate = sample_rate.unwrap_or(Some(48000));
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

/// The `lk agent console` socket: pass it as the session's audio input and output and as a
/// `RemoteSessionHost`'s transport and console.
#[napi(js_name = "ConsoleIo")]
pub struct ConsoleIo {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::remote_session::console::ConsoleIo>,
}

#[napi]
impl ConsoleIo {
}

/// The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to write
/// and how it is normalized and lowered before synthesis.
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
    pub(crate) inner: std::sync::Arc<lk_agents_abi::voice::io::PlaybackTracker>,
}

#[napi]
impl PlaybackTracker {
    #[napi(constructor)]
    pub fn new() -> Self {
        PlaybackTracker { inner: std::sync::Arc::new(lk_agents_abi::voice::io::PlaybackTracker::new()) }
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

    #[napi(js_name = "noteCapture")]
    pub fn note_capture(&self) {
        self.inner.note_capture();
    }

    #[napi(js_name = "noteFlush")]
    pub fn note_flush(&self) {
        self.inner.note_flush();
    }

    #[napi(js_name = "waitForPlayout", ts_return_type = "Promise<unknown>")]
    pub fn wait_for_playout<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, (f64, bool, Option<String>)>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.wait_for_playout_for_host().await)) }, |env, out| match out {
            Ok(out) => Ok(out),
            Err(e) => Err(e.into_napi(env)),
        })
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

#[napi(js_name = "DataStreamAudioOutput")]
pub struct DataStreamAudioOutput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::avatar::DataStreamAudioOutput>,
}

#[napi]
impl DataStreamAudioOutput {
    #[napi(getter, js_name = "label")]
    pub fn label(&self) -> String {
        self.inner.label_for_host()
    }

    #[napi(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate_for_host()
    }
}

#[napi(js_name = "AvatarRunner")]
pub struct AvatarRunner {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::avatar_runner::AvatarRunner>,
}

#[napi]
impl AvatarRunner {
    #[napi(js_name = "start", ts_return_type = "Promise<void>")]
    pub fn start<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.start().await.map_err(crate::generated::errors::runner_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "waitForComplete", ts_return_type = "Promise<void>")]
    pub fn wait_for_complete<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.wait_for_complete().await.map_err(crate::generated::errors::runner_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
    }

    #[napi(js_name = "clearBuffer")]
    pub fn clear_buffer(&self) {
        self.inner.clear_buffer();
    }

    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(inner.aclose().await.map_err(crate::generated::errors::runner_error_to_node)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
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

/// The host's view: `_core_rtc.LocalParticipant._core_stream_bytes` spreads it into
/// `stream_bytes(..)`.
#[napi(js_name = "StreamBytesOptions")]
pub struct StreamBytesOptions {
    pub(crate) inner: lk_agents_abi::voice::rtc::StreamBytesOptions,
}

#[napi]
impl StreamBytesOptions {
    #[napi(getter, js_name = "name")]
    pub fn name(&self) -> String {
        self.inner.name()
    }

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

    #[napi(js_name = "byteStreams")]
    pub fn byte_streams(&self) -> Vec<(String, Vec<String>, std::collections::HashMap<String, String>, i64, bool)> {
        self.inner.byte_streams().into_iter().map(|v| { let (t0, t1, t2, t3, t4,) = v; (t0, t1, t2.into_iter().map(|(k, v)| (k, v)).collect::<std::collections::HashMap<_, _>>(), i64::try_from(t3).unwrap_or(i64::MAX), t4) }).collect::<Vec<_>>()
    }

    #[napi(js_name = "rpc")]
    pub fn rpc(&self, method: String, caller: String, payload: String) -> Option<String> {
        self.inner.rpc(method, caller, payload)
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

    #[napi(js_name = "pressDtmf")]
    pub fn press_dtmf(&self, digit: String) {
        self.inner.press_dtmf(digit);
    }

    #[napi(js_name = "dtmfPublished")]
    pub fn dtmf_published(&self) -> Vec<(u32, String)> {
        self.inner.dtmf_published()
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

/// `stt.StreamAdapter`'s engine, as the host builds it; the public class stays Python.
#[napi(js_name = "SttStreamAdapter")]
pub struct SttStreamAdapter {
    pub(crate) inner: std::sync::Arc<lk_agents_abi::stt::stream_adapter::SttStreamAdapter>,
}

#[napi]
impl SttStreamAdapter {
}

/// `_core.RetryingStt`: what a host STT bound to `conn_options` is to the core.
#[napi(js_name = "RetryingStt")]
pub struct RetryingStt {
    pub(crate) inner: std::sync::Arc<lk_agents_abi::stt::retry::RetryingStt>,
}

#[napi]
impl RetryingStt {
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

/// `inference/stt.py`'s WebSocket, over `lk-agents-inference::stt::NativeStt`.
#[napi(js_name = "InferenceStt")]
pub struct InferenceStt {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::stt::NativeStt>,
}

#[napi]
impl InferenceStt {
    #[napi(js_name = "updateKeyterms")]
    pub fn update_keyterms(&self, keyterms: Vec<String>) {
        self.inner.update_keyterms_for_host(keyterms);
    }

    #[napi(js_name = "pushConversationItem")]
    pub fn push_conversation_item(&self, role: String, text: String) {
        self.inner.push_conversation_item(&role, &text);
    }
}

/// `inference/vad.py`: Silero VAD, the machine in Rust and the model native when the build links
/// one, else the host's.
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
/// mirror are in Rust. Not a `RealtimeModel` subclass (the ABC stays Python); `VoiceSession`
/// takes it as its realtime model, and nothing per-turn crosses. The keyword constructor,
/// `update_options` and `bind_identification` are `realtime::python`'s.
#[napi(js_name = "NativeRealtimeModel")]
pub struct NativeRealtimeModel {
    pub(crate) inner: std::sync::Arc<lk_agents_inference::realtime::NativeRealtime>,
}

#[napi]
impl NativeRealtimeModel {
    #[napi(getter, js_name = "provider")]
    pub fn provider(&self) -> String {
        self.inner.provider()
    }

    #[napi(getter, js_name = "model")]
    pub fn model(&self) -> String {
        self.inner.model()
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

    #[napi(getter, js_name = "generation")]
    pub fn generation(&self) -> i64 {
        i64::try_from(self.inner.generation()).unwrap_or(i64::MAX)
    }

    #[napi(getter, js_name = "lastAcquireTime")]
    pub fn last_acquire_time(&self) -> f64 {
        self.inner.last_acquire_time()
    }

    #[napi(getter, js_name = "lastConnectionReused")]
    pub fn last_connection_reused(&self) -> bool {
        self.inner.last_connection_reused()
    }

    #[napi(js_name = "connected")]
    pub fn connected(&mut self, env: Env, id: i64, now: f64, generation: i64, acquire_time: Option<f64>) -> napi::Result<bool> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            let generation = <u64>::try_from(generation).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("generation: {e}")))?;
            Ok(self.inner.connected(id, now, generation, acquire_time))
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

    #[napi(js_name = "closeAll")]
    pub fn close_all(&mut self) {
        self.inner.close_all();
    }

    #[napi(js_name = "nextClosing")]
    pub fn next_closing(&self) -> Option<i64> {
        self.inner.next_closing().map(|v| i64::try_from(v).unwrap_or(i64::MAX))
    }

    #[napi(js_name = "closed")]
    pub fn closed(&mut self, env: Env, id: i64) -> napi::Result<()> {
        crate::host_error::run(&env, || {
            let id = <u64>::try_from(id).map_err(|e| crate::host_error::JsError::new("OverflowError", format!("id: {e}")))?;
            self.inner.closed(id);
            Ok(())
        })
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

/// `session.input.audio` when the room is the source, as `_core.RoomAudioInput`: built only by
/// the session, so there is no constructor.
#[napi(js_name = "RoomAudioInput")]
pub struct RoomAudioInput {
    pub(crate) inner: std::sync::Arc<lk_agents_voice::room_io::_input::ParticipantAudioInput>,
}

#[napi]
impl RoomAudioInput {
    #[napi(js_name = "aclose", ts_return_type = "Promise<void>")]
    pub fn aclose<'env>(&self, env: &'env Env) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, ()>> {
        let inner = std::sync::Arc::clone(&self.inner);
        env.spawn_future_with_callback(async move { Ok(Ok::<_, crate::host_error::JsError>(inner.aclose().await)) }, |env, out| match out {
            Ok(out) => Ok(()),
            Err(e) => Err(e.into_napi(env)),
        })
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
