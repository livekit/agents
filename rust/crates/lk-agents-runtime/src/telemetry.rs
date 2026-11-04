//! The process's telemetry (docs/internals/telemetry.md "Destinations"): one tracer every
//! `ProcessEnv` lends, and one OTLP exporter it records to once the host installs telemetry; the
//! `function_tool` span the voice and text executors share; and the usage counters a metrics
//! record feeds.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use lk_agents_abi::FunctionCall;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::decl::{BoolString, GenAiContent, Label, json};
use lk_agents_abi::telemetry::host::TelemetryConfigError;
use lk_agents_abi::telemetry::instruments::EventLoopBlockedDuration;
use lk_agents_abi::telemetry::instruments::{
    ConnectionAcquireTime, ExecuteToolDuration, InterruptionNumRequests, LlmInputAudioTokens,
    LlmInputCachedTokens, LlmInputTextTokens, LlmInputTokens, LlmOutputAudioTokens,
    LlmOutputTextTokens, LlmOutputTokens, LlmSessionDuration, ModelLabels, SttAudioDuration,
    TtsAudioDuration, TtsCharacters,
};
use lk_agents_abi::telemetry::record::Key;
use lk_agents_abi::telemetry::spans::agent_turn::{FunctionTool, FunctionToolEnd};
use lk_agents_abi::telemetry::spans::session::{BlockingSeverity, EventLoopBlocked};
use lk_agents_net::telemetry::ExporterSlot;
use lk_agents_net::telemetry::otlp::{AppConfig, Exporter, Limits, ProcessIdentity, SdkName};
use parking_lot::Mutex;
use serde_json::Value as JsonValue;

// What the runtime's logic names, spelled `telemetry::` there (`telemetry.md`, rule 4).
pub(crate) use lk_agents_abi::telemetry::decl::{Content, Json, Measured, Seconds};
pub(crate) use lk_agents_abi::telemetry::instruments::{AudioConsumer, TurnLlmTtft, TurnTtsTtfb};
pub(crate) use lk_agents_abi::telemetry::record::{JobKey, SpanId, SpanRef, TraceContext, TraceId};
#[cfg(test)]
pub(crate) use lk_agents_abi::telemetry::record::{Record, RecordSink, Value};
pub(crate) use lk_agents_abi::telemetry::report::SessionReport;
pub(crate) use lk_agents_abi::telemetry::spans::agent_turn::{
    AgentTurn, LlmNode, LlmNodeEnd, RealtimeInference, RealtimeMetrics, TtsNode, TtsNodeEnd,
};
pub(crate) use lk_agents_abi::telemetry::spans::common::{GenAiOperation, OutputType, ToolType};
pub(crate) use lk_agents_abi::telemetry::spans::inference::{
    RealtimeResponse, RealtimeUsage, TtsRequestRun, UtcInstant,
};
pub(crate) use lk_agents_abi::telemetry::spans::job::{
    FirstFrame, PreConnectAudio, PublishAudioOutput, PublishAudioOutputEnd, TrackSubscribed,
    WaitForAudioTrack, WaitForAudioTrackEnd, WaitForParticipant,
};
pub(crate) use lk_agents_abi::telemetry::spans::session::{
    AgentSession, Amd, KeytermDetection, SessionStart, SetupToolsets, StartAgentActivity,
    UpdateAgent,
};
pub(crate) use lk_agents_abi::telemetry::spans::text::{
    A2aRequest, AgentRun, AgentRunEnd, RunStatus,
};
pub(crate) use lk_agents_core::telemetry::gen_ai;
pub(crate) use lk_agents_core::telemetry::provider::Context;
pub(crate) use lk_agents_core::telemetry::span::seeded_ids;
pub(crate) use lk_agents_inference::telemetry::{LlmTrace, TtsTrace, model_labels, provider_name};
pub(crate) use lk_agents_net::telemetry::{
    Cx, ErrorType, Failure, ScopedSpan, Tracer, check, end_with, event, fail, measure,
    parse_traceparent, random_seed, reserve, start,
};
pub(crate) use lk_agents_voice::telemetry::{
    Activation, JobIds, Opening, SessionTelemetry, participant_attrs,
};
// The extension's declared classes the core implements (`crate::cores`).
pub use lk_agents_core::telemetry::provider::{LlmMetricState, ProviderMetrics, TtsMetricState};
pub use lk_agents_core::telemetry::records::MetricsRecords;

/// How long the process-exit flush may take.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

struct Process {
    slot: Arc<ExporterSlot>,
    tracer: Tracer,
    /// `configure()`'s application endpoint, applied at installation when it comes first.
    app: Mutex<Option<AppConfig>>,
    /// Configuration conflicts the host found (a tracer provider it cannot bridge).
    conflicts: AtomicU64,
    /// The rate limit on the process's `event_loop_blocked` spans.
    blocked: Blocked,
    /// The voice sessions running, which a stall of their job's loop is folded into.
    sessions: Mutex<BTreeMap<u64, Stalled>>,
}

/// A running session as a stall reaches it: its job, its `agent_session` and its loop.
struct Stalled {
    job: Option<JobKey>,
    session: SpanRef,
    notify: Arc<dyn Fn(Duration) + Send + Sync>,
}

/// A session's registration for its job's stalls, withdrawn when dropped.
pub(crate) struct StallWatch(u64);

impl Drop for StallWatch {
    fn drop(&mut self) {
        PROCESS.sessions.lock().remove(&self.0);
    }
}

/// `session` (its `agent_session`) hears of every stall of `job`'s loop through `notify`, until
/// the returned watch drops.
pub(crate) fn watch_stalls(
    job: Option<JobKey>,
    session: SpanRef,
    notify: Arc<dyn Fn(Duration) + Send + Sync>,
) -> StallWatch {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PROCESS.sessions.lock().insert(id, Stalled { job, session, notify });
    StallWatch(id)
}

static PROCESS: LazyLock<Process> = LazyLock::new(|| {
    let slot = Arc::new(ExporterSlot::default());
    let tracer = Tracer::process(slot.clone(), crate::process::host_clock(), seed());
    if let Some(capture) = capture_from_env() {
        tracer.set_capture_content(capture);
    }
    Process {
        slot,
        tracer,
        app: Mutex::new(None),
        conflicts: AtomicU64::new(0),
        blocked: Blocked::default(),
        sessions: Mutex::new(BTreeMap::new()),
    }
});

fn seed() -> u64 {
    lk_agents_net::telemetry::random_seed()
}

/// The process tracer.
pub fn tracer() -> Tracer {
    PROCESS.tracer.clone()
}

/// A forked child abandons its parent's exporter and draws its own ids.
pub(crate) fn reinit() {
    PROCESS.slot.forget();
    PROCESS.tracer.reseed(seed());
}

/// The process-exit flush, bounded by [`SHUTDOWN_TIMEOUT`]. The exporter leaves the slot, so a
/// server started again in this process installs a new one instead of a closed one.
pub async fn shutdown() {
    if let Some(exporter) = PROCESS.slot.take() {
        exporter.shutdown(SHUTDOWN_TIMEOUT).await;
    }
}

fn allow_pii_from_env() -> bool {
    std::env::var("LIVEKIT_TELEMETRY_ALLOW_PII").is_ok_and(|v| {
        !matches!(v.trim().to_lowercase().as_str(), "" | "0" | "false" | "no" | "off")
    })
}

fn capture_from_env() -> Option<bool> {
    let v = std::env::var("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT").ok()?;
    Some(!matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no" | "off"))
}

fn env_app() -> Option<AppConfig> {
    AppConfig::from_env().map(|config| AppConfig { allow_pii: allow_pii_from_env(), ..config })
}

pub struct Telemetry;

impl lk_agents_abi::telemetry::host::Telemetry for Telemetry {
    fn _core_telemetry_install(sdk_version: String) {
        let process = &*PROCESS;
        if process.slot.get().is_some() {
            return;
        }
        let spawner = match crate::process::host_spawner() {
            Ok(spawner) => spawner,
            Err(error) => {
                return log::warn!(target: "livekit.agents", "telemetry not started: {error}");
            }
        };
        let exporter = process.slot.install(|| {
            let identity = ProcessIdentity::from_env(SdkName::Python, sdk_version);
            Exporter::start(&*spawner, crate::process::host_clock(), identity, Limits::default())
        });
        exporter.configure_app(process.app.lock().clone().or_else(env_app));
    }

    fn _core_telemetry_configure(
        endpoint: Option<String>,
        headers: Option<BTreeMap<String, String>>,
        metadata: Option<BTreeMap<String, JsonValue>>,
        allow_pii: Option<bool>,
    ) -> Result<(), TelemetryConfigError> {
        let mut config = env_app().unwrap_or_else(|| AppConfig {
            allow_pii: allow_pii_from_env(),
            ..AppConfig::default()
        });
        if let Some(endpoint) = endpoint {
            let base = AppConfig::with_endpoint(&endpoint);
            config.traces_endpoint = base.traces_endpoint;
            config.metrics_endpoint = base.metrics_endpoint;
            if config.resource.is_empty() {
                config.resource = base.resource;
            }
        }
        if let Some(headers) = headers {
            config.headers = headers.into_iter().collect();
        }
        if let Some(metadata) = metadata {
            config.metadata = metadata
                .iter()
                .filter_map(|(k, v)| Some((Key::Owned(k.clone()), json(v)?)))
                .collect();
        }
        if let Some(allow_pii) = allow_pii {
            config.allow_pii = allow_pii;
        }
        let config = (config.traces_endpoint.is_some() || config.metrics_endpoint.is_some())
            .then_some(config);
        *PROCESS.app.lock() = config.clone();
        if let Some(exporter) = PROCESS.slot.get() {
            exporter.configure_app(config);
        }
        Ok(())
    }

    fn _core_telemetry_diagnostics() -> BTreeMap<String, u64> {
        let d = PROCESS.slot.get().map(|e| e.diagnostics()).unwrap_or_default();
        let host_conflicts = PROCESS.conflicts.load(Ordering::Relaxed);
        BTreeMap::from([
            ("dropped".to_owned(), d.dropped),
            ("failed".to_owned(), d.failed),
            ("conflicts".to_owned(), d.conflicts + host_conflicts),
            ("incomplete_cleanup".to_owned(), d.incomplete_cleanup),
            ("swept_spans".to_owned(), d.swept_spans),
            ("abandoned_sessions".to_owned(), d.abandoned_sessions),
        ])
    }

    fn _core_telemetry_conflict() {
        PROCESS.conflicts.fetch_add(1, Ordering::Relaxed);
    }

    fn _core_set_capture_content(enabled: bool) {
        PROCESS.tracer.set_capture_content(enabled);
    }

    async fn _core_telemetry_shutdown() {
        shutdown().await;
    }

    fn _core_event_loop_blocked(
        duration: f64,
        parent: Option<(u128, u64, bool)>,
        job: Option<String>,
        task: Option<String>,
    ) {
        let parent = parent.filter(|(trace, span, _)| *trace != 0 && *span != 0).map(
            |(trace, span, sampled)| SpanRef {
                trace_id: TraceId(trace),
                span_id: SpanId(span),
                sampled,
            },
        );
        let job = job.map(JobKey);
        // The job's sessions count it; its span goes under the first one's `agent_session`.
        let sessions: Vec<_> = (PROCESS.sessions.lock().values())
            .filter(|s| s.job == job)
            .map(|s| (s.session, s.notify.clone()))
            .collect();
        let cx = PROCESS.tracer.root().continuing(parent).with_job(job);
        let cx = match sessions.first() {
            Some((session, _)) => cx.under::<AgentSession>(*session),
            None => cx,
        };
        for (_, notify) in &sessions {
            notify(Duration::from_secs_f64(duration.max(0.0)));
        }
        event_loop_blocked(&cx, &PROCESS.blocked, duration, task);
    }
}

/// A stall's report decides its severity from these, as 1.8.10's monitor did.
const BLOCKED_WARNING: f64 = 0.1;
const BLOCKED_ERROR: f64 = 0.5;
/// At most this many `event_loop_blocked` spans a minute; the rest are counted.
const BLOCKED_SPANS_PER_MINUTE: usize = 30;

/// The spans reported in the last minute, and the stalls suppressed since the last one.
#[derive(Default)]
pub(crate) struct Blocked(Mutex<(std::collections::VecDeque<u64>, u32)>);

/// One stall: its measurement always, its span within the rate limit.
pub(crate) fn event_loop_blocked(cx: &Cx, blocked: &Blocked, duration: f64, task: Option<String>) {
    let severity =
        if duration >= BLOCKED_ERROR { BlockingSeverity::Error } else { BlockingSeverity::Warning };
    lk_agents_net::telemetry::measure(cx, &EventLoopBlockedDuration { severity }, duration);
    let now = cx.now_nanos();
    let suppressed = {
        let (sent, suppressed) = &mut *blocked.0.lock();
        while sent.front().is_some_and(|at| now.saturating_sub(*at) >= 60_000_000_000) {
            sent.pop_front();
        }
        if sent.len() >= BLOCKED_SPANS_PER_MINUTE {
            *suppressed += 1;
            return;
        }
        sent.push_back(now);
        std::mem::take(suppressed)
    };
    let start = now.saturating_sub((duration * 1e9) as u64);
    let span = lk_agents_net::telemetry::start_at(
        cx,
        EventLoopBlocked {
            duration: Seconds(duration),
            threshold: Seconds(BLOCKED_WARNING),
            gc_time: None,
            cpu_time: None,
            severity,
            task,
            stack: None,
            import: None,
            suppressed: (suppressed > 0).then_some(suppressed),
        },
        start,
    );
    if severity == BlockingSeverity::Error {
        lk_agents_net::telemetry::fail(
            &span,
            &lk_agents_net::telemetry::Failure {
                kind: ErrorType::EventLoopBlocked,
                message: format!("the event loop was blocked for {duration:.3}s"),
            },
        );
    }
    lk_agents_net::telemetry::end_with(span, ());
}

/// One tool call's span, opened under `cx` (its `agent_turn` or `agent_run`).
pub(crate) fn function_tool(
    cx: &Cx,
    call: &FunctionCall,
    agent: &str,
    description: Option<&str>,
) -> ScopedSpan<FunctionTool> {
    let arguments: serde_json::Value =
        serde_json::from_str(&call.arguments).unwrap_or(JsonValue::String(call.arguments.clone()));
    let capture = cx.capture_content();
    lk_agents_net::telemetry::start(
        cx,
        FunctionTool {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: Content(Json(arguments.clone())),
            operation: GenAiOperation::ExecuteTool,
            tool_name: call.name.clone(),
            tool_type: ToolType::Function,
            call_id: call.call_id.clone(),
            agent_name: agent.to_owned(),
            description: description.filter(|_| capture).map(|d| GenAiContent(d.to_owned())),
            call_arguments: capture.then(|| GenAiContent(Json(arguments))),
        },
    )
}

/// Ends a tool call's span with what the body came to, and records its duration.
pub(crate) fn end_function_tool(
    span: ScopedSpan<FunctionTool>,
    call: &FunctionCall,
    started: u64,
    outcome: &ToolOutcome,
) {
    let (output, error) = match outcome {
        ToolOutcome::Value { text, .. } | ToolOutcome::Completed { text, .. } => {
            (text.clone(), false)
        }
        ToolOutcome::ToolError(message) | ToolOutcome::ModelRetry(message) => {
            (message.clone(), true)
        }
        ToolOutcome::Failed(message) | ToolOutcome::Rejected(message) => (message.clone(), true),
        ToolOutcome::InputRequired { question, .. } => (question.clone(), false),
        ToolOutcome::Internal | ToolOutcome::Invalid => (String::new(), true),
        _ => (String::new(), false),
    };
    let cx = span.cx();
    let error_type = error.then_some(ErrorType::ToolError);
    let seconds = cx.now_nanos().saturating_sub(started) as f64 / 1e9;
    lk_agents_net::telemetry::measure(
        cx,
        &ExecuteToolDuration {
            operation: GenAiOperation::ExecuteTool,
            tool: Label(call.name.clone()),
            error_type,
        },
        seconds,
    );
    let capture = cx.capture_content();
    let result = capture.then(|| GenAiContent(output.clone()));
    lk_agents_net::telemetry::end_with(
        span,
        FunctionToolEnd { output: Content(output), is_error: error, error_type, result },
    );
}

/// The `lk.agents.usage.*` counters and the connection time one metrics record carries, as
/// 1.8.10 recorded them: a zero records nothing; the labels are the record's model.
pub(crate) fn record_usage(cx: &Cx, metrics: &AgentMetrics) {
    let name = |v: Option<&String>| {
        Label(v.filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| "unknown".to_owned()))
    };
    let md = metrics.metadata();
    let model = || ModelLabels {
        provider: name(md.and_then(|m| m.model_provider.as_ref())),
        model: name(md.and_then(|m| m.model_name.as_ref())),
    };
    let count = |n: Option<u64>| n.filter(|n| *n != 0);
    let secs = |s: Option<f64>| s.filter(|s| *s != 0.0);
    macro_rules! add {
        ($instrument:ident, $value:expr) => {
            if let Some(v) = $value {
                lk_agents_net::telemetry::measure(&cx, &$instrument { model: model() }, v);
            }
        };
    }
    let connection = match metrics {
        AgentMetrics::Llm(m) => {
            add!(LlmInputTokens, count(Some(m.tokens.input_tokens)));
            add!(LlmInputCachedTokens, count(m.tokens.cache_read_tokens));
            add!(LlmOutputTokens, count(Some(m.tokens.output_tokens)));
            None
        }
        AgentMetrics::RealtimeModel(m) => {
            let t = &m.tokens;
            add!(LlmInputTokens, count(Some(t.input_tokens)));
            add!(LlmInputCachedTokens, count(t.cache_read_tokens));
            add!(LlmOutputTokens, count(Some(t.output_tokens)));
            add!(LlmInputAudioTokens, count(t.input_audio_tokens));
            add!(LlmInputTextTokens, count(t.input_text_tokens));
            add!(LlmOutputAudioTokens, count(t.output_audio_tokens));
            add!(LlmOutputTextTokens, count(t.output_text_tokens));
            add!(LlmSessionDuration, secs(m.connection_duration));
            Some((m.acquire_time, m.connection_reused))
        }
        AgentMetrics::Tts(m) => {
            add!(TtsCharacters, count(Some(m.characters_count.into())));
            add!(TtsAudioDuration, secs(Some(m.audio_duration)));
            Some((m.acquire_time, m.connection_reused))
        }
        AgentMetrics::Stt(m) => {
            add!(SttAudioDuration, secs(m.audio_duration));
            Some((m.acquire_time, m.connection_reused))
        }
        AgentMetrics::Interruption(m) => {
            add!(InterruptionNumRequests, count(Some(m.num_requests.into())));
            None
        }
        _ => None,
    };
    if let Some((Some(acquire), reused)) = connection
        && acquire > 0.0
    {
        lk_agents_net::telemetry::measure(
            cx,
            &ConnectionAcquireTime { model: model(), reused: BoolString(reused) },
            acquire,
        );
    }
}
