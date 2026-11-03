//! The process's telemetry (docs/internals/telemetry.md "Destinations"): one tracer every
//! `ProcessEnv` lends, and one OTLP exporter it records to once the host installs telemetry; and
//! the `function_tool` span the voice and text executors share.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use lk_agents_abi::FunctionCall;
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::telemetry::decl::{Content, GenAiContent, Json, Label, json};
use lk_agents_abi::telemetry::host::TelemetryConfigError;
use lk_agents_abi::telemetry::instruments::ExecuteToolDuration;
use lk_agents_abi::telemetry::record::Key;
use lk_agents_abi::telemetry::spans::agent_turn::{FunctionTool, FunctionToolEnd};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_net::otlp::{AppConfig, Exporter, Limits, ProcessIdentity, Protocol, SdkName};
use lk_agents_net::telemetry::{Cx, ExporterSlot, ScopedSpan, Tracer};
use parking_lot::Mutex;
use serde_json::Value;

/// How long the process-exit flush may take.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

struct Process {
    slot: Arc<ExporterSlot>,
    tracer: Tracer,
    /// `configure()`'s application endpoint, applied at installation when it comes first.
    app: Mutex<Option<AppConfig>>,
    /// Configuration conflicts the host found (a tracer provider it cannot bridge).
    conflicts: AtomicU64,
}

static PROCESS: LazyLock<Process> = LazyLock::new(|| {
    let slot = Arc::new(ExporterSlot::default());
    let tracer = Tracer::process(slot.clone(), crate::process::host_clock(), seed());
    if let Some(capture) = capture_from_env() {
        tracer.set_capture_content(capture);
    }
    Process { slot, tracer, app: Mutex::new(None), conflicts: AtomicU64::new(0) }
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

/// The process-exit flush, bounded by [`SHUTDOWN_TIMEOUT`].
pub async fn shutdown() {
    if let Some(exporter) = PROCESS.slot.get() {
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
        protocol: Option<String>,
        metadata: Option<BTreeMap<String, Value>>,
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
        if let Some(protocol) = protocol {
            config.protocol = match protocol.as_str() {
                "http/protobuf" => Protocol::Protobuf,
                "http/json" => Protocol::Json,
                other => {
                    return Err(TelemetryConfigError(format!(
                        "protocol must be \"http/protobuf\" or \"http/json\", not {other:?}"
                    )));
                }
            };
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
        ])
    }

    fn _core_telemetry_conflict() {
        PROCESS.conflicts.fetch_add(1, Ordering::Relaxed);
    }

    fn _core_set_capture_content(enabled: bool) {
        PROCESS.tracer.set_capture_content(enabled);
    }
}

/// One tool call's span, opened under `cx` (its `agent_turn` or `agent_run`).
pub(crate) fn function_tool(
    cx: &Cx,
    call: &FunctionCall,
    agent: &str,
    description: Option<&str>,
) -> ScopedSpan<FunctionTool> {
    let arguments: serde_json::Value =
        serde_json::from_str(&call.arguments).unwrap_or(Value::String(call.arguments.clone()));
    let capture = cx.tracer().capture_content();
    cx.start_span(FunctionTool {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: Content(Json(arguments.clone())),
        operation: GenAiOperation::ExecuteTool,
        tool_name: call.name.clone(),
        tool_type: "function".to_owned(),
        call_id: call.call_id.clone(),
        agent_name: agent.to_owned(),
        description: description.filter(|_| capture).map(|d| GenAiContent(d.to_owned())),
        call_arguments: capture.then(|| GenAiContent(Json(arguments))),
    })
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
    let error_type = error.then(|| "tool_error".to_owned());
    let seconds = cx.tracer().now_nanos().saturating_sub(started) as f64 / 1e9;
    cx.measure(
        &ExecuteToolDuration {
            operation: GenAiOperation::ExecuteTool,
            tool: Label(call.name.clone()),
            error_type: error_type.clone().map(Label),
        },
        seconds,
    );
    let capture = cx.tracer().capture_content();
    let result = capture.then(|| GenAiContent(output.clone()));
    span.end(FunctionToolEnd { output: Content(output), is_error: error, error_type, result });
}
