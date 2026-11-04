//! What the exporter is told: the process and job identity, the Cloud endpoint and a job's
//! recording decision, and the application endpoint (`OTEL_EXPORTER_OTLP_*` or explicit).

use lk_agents_abi::telemetry::record::KeyValues;

/// `lk.sdk.name`: the SDK a process runs, a closed set Agent Insights reads. The Python binding
/// passes [`SdkName::Python`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdkName {
    Python,
    Node,
}

impl SdkName {
    pub fn as_str(self) -> &'static str {
        match self {
            SdkName::Python => "PYTHON",
            SdkName::Node => "NODE",
        }
    }
}

/// The process's identity on every resource (`lk.sdk.*`) and Cloud record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub sdk_name: SdkName,
    /// `lk.sdk.version`: the framework's version.
    pub sdk_version: String,
    /// `LIVEKIT_AGENT_ID`, set by LiveKit Cloud.
    pub cloud_agent_id: Option<String>,
    /// `LIVEKIT_AGENT_DEPLOYMENT`, set by LiveKit Cloud.
    pub deployment_id: Option<String>,
}

impl ProcessIdentity {
    pub fn from_env(sdk_name: SdkName, sdk_version: impl Into<String>) -> Self {
        let var = |name| std::env::var(name).ok().filter(|v| !v.is_empty());
        ProcessIdentity {
            sdk_name,
            sdk_version: sdk_version.into(),
            cloud_agent_id: var("LIVEKIT_AGENT_ID"),
            deployment_id: var("LIVEKIT_AGENT_DEPLOYMENT"),
        }
    }
}

/// A job's identity, known when the job is received.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JobIdentity {
    pub room_id: String,
    pub job_id: String,
    pub room_name: String,
    pub agent_name: Option<String>,
}

/// Where Cloud telemetry goes and the credentials it is signed with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudEndpoint {
    /// The observability base URL (`strings::observability_url`).
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
}

/// A job's Cloud recording decision.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CloudRecording {
    pub traces: bool,
    /// The session report's records.
    pub logs: bool,
    pub metrics: bool,
    pub redaction: bool,
    pub simulation: Option<Simulation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Simulation {
    pub run_id: Option<String>,
    pub job_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    #[default]
    Protobuf,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Temporality {
    #[default]
    Cumulative,
    Delta,
}

/// The application's OTLP/HTTP endpoint: traces and metrics, never report records.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppConfig {
    /// Full URL; `None` sends no spans.
    pub traces_endpoint: Option<String>,
    /// Full URL; `None` sends no metrics.
    pub metrics_endpoint: Option<String>,
    pub headers: Vec<(String, String)>,
    pub protocol: Protocol,
    pub gzip: bool,
    pub temporality: Temporality,
    /// Resource attributes, `service.name` included.
    pub resource: Vec<(String, String)>,
    /// Stamped on every span.
    pub metadata: KeyValues,
    /// Keep `lk.pii.*`, GenAI content and exception text.
    pub allow_pii: bool,
}

const SERVICE_NAME: &str = "service.name";
const DEFAULT_SERVICE: &str = "livekit-agents";

impl AppConfig {
    /// Both signals under an OTLP base endpoint (`.../v1/traces`, `.../v1/metrics`).
    pub fn with_endpoint(endpoint: &str) -> Self {
        let mut config = AppConfig::default();
        config.set_base(endpoint);
        config.resource = vec![(SERVICE_NAME.into(), DEFAULT_SERVICE.into())];
        config
    }

    fn set_base(&mut self, endpoint: &str) {
        let base = endpoint.trim_end_matches('/');
        self.traces_endpoint = Some(format!("{base}/v1/traces"));
        self.metrics_endpoint = Some(format!("{base}/v1/metrics"));
    }

    /// From the standard `OTEL_*` variables; `None` when no endpoint is set.
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
    }

    pub(crate) fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let mut config = AppConfig::default();
        if let Some(base) = var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            config.set_base(&base);
        }
        if let Some(url) = var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") {
            config.traces_endpoint = Some(url);
        }
        if let Some(url) = var("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") {
            config.metrics_endpoint = Some(url);
        }
        if config.traces_endpoint.is_none() && config.metrics_endpoint.is_none() {
            return None;
        }
        config.headers = var("OTEL_EXPORTER_OTLP_HEADERS").map(|v| pairs(&v)).unwrap_or_default();
        config.protocol = match var("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref() {
            Some("http/json") => Protocol::Json,
            Some("http/protobuf") | None => Protocol::Protobuf,
            Some(other) => {
                log::warn!(
                    target: "livekit.agents",
                    "unsupported OTLP protocol {other:?}; using http/protobuf"
                );
                Protocol::Protobuf
            }
        };
        config.gzip = var("OTEL_EXPORTER_OTLP_COMPRESSION").as_deref() == Some("gzip");
        config.temporality = match var("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE")
            .as_deref()
        {
            Some(v) if v.eq_ignore_ascii_case("delta") || v.eq_ignore_ascii_case("lowmemory") => {
                Temporality::Delta
            }
            _ => Temporality::Cumulative,
        };
        let mut resource = var("OTEL_RESOURCE_ATTRIBUTES").map(|v| pairs(&v)).unwrap_or_default();
        resource.retain(|(k, _)| k != SERVICE_NAME || var("OTEL_SERVICE_NAME").is_none());
        let service = var("OTEL_SERVICE_NAME").unwrap_or_else(|| DEFAULT_SERVICE.into());
        if !resource.iter().any(|(k, _)| k == SERVICE_NAME) {
            resource.insert(0, (SERVICE_NAME.into(), service));
        }
        config.resource = resource;
        Some(config)
    }
}

/// `k1=v1,k2=v2` with percent-encoded values, as the OTel environment specification writes
/// headers and resource attributes. Malformed entries are skipped.
fn pairs(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|entry| {
            let (key, value) = entry.split_once('=')?;
            let key = percent_decode(key.trim())?;
            (!key.is_empty()).then_some(())?;
            Some((key, percent_decode(value.trim())?))
        })
        .collect()
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(entries: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let entries: Vec<(String, String)> =
            entries.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect();
        move |name| entries.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    // Catches: signal endpoints not overriding the base, or headers and resource attributes
    // read without percent-decoding.
    #[test]
    fn the_environment_follows_the_otel_specification() {
        assert_eq!(AppConfig::from_vars(vars(&[("OTEL_SERVICE_NAME", "x")])), None);
        let config = AppConfig::from_vars(vars(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"),
            ("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://metrics/custom"),
            ("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer%20abc, x-team = voice ,bad"),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip"),
            ("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "Delta"),
            ("OTEL_RESOURCE_ATTRIBUTES", "service.name=ignored,deployment.environment=prod%2Ceu"),
            ("OTEL_SERVICE_NAME", "front-desk"),
        ]))
        .unwrap();
        assert_eq!(config.traces_endpoint.as_deref(), Some("http://collector:4318/v1/traces"));
        assert_eq!(config.metrics_endpoint.as_deref(), Some("http://metrics/custom"));
        assert_eq!(
            config.headers,
            vec![("authorization".into(), "Bearer abc".into()), ("x-team".into(), "voice".into())]
        );
        assert_eq!((config.protocol, config.gzip), (Protocol::Json, true));
        assert_eq!(config.temporality, Temporality::Delta);
        assert_eq!(
            config.resource,
            vec![
                ("service.name".into(), "front-desk".into()),
                ("deployment.environment".into(), "prod,eu".into())
            ]
        );
        let traces_only = AppConfig::from_vars(vars(&[(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://t/v1/traces",
        )]))
        .unwrap();
        assert_eq!(traces_only.metrics_endpoint, None);
        assert_eq!(traces_only.temporality, Temporality::Cumulative);
        assert_eq!(traces_only.resource, vec![("service.name".into(), "livekit-agents".into())]);
    }
}
