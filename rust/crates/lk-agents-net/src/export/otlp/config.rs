use lk_agents_abi::telemetry::record::{FRAMEWORK, KeyValues};
use lk_agents_abi::telemetry::semconv::otel::SERVICE_NAME;
use lk_agents_abi::worker::job::Simulation;

use super::Signal;

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

/// The session's identity, known when its job is received; `None` is unknown and never stamped.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionIdentity {
    pub room_id: Option<String>,
    pub job_id: Option<String>,
    pub room_name: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Temporality {
    #[default]
    Cumulative,
    Delta,
}

/// One signal's OTLP/HTTP destination. The specification configures endpoint, headers and
/// compression per signal, each falling back to the shared variable.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    /// Full URL (`.../v1/traces`).
    pub url: String,
    pub headers: Vec<(String, String)>,
    /// `Content-Encoding: gzip`; the specification's default is no compression.
    pub gzip: bool,
}

/// The application's OTLP/HTTP endpoint: traces and metrics, never report records.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppConfig {
    /// `None` sends no spans.
    pub traces: Option<Target>,
    /// `None` sends no metrics.
    pub metrics: Option<Target>,
    pub temporality: Temporality,
    /// Resource attributes, `service.name` included.
    pub resource: Vec<(String, String)>,
    /// Stamped on every span.
    pub metadata: KeyValues,
    /// Keep `lk.pii.*`, GenAI content and exception text.
    pub allow_pii: bool,
}

impl AppConfig {
    /// Both signals under an OTLP base endpoint, uncompressed and without headers.
    pub fn with_endpoint(endpoint: &str) -> Self {
        let mut config = AppConfig { resource: framework_resource(), ..AppConfig::default() };
        config.set_endpoint(endpoint);
        config
    }

    /// Both signals under an OTLP base endpoint (`.../v1/traces`, `.../v1/metrics`); a signal
    /// already configured keeps its headers and compression.
    pub fn set_endpoint(&mut self, endpoint: &str) {
        for (signal, target) in
            [(Signal::Traces, &mut self.traces), (Signal::Metrics, &mut self.metrics)]
        {
            let url = signal_url(endpoint, signal);
            match target {
                Some(target) => target.url = url,
                None => *target = Some(Target { url, headers: Vec::new(), gzip: false }),
            }
        }
    }

    /// The same headers on every configured signal.
    pub fn set_headers(&mut self, headers: &[(String, String)]) {
        for target in [&mut self.traces, &mut self.metrics].into_iter().flatten() {
            target.headers = headers.to_vec();
        }
    }

    /// From the standard `OTEL_*` variables; `None` when no endpoint is set.
    pub fn from_env() -> Option<Self> {
        Self::from_vars(|name| std::env::var(name).ok().filter(|v| !v.is_empty()))
    }

    pub(crate) fn from_vars(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let base = var("OTEL_EXPORTER_OTLP_ENDPOINT");
        let target = |signal: Signal| {
            let name = signal.path().to_ascii_uppercase();
            let own = |key: &str| {
                var(&format!("OTEL_EXPORTER_OTLP_{name}_{key}"))
                    .or_else(|| var(&format!("OTEL_EXPORTER_OTLP_{key}")))
            };
            let url = var(&format!("OTEL_EXPORTER_OTLP_{name}_ENDPOINT"))
                .or_else(|| base.as_deref().map(|base| signal_url(base, signal)))?;
            let gzip = match own("COMPRESSION").as_deref() {
                Some("gzip") => true,
                None | Some("none") => false,
                Some(other) => {
                    log::warn!(
                        target: lk_agents_abi::LOG_TARGET,
                        "unsupported OTLP compression {other:?}; sending uncompressed"
                    );
                    false
                }
            };
            let headers = own("HEADERS").map(|v| pairs(&v)).unwrap_or_default();
            Some(Target { url, headers, gzip })
        };
        let mut config = AppConfig {
            traces: target(Signal::Traces),
            metrics: target(Signal::Metrics),
            ..AppConfig::default()
        };
        if config.traces.is_none() && config.metrics.is_none() {
            return None;
        }
        if let Some(other) = var("OTEL_EXPORTER_OTLP_PROTOCOL").filter(|p| p != "http/protobuf") {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                log::warn!(
                    target: lk_agents_abi::LOG_TARGET,
                    "unsupported OTLP protocol {other:?}; using http/protobuf"
                );
            });
        }
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
        let service = var("OTEL_SERVICE_NAME").unwrap_or_else(|| FRAMEWORK.into());
        if !resource.iter().any(|(k, _)| k == SERVICE_NAME) {
            resource.insert(0, (SERVICE_NAME.into(), service));
        }
        config.resource = resource;
        Some(config)
    }
}

/// The resource of an application that names no service: `service.name` is the framework.
pub fn framework_resource() -> Vec<(String, String)> {
    vec![(SERVICE_NAME.into(), FRAMEWORK.into())]
}

/// `signal`'s path under an OTLP base endpoint.
fn signal_url(base: &str, signal: Signal) -> String {
    format!("{}/v1/{}", base.trim_end_matches('/'), signal.path())
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

    // Catches: a signal's own endpoint, headers or compression not overriding the shared one, or
    // headers and resource attributes read without percent-decoding.
    #[test]
    fn the_environment_follows_the_otel_specification() {
        assert_eq!(AppConfig::from_vars(vars(&[("OTEL_SERVICE_NAME", "x")])), None);
        let config = AppConfig::from_vars(vars(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/"),
            ("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://metrics/custom"),
            ("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer%20abc, x-team = voice ,bad"),
            ("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip"),
            ("OTEL_EXPORTER_OTLP_METRICS_COMPRESSION", "none"),
            ("OTEL_EXPORTER_OTLP_METRICS_HEADERS", "x-metrics=1"),
            ("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "Delta"),
            ("OTEL_RESOURCE_ATTRIBUTES", "service.name=ignored,deployment.environment=prod%2Ceu"),
            ("OTEL_SERVICE_NAME", "front-desk"),
        ]))
        .unwrap();
        let shared_headers =
            vec![("authorization".into(), "Bearer abc".into()), ("x-team".into(), "voice".into())];
        assert_eq!(
            config.traces,
            Some(Target {
                url: "http://collector:4318/v1/traces".into(),
                headers: shared_headers,
                gzip: true,
            })
        );
        assert_eq!(
            config.metrics,
            Some(Target {
                url: "http://metrics/custom".into(),
                headers: vec![("x-metrics".into(), "1".into())],
                gzip: false,
            })
        );
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
        assert_eq!(traces_only.metrics, None);
        assert_eq!(traces_only.temporality, Temporality::Cumulative);
        assert_eq!(traces_only.resource, vec![("service.name".into(), "livekit-agents".into())]);
    }
}
