//! Native realtime clients and their presence-aware option boundary.

use super::llm::ToolChoice;
use super::realtime::RealtimeCapabilities;
use crate::errors::{ApiError, HostError};
use crate::{ApiConnectOptions, HostRef};

/// A supplied JSON option, including explicit null. An absent wrapper means omitted.
#[crate::host(record(module = "realtime_options", extract, emit))]
#[derive(Debug, Clone)]
pub struct RealtimeJsonOption {
    pub value: serde_json::Value,
}

#[crate::host(record(module = "realtime_options", extract, emit))]
#[derive(Debug, Clone)]
pub struct RealtimeToolChoice {
    pub value: Option<ToolChoice>,
}

#[crate::host(record(module = "realtime_options", extract, emit))]
#[derive(Debug, Clone)]
pub struct RealtimeDurationOption {
    pub value: Option<f64>,
}

/// Omission preserves a setting; a supplied nullable wrapper clears it explicitly.
#[crate::host(record(module = "realtime_options", extract, emit))]
#[derive(Debug, Clone, Default)]
pub struct RealtimeOptions {
    #[host(default = None)]
    pub input_audio_transcription: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub input_audio_noise_reduction: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub turn_detection: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub tool_choice: Option<RealtimeToolChoice>,
    #[host(default = None)]
    pub speed: Option<f64>,
    #[host(default = None)]
    pub tracing: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub truncation: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub reasoning: Option<RealtimeJsonOption>,
    #[host(default = None)]
    pub max_session_duration: Option<RealtimeDurationOption>,
    #[host(default = None)]
    pub voice: Option<String>,
    #[host(default = None)]
    pub max_response_output_tokens: Option<RealtimeJsonOption>,
}

#[crate::host(record(module = "realtime_options", extract, emit))]
#[derive(Debug, Clone)]
pub struct NativeRealtimeConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub conn_options: ApiConnectOptions,
    pub options: RealtimeOptions,
    #[host(default = None)]
    pub dialect: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = None)]
    pub voice: Option<String>,
    #[host(default = None)]
    pub modalities: Option<Vec<String>>,
    #[host(default = None)]
    pub azure_deployment: Option<String>,
    #[host(default = None)]
    pub entra_token: Option<String>,
    #[host(default = false)]
    pub azure_defaults: bool,
    #[host(default = None)]
    pub api_secret: Option<String>,
    #[host(default = None)]
    pub inference_provider: Option<String>,
    #[host(default = None)]
    pub inference_class: Option<String>,
    #[host(default = None)]
    pub label: Option<String>,
}

#[crate::host(error(with = crate::llm::realtime_native::configuration_error_to_py))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum RealtimeConfigurationError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error(transparent)]
    Host(#[from] HostError),
}

#[cfg(feature = "python")]
pub fn configuration_error_to_py(error: RealtimeConfigurationError) -> pyo3::PyErr {
    use crate::python::HostException as _;
    match error {
        RealtimeConfigurationError::Invalid(message) => {
            pyo3::exceptions::PyValueError::new_err(message)
        }
        RealtimeConfigurationError::Api(error) => error.to_py_err(),
        RealtimeConfigurationError::Host(error) => error.to_py_err(),
    }
}

#[crate::host(class(name = "NativeRealtimeModel", native = crate::llm::realtime::Realtime))]
pub trait NativeRealtimeModel: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(config: NativeRealtimeConfig) -> Result<Self, RealtimeConfigurationError>;
    #[host(getter)]
    fn provider(&self) -> String;
    #[host(getter)]
    fn model(&self) -> String;
    #[host(getter)]
    fn label(&self) -> String;
    #[host(call)]
    fn capabilities(&self) -> RealtimeCapabilities;
    #[host(call)]
    fn update_options(&self, options: RealtimeOptions);
    #[host(call)]
    fn bind_identification(&self, headers: Vec<(String, String)>);
}

/// The generated facade holds native resources; host provider objects use the shared bridge.
#[crate::host(class(name = "RealtimeFallbackModel", native = crate::llm::realtime::Realtime))]
#[async_trait::async_trait]
pub trait RealtimeFallbackModel: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        models: Vec<HostRef>,
        #[host(kw, default = 10.0)] cooldown: f64,
        #[host(default = true)] regenerate_on_swap: bool,
        #[host(default = None)] on_availability: Option<HostRef>,
    ) -> Result<Self, RealtimeConfigurationError>;
    #[host(getter)]
    fn active_index(&self) -> usize;
    #[host(call)]
    fn capabilities(&self) -> RealtimeCapabilities;
    #[host(await)]
    async fn restart(&self, #[host(kw, default = false)] switch_model: bool);
}
