//! Session recording uploads. The runtime owns credentials, files, attempts and transport.

/// The immutable input of one recording upload; no host objects or file contents are retained.
#[derive(Debug, Clone)]
pub struct RecordingUpload {
    pub base_url: String,
    pub room_id: String,
    pub job_id: String,
    pub simulated: bool,
    pub redaction_enabled: bool,
    pub start_time_millis: i64,
    pub transcript: Option<String>,
    pub audio_path: Option<String>,
    pub disabled: bool,
}

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct UploadError(pub String);

/// Credentials for the session recording endpoint and host OTLP exporters.
#[crate::host(class(name = "_core_SessionReportUpload"))]
#[async_trait::async_trait]
pub trait SessionReportUpload: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(api_key: String, api_secret: String) -> Result<Self, UploadError>;

    /// A fresh LiveKit observability credential. No token policy lives in the host adapter.
    #[host(call)]
    fn bearer(&self) -> Result<String, UploadError>;

    /// Recording options are explicit consent for audio; transcript also requires GenAI content
    /// capture. Project redaction excludes both raw content formats.
    #[host(call)]
    fn content_policy(
        &self,
        transcript_requested: bool,
        audio_requested: bool,
        capture_content: bool,
        redaction_enabled: bool,
    ) -> (bool, bool);

    /// Project and serialize a native history on bounded runtime workers, with no host objects
    /// or Python JSON allocation on the session loop.
    #[host(await)]
    async fn transcript(&self, items: Vec<crate::ChatItem>) -> Result<String, UploadError>;

    /// Returns false only when the project disabled recording. Export failures are reported once
    /// to the caller, which decides where to surface them without failing the conversation.
    #[host(await)]
    #[allow(clippy::too_many_arguments)]
    async fn upload(
        &self,
        base_url: String,
        room_id: String,
        job_id: String,
        simulated: bool,
        redaction_enabled: bool,
        start_time_millis: i64,
        transcript: Option<String>,
        audio_path: Option<String>,
        disabled: bool,
    ) -> Result<bool, UploadError>;
}
