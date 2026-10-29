//! The Cloud session report. `lk-agents-net` owns its credentials, files, attempts and transport.

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct UploadError(pub String);

/// Credentials for the session recording endpoint and host OTLP exporters.
#[crate::host(class(name = "_core_SessionReportUpload"))]
pub trait SessionReportUpload: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env, api_key: String, api_secret: String) -> Result<Self, UploadError>;

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

/// Incremental report preparation. Host adapters admit one item's shape before conversion and
/// append it off their event loop; no whole-history extraction crosses a binding call.
#[crate::host(class(name = "_core_PreparedTranscript"))]
pub trait PreparedTranscript: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env) -> Result<Self, UploadError>;

    /// Maximum items per append batch before the adapter returns control to its event loop.
    #[host(getter)]
    fn input_batch_size(&self) -> usize;

    /// Which media parts belong to the transcript's native projection (image, audio).
    #[host(call)]
    fn content_policy(&self) -> (bool, bool);

    /// Validate host collection lengths and the shared retained-byte verdict before converting an
    /// item's fields. These scalar facts cannot allocate an unbounded native item.
    #[host(call)]
    fn check_input(&self, items: usize, item_bytes: usize) -> Result<(), UploadError>;

    #[host(call)]
    fn append(&self, item: crate::ChatItem) -> Result<(), UploadError>;

    #[host(call)]
    fn finish(&self) -> Result<String, UploadError>;

    /// Append one already projected event to the same bounded recording preparation.
    #[host(call)]
    fn append_event(&self, event: serde_json::Value) -> Result<bool, UploadError>;

    /// Serialize and write a console report without converting its history back into host objects.
    #[host(await)]
    async fn finish_report(
        &self,
        report: super::report::projection::SessionReportSnapshot,
        path: String,
    ) -> Result<(), UploadError>;
}
