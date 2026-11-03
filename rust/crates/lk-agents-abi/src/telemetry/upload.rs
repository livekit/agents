//! The Cloud credentials the host's own OTLP exporters sign with. The session report's recording
//! upload is `lk-agents-net`'s, with no host contract.

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct UploadError(pub String);

/// Credentials for the host's OTLP exporters.
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
}
