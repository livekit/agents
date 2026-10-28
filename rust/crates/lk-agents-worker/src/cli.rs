//! Legacy CLI host declarations, compiled by the runtime until static async calls are generated.
use crate::worker::cli::LogFields;
use lk_agents_abi::errors::HostError;
use std::sync::Arc;

/// A job running under the dev server, kept across reloads: the `Job` protobuf as bytes, the
/// accept arguments (name, identity, metadata), the server URL, the token, the worker id, and
/// whether it is a fake job.
pub type DevJob = (Vec<u8>, String, String, String, String, String, String, bool);

/// The agent server the dev channel reports on.
#[lk_agents_abi::host(proxy(live))]
#[async_trait::async_trait]
pub trait DevServer: Send + Sync {
    /// The jobs running now.
    fn running_jobs(&self) -> Vec<DevJob>;
    /// Resume the jobs the CLI kept from before the reload, on this server's URL and a fresh token.
    async fn reload_jobs(&self, jobs: Vec<DevJob>) -> Result<(), HostError>;
}

/// Format one host log record.
#[lk_agents_abi::host(export(pid = false))]
pub fn format_log_record(
    style: &str,
    record: LogFields,
    message: String,
    asctime: String,
    exception: Option<String>,
    stack: Option<String>,
    ensure_ascii: bool,
) -> String {
    crate::worker::cli::format_log_record(
        style,
        record,
        message,
        asctime,
        exception,
        stack,
        ensure_ascii,
    )
}

/// Serve the development CLI connection.
#[lk_agents_abi::host(export)]
pub async fn dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    api_key: String,
    api_secret: String,
    server: Arc<dyn DevServer>,
) {
    crate::worker::cli::dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await;
}
