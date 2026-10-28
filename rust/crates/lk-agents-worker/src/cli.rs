//! Legacy CLI host declarations, compiled by the runtime until static async calls are generated.
use crate::worker::cli::LogFields;
use std::sync::Arc;

pub use lk_agents_abi::worker::cli::{DevJob, DevServer};

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
