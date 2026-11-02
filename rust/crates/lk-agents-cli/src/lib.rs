//! Agent process launcher: command line, remote CLI connection, logs and shutdown.

use std::sync::Arc;

pub mod args;
pub mod logging;
pub mod remote_cli;
pub mod shutdown;

pub use args::parse_cli;
pub use logging::{LogRecord, format_log_record};
pub use remote_cli::dev_channel;

const LOG: &str = "livekit.agents";

/// The launcher's functions as the host calls them.
pub struct Commands;

impl lk_agents_abi::cli::Cli for Commands {
    fn parse_cli(argv: Vec<String>) -> lk_agents_abi::cli::CliParseResult {
        parse_cli(argv)
    }

    fn format_log_record(record: LogRecord) -> String {
        format_log_record(record)
    }

    async fn dev_channel(
        cli_addr: String,
        agent_name: String,
        url: String,
        api_key: String,
        api_secret: String,
        server: Arc<dyn lk_agents_abi::cli::DevServer>,
    ) {
        dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await;
    }
}
