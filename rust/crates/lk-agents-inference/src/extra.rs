//! The gateway session's `connection` block every client sends.

use lk_agents_abi::ApiConnectOptions;
use serde::Serialize;

#[derive(Serialize)]
pub struct Connection {
    retries: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<f64>,
}

impl From<&ApiConnectOptions> for Connection {
    fn from(options: &ApiConnectOptions) -> Self {
        Self { retries: options.max_retry, timeout: options.connect_timeout }
    }
}
