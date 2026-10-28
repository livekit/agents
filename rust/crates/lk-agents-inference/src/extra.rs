//! The gateway session's `connection` block every client sends.

use lk_agents_abi::ApiConnectOptions;
use serde_json::{Value, json};

/// The session-create `connection` block: the gateway retries its provider connection `retries`
/// times, each try bounded by `timeout` (our `connect_timeout`; absent, the gateway's default).
pub fn connection(conn_options: &ApiConnectOptions) -> Value {
    let mut block = json!({"retries": conn_options.max_retry});
    if let Some(timeout) = conn_options.connect_timeout {
        block["timeout"] = json!(timeout);
    }
    block
}
