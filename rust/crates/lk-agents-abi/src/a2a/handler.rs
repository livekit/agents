//! Host return classification; the core decides the transport outcome.

use crate::{
    ChatItem,
    a2a::turn::{HandlerOutput, PendingCall},
    storage::StorageError,
    usage::Usage,
};
use serde_json::Value;

#[crate::host(record(module = "storage_handler", extract))]
#[derive(Clone)]
pub enum HandlerOutcome {
    Value { value: Value },
    Completed { value: Value, verbatim: bool },
    InputRequired { question: String },
    Failed { error: String },
    Rejected { reason: String },
    Working { message: Option<String> },
}

#[crate::host(record(module = "storage_handler", extract))]
#[derive(Clone)]
pub struct HandlerRun {
    pub items: Vec<ChatItem>,
    pub usage: Usage,
    pub verbatim: Option<String>,
    pub question: Option<String>,
    pub pending_call: Option<PendingCall>,
}

#[crate::host(functions)]
pub trait HandlerOutcomes {
    #[host(function)]
    fn normalize_handler_output(
        output: HandlerOutput,
        outcome: HandlerOutcome,
        runs: Vec<HandlerRun>,
    ) -> Result<HandlerOutput, StorageError>;
}
