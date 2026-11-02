//! One synchronous LLM request owner, including retries, response assembly, and cancellation.
//! The gateway adapter performs its effects and supplies observations and time.

mod request;

pub use request::{AttemptFailure, LlmRequest, LlmRequestEvent, LlmRequestId, RequestPolicy};
