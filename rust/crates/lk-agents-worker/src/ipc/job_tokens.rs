//! Job token requests from a job process, in token mode.
//!
//! The job sends a `CreateJobToken` message to its worker. The worker asks the server and sends
//! back a `JobToken` message. The job matches the answer to the request by its request id.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lk_agents_abi::Clock;
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_net::job_token::{Issue, Issued};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use super::ipc_channel::Channel;
use super::wire::{JobTokenAnswer, ToChild, ToParent};

/// How long a job waits for its worker's answer.
///
/// This is longer than the worker's own wait for the server, [`crate::JOB_TOKEN_TIMEOUT`]. So this
/// timeout fires only when the worker itself does not answer.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(6);

/// Sends the job's token requests to the worker over the job's IPC channel.
pub(crate) struct WorkerIssue {
    parent: Arc<Channel<ToParent, ToChild>>,
    clock: Arc<dyn Clock>,
    next: AtomicU64,
    pending: Mutex<BTreeMap<u64, oneshot::Sender<JobTokenAnswer>>>,
}

/// Removes the request when the asker stops waiting, for any reason.
struct Waiting<'a>(&'a WorkerIssue, u64);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.pending.lock().remove(&self.1);
    }
}

impl WorkerIssue {
    pub(crate) fn new(parent: Arc<Channel<ToParent, ToChild>>, clock: Arc<dyn Clock>) -> Self {
        Self { parent, clock, next: AtomicU64::new(0), pending: Mutex::default() }
    }

    /// Delivers the worker's answer to a request. When nobody waits for the answer, it is
    /// dropped.
    pub(crate) fn answer(&self, request_id: u64, answer: JobTokenAnswer) {
        let waiting = self.pending.lock().remove(&request_id);
        if let Some(waiting) = waiting {
            // The asker stopped waiting, so nobody reads the answer.
            let _ = waiting.send(answer);
        }
    }
}

#[async_trait::async_trait]
impl Issue for WorkerIssue {
    async fn issue(&self, claims: &str) -> Result<Issued, ApiError> {
        let request_id = self.next.fetch_add(1, Ordering::Relaxed);
        let (answer, answered) = oneshot::channel();
        self.pending.lock().insert(request_id, answer);
        let _waiting = Waiting(self, request_id);
        let request = ToParent::CreateJobToken { request_id, claims: claims.to_owned() };
        self.parent.send(request).await.map_err(|error| {
            ApiError::new(
                ErrorKind::Connection,
                format!("cannot ask the worker for a job token: {error}"),
            )
        })?;
        match lk_agents_abi::timeout(&*self.clock, ANSWER_TIMEOUT, answered).await {
            Ok(Ok(JobTokenAnswer::Issued { token, expires_at })) => {
                Ok(Issued { token: token.0, expires_at })
            }
            Ok(Ok(JobTokenAnswer::Refused(error))) => Err(ApiError::new(ErrorKind::Auth, error)),
            Ok(Err(_)) => Err(ApiError::new(
                ErrorKind::Connection,
                "the job stopped before the worker answered its job token request",
            )),
            Err(_) => Err(ApiError::new(
                ErrorKind::Timeout,
                format!(
                    "the worker did not answer the job token request within {}s",
                    ANSWER_TIMEOUT.as_secs()
                ),
            )),
        }
    }
}

#[cfg(test)]
#[path = "job_tokens_tests.rs"]
mod tests;
