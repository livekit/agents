//! Job token requests on the worker connection, for token mode.
//!
//! A job asks for a job token. The worker sends a `CreateJobTokenRequest` to the server and
//! matches the answer to the request by its request id. The worker sends a request only while it
//! is registered. When the connection closes, every open request fails.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use livekit_protocol as proto;
use lk_agents_net::job_token::Issued;
use tokio::sync::oneshot;

use super::jobs::ConnState;
use super::{JobId, Outbox, Worker, WorkerClient, WorkerCommand};

/// How long a job token request waits for the server's answer.
pub const JOB_TOKEN_TIMEOUT: Duration = Duration::from_secs(5);

/// The lowest `ServerInfo.agent_protocol` that supports `CreateJobTokenRequest`. An older server
/// closes the worker connection when it gets one.
pub const JOB_TOKEN_AGENT_PROTOCOL: i32 = 2;

/// The reason that a job got no job token.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct JobTokenError(pub String);

/// Asks the server for a job token, for a job that runs on this worker. The claims are JSON.
pub type JobTokenIssuer = Arc<
    dyn Fn(JobId, String) -> lk_agents_abi::BoxFuture<'static, Result<Issued, JobTokenError>>
        + Send
        + Sync,
>;

type Answer = oneshot::Sender<Result<Issued, JobTokenError>>;

/// The requests that wait for an answer from the server, by request id. The client adds a request
/// and the worker loop answers it.
#[derive(Default)]
pub(crate) struct JobTokenRequests(parking_lot::Mutex<BTreeMap<String, Answer>>);

impl JobTokenRequests {
    fn answer(&self, request_id: &str, result: Result<Issued, JobTokenError>) {
        let answer = self.0.lock().remove(request_id);
        if let Some(answer) = answer {
            // The asker stopped waiting (timeout or drop), so nobody reads the answer.
            let _ = answer.send(result);
        }
    }

    /// Fails every open request. Its answer cannot come, because the connection is gone.
    pub(crate) fn fail_all(&self, reason: &str) {
        let pending = std::mem::take(&mut *self.0.lock());
        for (_, answer) in pending {
            let _ = answer.send(Err(JobTokenError(reason.into())));
        }
    }
}

/// Removes the request when the asker stops waiting, for any reason.
struct Waiting<'a>(&'a JobTokenRequests, &'a str);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.0.lock().remove(self.1);
    }
}

impl WorkerClient {
    /// Asks the server for a job token for the job `job_id`, which runs on this worker.
    ///
    /// `claims` is JSON with the same fields as the payload of an access token, such as
    /// `identity` and `video`. The server decides which grants it allows and how long the token
    /// lives.
    pub async fn create_job_token(
        &self,
        job_id: &str,
        claims: &str,
    ) -> Result<Issued, JobTokenError> {
        let request_id = lk_agents_net::strings::shortuuid("JT_");
        let (answer, answered) = oneshot::channel();
        self.job_tokens.0.lock().insert(request_id.clone(), answer);
        let _waiting = Waiting(&self.job_tokens, &request_id);
        let command = WorkerCommand::CreateJobToken {
            request_id: request_id.clone(),
            job_id: job_id.to_owned(),
            claims: claims.to_owned(),
        };
        if self.send(command).is_err() {
            return Err(JobTokenError("worker stopped or command queue full".into()));
        }
        match lk_agents_abi::timeout(&*self.clock, JOB_TOKEN_TIMEOUT, answered).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(JobTokenError(
                "the worker stopped before the LiveKit server answered the job token request"
                    .into(),
            )),
            Err(_) => Err(JobTokenError(format!(
                "the LiveKit server did not answer the job token request within {}s",
                JOB_TOKEN_TIMEOUT.as_secs()
            ))),
        }
    }
}

impl Worker {
    /// Stops the worker in token mode when the server cannot issue job tokens.
    ///
    /// The worker does not retry, because the server gives the same answer to each registration.
    /// Key mode does not need job tokens, so it skips this check.
    pub(super) fn check_token_mode_server(
        &self,
        response: &proto::RegisterWorkerResponse,
    ) -> Result<(), super::RuntimeError> {
        let config = &self.runtime_cfg;
        let token = config.token.lock().clone();
        let token_mode =
            lk_agents_net::job_token::token_mode(&config.api_key, &config.api_secret, &token);
        let agent_protocol = response.server_info.as_ref().map_or(0, |info| info.agent_protocol);
        if token_mode && agent_protocol < JOB_TOKEN_AGENT_PROTOCOL {
            return Err(super::RuntimeError::Configuration(format!(
                "the LiveKit server does not support token mode (LIVEKIT_TOKEN): it reports agent \
                 protocol {agent_protocol}, token mode needs {JOB_TOKEN_AGENT_PROTOCOL}. Upgrade \
                 the server, or set LIVEKIT_API_KEY and LIVEKIT_API_SECRET in place of \
                 LIVEKIT_TOKEN"
            )));
        }
        Ok(())
    }

    /// Sends a job token request to the server. When the worker is not registered, the request
    /// fails at once.
    pub(super) fn request_job_token(
        &mut self,
        request_id: String,
        job_id: JobId,
        claims: String,
        outbox: &mut Outbox,
    ) {
        if self.conn != ConnState::Registered {
            let error = "cannot request a job token: the worker is not connected to the LiveKit \
                         server";
            self.job_tokens.answer(&request_id, Err(JobTokenError(error.into())));
            return;
        }
        let request = proto::CreateJobTokenRequest { request_id, job_id, claims };
        let message = proto::WorkerMessage {
            message: Some(proto::worker_message::Message::CreateJobToken(request)),
        };
        self.queue(message, outbox);
    }

    /// Handles the server's answer to a job token request.
    pub(super) fn on_job_token(&mut self, response: proto::CreateJobTokenResponse) {
        let result = if response.error.is_empty() {
            Ok(Issued { token: response.token, expires_at: response.expires_at })
        } else {
            Err(JobTokenError(format!(
                "the LiveKit server refused the job token: {}",
                response.error
            )))
        };
        self.job_tokens.answer(&response.request_id, result);
    }

    /// Adds the kept jobs of a reload to this worker.
    ///
    /// The worker tells the server about them with a `MigrateJobRequest`. When the worker is
    /// registered, it sends the request now. Otherwise the request goes out with the next
    /// registration. In token mode this happens before the worker asks for the jobs' new join
    /// tokens.
    pub(super) fn adopt_jobs(&mut self, jobs: Vec<crate::RunningJobInfo>, outbox: &mut Outbox) {
        let jobs: Vec<_> = jobs
            .into_iter()
            .filter(|info| !self.active.iter().any(|active| active.job.id == info.job.id))
            .collect();
        if jobs.is_empty() {
            return;
        }
        if self.conn == ConnState::Registered {
            self.queue(super::jobs::migrate_jobs(&jobs), outbox);
        }
        for info in jobs {
            self.dispositions.insert(info.job.id.clone(), proto::JobStatus::JsRunning);
            self.active.push(info);
        }
    }
}

#[cfg(test)]
#[path = "job_tokens_tests.rs"]
mod tests;
