use super::WorkerHost;
use crate::{JobDecision, JobRequest};
use lk_agents_abi::Spawner;
use lk_agents_net::{
    signal,
    task::{ScopedTask, TaskSet},
};
use std::sync::Arc;
use std::task::Poll;

/// A pending decision owns both its callback and the result it must consume. Removing the
/// pending state cancels the callback; task death resolves its result as a rejection.
pub(super) struct Decision {
    task: ScopedTask,
    result: lk_agents_abi::BoxFuture<'static, JobDecision>,
}

impl std::fmt::Debug for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decision").field("task", &self.task).finish_non_exhaustive()
    }
}

impl Decision {
    pub(super) fn spawn(
        host: Arc<dyn WorkerHost>,
        spawner: &dyn Spawner,
        tasks: &Arc<TaskSet>,
        request: JobRequest,
    ) -> Self {
        let (producer, decision) = signal::once();
        let task = tasks.scoped(spawner, "job decision", async move {
            producer.resolve(host.on_job_request(request).await);
        });
        Self {
            task,
            result: Box::pin(async move { decision.wait().await.unwrap_or(JobDecision::Reject) }),
        }
    }
    pub(super) fn poll(&mut self, cx: &mut std::task::Context<'_>) -> Poll<JobDecision> {
        self.result.as_mut().poll(cx)
    }
}
