use lk_agents_abi::llm::toolsets::{self, Lease, Lifetime, ToolsetHooks};
use lk_agents_abi::{BoxFuture, errors::HostError};
use lk_agents_core::toolsets::Group as Policy;
use lk_agents_net::signal;
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// Every acquired lease settles before a failure or close returns.

type Outcome = Result<(), HostError>;
type Reply = signal::Cell<Option<Outcome>>;
type Settle = signal::Producer<Option<Outcome>>;
type Entry = (Arc<dyn Lifetime>, Arc<dyn ToolsetHooks>);

pub struct Group(Arc<Inner>);
struct Inner {
    env: ProcessEnv,
    entries: Mutex<Option<Vec<Entry>>>,
    /// The driver's writers, taken once by whoever launches it: its end ends every wait.
    settle: Mutex<Option<(Settle, Settle)>>,
    closing: CancellationToken,
    started: Reply,
    closed: Reply,
    tasks: Arc<TaskSet>,
}
impl Group {
    pub(super) fn from_entries(env: &ProcessEnv, entries: Vec<Entry>) -> Self {
        let ((start, started), (close, closed)) = (signal::once(), signal::once());
        Self(Arc::new(Inner {
            env: env.clone(),
            entries: Mutex::new(Some(entries)),
            settle: Mutex::new(Some((start, close))),
            closing: CancellationToken::new(),
            started,
            closed,
            tasks: Arc::default(),
        }))
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        self.0.closing.cancel();
    }
}

impl toolsets::ToolsetGroup for Group {
    type Env = ProcessEnv;
    fn new(env: &ProcessEnv, toolsets: Vec<Arc<dyn ToolsetHooks>>) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        for hooks in toolsets {
            let lifetime = hooks.lifetime().clone();
            if !entries.iter().any(|(other, _)| Arc::ptr_eq(other, &lifetime)) {
                entries.push((lifetime, hooks));
            }
        }
        Self::from_entries(env, entries)
    }

    async fn start(&self) -> Outcome {
        self.0.ensure_driver();
        let mut guard = OpeningWait { inner: self.0.clone(), armed: true };
        let result = wait(&self.0.started).await;
        guard.armed = false;
        result
    }
    async fn aclose(&self) -> Outcome {
        self.0.close().await
    }
}
impl toolsets::LeaseGroup for Group {
    fn close(&self) -> BoxFuture<'_, Outcome> {
        Box::pin(self.0.close())
    }
}
struct OpeningWait {
    inner: Arc<Inner>,
    armed: bool,
}
impl Drop for OpeningWait {
    fn drop(&mut self) {
        if self.armed {
            self.inner.closing.cancel();
        }
    }
}
async fn wait(reply: &Reply) -> Outcome {
    reply.wait().await.unwrap_or_else(|_| Err(HostError::new("toolset group ended")))
}
impl Inner {
    fn ensure_driver(self: &Arc<Self>) {
        let Some((started, closed)) = self.settle.lock().take() else { return };
        match (self.env.spawner)() {
            Ok(spawner) => {
                let inner = self.clone();
                self.tasks.spawn(&*spawner, "toolset lease group", async move {
                    inner.run(started, closed).await
                });
            }
            Err(error) => {
                let retired = self.entries.lock().take();
                drop(retired);
                started.resolve(Err(HostError::new(error.to_string())));
                closed.resolve(Ok(()));
            }
        }
    }
    async fn close(self: &Arc<Self>) -> Outcome {
        self.closing.cancel();
        self.ensure_driver();
        wait(&self.closed).await
    }
    async fn run(self: Arc<Self>, started: Settle, closed: Settle) {
        let mut started = Some(started);
        let entries = self.entries.lock().take().unwrap_or_default();
        let mut policy = Policy::new(entries.len());
        let mut leases: Vec<Arc<dyn Lease>> = Vec::new();
        let mut failure = None;
        if self.closing.is_cancelled() {
            policy.request_close();
        }
        while let Some(index) = policy.next() {
            let (lifetime, hooks) = &entries[index];
            let ticket = lifetime.acquire(hooks.clone());
            let result = tokio::select! {
                biased;
                () = self.closing.cancelled() => {
                    policy.request_close();
                    let cancelled = ticket.cancel().await;
                    Err(cancelled.err().unwrap_or_else(|| HostError::new("toolset initialization was cancelled")))
                }
                result = ticket.result() => result,
            };
            policy.acquired(result.is_ok());
            match result {
                Ok(lease) => leases.push(lease),
                Err(error) => {
                    failure = Some(error);
                }
            }
            if self.closing.is_cancelled() {
                policy.request_close();
            }
        }
        if policy.ready() {
            if let Some(started) = started.take() {
                started.resolve(Ok(()));
            }
            self.closing.cancelled().await;
            policy.request_close();
        }
        // Closing every lease is an independent effect. Do not let one failure abandon siblings.
        let results =
            futures_util::future::join_all(policy.releases().map(|index| leases[index].close()))
                .await;
        let close_error = results.into_iter().find_map(Result::err);
        if let Some(started) = started {
            let failure =
                failure.unwrap_or_else(|| HostError::new("toolset initialization was cancelled"));
            started.resolve(Err(failure));
        }
        drop(leases);
        drop(entries);
        closed.resolve(close_error.map_or(Ok(()), Err));
    }
}
