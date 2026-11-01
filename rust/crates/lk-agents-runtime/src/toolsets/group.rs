//! Executes a scope's native toolset lease plan; all acquired leases settle before failure/close.

use lk_agents_abi::llm::toolsets::{self, Lease, Lifetime, ToolsetHooks};
use lk_agents_abi::{BoxFuture, errors::HostError};
use lk_agents_core::toolsets::Group as Policy;
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::watch;

type Outcome = Result<(), HostError>;
type Reply = watch::Sender<Option<Outcome>>;
type Entry = (Arc<dyn Lifetime>, Arc<dyn ToolsetHooks>);

pub struct Group(Arc<Inner>);
struct Inner {
    env: ProcessEnv,
    entries: Mutex<Option<Vec<Entry>>>,
    launched: AtomicBool,
    closing: watch::Sender<bool>,
    started: Reply,
    closed: Reply,
    tasks: Arc<TaskSet>,
}
impl Group {
    pub(super) fn from_entries(env: &ProcessEnv, entries: Vec<Entry>) -> Self {
        Self(Arc::new(Inner {
            env: env.clone(),
            entries: Mutex::new(Some(entries)),
            launched: AtomicBool::new(false),
            closing: watch::channel(false).0,
            started: watch::channel(None).0,
            closed: watch::channel(None).0,
            tasks: Arc::default(),
        }))
    }
}
impl Drop for Group {
    fn drop(&mut self) {
        self.0.closing.send_replace(true);
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
            self.inner.closing.send_replace(true);
        }
    }
}
async fn wait(reply: &Reply) -> Outcome {
    let mut events = reply.subscribe();
    loop {
        if let Some(result) = events.borrow_and_update().clone() {
            return result;
        }
        if events.changed().await.is_err() {
            return Err(HostError::new("toolset group ended"));
        }
    }
}
impl Inner {
    fn ensure_driver(self: &Arc<Self>) {
        if self.launched.swap(true, Ordering::SeqCst) {
            return;
        }
        match (self.env.spawner)() {
            Ok(spawner) => {
                let inner = self.clone();
                self.tasks.spawn(
                    &*spawner,
                    "toolset lease group",
                    async move { inner.run().await },
                );
            }
            Err(error) => {
                let retired = self.entries.lock().take();
                drop(retired);
                self.started.send_replace(Some(Err(HostError::new(error))));
                self.closed.send_replace(Some(Ok(())));
            }
        }
    }
    async fn close(self: &Arc<Self>) -> Outcome {
        self.closing.send_replace(true);
        self.ensure_driver();
        wait(&self.closed).await
    }
    async fn run(self: Arc<Self>) {
        let entries = self.entries.lock().take().unwrap_or_default();
        let mut closing = self.closing.subscribe();
        let mut policy = Policy::new(entries.len());
        let mut leases: Vec<Arc<dyn Lease>> = Vec::new();
        let mut failure = None;
        if *closing.borrow_and_update() {
            policy.request_close();
        }
        while let Some(index) = policy.next() {
            let (lifetime, hooks) = &entries[index];
            let ticket = lifetime.acquire(hooks.clone());
            let result = tokio::select! {
                biased;
                _ = closing.changed() => {
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
            if *closing.borrow_and_update() {
                policy.request_close();
            }
        }
        if policy.ready() {
            self.started.send_replace(Some(Ok(())));
            while !*closing.borrow_and_update() {
                if closing.changed().await.is_err() {
                    break;
                }
            }
            policy.request_close();
        }
        // Closing every lease is an independent effect. Do not let one failure abandon siblings.
        let results =
            futures_util::future::join_all(policy.releases().map(|index| leases[index].close()))
                .await;
        let close_error = results.into_iter().find_map(Result::err);
        if self.started.borrow().is_none() {
            self.started
                .send_replace(Some(Err(failure
                    .unwrap_or_else(|| HostError::new("toolset initialization was cancelled")))));
        }
        drop(leases);
        drop(entries);
        self.closed.send_replace(Some(close_error.map_or(Ok(()), Err)));
    }
}
