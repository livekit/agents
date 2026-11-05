//! Executes the core's toolset lifetime effects. Ready resources retain no host callbacks.

use crate::host::Cancellation;
use crate::telemetry;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::toolsets::{self, Acquisition, Lease, ToolsetHooks};
use lk_agents_abi::{BoxFuture, OwnedTask, SpawnerExt};
use lk_agents_core::toolsets::{Lifetime as Policy, ToolsetEffect};
use lk_agents_net::env::ProcessEnv;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::watch;

type Outcome = Result<(), HostError>;
type Reply = watch::Sender<Option<Outcome>>;
/// The running operation: its epoch, its task and its cancellation.
type Running = (u64, OwnedTask, Arc<Cancellation>);

/// What a launched operation does to the toolset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    Setup,
    Close,
}

struct TicketData {
    id: u64,
    hooks: Arc<dyn ToolsetHooks>,
    ready: Reply,
    released: Reply,
    releasing: AtomicBool,
}
struct State {
    policy: Policy,
    next: u64,
    pending: BTreeMap<u64, Arc<TicketData>>,
    releases: BTreeMap<u64, Reply>,
    epoch: u64,
    operation: Option<(u64, OwnedTask, Arc<Cancellation>)>,
}
struct Inner {
    env: ProcessEnv,
    state: Mutex<State>,
}

pub struct Lifetime(Arc<Inner>);
impl toolsets::ToolsetLifetime for Lifetime {
    type Env = ProcessEnv;
    fn new(env: &ProcessEnv) -> Self {
        Self(Arc::new(Inner {
            env: env.clone(),
            state: Mutex::new(State {
                policy: Policy::default(),
                next: 0,
                pending: BTreeMap::new(),
                releases: BTreeMap::new(),
                epoch: 0,
                operation: None,
            }),
        }))
    }
}
impl toolsets::Lifetime for Lifetime {
    fn acquire(&self, hooks: Arc<dyn ToolsetHooks>) -> Arc<dyn Acquisition> {
        let (data, effects) = {
            let mut state = self.0.state.lock();
            state.next += 1;
            let id = state.next;
            let data = Arc::new(TicketData {
                id,
                hooks,
                ready: watch::channel(None).0,
                released: watch::channel(None).0,
                releasing: AtomicBool::new(false),
            });
            state.pending.insert(id, data.clone());
            let effects = state.policy.acquire(id);
            (data, effects)
        };
        self.0.apply(effects, None, data.hooks.clone());
        Arc::new(Ticket { inner: self.0.clone(), data, claimed: AtomicBool::new(false) })
    }
}

struct Ticket {
    inner: Arc<Inner>,
    data: Arc<TicketData>,
    claimed: AtomicBool,
}
impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.claimed.load(Ordering::SeqCst) {
            self.inner.release(&self.data);
        }
    }
}
impl Acquisition for Ticket {
    fn result(&self) -> BoxFuture<'_, Result<Arc<dyn Lease>, HostError>> {
        Box::pin(async move {
            wait(&self.data.ready).await?;
            if self.data.releasing.load(Ordering::SeqCst) {
                return Err(cancelled());
            }
            if self.claimed.swap(true, Ordering::SeqCst) {
                return Err(HostError::new("toolset acquisition already consumed"));
            }
            Ok(Arc::new(ResourceLease { inner: self.inner.clone(), data: self.data.clone() })
                as Arc<dyn Lease>)
        })
    }
    fn cancel(&self) -> BoxFuture<'_, Outcome> {
        self.inner.release(&self.data);
        Box::pin(wait(&self.data.released))
    }
}
struct ResourceLease {
    inner: Arc<Inner>,
    data: Arc<TicketData>,
}
impl Drop for ResourceLease {
    fn drop(&mut self) {
        self.inner.release(&self.data);
    }
}
impl Lease for ResourceLease {
    fn close(&self) -> BoxFuture<'_, Outcome> {
        self.inner.release(&self.data);
        Box::pin(wait(&self.data.released))
    }
}

/// Closes the successful framework-owned children of a failed setup, never the failed parent's
/// authored close callback. This also settles child cleanup when setup was cancelled.
async fn roll_back_children(hooks: &dyn ToolsetHooks) {
    match hooks.children() {
        Ok(Some(children)) => {
            if let Err(error) = children.close().await {
                log::warn!("toolset setup rollback: {error}");
            }
        }
        Ok(None) => {}
        Err(error) => log::warn!("toolset child lookup during rollback: {error}"),
    }
}

fn cancelled() -> HostError {
    HostError::new("toolset initialization was cancelled")
}
async fn wait(reply: &Reply) -> Outcome {
    let mut events = reply.subscribe();
    loop {
        if let Some(result) = events.borrow_and_update().clone() {
            return result;
        }
        if events.changed().await.is_err() {
            return Err(cancelled());
        }
    }
}

impl Inner {
    fn release(self: &Arc<Self>, data: &Arc<TicketData>) {
        if data.releasing.swap(true, Ordering::SeqCst) {
            return;
        }
        let effects = {
            let mut state = self.state.lock();
            state.releases.insert(data.id, data.released.clone());
            state.policy.release(data.id)
        };
        self.apply(effects, None, data.hooks.clone());
    }

    fn apply(
        self: &Arc<Self>,
        effects: Vec<ToolsetEffect>,
        error: Option<HostError>,
        hooks: Arc<dyn ToolsetHooks>,
    ) {
        for effect in effects {
            match effect {
                ToolsetEffect::Setup(id) => self.launch_setup(id, &hooks),
                ToolsetEffect::CancelSetup => self.cancel_setup(),
                ToolsetEffect::Grant(id) => self.answer_ticket(id, Ok(())),
                ToolsetEffect::Fail(id) => {
                    self.answer_ticket(id, Err(error.clone().unwrap_or_else(cancelled)))
                }
                ToolsetEffect::Released(id) => self.answer_release(id, &error),
                ToolsetEffect::Close => self.launch(hooks.clone(), Operation::Close),
            }
        }
    }

    /// Sets the toolset up with the waiting ticket's hooks, else with the caller's.
    fn launch_setup(self: &Arc<Self>, id: u64, hooks: &Arc<dyn ToolsetHooks>) {
        let data = self.state.lock().pending.get(&id).cloned();
        self.launch(
            data.map_or_else(|| hooks.clone(), |data| data.hooks.clone()),
            Operation::Setup,
        );
    }

    /// Cancels the running setup when the policy is cancelling it.
    fn cancel_setup(&self) {
        let cancel = {
            let state = self.state.lock();
            state
                .policy
                .cancelling_setup()
                .then(|| state.operation.as_ref().map(|(_, _, cancel)| cancel.clone()))
                .flatten()
        };
        if let Some(cancel) = cancel {
            cancel.cancel();
        }
    }

    /// Answers a waiting ticket's acquisition.
    fn answer_ticket(&self, id: u64, outcome: Outcome) {
        let data = self.state.lock().pending.remove(&id);
        if let Some(data) = data {
            data.ready.send_replace(Some(outcome));
        }
    }

    /// Answers a release; a ticket still waiting is cancelled.
    fn answer_release(&self, id: u64, error: &Option<HostError>) {
        let (reply, pending) = {
            let mut state = self.state.lock();
            (state.releases.remove(&id), state.pending.remove(&id))
        };
        if let Some(pending) = pending {
            pending.ready.send_replace(Some(Err(cancelled())));
        }
        if let Some(reply) = reply {
            reply.send_replace(Some(error.clone().map_or(Ok(()), Err)));
        }
    }

    fn launch(self: &Arc<Self>, hooks: Arc<dyn ToolsetHooks>, operation: Operation) {
        let epoch = {
            let mut state = self.state.lock();
            state.epoch += 1;
            state.epoch
        };
        let spawner = match (self.env.spawner)() {
            Ok(spawner) => spawner,
            Err(error) => {
                self.completed(epoch, operation, Err(HostError::new(error)), hooks);
                return;
            }
        };
        let result = Arc::new(Mutex::new(None));
        let (work_result, work_hooks) = (result.clone(), hooks.clone());
        let inner = self.clone();
        let cancel = Arc::new(Cancellation::default());
        let work_cancel = cancel.clone();
        let task = spawner.spawn_settled(
            async move {
                let outcome = match operation {
                    Operation::Setup => work_hooks.setup(&telemetry::Cx::NONE, work_cancel).await,
                    Operation::Close => work_hooks.close(&telemetry::Cx::NONE).await,
                };
                *work_result.lock() = Some(outcome);
            },
            move |_| async move {
                let outcome = result.lock().take().unwrap_or_else(|| Err(cancelled()));
                if operation == Operation::Setup && outcome.is_err() {
                    roll_back_children(hooks.as_ref()).await;
                }
                inner.completed(epoch, operation, outcome, hooks);
            },
        );
        let (retired, cancelling) = self.adopt_operation(epoch, task, &cancel, operation);
        if cancelling {
            cancel.cancel();
        }
        drop(retired);
    }

    /// Stores the operation's task unless a later one replaced it; returns what it retires (to
    /// drop outside the lock) and whether a setup must be cancelled at once.
    fn adopt_operation(
        &self,
        epoch: u64,
        task: OwnedTask,
        cancel: &Arc<Cancellation>,
        operation: Operation,
    ) -> (Option<Running>, bool) {
        let mut state = self.state.lock();
        if state.epoch != epoch {
            return (Some((epoch, task, cancel.clone())), false);
        }
        let cancelling = operation == Operation::Setup && state.policy.cancelling_setup();
        (state.operation.replace((epoch, task, cancel.clone())), cancelling)
    }

    fn completed(
        self: &Arc<Self>,
        epoch: u64,
        operation: Operation,
        outcome: Outcome,
        hooks: Arc<dyn ToolsetHooks>,
    ) {
        let (effects, retired) = {
            let mut state = self.state.lock();
            if state.epoch != epoch {
                return;
            }
            let retired = state.operation.take();
            // Invalidate a task completing before its handle was stored.
            state.epoch += 1;
            let effects = match operation {
                Operation::Setup => state.policy.opened(outcome.is_ok()),
                Operation::Close => state.policy.closed(),
            };
            (effects, retired)
        };
        drop(retired);
        self.apply(effects, outcome.err(), hooks);
    }
}
