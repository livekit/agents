//! `FallbackChain`: the availability + background-recovery machine the LLM, STT and TTS adapters
//! share. Scanned from index 0 on every request (`llm:274`), so a recovered primary takes the next
//! one; probes re-issue the real request (`llm:174-200`, `stt:201-235`, `tts:504-531`).

use std::sync::Arc;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::Mutex;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use super::machine::{ChainEffect, ChainEvent, ChainRequest, FallbackHealth, Modality};
use async_trait::async_trait;
use lk_agents_abi::AbortHandle;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::Clock;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiErrorKind;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use tokio::sync::{oneshot, watch};

use crate::fallback::policy::{ChainPolicy, conn_options_for_attempt};

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    /// Clonable because recovery replays it.
    type Req: Clone + Send + Sync + 'static;
    type Session: Send + 'static;

    fn label(&self) -> String;

    fn modality(&self) -> Modality {
        Modality::Llm
    }

    /// `conn` was rewritten by [`conn_options_for_attempt`](crate::fallback::conn_options_for_attempt); pass it verbatim.
    async fn open(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError>;

    /// `Ok(Recovery::NotYet)` is a clean decline (`stt/fallback_adapter.py:465-466`), not an error.
    /// `stop` turns true when the adapter closes: a probe then closes what it opened and returns
    /// (the reference cancels the probe task, whose `async with stream` closes the stream).
    async fn recover(
        &self,
        req: &Self::Req,
        conn: &ApiConnectOptions,
        slot: RecoverySlot,
        stop: watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    Recovered,
    NotYet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoverySlot {
    OneShot = 0,
    Streaming = 1,
}

/// Policy and acceptance time of one operation, shared by its serving and recovery work.
pub(crate) struct ChainCall {
    pub conn_options: ApiConnectOptions,
    pub accepted_at: Ts,
    pub slot: RecoverySlot,
}

pub enum Attempt<T> {
    Served(T),
    Failed {
        error: ProviderError,
        /// Whether anything reached the consumer before the failure; prevents replay.
        produced_output: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOutcome<T> {
    Served(T),
}

pub use lk_agents_abi::llm::fallback::{AvailabilityChanged, ChainObserver};

pub struct IgnoreEvents;
impl ChainObserver for IgnoreEvents {
    fn availability_changed(&self, _ev: AvailabilityChanged) {}
}

#[derive(Debug)]
struct Status {
    resource: ResourceId,
    closed: bool,
    recovery: [Option<Probe>; 2],
    active: Vec<Probe>,
}

struct ChainState {
    health: FallbackHealth,
    backends: Vec<Status>,
}

/// A recovery probe in flight. `done` resolves when its task ends, finished or dropped.
#[derive(Debug)]
struct Probe {
    handle: AbortHandle,
    done: oneshot::Receiver<()>,
}

/// `ValueError("at least one … instance must be provided.")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoBackends;

impl std::fmt::Display for NoBackends {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("at least one backend instance must be provided")
    }
}
impl std::error::Error for NoBackends {}

pub struct FallbackChain<B: Backend> {
    entries: Vec<Arc<B>>,
    labels: Vec<String>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    status: Arc<Mutex<ChainState>>,
    policy: ChainPolicy,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    observer: Arc<dyn ChainObserver>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `_active_instance`, labelling metrics & traces. Starts at 0.
    active: AtomicUsize,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `_attempt_id`, numbering every attempt.
    attempts: AtomicU64,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `tts/fallback_adapter.py:102,155-159`. Set before the cancel sweep so an in-flight stream
    /// cannot start a probe nothing will cancel.
    closed: AtomicBool,
    /// Every probe's `stop`; see [`Backend::recover`].
    stop: watch::Sender<bool>,
}

impl<B: Backend> FallbackChain<B> {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(
        entries: Vec<Arc<B>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, NoBackends> {
        if entries.is_empty() {
            return Err(NoBackends);
        }
        let labels = entries.iter().map(|b| b.label()).collect();
        let status: Vec<Status> = entries
            .iter()
            .enumerate()
            .map(|(index, _)| Status {
                resource: ResourceId(index as u64),
                closed: false,
                recovery: [None, None],
                active: Vec::new(),
            })
            .collect();
        Ok(FallbackChain {
            entries,
            labels,
            status: Arc::new(Mutex::new(ChainState {
                health: FallbackHealth::default(),
                backends: status,
            })),
            policy,
            clock,
            spawner,
            observer,
            active: AtomicUsize::new(0),
            attempts: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            stop: watch::channel(false).0,
        })
    }

    /// Construct from plugin resources already registered by the runtime.
    pub fn registered(
        entries: Vec<(ResourceId, Arc<B>)>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, NoBackends> {
        let (resources, entries): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        let chain = Self::new(entries, policy, clock, spawner, observer)?;
        for (backend, resource) in
            chain.status.lock().expect("chain status").backends.iter_mut().zip(resources)
        {
            backend.resource = resource;
        }
        Ok(chain)
    }

    pub(crate) fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
    }

    pub(crate) fn teardown_deadline(&self) -> lk_agents_abi::Ts {
        super::channel::teardown_deadline(self.clock.as_ref())
    }

    pub fn entries(&self) -> &[Arc<B>] {
        &self.entries
    }

    pub fn active_index(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    pub fn active(&self) -> &Arc<B> {
        &self.entries[self.active_index()]
    }

    /// Call when the first item reaches the consumer (`llm:200-205`, `tts:197-201`), so a stream
    /// that emits then dies is still attributed.
    pub fn mark_serving(&self, index: usize) {
        self.active.store(index, Ordering::Relaxed);
    }

    pub fn is_available(&self, index: usize) -> bool {
        let state = self.status.lock().expect("chain status");
        state.backends.get(index).is_some_and(|backend| state.health.is_available(backend.resource))
    }

    fn order(&self) -> Vec<usize> {
        let state = self.status.lock().expect("chain status");
        let resources: Vec<_> = state.backends.iter().map(|backend| backend.resource).collect();
        state.health.order(&resources, self.policy.sticky)
    }

    fn select(&self, index: usize, selection_id: u64) {
        let mut state = self.status.lock().expect("chain status");
        if let Some(backend) = state.backends.get(index) {
            let resource = backend.resource;
            state.health.select(resource, selection_id);
        }
    }

    pub async fn run<S, T, F>(
        &self,
        req: &B::Req,
        caller_conn: &ApiConnectOptions,
        slot: RecoverySlot,
        state: &mut S,
        attempt: F,
    ) -> Result<ChainOutcome<T>, ProviderError>
    where
        S: Clone + Send + 'static,
        T: Send + 'static,
        F: for<'a> Fn(usize, &'a B, B::Session, &'a mut S) -> BoxFuture<'a, Attempt<T>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        self.run_cancellable(
            req,
            ChainCall { conn_options: *caller_conn, accepted_at: self.clock.now(), slot },
            state,
            self.stop.subscribe(),
            move |index, backend, session, state, _stop| attempt(index, backend, session, state),
        )
        .await
    }

    /// The consumer's cancellation belongs to this request, while chain close stops all requests.
    pub(crate) async fn run_cancellable<S, T, F>(
        &self,
        req: &B::Req,
        call: ChainCall,
        state: &mut S,
        mut cancel: watch::Receiver<bool>,
        attempt: F,
    ) -> Result<ChainOutcome<T>, ProviderError>
    where
        S: Clone + Send + 'static,
        T: Send + 'static,
        F: for<'a> Fn(
                usize,
                &'a B,
                B::Session,
                &'a mut S,
                watch::Receiver<bool>,
            ) -> BoxFuture<'a, Attempt<T>>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let ChainCall { conn_options, accepted_at: started, slot } = call;
        let caller_conn = &conn_options;
        let deadline = caller_conn
            .total_timeout
            .map(|seconds| started.saturating_add(Duration::from_secs_f64(seconds)));
        if self.closed.load(Ordering::SeqCst) || *cancel.borrow() {
            return Err(ApiError::status("fallback closed", 499, None, None).into());
        }
        if deadline.is_some_and(|at| self.clock.now() >= at) {
            return Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Total).into());
        }
        let all_failed = {
            let status = self.status.lock().expect("chain status");
            status.backends.iter().all(|entry| !status.health.is_available(entry.resource))
        };
        if all_failed {
            self.observer.all_unavailable();
        }
        let entries = self
            .order()
            .into_iter()
            .map(|index| {
                (
                    index,
                    self.status.lock().expect("chain status").backends[index].resource,
                    self.labels[index].clone(),
                )
            })
            .collect();
        let mut request = ChainRequest::new(
            self.entries[0].modality(),
            entries,
            ChainPolicy {
                retry_interval: Some(
                    self.policy.retry_interval.unwrap_or(caller_conn.retry_interval),
                ),
                ..self.policy
            },
            deadline,
            &self.status.lock().expect("chain status").health,
        );
        let mut effects = request.handle(ChainEvent::Wake, started);
        let recover_conn = conn_options_for_attempt(caller_conn, &self.policy);
        for index in self.order() {
            if !self.is_available(index) {
                self.try_recovery(index, slot, req, &recover_conn, deadline);
            }
        }
        loop {
            let Some(effect) = effects.pop() else {
                effects = request.handle(ChainEvent::Wake, self.clock.now());
                continue;
            };
            match effect {
                ChainEffect::Finished(result) => {
                    return result.map(|()| unreachable!("success returns its value"));
                }
                ChainEffect::WakeAt(at) => {
                    let mut closed = self.stop.subscribe();
                    let clock = self.clock.clone();
                    let wake = wait_until(clock.as_ref(), at, &mut cancel, &mut closed).await;
                    effects = request.handle(
                        if wake { ChainEvent::Wake } else { ChainEvent::Cancel },
                        self.clock.now(),
                    );
                }
                ChainEffect::LlmRequestStart { attempt: id, index, .. }
                | ChainEffect::SttStreamOpen { attempt: id, index, .. }
                | ChainEffect::TtsSynthesisStart { attempt: id, index, .. } => {
                    if self.closed.load(Ordering::SeqCst) || *cancel.borrow() {
                        effects = request.handle(ChainEvent::Cancel, self.clock.now());
                        continue;
                    }
                    // Recheck supplied time immediately before executing an authorized launch.
                    if deadline.is_some_and(|at| self.clock.now() >= at) {
                        effects = request.handle(ChainEvent::Wake, self.clock.now());
                        continue;
                    }
                    let attempt_id = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
                    self.select(index, attempt_id);
                    let mut conn = conn_options_for_attempt(caller_conn, &self.policy);
                    let remaining = deadline
                        .map(|at| at.saturating_duration_since(self.clock.now()).as_secs_f64());
                    conn.total_timeout = match (conn.total_timeout, remaining) {
                        (Some(attempt), Some(total)) => Some(attempt.min(total)),
                        (attempt, total) => attempt.or(total),
                    };
                    let backend = Arc::clone(&self.entries[index]);
                    let request_input = req.clone();
                    let mut attempt_state = state.clone();
                    let attempt_fn = attempt.clone();
                    let (stop_tx, stop_rx) = watch::channel(false);
                    let mut closed = self.stop.subscribe();
                    let clock = self.clock.clone();
                    let attempt_deadline = deadline;
                    let (result_tx, result_rx) = oneshot::channel();
                    let (done_tx, done) = oneshot::channel();
                    let handle = self.spawner.spawn_unboxed(async move {
                        let _done = done_tx;
                        let outcome = match backend.open(&request_input, &conn).await {
                            Ok(session) => {
                                attempt_fn(
                                    index,
                                    backend.as_ref(),
                                    session,
                                    &mut attempt_state,
                                    stop_rx,
                                )
                                .await
                            }
                            Err(error) => Attempt::Failed { error, produced_output: false },
                        };
                        let _ = result_tx.send((outcome, attempt_state));
                    });
                    {
                        let mut statuses = self.status.lock().expect("chain status");
                        let active = &mut statuses.backends[index].active;
                        active.retain(|probe| !probe.handle.is_finished());
                        active.push(Probe { handle, done });
                    }
                    let (outcome, updated_state) = execute_attempt(
                        result_rx,
                        clock.as_ref(),
                        attempt_deadline,
                        &mut cancel,
                        &mut closed,
                        stop_tx,
                    )
                    .await;
                    if let Some(updated_state) = updated_state {
                        *state = updated_state;
                    }
                    match outcome {
                        Attempt::Served(value) => {
                            let event = if self.closed.load(Ordering::SeqCst) || *cancel.borrow() {
                                ChainEvent::Cancel
                            } else {
                                ChainEvent::Succeeded { attempt: id }
                            };
                            let terminal = request.handle(event, self.clock.now());
                            if let Some(ChainEffect::Finished(Err(error))) = terminal
                                .into_iter()
                                .find(|effect| matches!(effect, ChainEffect::Finished(_)))
                            {
                                return Err(error);
                            }
                            self.mark_serving(index);
                            self.select(index, attempt_id);
                            self.set_available(index, true);
                            return Ok(ChainOutcome::Served(value));
                        }
                        Attempt::Failed { error, produced_output } => {
                            if error.api_kind() != Some(ApiErrorKind::Cancelled)
                                && !(error.api_kind() == Some(ApiErrorKind::Timeout)
                                    && deadline.is_some_and(|at| self.clock.now() >= at))
                            {
                                self.select(index, 0);
                                self.set_available(index, false);
                            }
                            effects = request.handle(
                                ChainEvent::Failed { attempt: id, error, produced_output },
                                self.clock.now(),
                            );
                            if !produced_output
                                && !self.closed.load(Ordering::SeqCst)
                                && !*cancel.borrow()
                            {
                                self.try_recovery(index, slot, req, &recover_conn, deadline);
                            }
                        }
                    }
                }
                ChainEffect::Cancel { .. } => {}
            }
        }
    }

    /// Unless a live probe is already on `slot`.
    pub fn try_recovery(
        &self,
        index: usize,
        slot: RecoverySlot,
        req: &B::Req,
        conn: &ApiConnectOptions,
        deadline: Option<Ts>,
    ) {
        if self.closed.load(Ordering::SeqCst) || deadline.is_some_and(|at| self.clock.now() >= at) {
            // `tts:218-220,499-502`. Nothing would cancel a probe started from here.
            return;
        }

        let generation = {
            let s = self.status.lock().expect("chain status");
            let Some(status) = s.backends.get(index) else { return };
            if status.closed {
                return;
            }
            if status.recovery[slot as usize].as_ref().is_some_and(|p| !p.handle.is_finished()) {
                return;
            }
            s.health.probe_generation(status.resource)
        };

        let backend = Arc::clone(&self.entries[index]);
        let label = self.labels[index].clone();
        let req = req.clone();
        let mut conn = *conn;
        let observer = Arc::clone(&self.observer);
        let status_for_task = Arc::clone(&self.status);
        let mut closed = self.stop.subscribe();
        let clock = self.clock.clone();
        let (done_tx, done) = oneshot::channel::<()>();

        let handle = self.spawner.spawn_unboxed(async move {
            // Dropped with the task, however it ends.
            let _done = done_tx;
            if *closed.borrow() || deadline.is_some_and(|at| clock.now() >= at) {
                return;
            }
            conn.total_timeout = deadline.map(|at| {
                at.saturating_duration_since(clock.now()).as_secs_f64()
            });
            let (stop, stopped) = watch::channel(false);
            let recovery = backend.recover(&req, &conn, slot, stopped);
            let limit = async {
                match deadline {
                    Some(at) => clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            };
            use futures_util::future::{Either, select};
            let interruption = select(Box::pin(limit), Box::pin(closed.changed()));
            let outcome = match select(Box::pin(recovery), Box::pin(interruption)).await {
                Either::Left((outcome, _)) => outcome,
                Either::Right((_, mut recovery)) => {
                    let _ = stop.send(true);
                    let cleanup = lk_agents_abi::Closed::bounded(
                        clock.as_ref(), super::channel::teardown_deadline(clock.as_ref()),
                        async { let _ = (&mut recovery).await; },
                    ).await;
                    if cleanup == lk_agents_abi::Closed::TimedOut {
                        log::error!("fallback probe cleanup incomplete; task retains ownership until settlement");
                        let _ = recovery.await;
                    }
                    return;
                }
            };
            if *closed.borrow() || deadline.is_some_and(|at| clock.now() >= at) {
                return;
            }
            match outcome {
                Ok(Recovery::Recovered) => {
                    let changed = {
                        let mut s = status_for_task.lock().expect("chain status");
                        match s.backends.get(index) {
                            Some(st) if !st.closed => {
                                let resource = st.resource;
                                s.health.recovered(resource, generation)
                            }
                            _ => false,
                        }
                    };
                    if changed {
                        observer.availability_changed(AvailabilityChanged {
                            index,
                            label,
                            available: true,
                        });
                    }
                }
                // Both are "still down"; the next request schedules another probe.
                Ok(Recovery::NotYet) | Err(_) => {}
            }
        });

        let mut s = self.status.lock().expect("chain status");
        let stale = s.backends.get(index).is_some_and(|status| {
            status.closed || s.health.probe_generation(status.resource) != generation
        });
        if let Some(status) = s.backends.get_mut(index) {
            if stale {
                status.active.push(Probe { handle, done });
            } else {
                status.recovery[slot as usize] = Some(Probe { handle, done });
            }
        }
    }

    /// Marks the chain closed, tells every probe to stop, and hands back the ones in flight.
    fn begin_close(&self) -> Vec<Probe> {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.stop.send(true);
        let mut s = self.status.lock().expect("chain status");
        s.backends
            .iter_mut()
            .flat_map(|status| {
                status.closed = true;
                let mut probes: Vec<_> =
                    status.recovery.iter_mut().filter_map(Option::take).collect();
                probes.append(&mut status.active);
                probes
            })
            .collect()
    }

    /// Idempotent: signal all active and recovery work; spawned tasks retain cleanup ownership.
    pub fn aclose(&self) {
        let _ = self.begin_close();
    }

    /// The host's `aclose()` (`cancel_and_wait` on each recovery task): every probe closes what it
    /// opened, awaited up to the teardown deadline. Late cleanup remains owned by its task.
    pub async fn close(&self) {
        let probes = self.begin_close();
        let (handles, dones): (Vec<_>, Vec<_>) =
            probes.into_iter().map(|p| (p.handle, p.done)).unzip();
        let closed = lk_agents_abi::Closed::bounded(
            self.clock.as_ref(),
            self.teardown_deadline(),
            async move {
                for done in dones {
                    let _ = done.await;
                }
            },
        )
        .await;
        if closed == lk_agents_abi::Closed::TimedOut {
            log::error!("fallback cleanup incomplete; tasks retain ownership until settlement");
        }
        drop(handles);
    }

    fn set_available(&self, index: usize, available: bool) {
        let changed = {
            let mut s = self.status.lock().expect("chain status");
            match s.backends.get(index) {
                Some(st) if !st.closed => {
                    let resource = st.resource;
                    s.health.observe(resource, available)
                }
                _ => false,
            }
        };
        // Only on a transition: a second failure of a down provider emits nothing.
        if changed {
            self.observer.availability_changed(AvailabilityChanged {
                index,
                label: self.labels[index].clone(),
                available,
            });
        }
    }
}

async fn wait_until(
    clock: &dyn Clock,
    at: Ts,
    cancel: &mut watch::Receiver<bool>,
    closed: &mut watch::Receiver<bool>,
) -> bool {
    use futures_util::future::{Either, select};
    if *cancel.borrow() || *closed.borrow() {
        return false;
    }
    let stop = select(Box::pin(cancel.changed()), Box::pin(closed.changed()));
    matches!(select(Box::pin(clock.wait_until(at)), Box::pin(stop)).await, Either::Left(_))
}

async fn execute_attempt<T, S>(
    result: oneshot::Receiver<(Attempt<T>, S)>,
    clock: &dyn Clock,
    deadline: Option<Ts>,
    cancel: &mut watch::Receiver<bool>,
    closed: &mut watch::Receiver<bool>,
    stop: watch::Sender<bool>,
) -> (Attempt<T>, Option<S>) {
    use futures_util::future::{Either, select};
    let limit = async {
        match deadline {
            Some(at) => clock.wait_until(at).await,
            None => std::future::pending().await,
        }
    };
    let cancellation = select(Box::pin(cancel.changed()), Box::pin(closed.changed()));
    let interruption = select(Box::pin(limit), Box::pin(cancellation));
    let (error, mut result) = match select(Box::pin(result), Box::pin(interruption)).await {
        Either::Left((Ok((outcome, state)), _)) => return (outcome, Some(state)),
        Either::Left((Err(_), _)) => {
            return (
                Attempt::Failed {
                    error: ApiError::connection(Some("fallback attempt lost its outcome".into()))
                        .into(),
                    produced_output: true,
                },
                None,
            );
        }
        Either::Right((reason, result)) => {
            let error = match reason {
                Either::Left(_) => ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Total),
                Either::Right(_) => ApiError::status("fallback cancelled", 499, None, None),
            };
            (error, result)
        }
    };
    let _ = stop.send(true);
    let mut settled = None;
    let cleanup =
        lk_agents_abi::Closed::bounded(clock, super::channel::teardown_deadline(clock), async {
            settled = (&mut result).await.ok();
        })
        .await;
    let (produced_output, state) = match settled {
        Some((Attempt::Failed { produced_output, .. }, state)) => (produced_output, Some(state)),
        Some((_, state)) => (true, Some(state)),
        None => (true, None),
    };
    if cleanup == lk_agents_abi::Closed::TimedOut {
        log::error!("fallback attempt cleanup incomplete; task retains ownership until settlement");
    }
    (Attempt::Failed { error: error.into(), produced_output }, state)
}
