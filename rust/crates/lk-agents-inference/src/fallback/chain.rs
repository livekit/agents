//! `FallbackChain`: the availability + background-recovery machine the LLM, STT and TTS adapters
//! share. Scanned from index 0 on every request (`llm:274`), so a recovered primary takes the next
//! one; probes re-issue the real request (`llm:174-200`, `stt:201-235`, `tts:504-531`).

use crate::effects::FallbackChainEffect;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use super::machine::{ChainEvent, ChainRequest, FallbackPolicy};
use async_trait::async_trait;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::Clock;
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::OwnedTask;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiErrorKind;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use lk_agents_net::task::{TaskId, TaskSet};
use tokio::sync::{oneshot, watch};

use crate::fallback::{ChainPolicy, conn_options_for_attempt};

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    /// Clonable because recovery replays it.
    type Req: Clone + Send + Sync + 'static;
    type Session: Send + 'static;

    fn label(&self) -> String;

    fn resource_id(&self) -> ResourceId;

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
        slot: RecoveryMode,
        stop: watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    Recovered,
    NotYet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryMode {
    OneShot = 0,
    Streaming = 1,
}

/// Policy and acceptance time of one operation, shared by its serving and recovery work.
pub(crate) struct ChainCall {
    pub conn_options: ApiConnectOptions,
    pub accepted_at: Ts,
    pub slot: RecoveryMode,
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
    health: FallbackPolicy,
    backends: Vec<Status>,
}

/// A recovery probe or attempt in flight; it settles when its work ends, finished or dropped.
#[derive(Debug)]
struct Probe {
    task: OwnedTask,
}

/// A request stream's task in its chain's [`TaskSet`].
pub(crate) struct StreamTask {
    streams: Arc<TaskSet>,
    id: TaskId,
}

impl StreamTask {
    #[cfg(test)]
    pub(crate) fn new(streams: Arc<TaskSet>, id: TaskId) -> Self {
        StreamTask { streams, id }
    }

    pub(crate) fn cancel(&self) {
        self.streams.cancel(self.id);
    }
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
    status: Arc<Mutex<ChainState>>,
    policy: ChainPolicy,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    observer: Arc<dyn ChainObserver>,
    /// `_active_instance`, labelling metrics & traces. Starts at 0.
    active: AtomicUsize,
    /// `_attempt_id`, numbering every attempt.
    attempts: AtomicU64,
    /// `tts/fallback_adapter.py:102,155-159`. Set before the cancel sweep so an in-flight stream
    /// cannot start a probe nothing will cancel.
    closed: AtomicBool,
    /// Every probe's `stop`; see [`Backend::recover`].
    stop: watch::Sender<bool>,
    /// The request streams' tasks: a dropped stream only signals its task to stop, which then
    /// closes its member and settles under the chain.
    streams: Arc<TaskSet>,
}

impl<B: Backend> FallbackChain<B> {
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
            .map(|backend| Status {
                resource: backend.resource_id(),
                closed: false,
                recovery: [None, None],
                active: Vec::new(),
            })
            .collect();
        Ok(FallbackChain {
            entries,
            labels,
            status: Arc::new(Mutex::new(ChainState {
                health: FallbackPolicy::default(),
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
            streams: Arc::default(),
        })
    }

    #[track_caller]
    pub(crate) fn spawn_stream(
        &self,
        work: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> StreamTask {
        let id = self.streams.spawn(self.spawner.as_ref(), "fallback stream", work);
        StreamTask { streams: Arc::clone(&self.streams), id }
    }

    pub(crate) fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
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
        let state = self.status.lock();
        state.backends.get(index).is_some_and(|backend| state.health.is_available(backend.resource))
    }

    fn select(&self, index: usize, selection_id: u64) {
        let mut state = self.status.lock();
        if let Some(backend) = state.backends.get(index) {
            let resource = backend.resource;
            state.health.select(resource, selection_id);
        }
    }

    pub async fn run<S, T, F>(
        &self,
        req: &B::Req,
        caller_conn: &ApiConnectOptions,
        slot: RecoveryMode,
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
        cancel: watch::Receiver<bool>,
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
        self.admit(req, call, cancel)?.drive(state, attempt).await
    }

    /// Opens one request: refuses it when stopped or past its deadline, refreshes health, builds
    /// the request's machine and probes every unavailable entry.
    fn admit<'c, T, S>(
        &'c self,
        req: &'c B::Req,
        call: ChainCall,
        cancel: watch::Receiver<bool>,
    ) -> Result<ChainRun<'c, B, T, S>, ProviderError> {
        let ChainCall { conn_options: caller_conn, accepted_at: started, slot } = call;
        let deadline = caller_conn
            .total_timeout
            .map(|seconds| started.saturating_add(Duration::from_secs_f64(seconds)));
        if self.closed.load(Ordering::SeqCst) || *cancel.borrow() {
            return Err(ApiError::status("fallback closed", 499, None, None).into());
        }
        if deadline.is_some_and(|at| self.clock.now() >= at) {
            return Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Total).into());
        }
        if self.refresh_health() {
            self.observer.all_unavailable();
        }
        let (mut request, recovery_order) = self.chain_request(&caller_conn, deadline);
        let effects = request.handle(ChainEvent::Wake, started).into();
        let recover_conn = conn_options_for_attempt(&caller_conn, &self.policy);
        for index in recovery_order {
            if !self.is_available(index) {
                self.try_recovery(index, slot, req, &recover_conn, deadline);
            }
        }
        Ok(ChainRun {
            chain: self,
            req,
            caller_conn,
            recover_conn,
            slot,
            deadline,
            cancel,
            request,
            effects,
            active: None,
            served: None,
        })
    }

    /// Ages every entry's health to now; true when none is available.
    fn refresh_health(&self) -> bool {
        let mut status = self.status.lock();
        let priority: Vec<_> = status.backends.iter().map(|entry| entry.resource).collect();
        status.health.refresh(&priority, self.clock.now());
        status.backends.iter().all(|entry| !status.health.is_available(entry.resource))
    }

    /// The request's machine over the entries as they stand, and the order to probe them in.
    fn chain_request(
        &self,
        caller_conn: &ApiConnectOptions,
        deadline: Option<Ts>,
    ) -> (ChainRequest, Vec<usize>) {
        let status = self.status.lock();
        let priority: Vec<_> = status.backends.iter().map(|entry| entry.resource).collect();
        let entries = status
            .backends
            .iter()
            .enumerate()
            .map(|(index, backend)| (index, backend.resource, self.labels[index].clone()))
            .collect();
        let policy = ChainPolicy {
            retry_interval: Some(self.policy.retry_interval.unwrap_or(caller_conn.retry_interval)),
            ..self.policy
        };
        (
            ChainRequest::new(entries, policy, deadline, &status.health),
            status.health.order(&priority, false),
        )
    }

    /// Holds a launched attempt under its entry until it settles; finished ones are pruned.
    fn track_attempt(&self, index: usize, task: OwnedTask) {
        let mut statuses = self.status.lock();
        let active = &mut statuses.backends[index].active;
        active.retain(|probe| !probe.task.is_finished());
        active.push(Probe { task });
    }

    /// Unless a live probe is already on `slot`.
    pub fn try_recovery(
        &self,
        index: usize,
        slot: RecoveryMode,
        req: &B::Req,
        conn: &ApiConnectOptions,
        deadline: Option<Ts>,
    ) {
        if self.closed.load(Ordering::SeqCst) || deadline.is_some_and(|at| self.clock.now() >= at) {
            // `tts:218-220,499-502`. Nothing would cancel a probe started from here.
            return;
        }
        let Some(generation) = self.free_probe_slot(index, slot) else { return };
        let probe = RecoveryProbe {
            backend: Arc::clone(&self.entries[index]),
            label: self.labels[index].clone(),
            req: req.clone(),
            conn: *conn,
            observer: Arc::clone(&self.observer),
            status: Arc::clone(&self.status),
            closed: self.stop.subscribe(),
            clock: self.clock.clone(),
            deadline,
            slot,
            index,
            generation,
        };
        let task = self.spawner.spawn(probe.run());
        self.register_probe(index, slot, generation, task);
    }

    /// The entry's probe generation, when the entry is open and `slot` has no live probe.
    fn free_probe_slot(&self, index: usize, slot: RecoveryMode) -> Option<u64> {
        let s = self.status.lock();
        let status = s.backends.get(index).filter(|status| !status.closed)?;
        if status.recovery[slot as usize].as_ref().is_some_and(|p| !p.task.is_finished()) {
            return None;
        }
        Some(s.health.probe_generation(status.resource))
    }

    /// A probe whose entry closed or whose generation went stale while it spawned is still owned
    /// by the entry, but no longer holds `slot`.
    fn register_probe(&self, index: usize, slot: RecoveryMode, generation: u64, task: OwnedTask) {
        let mut s = self.status.lock();
        let stale = s.backends.get(index).is_some_and(|status| {
            status.closed || s.health.probe_generation(status.resource) != generation
        });
        if let Some(status) = s.backends.get_mut(index) {
            if stale {
                status.active.push(Probe { task });
            } else {
                status.recovery[slot as usize] = Some(Probe { task });
            }
        }
    }

    /// Marks the chain closed, tells every probe to stop, and hands back the ones in flight.
    fn begin_close(&self) -> Vec<Probe> {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.stop.send(true);
        let mut s = self.status.lock();
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
        let probes = self.begin_close();
        self.retain(probes);
    }

    /// The chain keeps owning probes that were told to stop; they settle on their own.
    fn retain(&self, probes: Vec<Probe>) {
        let mut s = self.status.lock();
        if let Some(status) = s.backends.first_mut() {
            status.active.extend(probes.into_iter().filter(|p| !p.task.is_finished()));
        }
    }

    /// The host's `aclose()` (`cancel_and_wait` on each recovery task): every probe closes what it
    /// opened, awaited up to the teardown deadline. Late cleanup remains owned by its task.
    pub async fn close(&self) {
        let mut probes = self.begin_close();
        let closed = lk_agents_abi::Closed::bounded(
            self.clock.as_ref(),
            self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT),
            async {
                for probe in &mut probes {
                    let _ = (&mut probe.task).await;
                }
            },
        )
        .await;
        if closed == lk_agents_abi::Closed::TimedOut {
            log::error!("fallback cleanup incomplete; tasks retain ownership until settlement");
            self.retain(probes);
        }
    }

    fn observe_result(&self, index: usize, result: Result<(), &ProviderError>) -> bool {
        let available = result.is_ok();
        let changed = {
            let mut s = self.status.lock();
            match s.backends.get(index) {
                Some(st) if !st.closed => {
                    let resource = st.resource;
                    match result {
                        Ok(()) => Some(s.health.observe(resource, true, None)),
                        Err(error) => s.health.failed(resource, error.api_kind(), None),
                    }
                }
                _ => None,
            }
        };
        // Only on a transition: a second failure of a down provider emits nothing.
        if changed == Some(true) {
            self.observer.availability_changed(AvailabilityChanged {
                index,
                label: self.labels[index].clone(),
                available,
            });
        }
        changed.is_some()
    }
}

/// One request's loop: it runs the request machine's effects and settles each attempt it launches.
struct ChainRun<'c, B: Backend, T, S> {
    chain: &'c FallbackChain<B>,
    req: &'c B::Req,
    caller_conn: ApiConnectOptions,
    recover_conn: ApiConnectOptions,
    slot: RecoveryMode,
    deadline: Option<Ts>,
    cancel: watch::Receiver<bool>,
    request: ChainRequest,
    effects: VecDeque<FallbackChainEffect>,
    active: Option<RunningAttempt<T, S>>,
    /// The served value with its entry and selection id, until the machine finishes.
    served: Option<(T, usize, u64)>,
}

impl<B: Backend, T: Send + 'static, S: Clone + Send + 'static> ChainRun<'_, B, T, S> {
    async fn drive<F>(mut self, state: &mut S, attempt: F) -> Result<ChainOutcome<T>, ProviderError>
    where
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
        loop {
            let Some(effect) = self.effects.pop_front() else {
                self.await_active(state).await;
                continue;
            };
            match effect {
                FallbackChainEffect::Finished(result) => return self.finish(result),
                FallbackChainEffect::WakeAt(at) => self.sleep_until(at).await,
                FallbackChainEffect::Start { attempt: id, index, .. } => {
                    self.launch(id, index, state, &attempt)
                }
                FallbackChainEffect::Cancel { attempt: id } => self.cancel(id, state).await,
            }
        }
    }

    fn handle(&mut self, event: ChainEvent) {
        self.effects.extend(self.request.handle(event, self.chain.clock.now()));
    }

    /// Waits for the running attempt to settle or be interrupted; with none running the machine
    /// is woken to decide what comes next.
    async fn await_active(&mut self, state: &mut S) {
        let Some(mut active) = self.active.take() else {
            self.handle(ChainEvent::Wake);
            return;
        };
        let mut closed = self.chain.stop.subscribe();
        let clock = self.chain.clock.as_ref();
        let observation = active.observe(clock, self.deadline, &mut self.cancel, &mut closed).await;
        let event = match observation {
            AttemptObservation::Interrupted(event) => event,
            AttemptObservation::Completed(outcome, updated_state) => {
                if let Some(updated_state) = updated_state {
                    *state = updated_state;
                }
                match outcome {
                    Attempt::Served(value) => self.served(&active, value),
                    Attempt::Failed { error, produced_output } => {
                        self.failed(&active, error, produced_output)
                    }
                }
            }
        };
        let settled = matches!(event, ChainEvent::Succeeded { .. } | ChainEvent::Failed { .. });
        if !settled && self.served.is_none() {
            self.active = Some(active);
        }
        self.handle(event);
    }

    fn served(&mut self, active: &RunningAttempt<T, S>, value: T) -> ChainEvent {
        self.served = Some((value, active.index, active.attempt_id));
        if self.chain.closed.load(Ordering::SeqCst) || *self.cancel.borrow() {
            ChainEvent::Cancel
        } else {
            ChainEvent::Succeeded { attempt: active.id }
        }
    }

    /// A timeout at the request's own deadline is not the entry's fault; any other failure marks
    /// it down and, unless output already reached the consumer, starts its recovery probe.
    fn failed(
        &self,
        active: &RunningAttempt<T, S>,
        error: ProviderError,
        produced_output: bool,
    ) -> ChainEvent {
        let (chain, index) = (self.chain, active.index);
        let failed = !(error.api_kind() == Some(ApiErrorKind::Timeout)
            && self.deadline.is_some_and(|at| chain.clock.now() >= at))
            && chain.observe_result(index, Err(&error));
        if failed {
            chain.select(index, 0);
        }
        if failed
            && !produced_output
            && !chain.closed.load(Ordering::SeqCst)
            && !*self.cancel.borrow()
        {
            chain.try_recovery(index, self.slot, self.req, &self.recover_conn, self.deadline);
        }
        ChainEvent::Failed { attempt: active.id, error, produced_output }
    }

    fn finish(
        &mut self,
        result: Result<(), ProviderError>,
    ) -> Result<ChainOutcome<T>, ProviderError> {
        result?;
        let (value, index, attempt_id) =
            self.served.take().expect("the machine succeeds only after an attempt served");
        self.chain.mark_serving(index);
        self.chain.select(index, attempt_id);
        self.chain.observe_result(index, Ok(()));
        Ok(ChainOutcome::Served(value))
    }

    async fn sleep_until(&mut self, at: Ts) {
        let mut closed = self.chain.stop.subscribe();
        let clock = self.chain.clock.clone();
        let wake = wait_until(clock.as_ref(), at, &mut self.cancel, &mut closed).await;
        self.handle(if wake { ChainEvent::Wake } else { ChainEvent::Cancel });
    }

    /// Spawns entry `index`'s attempt, rechecking the stop and the deadline first: the machine
    /// authorized it at an earlier instant.
    fn launch<F>(&mut self, id: u64, index: usize, state: &S, serve: &F)
    where
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
        let chain = self.chain;
        if chain.closed.load(Ordering::SeqCst) || *self.cancel.borrow() {
            return self.handle(ChainEvent::Cancel);
        }
        // Recheck supplied time immediately before executing an authorized launch.
        if self.deadline.is_some_and(|at| chain.clock.now() >= at) {
            return self.handle(ChainEvent::Wake);
        }
        let attempt_id = chain.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        chain.select(index, attempt_id);
        let (stop, stop_rx) = watch::channel(false);
        let (result_tx, result) = oneshot::channel();
        let launch = AttemptLaunch {
            backend: Arc::clone(&chain.entries[index]),
            req: self.req.clone(),
            conn: conn_options_for_attempt(&self.caller_conn, &chain.policy),
            deadline: self.deadline,
            clock: chain.clock.clone(),
            closed: chain.stop.subscribe(),
            stop: stop_rx,
            state: state.clone(),
            serve: serve.clone(),
            index,
        };
        let task = chain.spawner.spawn(async move {
            let _ = result_tx.send(launch.run().await);
        });
        chain.track_attempt(index, task);
        self.active = Some(RunningAttempt { id, index, attempt_id, result, stop });
    }

    async fn cancel(&mut self, attempt: u64, state: &mut S) {
        let Some(running) = self.active.as_mut().filter(|running| running.id == attempt) else {
            return;
        };
        if let Some(updated_state) = running.cancel(self.chain.clock.as_ref()).await {
            *state = updated_state;
        }
    }
}

/// One attempt's task: it opens its entry and serves the session, handing back its outcome and
/// the state it served with.
struct AttemptLaunch<B: Backend, S, F> {
    backend: Arc<B>,
    req: B::Req,
    conn: ApiConnectOptions,
    deadline: Option<Ts>,
    clock: Arc<dyn Clock>,
    closed: watch::Receiver<bool>,
    stop: watch::Receiver<bool>,
    state: S,
    serve: F,
    index: usize,
}

impl<B: Backend, S, F> AttemptLaunch<B, S, F> {
    async fn run<T>(mut self) -> (Attempt<T>, S)
    where
        F: for<'a> Fn(
            usize,
            &'a B,
            B::Session,
            &'a mut S,
            watch::Receiver<bool>,
        ) -> BoxFuture<'a, Attempt<T>>,
    {
        // Admission can outlive its authorization while the executor is busy.
        let now = self.clock.now();
        if let Some(error) = self.refusal(now) {
            return (Attempt::Failed { error, produced_output: false }, self.state);
        }
        let remaining = self.deadline.map(|at| at.saturating_duration_since(now).as_secs_f64());
        self.conn.total_timeout = match (self.conn.total_timeout, remaining) {
            (Some(attempt), Some(total)) => Some(attempt.min(total)),
            (attempt, total) => attempt.or(total),
        };
        let session = match self.backend.open(&self.req, &self.conn).await {
            Ok(session) => session,
            Err(error) => return (Attempt::Failed { error, produced_output: false }, self.state),
        };
        let outcome =
            (self.serve)(self.index, self.backend.as_ref(), session, &mut self.state, self.stop)
                .await;
        (outcome, self.state)
    }

    /// Why an attempt admitted late must not open: it was stopped, or its deadline passed.
    fn refusal(&self, now: Ts) -> Option<ProviderError> {
        if *self.stop.borrow() || *self.closed.borrow() {
            return Some(ApiError::status("fallback cancelled", 499, None, None).into());
        }
        self.deadline.is_some_and(|at| now >= at).then(|| {
            ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Total)
                .with_retryable(false)
                .into()
        })
    }
}

/// A background recovery probe of one entry: it replays the request and, when the entry
/// answers, marks it available again.
struct RecoveryProbe<B: Backend> {
    backend: Arc<B>,
    label: String,
    req: B::Req,
    conn: ApiConnectOptions,
    observer: Arc<dyn ChainObserver>,
    status: Arc<Mutex<ChainState>>,
    closed: watch::Receiver<bool>,
    clock: Arc<dyn Clock>,
    deadline: Option<Ts>,
    slot: RecoveryMode,
    index: usize,
    generation: u64,
}

impl<B: Backend> RecoveryProbe<B> {
    async fn run(mut self) {
        if *self.closed.borrow() || self.deadline.is_some_and(|at| self.clock.now() >= at) {
            return;
        }
        self.conn.total_timeout =
            self.deadline.map(|at| at.saturating_duration_since(self.clock.now()).as_secs_f64());
        let Some(outcome) = self.until_interrupted().await else { return };
        if *self.closed.borrow() || self.deadline.is_some_and(|at| self.clock.now() >= at) {
            return;
        }
        // `NotYet` and an error are both "still down"; the next request schedules another probe.
        if let Ok(Recovery::Recovered) = outcome {
            self.mark_recovered();
        }
    }

    /// The probe's verdict, or `None` when the deadline or the chain's close came first: the
    /// probe is then told to stop and awaited, bounded, to close what it opened.
    async fn until_interrupted(&mut self) -> Option<Result<Recovery, ProviderError>> {
        use futures_util::future::{Either, select};
        let (stop, stopped) = watch::channel(false);
        let recovery = self.backend.recover(&self.req, &self.conn, self.slot, stopped);
        let limit = crate::wait_for_deadline(self.clock.as_ref(), self.deadline);
        let interruption = select(Box::pin(limit), Box::pin(self.closed.changed()));
        let mut recovery = match select(Box::pin(recovery), Box::pin(interruption)).await {
            Either::Left((outcome, _)) => return Some(outcome),
            Either::Right((_, recovery)) => recovery,
        };
        let _ = stop.send(true);
        let cleanup = lk_agents_abi::Closed::bounded(
            self.clock.as_ref(),
            self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT),
            async {
                let _ = (&mut recovery).await;
            },
        )
        .await;
        if cleanup == lk_agents_abi::Closed::TimedOut {
            log::error!(
                "fallback probe cleanup incomplete; task retains ownership until settlement"
            );
            let _ = recovery.await;
        }
        None
    }

    /// Unless the entry closed or a newer generation superseded this probe; the observer hears
    /// it with no lock held.
    fn mark_recovered(self) {
        let changed = {
            let mut s = self.status.lock();
            match s.backends.get(self.index) {
                Some(st) if !st.closed => {
                    let resource = st.resource;
                    s.health.recovered(resource, self.generation)
                }
                _ => false,
            }
        };
        if changed {
            self.observer.availability_changed(AvailabilityChanged {
                index: self.index,
                label: self.label,
                available: true,
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

enum AttemptObservation<T, S> {
    Completed(Attempt<T>, Option<S>),
    Interrupted(ChainEvent),
}

/// The runtime owns an exchange until settlement; only a core Cancel effect signals its stop.
struct RunningAttempt<T, S> {
    /// The machine's id for the attempt.
    id: u64,
    index: usize,
    /// The selection id it was chosen under.
    attempt_id: u64,
    result: oneshot::Receiver<(Attempt<T>, S)>,
    stop: watch::Sender<bool>,
}

impl<T, S> RunningAttempt<T, S> {
    async fn observe(
        &mut self,
        clock: &dyn Clock,
        deadline: Option<Ts>,
        cancel: &mut watch::Receiver<bool>,
        closed: &mut watch::Receiver<bool>,
    ) -> AttemptObservation<T, S> {
        use futures_util::future::{Either, select};
        if *cancel.borrow() || *closed.borrow() {
            return AttemptObservation::Interrupted(ChainEvent::Cancel);
        }
        let limit = crate::wait_for_deadline(clock, deadline);
        let cancellation = select(Box::pin(cancel.changed()), Box::pin(closed.changed()));
        let interruption = select(Box::pin(limit), Box::pin(cancellation));
        match select(Box::pin(&mut self.result), Box::pin(interruption)).await {
            Either::Left((Ok((outcome, state)), _)) => {
                AttemptObservation::Completed(outcome, Some(state))
            }
            Either::Left((Err(_), _)) => AttemptObservation::Completed(
                Attempt::Failed {
                    error: ApiError::connection(Some("fallback attempt lost its outcome".into()))
                        .into(),
                    produced_output: true,
                },
                None,
            ),
            Either::Right((Either::Left(_), _)) => {
                AttemptObservation::Interrupted(ChainEvent::Wake)
            }
            Either::Right((Either::Right(_), _)) => {
                AttemptObservation::Interrupted(ChainEvent::Cancel)
            }
        }
    }

    async fn cancel(&mut self, clock: &dyn Clock) -> Option<S> {
        let _ = self.stop.send(true);
        let mut settled = None;
        let cleanup = lk_agents_abi::Closed::bounded(
            clock,
            clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT),
            async {
                settled = (&mut self.result).await.ok();
            },
        )
        .await;
        if cleanup == lk_agents_abi::Closed::TimedOut {
            log::error!(
                "fallback attempt cleanup incomplete; task retains ownership until settlement"
            );
        }
        settled.map(|(_, state)| state)
    }
}
