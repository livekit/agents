//! `FallbackChain`: the availability + background-recovery machine the LLM, STT and TTS adapters
//! share. Scanned from index 0 on every request (`llm:274`), so a recovered primary takes the next
//! one; probes re-issue the real request (`llm:174-200`, `stt:201-235`, `tts:504-531`).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use lk_agents_abi::{Clock, Ts};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::AbortHandle;
use tokio::sync::{oneshot, watch};

use crate::fallback::policy::{
    AfterOutput, ChainPolicy, conn_options_for_attempt, conn_options_for_recovery,
};

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    /// Clonable because recovery replays it.
    type Req: Clone + Send + Sync + 'static;
    type Session: Send + 'static;

    fn label(&self) -> String;

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

pub enum Attempt<T> {
    Served(T),
    Failed {
        error: ProviderError,
        /// Whether anything reached the consumer before the failure; feeds [`AfterOutput`].
        produced_output: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainOutcome<T> {
    Served(T),
    /// A backend failed after output and the policy is [`AfterOutput::Stop`]: the caller sees
    /// success (`tts:295-297`).
    Truncated,
}

pub use lk_agents_abi::llm::fallback::AvailabilityChanged;

/// The chain's one outbound port. `metrics_collected` re-emission lives in `lk-agents-py`.
///
/// A host adapter is this trait through its bare `#[host(proxy)]` (`PyChainObserver`): the
/// `FallbackAdapter` itself, whose `_core_availability_changed` maps the index back to its instance.
#[lk_agents_abi::host(proxy(host = "livekit.agents.llm.FallbackAdapter"))]
pub trait ChainObserver: Send + Sync {
    fn availability_changed(&self, ev: AvailabilityChanged);

    /// `logger.error("all X are unavailable, retrying..")`. The chain then tries every entry
    /// anyway rather than failing fast, which is what `all_failed` does in all three adapters.
    fn all_unavailable(&self) {}
}

pub struct IgnoreEvents;
impl ChainObserver for IgnoreEvents {
    fn availability_changed(&self, _ev: AvailabilityChanged) {}
}

#[derive(Debug)]
struct Status {
    available: bool,
    recovery: [Option<Probe>; 2],
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
    status: Arc<Mutex<Vec<Status>>>,
    policy: ChainPolicy,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    observer: Arc<dyn ChainObserver>,
    /// `_active_instance`, labelling metrics & traces. Starts at 0.
    active: AtomicUsize,
    /// `tts/fallback_adapter.py:102,155-159`. Set before the cancel sweep so an in-flight stream
    /// cannot start a probe nothing will cancel.
    closed: AtomicBool,
    /// Every probe's `stop`; see [`Backend::recover`].
    stop: watch::Sender<bool>,
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
        let status: Vec<Status> =
            entries.iter().map(|_| Status { available: true, recovery: [None, None] }).collect();
        Ok(FallbackChain {
            entries,
            labels,
            status: Arc::new(Mutex::new(status)),
            policy,
            clock,
            spawner,
            observer,
            active: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            stop: watch::channel(false).0,
        })
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
        self.status.lock().expect("chain status").get(index).is_some_and(|s| s.available)
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
        F: for<'a> Fn(usize, &'a B, B::Session, &'a mut S) -> BoxFuture<'a, Attempt<T>>,
    {
        let started = self.clock.now();
        let attempt_conn = conn_options_for_attempt(caller_conn, &self.policy);
        let recover_conn = conn_options_for_recovery(caller_conn, &self.policy);

        // Snapshot once, like the reference: a probe flipping an entry mid-scan must not change
        // the mode.
        let all_failed = {
            let s = self.status.lock().expect("chain status");
            s.iter().all(|st| !st.available)
        };
        if all_failed {
            self.observer.all_unavailable();
        }

        let mut last_error: Option<ProviderError> = None;

        for index in 0..self.entries.len() {
            if self.is_available(index) || all_failed {
                let backend = &self.entries[index];
                let outcome = match backend.open(req, &attempt_conn).await {
                    Ok(session) => attempt(index, backend.as_ref(), session, state).await,
                    // An `open` that raises is a failure with nothing produced.
                    Err(error) => Attempt::Failed { error, produced_output: false },
                };

                match outcome {
                    Attempt::Served(value) => {
                        self.mark_serving(index);
                        return Ok(ChainOutcome::Served(value));
                    }
                    Attempt::Failed { error, produced_output } => {
                        self.set_available(index, false);
                        // The previous member's error is dropped; for a host exception that
                        // releases its side-table key (RAII).
                        last_error = Some(error);

                        if produced_output {
                            match self.policy.after_output {
                                AfterOutput::Abort => {
                                    return Err(last_error.expect("just assigned"));
                                }
                                AfterOutput::Stop => return Ok(ChainOutcome::Truncated),
                                AfterOutput::Retry => {}
                            }
                        }
                    }
                }
            }

            // Every entry that did not serve, including skipped unavailable ones, gets a probe.
            self.try_recovery(index, slot, req, &recover_conn);
        }

        Err(self.exhausted(started, last_error))
    }

    /// Unless a live probe is already on `slot`.
    pub fn try_recovery(
        &self,
        index: usize,
        slot: RecoverySlot,
        req: &B::Req,
        conn: &ApiConnectOptions,
    ) {
        if self.closed.load(Ordering::SeqCst) {
            // `tts:218-220,499-502`. Nothing would cancel a probe started from here.
            return;
        }

        {
            let s = self.status.lock().expect("chain status");
            let Some(status) = s.get(index) else { return };
            if status.recovery[slot as usize].as_ref().is_some_and(|p| !p.handle.is_finished()) {
                return;
            }
        }

        let backend = Arc::clone(&self.entries[index]);
        let label = self.labels[index].clone();
        let req = req.clone();
        let conn = *conn;
        let observer = Arc::clone(&self.observer);
        let status_for_task = Arc::clone(&self.status);
        let stop = self.stop.subscribe();
        let (done_tx, done) = oneshot::channel::<()>();

        let handle = self.spawner.spawn_unboxed(async move {
            // Dropped with the task, however it ends.
            let _done = done_tx;
            match backend.recover(&req, &conn, slot, stop).await {
                Ok(Recovery::Recovered) => {
                    let changed = {
                        let mut s = status_for_task.lock().expect("chain status");
                        match s.get_mut(index) {
                            Some(st) if !st.available => {
                                st.available = true;
                                true
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
        if let Some(status) = s.get_mut(index) {
            status.recovery[slot as usize] = Some(Probe { handle, done });
        }
    }

    /// Marks the chain closed, tells every probe to stop, and hands back the ones in flight.
    fn begin_close(&self) -> Vec<Probe> {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.stop.send(true);
        let mut s = self.status.lock().expect("chain status");
        s.iter_mut().flat_map(|status| status.recovery.iter_mut().filter_map(Option::take)).collect()
    }

    /// Idempotent; see `closed`. Aborts every probe in flight at once.
    pub fn aclose(&self) {
        for probe in self.begin_close() {
            probe.handle.abort();
        }
    }

    /// The host's `aclose()` (`cancel_and_wait` on each recovery task): every probe closes what it
    /// opened, awaited up to the teardown deadline; one still running then is aborted.
    pub async fn close(&self) {
        let probes = self.begin_close();
        let (handles, dones): (Vec<_>, Vec<_>) = probes.into_iter().map(|p| (p.handle, p.done)).unzip();
        lk_agents_abi::Closed::bounded(self.clock.as_ref(), self.teardown_deadline(), async move {
            for done in dones {
                let _ = done.await;
            }
        })
        .await;
        for handle in handles {
            handle.abort();
        }
    }

    fn set_available(&self, index: usize, available: bool) {
        let changed = {
            let mut s = self.status.lock().expect("chain status");
            match s.get_mut(index) {
                Some(st) if st.available != available => {
                    st.available = available;
                    true
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

/// The reference's `APIConnectionError("all backends failed ...")` (`llm/fallback_adapter.py:
/// 305`). `last` contributes its message and is then dropped (releasing any side-table entry).
    fn exhausted(&self, started: Ts, last: Option<ProviderError>) -> ProviderError {
        let elapsed: Duration = self.clock.now().saturating_duration_since(started);
        let detail = last.map(|e| format!(": {}", e.message())).unwrap_or_default();
        ProviderError::Local(ApiError::connection(Some(format!(
            "all backends failed ({}) after {:.3} seconds{detail}",
            self.labels.join(", "),
            elapsed.as_secs_f64(),
        ))))
    }
}
