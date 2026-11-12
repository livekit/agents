use parking_lot::Mutex;
use std::sync::Arc;

use lk_agents_abi::Clock;
use lk_agents_net::env::ProcessError;

// Started on first use, never at import: `fork()` copies only the calling thread, so a runtime
// inherited across a fork has no workers.

struct Started {
    runtime: tokio::runtime::Runtime,
    http: Arc<lk_agents_net::http::GatewayHttp>,
}

struct State {
    started: Option<Started>,
    /// Work no session owns, on the runtime: inherited across a fork, it is abandoned with it.
    tasks: Option<Arc<lk_agents_net::task::TaskSet>>,
    /// The process that declared itself the runtime's owner.
    owner: Option<u32>,
}

static PROCESS: Mutex<State> = Mutex::new(State { started: None, tasks: None, owner: None });

fn state() -> parking_lot::MutexGuard<'static, State> {
    PROCESS.lock()
}

/// The runtime's handle, starting it in this process on first use.
pub fn handle() -> Result<tokio::runtime::Handle, ProcessError> {
    with_started(|started| started.runtime.handle().clone())
}

/// The runtime's handle if it started in this process; starts nothing.
pub fn started_handle() -> Option<tokio::runtime::Handle> {
    let state = state();
    let owned = state.owner == Some(std::process::id());
    state.started.as_ref().filter(|_| owned).map(|started| started.runtime.handle().clone())
}

fn with_started<T>(read: impl FnOnce(&Started) -> T) -> Result<T, ProcessError> {
    let mut state = state();
    owned(&mut state)?;
    let slot = &mut state.started;
    if let Some(started) = slot.as_ref() {
        return Ok(read(started));
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        // Codec/model work has its own admitted workers. Starting one async worker per CPU
        // here makes the first loop-facing constructor pay for unrelated native capacity.
        .worker_threads(2)
        // Blocking work runs on our pools (lk_agents_media::pool::media_pool,
        // lk_agents_net::blocking_io), never here: one thread is left for third-party crates
        // that still resolve names through Tokio (web-transport-quinn), so they cannot grow
        // a thread per stalled lookup.
        .max_blocking_threads(1)
        .thread_name("lk-agents-runtime")
        .enable_all()
        .build()
        .map_err(|e| ProcessError::Start(e.to_string()))?;
    let http = lk_agents_net::http::GatewayHttp::new()
        .map_err(|error| ProcessError::Start(error.to_string()))?;
    let http = Arc::new(http);
    let started = slot.insert(Started { runtime, http });
    Ok(read(started))
}

/// A process that never declared itself does so now; one holding another's declaration is
/// refused.
fn owned(state: &mut State) -> Result<(), ProcessError> {
    let pid = std::process::id();
    if *state.owner.get_or_insert(pid) == pid { Ok(()) } else { Err(ProcessError::Inherited) }
}

/// The fork guard every host entry passes first: a process that never declared itself does so
/// now; one that inherited another's declaration across a
/// fork is refused until it calls [`reinit`], since its runtime's threads did not survive.
pub fn check_pid() -> Result<(), ProcessError> {
    owned(&mut state())
}

/// Declares this process as the runtime's owner (in a forked child): a runtime
/// inherited from the parent is abandoned, never dropped, since dropping joins worker threads
/// the child does not have. The next [`handle`] starts a fresh one.
pub fn reinit() {
    let mut state = state();
    if state.owner.replace(std::process::id()) != Some(std::process::id()) {
        std::mem::forget(state.started.take());
        state.tasks = None;
        crate::telemetry::reinit();
    }
}

/// The host's clock: the wall clock.
pub fn host_clock() -> Arc<dyn Clock> {
    Arc::new(lk_agents_net::clock::RealClock::new())
}

/// The host's executor: the process runtime's.
pub fn host_spawner() -> Result<Arc<dyn lk_agents_abi::Spawner>, ProcessError> {
    Ok(Arc::new(lk_agents_net::task::TokioSpawner::new(handle()?)))
}

/// What this process lends the declared classes. Reading it starts nothing: the
/// runtime-bound handles start the process runtime on first use.
pub fn env() -> lk_agents_net::env::ProcessEnv {
    lk_agents_net::env::ProcessEnv {
        runtime: handle,
        spawner: host_spawner,
        clock: host_clock(),
        http: || with_started(|started| started.http.clone()),
        tracer: crate::telemetry::tracer(),
        tasks: Arc::clone(state().tasks.get_or_insert_with(Arc::default)),
    }
}

/// The process runtime's load: its async workers' busy time.
pub struct RuntimeLoad;

impl lk_agents_abi::metrics::RuntimeLoad for RuntimeLoad {
    /// Starts nothing: a runtime not started in this process reports no workers.
    fn runtime_busy_seconds() -> Vec<f64> {
        let Some(handle) = started_handle() else { return Vec::new() };
        let metrics = handle.metrics();
        (0..metrics.num_workers())
            .map(|worker| metrics.worker_total_busy_duration(worker).as_secs_f64())
            .collect()
    }
}

/// Lets the executor's other tasks run before continuing: one `Pending` that wakes itself.
/// Unlike `tokio::task::yield_now`, it assumes no Tokio runtime, so a deterministic scheduler
/// runs it as it is.
pub(crate) async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if std::mem::replace(&mut yielded, true) {
            return std::task::Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    })
    .await
}

/// The host's explicit process initialization, including fork recovery.
pub struct Initialization;

impl lk_agents_abi::worker::ipc_host::ProcessInitialization for Initialization {
    fn init(_role: lk_agents_abi::worker::ipc_host::ProcessRole) {
        reinit();
        // The sentence rules compile once per process (about 10ms), here rather than on the first
        // reply's way to the voice; a forked child inherits them compiled.
        drop(lk_agents_core::tokenize::sentences::SentenceSplitter::default());
    }
}

#[cfg(test)]
mod pid_tests {
    use super::*;

    // A declaration inherited from another process is refused until this one re-declares itself;
    // a process that never declared is usable at once.
    #[test]
    fn a_forked_child_is_refused_until_it_re_declares_itself() {
        state().owner = None;
        check_pid().expect("first use must self-declare rather than raise");
        state().owner = Some(std::process::id() ^ 0x5555);
        let err = check_pid().expect_err("a stale declaration must be refused");
        assert!(err.to_string().contains("inherited across a fork"), "{err}");
        reinit();
        check_pid().expect("usable once re-declared");
    }
}
