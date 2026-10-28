//! The process-wide Tokio runtime, started on first use and never at import: `fork()` copies only
//! the calling thread, so a runtime inherited across a fork has no workers. The runtime remembers
//! the process that started it and refuses to serve another. The host's clock is read here too.

use std::sync::{Arc, Mutex};

use lk_agents_abi::Clock;
use lk_agents_media::workers::{MediaWorkerError, MediaWorkerLimits, MediaWorkers};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessError {
    #[error(
        "the native runtime was started before this process was forked; it cannot run in the \
         child. Use the core only after the fork (or with the spawn start method)."
    )]
    Forked,
    #[error("the native runtime failed to start: {0}")]
    Start(String),
}

struct Started {
    pid: u32,
    runtime: tokio::runtime::Runtime,
    workers: MediaWorkers,
}

static RUNTIME: Mutex<Option<Started>> = Mutex::new(None);

/// The runtime's handle, starting it in this process on first use.
pub fn handle() -> Result<tokio::runtime::Handle, ProcessError> {
    with_started(|started| started.runtime.handle().clone())
}

/// The runtime the caller is on, else the process runtime: where a native worker's async side runs.
pub fn caller_handle() -> Result<tokio::runtime::Handle, ProcessError> {
    tokio::runtime::Handle::try_current().or_else(|_| handle())
}

/// The process-owned native admission pool. Clones share the same class and total capacity.
pub fn workers() -> Result<MediaWorkers, ProcessError> {
    with_started(|started| started.workers.clone())
}

fn with_started<T>(read: impl FnOnce(&Started) -> T) -> Result<T, ProcessError> {
    let pid = std::process::id();
    let mut slot = RUNTIME.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(started) = slot.as_ref() {
        return if started.pid == pid { Ok(read(started)) } else { Err(ProcessError::Forked) };
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        // Codec/model work has its own admitted workers. Starting one async worker per CPU
        // here makes the first loop-facing constructor pay for unrelated native capacity.
        .worker_threads(2)
        .thread_name("lk-agents-runtime")
        .enable_all()
        .build()
        .map_err(|e| ProcessError::Start(e.to_string()))?;
    let workers = MediaWorkers::new(MediaWorkerLimits::default())
        .map_err(|error| ProcessError::Start(error.to_string()))?;
    let started = slot.insert(Started { pid, runtime, workers });
    Ok(read(started))
}

/// Declares this process as the runtime's owner (`_core.init` in a forked child): a runtime
/// inherited from the parent is abandoned, never dropped, since dropping joins worker threads
/// the child does not have. The next [`handle`] starts a fresh one.
pub fn reinit() {
    let mut slot = RUNTIME.lock().unwrap_or_else(|p| p.into_inner());
    if slot.as_ref().is_some_and(|started| started.pid != std::process::id()) {
        std::mem::forget(slot.take());
    }
}

/// The host loop's clock (virtual in a virtual-time test), else the real one.
#[cfg(feature = "python")]
pub fn host_clock() -> Arc<dyn Clock> {
    <Arc<dyn Clock> as lk_agents_abi::python::Ambient>::ambient()
}

#[cfg(not(feature = "python"))]
pub fn host_clock() -> Arc<dyn Clock> {
    Arc::new(lk_agents_abi::RealClock::new())
}

/// This process as the inference clients' [`Env`](lk_agents_inference::env::Env): what a class
/// the host constructs runs on (decision 88).
pub struct Process;

impl lk_agents_inference::env::Env for Process {
    fn runtime() -> Result<tokio::runtime::Handle, String> {
        handle().map_err(|error| error.to_string())
    }

    #[cfg(feature = "python")]
    fn spawner() -> Result<Arc<dyn lk_agents_abi::Spawner>, String> {
        Ok(<Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient())
    }

    fn clock() -> Arc<dyn Clock> {
        host_clock()
    }

    fn workers() -> Result<MediaWorkers, String> {
        workers().map_err(|error| error.to_string())
    }

    #[cfg(feature = "media-images")]
    fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
        Arc::new(crate::images::NativeImages)
    }

    #[cfg(not(feature = "media-images"))]
    fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
        <lk_agents_inference::env::Standalone as lk_agents_inference::env::Env>::images()
    }
}

lk_agents_abi::inventory::submit! {
    lk_agents_abi::utils::workers::WorkerService {
        try_spawn: |class, job| {
            let workers = workers()
                .map_err(|error| MediaWorkerError::Unavailable(error.to_string()))?;
            let receipt = workers.try_spawn(class, job)?;
            Ok(Box::pin(async move { receipt.await.map_err(|error| error.to_string()) }))
        },
    }
}
