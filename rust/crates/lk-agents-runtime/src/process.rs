//! The process-wide Tokio runtime, started on first use and never at import: `fork()` copies only
//! the calling thread, so a runtime inherited across a fork has no workers. The runtime remembers
//! the process that started it and refuses to serve another. The host's ports (its time and frame
//! reader) are installed here once.

use std::sync::{Arc, Mutex};

use lk_agents_abi::Clock;
use lk_agents_abi::media::images::FrameReader;
use lk_agents_media::workers::{MediaWorkerLimits, MediaWorkers};
use lk_agents_voice::tts::processing::SpeechText;

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
    worker_io: lk_agents_net::worker_io::WorkerIo,
    report_admission: lk_agents_net::session_report::ReportAdmission,
    http: Arc<lk_agents_net::http::GatewayHttp>,
}

/// What the process holds: the runtime once started, and the host's ports handed to it at init.
/// The ports outlive a fork: the host objects are the child's too.
struct State {
    started: Option<Started>,
    /// The host's frame reader (decision 94), installed once as the host starts.
    frames: Option<Arc<dyn FrameReader>>,
    /// The host's time, installed once as the host loads.
    time: Option<Arc<dyn HostTime>>,
}

static PROCESS: Mutex<State> = Mutex::new(State { started: None, frames: None, time: None });

/// The host's clock and executor (decision 88). Each read resolves the caller's: a host whose
/// running loop is on virtual time answers with that loop's.
pub trait HostTime: Send + Sync {
    fn clock(&self) -> Arc<dyn Clock>;
    /// The running loop's executor when it has its own, else `None`: the process runtime's.
    fn spawner(&self) -> Option<Arc<dyn lk_agents_abi::Spawner>>;
}

fn state() -> std::sync::MutexGuard<'static, State> {
    PROCESS.lock().unwrap_or_else(|p| p.into_inner())
}

/// The runtime's handle, starting it in this process on first use.
pub fn handle() -> Result<tokio::runtime::Handle, ProcessError> {
    with_started(|started| started.runtime.handle().clone())
}

/// The runtime the caller is on, else the process runtime: where a native worker's async side runs.
pub fn caller_handle() -> Result<tokio::runtime::Handle, ProcessError> {
    tokio::runtime::Handle::try_current().or_else(|_| handle())
}

/// Process-owned admission reserved for worker setup and health sampling.
pub fn worker_io() -> Result<lk_agents_net::worker_io::WorkerIo, ProcessError> {
    with_started(|started| started.worker_io.clone())
}

/// The process-owned native admission pool. Clones share the same class and total capacity.
pub fn workers() -> Result<MediaWorkers, ProcessError> {
    with_started(|started| started.workers.clone())
}

fn with_started<T>(read: impl FnOnce(&Started) -> T) -> Result<T, ProcessError> {
    let pid = std::process::id();
    let mut state = state();
    let slot = &mut state.started;
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
    let worker_io = lk_agents_net::worker_io::WorkerIo::new(runtime.handle().clone(), 4)
        .map_err(|error| ProcessError::Start(error.to_string()))?;
    let report_admission = Default::default();
    let http = Arc::new(lk_agents_net::http::GatewayHttp::new().map_err(ProcessError::Start)?);
    let started = slot.insert(Started { pid, runtime, workers, worker_io, report_admission, http });
    Ok(read(started))
}

/// Declares this process as the runtime's owner (`_core.init` in a forked child): a runtime
/// inherited from the parent is abandoned, never dropped, since dropping joins worker threads
/// the child does not have. The next [`handle`] starts a fresh one.
pub fn reinit() {
    let mut state = state();
    if state.started.as_ref().is_some_and(|started| started.pid != std::process::id()) {
        std::mem::forget(state.started.take());
    }
}

fn host_time() -> Option<Arc<dyn HostTime>> {
    state().time.clone()
}

/// The host loop's clock (virtual in a virtual-time test), else the real one.
pub fn host_clock() -> Arc<dyn Clock> {
    match host_time() {
        Some(time) => time.clock(),
        None => Arc::new(lk_agents_abi::RealClock::new()),
    }
}

/// The host loop's executor (on virtual time in a virtual-time test), else the process runtime's.
pub fn host_spawner() -> Result<Arc<dyn lk_agents_abi::Spawner>, String> {
    match host_time().and_then(|time| time.spawner()) {
        Some(spawner) => Ok(spawner),
        None => Ok(Arc::new(lk_agents_abi::TokioSpawner::new(
            handle().map_err(|error| error.to_string())?,
        ))),
    }
}

/// What this process lends the declared classes (decision 100). Reading it starts nothing: the
/// runtime-bound handles start the process runtime on first use.
pub fn env() -> lk_agents_net::env::ProcessEnv {
    lk_agents_net::env::ProcessEnv {
        runtime: || handle().map_err(|error| error.to_string()),
        spawner: host_spawner,
        clock: host_clock(),
        http: || with_started(|started| started.http.clone()).map_err(|error| error.to_string()),
        report_admission: || {
            with_started(|started| started.report_admission.clone())
                .map_err(|error| error.to_string())
        },
        worker_io: || worker_io().map_err(|error| error.to_string()),
        workers: || workers().map_err(|error| error.to_string()),
        images: images(),
        strip_assistant_markup: lk_agents_voice::tts::markup::strip_expr_markup,
        speech_text: SpeechText::PROCESSORS,
    }
}

/// Image preparation with the host's frame reader, once the host installed one.
#[cfg(feature = "media-images")]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    Arc::new(Process::images())
}

#[cfg(not(feature = "media-images"))]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    Arc::new(lk_agents_net::env::NoImagePreparation)
}

/// The process the runtime owns: its host ports and image preparation.
pub struct Process;

impl Process {
    /// Takes the host's frame reader as the host starts; the first one installed stays.
    pub fn install_frame_reader(reader: Arc<dyn FrameReader>) {
        state().frames.get_or_insert(reader);
    }

    /// Takes the host's time as the host loads; the first one installed stays.
    pub fn install_host_time(time: Arc<dyn HostTime>) {
        state().time.get_or_insert(time);
    }

    /// Image preparation with the host's frame reader, once the host installed one.
    #[cfg(feature = "media-images")]
    pub fn images() -> lk_agents_media::images::Images {
        lk_agents_media::images::Images::new(state().frames.clone())
    }
}
