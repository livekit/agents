//! The process-wide Tokio runtime, started on first use and never at import: `fork()` copies only
//! the calling thread, so a runtime inherited across a fork has no workers. The runtime remembers
//! the process that started it and refuses to serve another. The host's clock is read here too.

use std::sync::{Arc, Mutex};

use lk_agents_abi::Clock;
use lk_agents_media::workers::{MediaWorkerError, MediaWorkerLimits, MediaWorkers};
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
    worker_io: lk_agents_worker::io::WorkerIo,
    report_admission: lk_agents_net::session_report::ReportAdmission,
    http: Arc<lk_agents_net::http::GatewayHttp>,
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

/// Process-owned admission reserved for worker setup and health sampling.
pub fn worker_io() -> Result<lk_agents_worker::io::WorkerIo, ProcessError> {
    with_started(|started| started.worker_io.clone())
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
    let worker_io = lk_agents_worker::io::WorkerIo::new(runtime.handle().clone(), 4)
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

/// The process services selected by host-constructed clients through the shared environment.
#[derive(Clone)]
pub struct Process;

impl lk_agents_net::env::Env for Process {
    fn http() -> Result<Arc<lk_agents_net::http::GatewayHttp>, String> {
        with_started(|started| started.http.clone()).map_err(|error| error.to_string())
    }

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

    fn report_admission() -> Result<lk_agents_net::session_report::ReportAdmission, String> {
        with_started(|started| started.report_admission.clone()).map_err(|error| error.to_string())
    }
}

impl lk_agents_inference::env::Env for Process {
    fn workers() -> Result<MediaWorkers, String> {
        workers().map_err(|error| error.to_string())
    }

    fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
        images()
    }

    fn strip_assistant_markup(text: &str) -> String {
        lk_agents_voice::tts::markup::strip_expr_markup(text)
    }
}

impl lk_agents_worker::env::Env for Process {
    fn worker_io() -> Result<lk_agents_worker::io::WorkerIo, String> {
        worker_io().map_err(|error| error.to_string())
    }
}

#[cfg(feature = "media-images")]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    Arc::new(crate::images::NativeImages)
}

#[cfg(not(feature = "media-images"))]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    <lk_agents_inference::env::Standalone as lk_agents_inference::env::Env>::images()
}

/// The voice core's speech text in the native synthesis clients and the judges' transcripts.
impl lk_agents_abi::tts::text::TtsTextProcessorFactory for Process {
    fn sentence_splitter(
        provider: Option<&str>,
        expressive: bool,
    ) -> Box<dyn lk_agents_abi::SentenceSplitter> {
        SpeechText::sentence_splitter(provider, expressive)
    }
    fn markup(provider: &str) -> Option<Box<dyn lk_agents_abi::SentenceTextProcessor>> {
        SpeechText::markup(provider)
    }
    fn transcript(expressive: bool) -> Box<dyn lk_agents_abi::tts::text::TranscriptProcessor> {
        SpeechText::transcript(expressive)
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
