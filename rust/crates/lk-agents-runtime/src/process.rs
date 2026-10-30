//! The process-wide Tokio runtime, started on first use and never at import: `fork()` copies only
//! the calling thread, so a runtime inherited across a fork has no workers. The process records
//! its owner and refuses to serve another. The host's ports (its time and frame reader) are
//! installed here once.

use lk_agents_abi::media::images::FrameReader;
use lk_agents_net::lock::Mutex;
use std::sync::Arc;

use lk_agents_abi::Clock;
use lk_agents_voice::tts::processing::SpeechText;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessError {
    #[error("the native runtime failed to start: {0}")]
    Start(String),
    #[error(
        "livekit.agents._core was inherited across a fork. The Rust core cannot be used in a \
         forked child without re-initialising it: call _core.init(role) in the child."
    )]
    Inherited,
}

struct Started {
    runtime: tokio::runtime::Runtime,
    http: Arc<lk_agents_net::http::GatewayHttp>,
}

/// What the process holds: the runtime once started, and the host's ports handed to it at init.
/// The ports outlive a fork: the host objects are the child's too.
struct State {
    started: Option<Started>,
    /// The host's frame reader, installed once as the host starts.
    frames: Option<Arc<dyn FrameReader>>,
    /// The process that declared itself the core's (`_core.init`, else its first entry or the
    /// runtime's start): the runtime, when started, is this process's.
    owner: Option<u32>,
}

static PROCESS: Mutex<State> = Mutex::new(State { started: None, frames: None, owner: None });

fn state() -> lk_agents_net::lock::MutexGuard<'static, State> {
    PROCESS.lock()
}

/// The runtime's handle, starting it in this process on first use.
pub fn handle() -> Result<tokio::runtime::Handle, ProcessError> {
    with_started(|started| started.runtime.handle().clone())
}

/// The runtime the caller is on, else the process runtime: where a native worker's async side runs.
pub fn caller_handle() -> Result<tokio::runtime::Handle, ProcessError> {
    tokio::runtime::Handle::try_current().or_else(|_| handle())
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
        .thread_name("lk-agents-runtime")
        .enable_all()
        .build()
        .map_err(|e| ProcessError::Start(e.to_string()))?;
    let http = Arc::new(lk_agents_net::http::GatewayHttp::new().map_err(ProcessError::Start)?);
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
/// now (the reference has no initialiser); one that inherited another's declaration across a
/// fork is refused until it calls [`reinit`], since its runtime's threads did not survive.
pub fn check_pid() -> Result<(), ProcessError> {
    owned(&mut state())
}

/// Declares this process as the runtime's owner (`_core.init` in a forked child): a runtime
/// inherited from the parent is abandoned, never dropped, since dropping joins worker threads
/// the child does not have. The next [`handle`] starts a fresh one.
pub fn reinit() {
    let mut state = state();
    if state.owner.replace(std::process::id()) != Some(std::process::id()) {
        std::mem::forget(state.started.take());
    }
}

/// The host's clock: the wall clock.
pub fn host_clock() -> Arc<dyn Clock> {
    Arc::new(lk_agents_net::clock::RealClock::new())
}

/// The host's executor: the process runtime's.
pub fn host_spawner() -> Result<Arc<dyn lk_agents_abi::Spawner>, String> {
    Ok(Arc::new(lk_agents_net::task::TokioSpawner::new(
        handle().map_err(|error| error.to_string())?,
    )))
}

/// What this process lends the declared classes. Reading it starts nothing: the
/// runtime-bound handles start the process runtime on first use.
pub fn env() -> lk_agents_net::env::ProcessEnv {
    lk_agents_net::env::ProcessEnv {
        runtime: || handle().map_err(|error| error.to_string()),
        spawner: host_spawner,
        clock: host_clock(),
        http: || with_started(|started| started.http.clone()).map_err(|error| error.to_string()),
        images: images(),
        strip_assistant_markup: lk_agents_voice::tts::markup::strip_expr_markup,
        speech_text: SpeechText::PROCESSORS,
    }
}

/// Image preparation with the host's frame reader, once the host installed one.
#[cfg(feature = "media-images")]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    Arc::new(lk_agents_media::images::Images::new(state().frames.clone()))
}

#[cfg(not(feature = "media-images"))]
fn images() -> Arc<dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync> {
    Arc::new(lk_agents_net::env::NoImagePreparation)
}

/// The process the runtime owns: its host ports.
pub struct Process;

impl lk_agents_abi::media::images::FrameReaderInstall for Process {
    /// The first one installed stays.
    fn install_frame_reader(reader: Arc<dyn FrameReader>) {
        state().frames.get_or_insert(reader);
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

#[cfg(test)]
mod pid_tests {
    use super::*;

    // The failure the guard exists for, and its remedy: a declaration inherited from another
    // process is refused until this one re-declares itself; a process that never declared is
    // usable at once (the reference needs no initialiser).
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
