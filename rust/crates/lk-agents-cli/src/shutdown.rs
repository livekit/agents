//! Process shutdown: one signal owner and one deadline for draining and final host cleanup.

use lk_agents_abi::cli::{CliError, CliShutdown};
use lk_agents_worker::config::DrainOptions;
use parking_lot::{Condvar, Mutex};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    deadline: Option<Instant>,
    closed: bool,
}

pub(crate) struct Watchdog {
    state: Mutex<State>,
    changed: Condvar,
    requested: tokio::sync::watch::Sender<bool>,
    options: DrainOptions,
}

pub struct Shutdown {
    watchdog: Arc<Watchdog>,
}

impl CliShutdown for Shutdown {
    fn new(timeout: f64, drain_timeout: f64) -> Result<Self, CliError> {
        let options = DrainOptions::new(timeout, drain_timeout).map_err(CliError::Invalid)?;
        let (requested, _) = tokio::sync::watch::channel(false);
        Ok(Self {
            watchdog: Arc::new(Watchdog {
                state: Mutex::default(),
                changed: Condvar::default(),
                requested,
                options,
            }),
        })
    }

    fn listen(&self) -> Result<(), CliError> {
        listen(&self.watchdog)
    }

    async fn wait(&self) {
        let mut requested = self.watchdog.requested.subscribe();
        let _ = requested.wait_for(|requested| *requested).await;
    }

    fn begin(&self) -> bool {
        self.watchdog.begin()
    }

    fn remaining(&self) -> f64 {
        let state = self.watchdog.state.lock();
        state
            .deadline
            .map_or(self.watchdog.options.total(), |at| {
                at.saturating_duration_since(Instant::now())
            })
            .as_secs_f64()
    }

    fn drain_remaining(&self) -> f64 {
        self.watchdog
            .options
            .drain_remaining(Duration::from_secs_f64(self.remaining()))
            .as_secs_f64()
    }

    fn closed(&self) {
        self.watchdog.state.lock().closed = true;
        self.watchdog.changed.notify_all();
        release(&self.watchdog);
    }
}

impl Watchdog {
    pub(crate) fn is_closed(&self) -> bool {
        self.state.lock().closed
    }

    pub(crate) fn begin(self: &Arc<Self>) -> bool {
        let mut state = self.state.lock();
        if state.closed || state.deadline.is_some() {
            return false;
        }
        let deadline = Instant::now() + self.options.total();
        state.deadline = Some(deadline);
        let watchdog = self.clone();
        let started = std::thread::Builder::new().name("agent-shutdown".into()).spawn(move || {
            let mut state = watchdog.state.lock();
            while !state.closed {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    drop(state);
                    terminate(b"livekit.agents: shutdown deadline exceeded; terminating the CLI\n");
                }
                watchdog.changed.wait_for(&mut state, remaining);
            }
        });
        if started.is_err() {
            terminate(b"livekit.agents: cannot start shutdown watchdog; terminating the CLI\n");
        }
        self.requested.send_replace(true);
        true
    }
}

pub(crate) fn terminate(_message: &[u8]) -> ! {
    #[cfg(unix)]
    // SAFETY: use an inherited descriptor without allowing a full stderr pipe to block exit.
    unsafe {
        let flags = libc::fcntl(libc::STDERR_FILENO, libc::F_GETFL);
        if flags >= 0
            && libc::fcntl(libc::STDERR_FILENO, libc::F_SETFL, flags | libc::O_NONBLOCK) == 0
        {
            libc::write(libc::STDERR_FILENO, _message.as_ptr().cast(), _message.len());
            libc::fcntl(libc::STDERR_FILENO, libc::F_SETFL, flags);
        }
        libc::_exit(1);
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn TerminateProcess(process: *mut std::ffi::c_void, code: u32) -> i32;
        }
        // SAFETY: the pseudohandle identifies this process; no host code participates in exit.
        unsafe {
            TerminateProcess(GetCurrentProcess(), 1);
        }
        std::process::abort();
    }
    #[cfg(not(any(unix, windows)))]
    std::process::abort();
}

// Process-wide signal registration has exactly one active launcher owner.
static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
static OWNER: Mutex<Weak<Watchdog>> = Mutex::new(Weak::new());

pub(crate) fn listen(watchdog: &Arc<Watchdog>) -> Result<(), CliError> {
    let mut owner = OWNER.lock();
    if watchdog.is_closed() {
        return Err(CliError::Invalid("cannot listen after shutdown has closed".into()));
    }
    if let Some(current) = owner.upgrade() {
        return if Arc::ptr_eq(&current, watchdog) {
            Ok(())
        } else {
            Err(CliError::Invalid("another launcher already owns process signals".into()))
        };
    }
    INSTALLED
        .get_or_init(|| ctrlc::set_handler(on_signal).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| {
            CliError::Invalid(format!("cannot install process signal handler: {error}"))
        })?;
    *owner = Arc::downgrade(watchdog);
    Ok(())
}

pub(crate) fn release(watchdog: &Arc<Watchdog>) {
    let mut owner = OWNER.lock();
    if owner.ptr_eq(&Arc::downgrade(watchdog)) {
        *owner = Weak::new();
    }
}

fn on_signal() {
    let owner = OWNER.lock().upgrade();
    if let Some(owner) = owner {
        owner.begin();
    } else {
        // The permanent OS handler must never swallow signals between launcher runs.
        terminate(b"livekit.agents: termination requested without an active launcher\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_launcher_can_own_signals_until_closed() {
        let first = Shutdown::new(60.0, 50.0).unwrap();
        let second = Shutdown::new(60.0, 50.0).unwrap();
        first.listen().unwrap();
        first.listen().unwrap();
        assert!(second.listen().is_err());
        first.closed();
        assert!(first.listen().is_err());
        second.listen().unwrap();
        second.closed();
    }

    #[test]
    fn repeated_requests_do_not_restart_the_budget() {
        let shutdown = Shutdown::new(60.0, 50.0).unwrap();
        assert!(shutdown.begin());
        let before = shutdown.remaining();
        assert!(!shutdown.begin());
        assert!(shutdown.remaining() <= before);
        shutdown.closed();
        assert!(!shutdown.begin());
    }
}
