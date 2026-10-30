//! One native process deadline shared by draining and final host cleanup.

use lk_agents_abi::cli::{CliError, CliShutdown};
use lk_agents_worker::shutdown::DrainOptions;
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
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
        crate::signals::listen(&self.watchdog)
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
        crate::signals::release(&self.watchdog);
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
