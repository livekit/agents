//! Native watchdog execution for the launcher's signal and loop-cleanup policy.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::cli::{EXIT_ESCALATION_TIMEOUT, ExitSignals};
use lk_agents_abi::worker::cli::CliSignals as CliSignalsPort;

pub struct CliSignals {
    state: Arc<(Mutex<ExitSignals>, Condvar)>,
    escalate: bool,
    #[cfg(unix)]
    main: usize,
}

impl CliSignalsPort for CliSignals {
    fn new(escalate: bool) -> Self {
        Self {
            state: Arc::new((Mutex::new(ExitSignals::new()), Condvar::new())),
            escalate,
            #[cfg(unix)]
            // SAFETY: pthread_self has no preconditions. The launcher owns this main thread
            // throughout its shutdown, so its identity remains valid while the watchdog runs.
            main: unsafe { libc::pthread_self() } as usize,
        }
    }

    fn signal(&self, exit_served: bool) -> String {
        let action = self.state.0.lock().unwrap_or_else(|p| p.into_inner()).signal(exit_served);
        if action == "schedule" && self.escalate {
            self.arm_watchdog(EXIT_ESCALATION_TIMEOUT, false);
        }
        action.into()
    }

    fn served(&self) {
        self.state.0.lock().unwrap_or_else(|p| p.into_inner()).served();
        self.state.1.notify_all();
    }

    fn draining(&self) {
        self.state.0.lock().unwrap_or_else(|p| p.into_inner()).draining();
        self.state.1.notify_all();
    }

    fn closing(&self) {
        let arm = self.state.0.lock().unwrap_or_else(|p| p.into_inner()).closing();
        self.state.1.notify_all();
        if arm {
            self.arm_watchdog(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, true);
        }
    }

    fn closed(&self) {
        self.state.0.lock().unwrap_or_else(|p| p.into_inner()).closed();
        self.state.1.notify_all();
    }
}

impl CliSignals {
    fn arm_watchdog(&self, timeout: Duration, cleanup: bool) {
        let state = self.state.clone();
        #[cfg(unix)]
        let main = self.main;
        std::thread::spawn(move || {
            let (lock, wake) = &*state;
            let guard = lock.lock().unwrap_or_else(|p| p.into_inner());
            let (mut guard, result) = wake
                .wait_timeout_while(guard, timeout, |policy| {
                    if cleanup { policy.cleanup_due() } else { policy.escalation_pending() }
                })
                .unwrap_or_else(|p| p.into_inner());
            if !result.timed_out() {
                return;
            }
            if cleanup && guard.cleanup_due() {
                drop(guard);
                terminate_incomplete_cleanup();
            }
            if !cleanup && guard.escalation_due() {
                drop(guard);
                #[cfg(unix)]
                // SAFETY: main is the live launcher's main thread, captured at construction.
                unsafe {
                    libc::pthread_kill(main as libc::pthread_t, libc::SIGTERM);
                }
            }
        });
    }
}

fn terminate_incomplete_cleanup() -> ! {
    #[cfg(unix)]
    // SAFETY: fd 2 belongs to this process and the static message remains valid for write.
    // The descriptor can be closed or unwritable: errors are ignored deliberately. Restore
    // its shared status flags before _exit, which cannot run blocking host/C exit handlers.
    unsafe {
        let flags = libc::fcntl(libc::STDERR_FILENO, libc::F_GETFL);
        if flags >= 0
            && libc::fcntl(libc::STDERR_FILENO, libc::F_SETFL, flags | libc::O_NONBLOCK) == 0
        {
            let message = b"livekit.agents: incomplete event-loop cleanup; terminating the CLI\n";
            libc::write(libc::STDERR_FILENO, message.as_ptr().cast(), message.len());
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
        // No synchronous write to an inherited Windows pipe is guaranteed nonblocking.
        // SAFETY: GetCurrentProcess returns the kernel's valid current-process pseudohandle.
        unsafe {
            TerminateProcess(GetCurrentProcess(), 1);
        }
        std::process::abort();
    }
    #[cfg(not(any(unix, windows)))]
    std::process::abort();
}
