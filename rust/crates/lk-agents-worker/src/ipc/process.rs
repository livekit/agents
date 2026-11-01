//! The process executor: the only code touching the OS process (exit observation, termination,
//! multiprocessing's start-method reaping receipts, memory samples), behind [`RunnerExecutor`].
//!
//! All polls are nonblocking. An exit observation never substitutes for the start method's
//! reaping receipt.

use lk_agents_abi::logging::LogWriter;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host;
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};

use super::Level;
use super::executor::{Connection, Exit, Runner, RunnerExecutor};
use super::ipc_channel::{self, Channel};
use super::owner::queue_log;

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartMethod {
    Spawn,
    ForkServer,
}

impl StartMethod {
    pub fn parse(method: &str) -> io::Result<Self> {
        match method {
            "spawn" => Ok(Self::Spawn),
            "forkserver" if cfg!(target_os = "linux") => Ok(Self::ForkServer),
            _ => {
                Err(io::Error::new(io::ErrorKind::InvalidInput, "unsupported process start method"))
            }
        }
    }
}

/// Immutable identity used by native signals, shared until the exit is observed.
pub struct ProcessControl {
    pub pid: u32,
    already_exited: bool,
    #[cfg(target_os = "linux")]
    pidfd: Option<OwnedFd>,
    #[cfg(target_os = "linux")]
    pidfd_error: Option<String>,
    #[cfg(target_os = "linux")]
    method: StartMethod,
    #[cfg(windows)]
    handle: OwnedHandle,
}

fn signal_pid(pid: u32) -> io::Result<i32> {
    i32::try_from(pid).ok().filter(|pid| *pid > 0).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "an invalid PID cannot receive process signals")
    })
}

impl ProcessControl {
    #[cfg(unix)]
    fn new(pid: u32, method: StartMethod) -> Self {
        #[cfg(not(target_os = "linux"))]
        let _ = method;
        #[cfg(target_os = "linux")]
        let (pidfd, already_exited, pidfd_error) = {
            // SAFETY: pid is a validated process identity, flags is zero.
            let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0_u32) };
            if fd < 0 {
                let error = io::Error::last_os_error();
                (None, error.raw_os_error() == Some(libc::ESRCH), Some(error.to_string()))
            } else {
                // SAFETY: pidfd_open returned a newly owned descriptor.
                (Some(unsafe { OwnedFd::from_raw_fd(fd as i32) }), false, None)
            }
        };
        Self {
            pid,
            #[cfg(not(target_os = "linux"))]
            already_exited: false,
            #[cfg(target_os = "linux")]
            already_exited,
            #[cfg(target_os = "linux")]
            pidfd,
            #[cfg(target_os = "linux")]
            pidfd_error,
            #[cfg(target_os = "linux")]
            method,
        }
    }

    #[cfg(unix)]
    fn signal(&self, signal: i32) -> io::Result<bool> {
        let pid = signal_pid(self.pid)?;
        if self.already_exited {
            return Ok(false);
        }
        #[cfg(target_os = "linux")]
        if let Some(fd) = &self.pidfd {
            // SAFETY: pidfd names this process even if its PID has been reused.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    signal,
                    std::ptr::null::<libc::siginfo_t>(),
                    0_u32,
                )
            };
            return if rc == 0 { Ok(true) } else { Err(io::Error::last_os_error()) };
        }
        #[cfg(target_os = "linux")]
        if self.method == StartMethod::ForkServer {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "cannot safely signal forkserver child without pidfd: {}",
                    self.pidfd_error.as_deref().unwrap_or("unavailable")
                ),
            ));
        }
        // SAFETY: a validated positive PID of our unreaped child; caller excludes reaping.
        let rc = unsafe { libc::kill(pid, signal) };
        if rc == 0 { Ok(true) } else { Err(io::Error::last_os_error()) }
    }

    pub fn kill(&self) -> io::Result<()> {
        #[cfg(unix)]
        {
            self.signal(libc::SIGKILL).map(drop)
        }
        #[cfg(windows)]
        {
            signal_pid(self.pid)?;
            if self.already_exited {
                return Ok(());
            }
            // SAFETY: owned process handle remains valid until monitor finishes.
            let rc = unsafe { windows::TerminateProcess(self.handle.as_raw_handle(), 1) };
            if rc != 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
        }
    }
}

enum ProcessIdentity {
    AwaitingPid,
    Process(Arc<ProcessControl>),
}

/// Owns the process and the start method's receipt descriptor until reaping is confirmed.
pub struct ProcessResource {
    identity: ProcessIdentity,
    method: StartMethod,
    #[cfg(unix)]
    sentinel: OwnedFd,
    #[cfg(unix)]
    _bootstrap_fds: Vec<OwnedFd>,
    #[cfg(windows)]
    _bootstrap_handles: Vec<OwnedHandle>,
    receipt: [u8; 8],
    received: usize,
    exitcode: Option<i64>,
}

impl ProcessResource {
    /// Takes the start method's existing finalizer handles from the bootstrap adapter.
    pub fn new(
        pid: u32,
        sentinel: i64,
        method: StartMethod,
        bootstrap_handles: Vec<i64>,
    ) -> io::Result<Self> {
        if (pid == 0 && method != StartMethod::ForkServer) || (cfg!(unix) && pid > i32::MAX as u32)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid process PID"));
        }
        #[cfg(unix)]
        let sentinel = {
            let fd = i32::try_from(sentinel)
                .ok()
                .filter(|fd| *fd >= 0)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid sentinel"))?;
            // SAFETY: bootstrap transfers ownership of this descriptor.
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            // SAFETY: fcntl receives our valid descriptor and plain flags.
            let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
            if flags >= 0 {
                // SAFETY: fcntl receives our valid descriptor and plain flags.
                let _ =
                    unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
            }
            fd
        };
        #[cfg(windows)]
        let handle = {
            if sentinel <= 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid process handle"));
            }
            // SAFETY: bootstrap transfers its duplicated process handle.
            unsafe { OwnedHandle::from_raw_handle(sentinel as *mut std::ffi::c_void) }
        };
        Ok(Self {
            #[cfg(unix)]
            identity: if pid == 0 && method == StartMethod::ForkServer {
                ProcessIdentity::AwaitingPid
            } else {
                ProcessIdentity::Process(Arc::new(ProcessControl::new(pid, method)))
            },
            #[cfg(windows)]
            identity: ProcessIdentity::Process(Arc::new(ProcessControl {
                pid,
                already_exited: false,
                handle,
            })),
            method,
            #[cfg(unix)]
            sentinel,
            #[cfg(unix)]
            _bootstrap_fds: bootstrap_handles
                .into_iter()
                .map(|fd| {
                    // SAFETY: the bootstrap finalizer transferred each distinct owned descriptor.
                    unsafe { OwnedFd::from_raw_fd(fd as i32) }
                })
                .collect(),
            #[cfg(windows)]
            _bootstrap_handles: bootstrap_handles
                .into_iter()
                .map(|handle| {
                    // SAFETY: the bootstrap finalizer transferred each distinct owned handle.
                    unsafe { OwnedHandle::from_raw_handle(handle as *mut std::ffi::c_void) }
                })
                .collect(),
            receipt: [0; 8],
            received: 0,
            exitcode: None,
        })
    }

    pub fn control(&self) -> Option<&Arc<ProcessControl>> {
        match &self.identity {
            ProcessIdentity::AwaitingPid => None,
            ProcessIdentity::Process(control) => Some(control),
        }
    }

    /// Forkserver writes its child PID before its later, separate reaping status. Bootstrap
    /// failure can occur before Python read that PID; the native owner continues that receipt.
    pub fn poll_identity(&mut self) -> io::Result<bool> {
        if self.control().is_some() {
            return Ok(true);
        }
        #[cfg(unix)]
        {
            let Some(pid) = self.read_forkserver_signed()? else {
                return Ok(false);
            };
            let pid = u32::try_from(pid).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "forkserver returned an invalid PID")
            })?;
            signal_pid(pid)?;
            self.identity =
                ProcessIdentity::Process(Arc::new(ProcessControl::new(pid, self.method)));
            self.received = 0;
            Ok(true)
        }
        #[cfg(windows)]
        {
            Err(io::Error::new(io::ErrorKind::Unsupported, "forkserver is Linux only"))
        }
    }

    #[cfg(unix)]
    fn read_forkserver_signed(&mut self) -> io::Result<Option<i64>> {
        if self.received == self.receipt.len() {
            return Ok(Some(i64::from_ne_bytes(self.receipt)));
        }
        let mut ready =
            libc::pollfd { fd: self.sentinel.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one owned receipt descriptor, zero timeout never waits.
        let polled = unsafe { libc::poll(&mut ready, 1, 0) };
        if polled == 0 {
            return Ok(None);
        }
        if polled < 0 {
            return Err(io::Error::last_os_error());
        }
        let data = &mut self.receipt[self.received..];
        // SAFETY: data is writable for its length, with a sole reader and available bytes.
        let count =
            unsafe { libc::read(self.sentinel.as_raw_fd(), data.as_mut_ptr().cast(), data.len()) };
        if count < 0 {
            let error = io::Error::last_os_error();
            return if matches!(error.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted)
            {
                Ok(None)
            } else {
                Err(error)
            };
        }
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "forkserver receipt truncated",
            ));
        }
        self.received += count as usize;
        Ok((self.received == self.receipt.len()).then(|| i64::from_ne_bytes(self.receipt)))
    }

    /// Observe OS exit without consuming the reaping receipt. Zombies are exited, not reaped.
    pub fn poll_exit(&mut self) -> io::Result<bool> {
        let Some(control) = self.control().cloned() else {
            return Ok(false);
        };
        if control.already_exited {
            return Ok(true);
        }
        #[cfg(target_os = "linux")]
        if self.method == StartMethod::ForkServer {
            let Some(fd) = control.pidfd.as_ref() else {
                // ESRCH is an OS exit observation, not a reaping receipt. A zombie still
                // exists for kill(pid, 0), so it cannot be mistaken for a reaped process.
                // SAFETY: a zero signal observes existence without sending a signal.
                if unsafe { libc::kill(control.pid as libc::pid_t, 0) } == 0 {
                    return Ok(false);
                }
                let error = io::Error::last_os_error();
                return if error.raw_os_error() == Some(libc::ESRCH) {
                    Ok(true)
                } else {
                    Err(error)
                };
            };
            let mut poll = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: poll receives one valid pollfd; zero timeout never waits.
            let rc = unsafe { libc::poll(&mut poll, 1, 0) };
            return if rc >= 0 { Ok(rc > 0) } else { Err(io::Error::last_os_error()) };
        }
        #[cfg(unix)]
        {
            // SAFETY: waitid initializes siginfo_t and WNOWAIT retains our child for reaping.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let rc = unsafe {
                libc::waitid(
                    libc::P_PID,
                    control.pid as libc::id_t,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if rc < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: waitid succeeded and initializes the SIGCHLD fields.
            let (pid, status) = unsafe { (info.si_pid(), info.si_status()) };
            if pid == 0 {
                return Ok(false);
            }
            self.exitcode = Some(if info.si_code == libc::CLD_EXITED {
                i64::from(status)
            } else {
                -i64::from(status)
            });
            Ok(true)
        }
        #[cfg(windows)]
        {
            // SAFETY: resource owns the process handle; zero timeout never waits.
            match unsafe { windows::WaitForSingleObject(control.handle.as_raw_handle(), 0) } {
                windows::WAIT_TIMEOUT => return Ok(false),
                windows::WAIT_OBJECT_0 => {}
                _ => return Err(io::Error::last_os_error()),
            }
            let mut code = 0;
            // SAFETY: resource owns the process handle and code is writable.
            let rc =
                unsafe { windows::GetExitCodeProcess(control.handle.as_raw_handle(), &mut code) };
            if rc == 0 {
                return Err(io::Error::last_os_error());
            }
            self.exitcode = Some(i64::from(code));
            Ok(true)
        }
    }

    pub fn exitcode(&self) -> Option<i64> {
        self.exitcode
    }

    /// The OS parent's wait receipt, or the forkserver's separate signed exit-status receipt.
    pub fn poll_reaped(&mut self) -> io::Result<Option<i64>> {
        if self.control().is_none() {
            return Ok(None);
        }
        #[cfg(unix)]
        if self.method == StartMethod::ForkServer {
            let code = self.read_forkserver_signed()?;
            self.exitcode = code;
            return Ok(code);
        }
        let control = self.control().expect("identity observed");
        #[cfg(unix)]
        {
            let mut status = 0;
            // SAFETY: the PID is our unreaped child; WNOHANG never waits.
            let pid =
                unsafe { libc::waitpid(control.pid as libc::pid_t, &mut status, libc::WNOHANG) };
            if pid < 0 {
                return Err(io::Error::last_os_error());
            }
            if pid == 0 {
                return Ok(None);
            }
            let code = if libc::WIFEXITED(status) {
                i64::from(libc::WEXITSTATUS(status))
            } else {
                -i64::from(libc::WTERMSIG(status))
            };
            self.exitcode = Some(code);
            Ok(Some(code))
        }
        #[cfg(windows)]
        {
            // Windows has no Unix zombie/reap phase. A signalled process handle is the
            // multiprocessing spawn wait receipt; ownership ends by closing our handle.
            // SAFETY: valid handle; a zero timeout never waits.
            match unsafe { windows::WaitForSingleObject(control.handle.as_raw_handle(), 0) } {
                windows::WAIT_OBJECT_0 => Ok(self.exitcode),
                windows::WAIT_TIMEOUT => Ok(None),
                _ => Err(io::Error::last_os_error()),
            }
        }
    }
}

/// Only installed in an isolated child process, never in a job thread.
pub fn terminate_current_process() {
    #[cfg(unix)]
    // SAFETY: _exit deliberately ends this isolated child without Python/GIL participation.
    unsafe {
        libc::_exit(1);
    }
    // Not ExitProcess: it runs DLL detach, which can deadlock on a loader or heap lock a dead
    // thread holds, exactly when an orphaned child must end.
    #[cfg(windows)]
    // SAFETY: terminates this isolated child process.
    unsafe {
        windows::TerminateProcess(windows::GetCurrentProcess(), 1);
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    pub const WAIT_OBJECT_0: u32 = 0;
    pub const WAIT_TIMEOUT: u32 = 258;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn TerminateProcess(handle: *mut c_void, code: u32) -> i32;
        pub fn GetExitCodeProcess(handle: *mut c_void, code: *mut u32) -> i32;
        pub fn WaitForSingleObject(handle: *mut c_void, timeout: u32) -> u32;
        pub fn GetCurrentProcess() -> *mut c_void;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::fd::IntoRawFd;
    use std::os::unix::net::UnixStream;
    use std::process::{Child, Command};
    use std::time::Duration;

    struct OwnedChild(Child);

    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn resource(child: &OwnedChild) -> ProcessResource {
        let (sentinel, _other) = UnixStream::pair().unwrap();
        ProcessResource::new(
            child.0.id(),
            sentinel.into_raw_fd().into(),
            StartMethod::Spawn,
            Vec::new(),
        )
        .unwrap()
    }

    fn await_exit(resource: &mut ProcessResource) {
        for _ in 0..1_000 {
            if resource.poll_exit().unwrap() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("child did not exit within the fixture deadline");
    }

    #[test]
    fn forkserver_receipt_reader_distinguishes_fragmented_pid_from_reaping_status() {
        use std::io::Write;
        let mut child = OwnedChild(Command::new("sleep").arg("30").spawn().unwrap());
        let (receipt, mut parent) = UnixStream::pair().unwrap();
        let mut resource = ProcessResource::new(
            0,
            receipt.into_raw_fd().into(),
            StartMethod::ForkServer,
            Vec::new(),
        )
        .unwrap();
        assert!(resource.control().is_none());
        let pid = i64::from(child.0.id()).to_ne_bytes();
        parent.write_all(&pid[..3]).unwrap();
        assert!(!resource.poll_identity().unwrap());
        assert!(resource.control().is_none(), "partial PID is never a signal target");
        parent.write_all(&pid[3..]).unwrap();
        assert!(resource.poll_identity().unwrap());
        assert_eq!(resource.control().unwrap().pid, child.0.id());
        assert_eq!(resource.poll_reaped().unwrap(), None, "PID is not the later wait receipt");
        child.0.kill().unwrap();
        let status = child.0.wait().unwrap();
        use std::os::unix::process::ExitStatusExt;
        let code = -i64::from(status.signal().unwrap());
        parent.write_all(&code.to_ne_bytes()).unwrap();
        assert_eq!(resource.poll_reaped().unwrap(), Some(code));
    }

    #[test]
    fn missing_forkserver_pid_retains_the_receipt_owner() {
        let (receipt, parent) = UnixStream::pair().unwrap();
        let descriptor = receipt.as_raw_fd();
        let mut resource = ProcessResource::new(
            0,
            receipt.into_raw_fd().into(),
            StartMethod::ForkServer,
            Vec::new(),
        )
        .unwrap();
        drop(parent);
        let error = (0..1_000)
            .find_map(|_| match resource.poll_identity() {
                Err(error) => Some(error),
                Ok(false) => {
                    std::thread::sleep(Duration::from_millis(1));
                    None
                }
                Ok(true) => panic!("EOF fabricated a process identity"),
            })
            .expect("EOF was not observed within the fixture deadline");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert!(resource.control().is_none());
        // SAFETY: F_GETFD observes whether the still-owned descriptor remains open.
        assert!(unsafe { libc::fcntl(descriptor, libc::F_GETFD) } >= 0);
        drop(resource);
    }

    #[test]
    fn process_signals_require_a_positive_process_identity() {
        assert_eq!(signal_pid(0).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(signal_pid(u32::MAX).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(signal_pid(42).unwrap(), 42);
    }

    #[test]
    fn native_exit_observation_does_not_reap_a_zombie() {
        let child = OwnedChild(Command::new("sh").args(["-c", "exit 17"]).spawn().unwrap());
        let mut process = resource(&child);
        await_exit(&mut process);
        assert_eq!(process.exitcode(), Some(17));
        assert!(process.poll_exit().unwrap(), "exit observation must leave the receipt available");
        assert_eq!(process.poll_reaped().unwrap(), Some(17));
        assert_eq!(process.poll_exit().unwrap_err().raw_os_error(), Some(libc::ECHILD));
    }

    #[test]
    fn native_termination_does_not_need_a_host_callback() {
        let child = OwnedChild(Command::new("sleep").arg("30").spawn().unwrap());
        let mut process = resource(&child);
        assert!(!process.poll_exit().unwrap());
        process.control().unwrap().kill().unwrap();
        await_exit(&mut process);
        assert_eq!(process.poll_reaped().unwrap(), Some(-i64::from(libc::SIGKILL)));
    }

    /// A forkserver child whose reaping receipt the test writes on `receipt`.
    fn forkserver_runner(child: &OwnedChild) -> (JobProcess, UnixStream) {
        let (sentinel, receipt) = UnixStream::pair().unwrap();
        let host = Arc::new(crate::ipc::pool::tests::Numbered(0));
        let resource = ProcessResource::new(
            child.0.id(),
            sentinel.into_raw_fd().into(),
            StartMethod::ForkServer,
            Vec::new(),
        )
        .unwrap();
        (JobProcess::watch(LogWriter::disabled(), host, resource, None), receipt)
    }

    async fn pending(runner: &JobProcess) -> bool {
        tokio::time::timeout(POLL * 5, runner.finished()).await.is_err()
    }

    fn write_receipt(receipt: &mut UnixStream, code: i64) {
        use std::io::Write;
        receipt.write_all(&code.to_ne_bytes()).unwrap();
    }

    // Catches: an OS exit ending the runner before the start method's reaping receipt, and the
    // exit code taken from the OS observation instead of the receipt.
    #[tokio::test]
    async fn finished_waits_for_the_reaping_receipt_and_reports_its_code() {
        let child = OwnedChild(Command::new("sh").args(["-c", "exit 17"]).spawn().unwrap());
        let (runner, mut receipt) = forkserver_runner(&child);
        for _ in 0..1_000 {
            if runner.process.lock().exited {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(runner.process.lock().exited, "child did not exit within the fixture deadline");
        assert!(pending(&runner).await, "an OS exit alone released the process");
        write_receipt(&mut receipt, 3);
        let exit = tokio::time::timeout(Duration::from_secs(5), runner.finished()).await.unwrap();
        assert_eq!(exit, Exit { code: 3, killed: false });
    }

    // Catches: a reaping receipt releasing a process the OS has not seen exit.
    #[tokio::test]
    async fn a_receipt_without_an_os_exit_keeps_the_process_owned() {
        let mut child = OwnedChild(Command::new("sleep").arg("30").spawn().unwrap());
        let (runner, mut receipt) = forkserver_runner(&child);
        write_receipt(&mut receipt, 9);
        assert!(pending(&runner).await, "a receipt released a live process");
        child.0.kill().unwrap();
        let exit = tokio::time::timeout(Duration::from_secs(5), runner.finished()).await.unwrap();
        assert_eq!(exit, Exit { code: 9, killed: false });
    }
}

/// The process's footprint in MB: its PSS where Linux reports it, else its RSS. Blocking: called
/// only on the supervisor's bounded sampling workers.
fn sample_memory(pid: u32) -> io::Result<f64> {
    #[cfg(target_os = "linux")]
    if let Ok(rollup) = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup"))
        && let Some(pss_kb) = rollup.lines().find_map(|line| {
            line.strip_prefix("Pss:")?.split_whitespace().next()?.parse::<f64>().ok()
        })
    {
        return Ok(pss_kb / 1024.0);
    }
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    let pid = Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let process = system
        .process(pid)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "process no longer exists"))?;
    Ok(process.memory() as f64 / (1024.0 * 1024.0))
}

/// A job process of the host's start method: the host spawns it on the job's end of the channel
/// and reports its receipt through [`ProcessExecutor::launched`].
pub struct ProcessExecutor {
    logging: LogWriter,
    host: Arc<dyn ipc_host::JobExecutor>,
    context: CallContext,
    receipt: Mutex<Option<ProcessResource>>,
    reported: Notify,
}

impl ProcessExecutor {
    pub fn new(
        host: Arc<dyn ipc_host::JobExecutor>,
        context: CallContext,
        logging: LogWriter,
    ) -> Self {
        Self { logging, host, context, receipt: Mutex::new(None), reported: Notify::new() }
    }

    /// The launch's receipt, reported on the launch worker right after the OS created the child.
    pub fn launched(
        &self,
        pid: u32,
        sentinel: i64,
        method: &str,
        owned: Vec<i64>,
    ) -> io::Result<()> {
        let resource = ProcessResource::new(pid, sentinel, StartMethod::parse(method)?, owned)?;
        let mut receipt = self.receipt.lock();
        if receipt.is_some() {
            return Err(io::Error::other("launch already reported"));
        }
        *receipt = Some(resource);
        self.reported.notify_one();
        Ok(())
    }
}

#[async_trait::async_trait]
impl RunnerExecutor for ProcessExecutor {
    /// Returns once a process exists to own: a host launch failing after its receipt kills it.
    async fn launch(&self) -> Result<(Box<dyn Runner>, Connection), String> {
        let (parent, child) = ipc_channel::pair().map_err(|error| error.to_string())?;
        let connection = Channel::from_endpoint(parent).map_err(|error| error.to_string())?;
        let (host, context) = (self.host.clone(), self.context.clone());
        let mut launching =
            TokioSpawner::current().spawn(async move { host.launch(&context, child).await });
        let launch = tokio::select! {
            biased;
            () = self.reported.notified() => None,
            launched = &mut launching => Some(launched),
        };
        let Some(resource) = self.receipt.lock().take() else {
            return Err(match launch {
                Some(Ok(Ok(()))) => "the launch reported no process".into(),
                Some(Ok(Err(error))) => error.to_string(),
                Some(Err(error)) => error.to_string(),
                None => unreachable!("a receipt was reported"),
            });
        };
        let launching = launch.is_none().then_some(launching);
        Ok((
            Box::new(JobProcess::watch(self.logging, self.host.clone(), resource, launching)),
            connection,
        ))
    }
}

/// The launched process, its termination request and what was observed of it.
struct Tracked {
    resource: ProcessResource,
    stopping: bool,
    killed: bool,
    exited: bool,
    /// An observation or signal failure was reported (once).
    failed: bool,
}

impl Tracked {
    fn poll(&mut self, host: &Arc<dyn ipc_host::JobExecutor>, logging: LogWriter) -> Option<Exit> {
        self.observe().unwrap_or_else(|error| {
            self.report(host, logging, format!("native process observation failed: {error}"));
            None
        })
    }

    /// The first failure only: the watcher keeps retrying.
    fn report(
        &mut self,
        host: &Arc<dyn ipc_host::JobExecutor>,
        logging: LogWriter,
        message: String,
    ) {
        if !std::mem::replace(&mut self.failed, true) {
            queue_log(logging, host.clone(), Level::Error, &message, Vec::new(), None);
        }
    }

    fn observe(&mut self) -> io::Result<Option<Exit>> {
        if !self.resource.poll_identity()? {
            return Ok(None);
        }
        if self.stopping {
            self.kill()?;
        }
        if !self.exited {
            self.exited = self.resource.poll_exit()?;
            if !self.exited {
                return Ok(None);
            }
        }
        let killed = self.killed;
        Ok(self.resource.poll_reaped()?.map(|code| Exit { code, killed }))
    }

    /// Kills it once its identity is known and until a signal was delivered.
    fn kill(&mut self) -> io::Result<()> {
        if self.killed || self.exited {
            return Ok(());
        }
        let Some(control) = self.resource.control() else { return Ok(()) };
        control.kill()?;
        self.killed = true;
        Ok(())
    }
}

/// Polls the process until its reaping receipt, finishing the host's launch on the way.
async fn watch(
    process: Arc<Mutex<Tracked>>,
    mut launching: Option<OwnedTask<Result<(), HostError>>>,
    exit: watch::Sender<Option<Exit>>,
    host: Arc<dyn ipc_host::JobExecutor>,
    logging: LogWriter,
) {
    loop {
        if let Some(launch) = launching.as_mut().filter(|launch| launch.is_finished()) {
            let cause = match launch.await {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error.to_string()),
                Err(error) => Some(error.to_string()),
            };
            launching = None;
            if let Some(cause) = cause {
                let message = format!("process launch failed: {cause}");
                queue_log(logging, host.clone(), Level::Error, &message, Vec::new(), None);
                process.lock().stopping = true;
            }
        }
        if let Some(code) = process.lock().poll(&host, logging) {
            exit.send_replace(Some(code));
            return;
        }
        tokio::time::sleep(POLL).await;
    }
}

const POLL: Duration = Duration::from_millis(10);

struct JobProcess {
    logging: LogWriter,
    host: Arc<dyn ipc_host::JobExecutor>,
    process: Arc<Mutex<Tracked>>,
    finished: watch::Receiver<Option<Exit>>,
    _watching: OwnedTask,
}

impl JobProcess {
    fn watch(
        logging: LogWriter,
        host: Arc<dyn ipc_host::JobExecutor>,
        resource: ProcessResource,
        launching: Option<OwnedTask<Result<(), HostError>>>,
    ) -> Self {
        let process = Arc::new(Mutex::new(Tracked {
            resource,
            stopping: false,
            killed: false,
            exited: false,
            failed: false,
        }));
        let (exit, finished) = watch::channel(None);
        let watching = TokioSpawner::current().spawn(watch(
            process.clone(),
            launching,
            exit,
            host.clone(),
            logging,
        ));
        Self { logging, host, process, finished, _watching: watching }
    }
}

#[async_trait::async_trait]
impl Runner for JobProcess {
    async fn finished(&self) -> Exit {
        let exit =
            self.finished.clone().wait_for(Option::is_some).await.ok().and_then(|exit| *exit);
        match exit {
            Some(exit) => exit,
            // the watcher was dropped with the runner: it never finishes
            None => std::future::pending().await,
        }
    }

    fn stop(&self) -> bool {
        let mut process = self.process.lock();
        process.stopping = true;
        if let Err(error) = process.kill() {
            process.report(
                &self.host,
                self.logging,
                format!("process termination failed: {error}"),
            );
        }
        true
    }

    fn memory_mb(&self) -> Option<f64> {
        let pid = self.process.lock().resource.control()?.pid;
        let error = match sample_memory(pid) {
            Ok(mb) => return Some(mb),
            Err(error) => error,
        };
        if !self.process.lock().exited {
            let message = format!("failed to get memory info for process: {error}");
            queue_log(self.logging, self.host.clone(), Level::Warning, &message, Vec::new(), None);
        }
        None
    }
}

/// Process bootstrap exports; IPC ownership stays with the worker.
pub struct Bootstrap;

impl lk_agents_abi::worker::ipc_host::ProcessBootstrap for Bootstrap {
    fn ipc_pair() -> Result<(i64, i64), lk_agents_abi::errors::HostError> {
        super::ipc_channel::pair()
            .map_err(|error| lk_agents_abi::errors::HostError::new(error.to_string()))
    }
}
