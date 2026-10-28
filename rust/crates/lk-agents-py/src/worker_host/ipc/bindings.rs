//! Conversions between the declared process classes and Python bootstrap objects.
use super::*;
use lk_agents_abi::python::{capture_host, intern_host, resolve_key};
use lk_agents_abi::{HostRef, errors::HostError, worker::ipc_host as port};

fn result<T>(call: impl FnOnce(Python<'_>) -> PyResult<T>) -> Result<T, HostError> {
    Python::attach(call).map_err(|error| Python::attach(|py| capture_host(py, &error)))
}

fn reference(
    call: impl for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>>,
) -> Result<HostRef, HostError> {
    result(|py| call(py).map(|value| intern_host(value.unbind())))
}

impl port::WorkerProcSupervisor for PyProcSupervisor {
    fn new(
        fd: i64,
        host: HostRef,
        pid: u32,
        initialize_timeout: f64,
        close_timeout: f64,
        memory_warn_mb: f64,
        memory_limit_mb: f64,
        ping_interval: f64,
        ping_timeout: f64,
        high_ping_threshold: f64,
        killable: bool,
    ) -> Result<Self, HostError> {
        result(|py| {
            Self::new(
                fd,
                resolve_key(py, host.id())?,
                pid,
                initialize_timeout,
                close_timeout,
                memory_warn_mb,
                memory_limit_mb,
                ping_interval,
                ping_timeout,
                high_ping_threshold,
                killable,
            )
        })
    }

    fn initialize(
        &self,
        asyncio_debug: bool,
        http_proxy: Option<String>,
    ) -> Result<HostRef, HostError> {
        reference(|py| self.initialize(py, asyncio_debug, http_proxy))
    }
    fn initialized(&self) -> bool {
        self.initialized()
    }
    fn _core_launched_process(
        &self,
        pid: u32,
        sentinel: i64,
        method: String,
        owned: Vec<i64>,
    ) -> Result<(), HostError> {
        result(|_| self._core_launched_process(pid, sentinel, &method, owned))
    }
    fn run(&self, joined: HostRef, host: HostRef) -> Result<HostRef, HostError> {
        reference(|py| self.run(py, resolve_key(py, joined.id())?, resolve_key(py, host.id())?))
    }
    fn _core_wait_exit(&self) -> Result<HostRef, HostError> {
        reference(|py| self._core_wait_exit(py))
    }
    fn _core_join(&self) -> Result<HostRef, HostError> {
        reference(|py| self._core_join(py))
    }
    fn aclose(&self) -> Result<HostRef, HostError> {
        reference(|py| self.aclose(py))
    }
    fn kill(&self) -> Result<HostRef, HostError> {
        reference(|py| self.kill(py))
    }
    fn start_job(
        &self,
        fields: lk_agents_abi::worker::job::StartJobFields,
    ) -> Result<HostRef, HostError> {
        reference(|py| self.start_job(py, fields))
    }
    fn killed(&self) -> bool {
        self.killed()
    }
}

impl port::WorkerIpcClient for PyIpcClient {
    fn new(fd: i64, initialize: HostRef, process: bool) -> Result<Self, HostError> {
        result(|py| Self::new(py, fd, resolve_key(py, initialize.id())?.bind(py).clone(), process))
    }
    fn asyncio_debug(&self) -> bool {
        self.asyncio_debug()
    }
    fn serve(&self, host: HostRef) -> Result<HostRef, HostError> {
        reference(|py| self.serve(py, resolve_key(py, host.id())?))
    }
    fn stop(&self) {
        self.stop();
    }
    fn aclose(&self) -> Result<HostRef, HostError> {
        reference(|py| self.aclose(py))
    }
}

impl port::WorkerProcPool for PyProcPool {
    fn new(
        host: HostRef,
        num_idle_processes: usize,
        initialize_timeout: f64,
        close_timeout: f64,
    ) -> Result<Self, HostError> {
        result(|py| {
            Self::new(
                py,
                resolve_key(py, host.id())?,
                num_idle_processes,
                initialize_timeout,
                close_timeout,
            )
        })
    }
    fn processes(&self) -> Vec<HostRef> {
        Python::attach(|py| self.processes(py).into_iter().map(intern_host).collect())
    }
    fn jobs_waiting(&self) -> usize {
        self.jobs_waiting()
    }
    fn target_idle_processes(&self) -> usize {
        self.target_idle_processes()
    }
    fn set_target_idle_processes(&self, n: usize) {
        self.set_target_idle_processes(n);
    }
    fn start(&self) -> Result<HostRef, HostError> {
        reference(|py| self.start(py))
    }
    fn launch_job(&self, info: HostRef) -> Result<HostRef, HostError> {
        reference(|py| self.launch_job(py, resolve_key(py, info.id())?))
    }
    fn join_jobs(&self) -> Result<HostRef, HostError> {
        reference(|py| self.join_jobs(py))
    }
    fn aclose(&self) -> Result<HostRef, HostError> {
        reference(|py| self.aclose(py))
    }
}

impl port::WorkerJobRun for PyJobRun {
    fn new(
        fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        url: String,
        recording: bool,
        session_end: Option<HostRef>,
        session_aclose_timeout: f64,
    ) -> Result<Self, HostError> {
        result(|py| {
            let session_end =
                session_end.map(|reference| resolve_key(py, reference.id())).transpose()?;
            Ok(Self::new(
                fake_job,
                session_end_timeout,
                redaction,
                url,
                recording,
                session_end.as_ref().map(|value| value.bind(py).clone()),
                session_aclose_timeout,
            ))
        })
    }
    fn add_shutdown_callback(&self, callback: HostRef) -> Result<(), HostError> {
        result(|py| self.add_shutdown_callback(resolve_key(py, callback.id())?.bind(py).clone()))
    }
    fn observability_url(&self) -> Option<String> {
        self.observability_url()
    }
    fn buffer_logs(&self) -> bool {
        self.buffer_logs()
    }
    fn shutdown(&self, reason: String, user_initiated: bool) {
        self.shutdown(reason, user_initiated);
    }
    fn connected(&self) {
        self.connected();
    }
    fn connect(&self, connect: HostRef) -> Result<HostRef, HostError> {
        reference(|py| self.connect(py, resolve_key(py, connect.id())?))
    }
    fn add_participant_entrypoint(
        &self,
        entrypoint: HostRef,
        kinds: Vec<i32>,
    ) -> Result<(), HostError> {
        result(|py| self.add_participant_entrypoint(py, resolve_key(py, entrypoint.id())?, kinds))
    }
    fn participant_joined(
        &self,
        ctx: HostRef,
        participant: HostRef,
        identity: String,
        kind: i32,
    ) -> Result<(), HostError> {
        result(|py| {
            self.participant_joined(
                py,
                resolve_key(py, ctx.id())?,
                resolve_key(py, participant.id())?,
                identity,
                kind,
            )
        })
    }
    fn participant_left(&self, attributes: std::collections::BTreeMap<String, String>) {
        self.participant_left(attributes.into_iter().collect());
    }
    fn api_call(
        &self,
        call: HostRef,
        name: String,
        console_result: HostRef,
    ) -> Result<HostRef, HostError> {
        reference(|py| {
            self.api_call(
                py,
                resolve_key(py, call.id())?,
                name,
                resolve_key(py, console_result.id())?,
            )
        })
    }
    fn close_tasks(&self) -> Result<HostRef, HostError> {
        reference(|py| self.close_tasks(py))
    }
    fn init_recording(
        &self,
        options: std::collections::BTreeMap<String, bool>,
    ) -> Result<Option<bool>, HostError> {
        result(|_| self.init_recording(options.into_iter().collect()))
    }
    fn claim_session(&self, session: HostRef) -> Result<bool, HostError> {
        result(|py| Ok(self.claim_session(py, resolve_key(py, session.id())?)))
    }
    fn primary_session(&self) -> Option<HostRef> {
        Python::attach(|py| self.primary_session(py).map(intern_host))
    }
    fn recording_initialized(&self) -> bool {
        self.recording_initialized()
    }
    fn redaction(&self) -> bool {
        self.redaction()
    }
    fn run(&self, host: HostRef) -> Result<HostRef, HostError> {
        reference(|py| self.run(py, resolve_key(py, host.id())?))
    }
}
