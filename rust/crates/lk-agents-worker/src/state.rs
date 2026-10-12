//! Worker registration, job lifecycle and drain: sans-IO, inputs in, effects out.

use crate::LOG;
use crate::config::*;
use crate::load::{LoadInputs, WorkerStatus};
use crate::protocol::{
    self, AvailabilityRequest, Job, JobAssignment, JobStatus, RegisterWorkerResponse,
    ServerMessage, WorkerMessage,
};
use crate::worker::{HostError, JobId};
use crate::{AcceptArguments, JobDecision, JobRequest, RunningJobInfo};
use lk_agents_abi::{TimerSet, Ts};
use std::collections::VecDeque;
use std::time::Duration;

/// Interned because [`TimerId`](lk_agents_abi::TimerId) requires `Copy` and `Job.id` is a `String`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(pub u32);

/// `Assignment` is the only per-job timer, bounded by the reserved slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkerTimer {
    /// `UPDATE_LOAD_INTERVAL`, 0.5 s. Lives for the whole worker, connected or not.
    LoadSample,
    /// `UPDATE_STATUS_INTERVAL`, 2.5 s. Per-connection: armed at register, cancelled on disconnect.
    StatusUpdate,
    DrainDeadline,
    /// `ASSIGNMENT_TIMEOUT`, 7.5 s, one per accepted-but-unassigned job.
    Assignment(Slot),
}

/// Work requested by [`WorkerState`], executed by the runtime.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStateEffect {
    Send(WorkerMessage),
    /// Call `WorkerHost::on_job_request`, then feed the answer back as
    /// [`WorkerStateInput::RequestDecision`] with the same slot.
    AskUser { slot: Slot, request: crate::JobRequest },
    /// Call `WorkerHost::launch_job`, then feed [`WorkerStateInput::JobLaunched`] back.
    LaunchJob { slot: Slot, info: Box<RunningJobInfo> },
    TerminateJob(JobId),
    /// Sample the load (user `load_fnc`, else cgroup CPU) and feed [`WorkerStateInput::LoadSampled`].
    SampleLoad,
    SetTargetIdleProcesses(u32),
    /// Emit a registration notification.
    Registered {
        worker_id: String,
        region: String,
        protocol: i32,
        deployment: String,
        agent_name: String,
    },
    /// `drain()` finished within its deadline.
    DrainFinished,
    /// `drain()` hit `drain_timeout`. The caller is still expected to close the worker afterwards.
    DrainTimedOut,
}

/// Events consumed by [`WorkerState`], supplied by the runtime.
#[allow(clippy::large_enum_variant)] // see `WorkerStateEffect`.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStateInput {
    /// The driver has a socket and is about to send `register`. Gates `update_worker`/`ping`.
    Connecting,
    /// The first `ServerMessage` came back and it was a `register`.
    Registered(RegisterWorkerResponse),
    Server(ServerMessage),
    RequestDecision { slot: Slot, decision: JobDecision },
    /// The transport finished writing an acceptance; start waiting for its assignment.
    AvailabilitySent { job_id: JobId },
    JobLaunched { slot: Slot, result: Result<JobId, HostError> },
    /// A launched job changed state. `JsSuccess`/`JsFailed` remove it from `active_jobs`.
    JobStatus { job_id: JobId, status: JobStatus },
    LoadSampled(f32),
    /// The socket closed. Per-connection timers stop; assignment timers keep running.
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConnState {
    Disconnected,
    /// Registration is pending; periodic status messages are suppressed.
    Connecting,
    Registered,
}

#[derive(Debug, Clone, PartialEq)]
enum PendingState {
    /// Waiting for a fresh load sample before checking capacity.
    AwaitingLoad,
    AwaitingUser,
    /// Accepted; waiting for `JobAssignment` under the 7.5 s timer.
    AwaitingAssignment(AcceptArguments),
    Launching(Box<RunningJobInfo>),
}

#[derive(Debug, Clone, PartialEq)]
struct Pending {
    slot: Slot,
    /// The job from the **`AvailabilityRequest`**. See [`WorkerState::on_assignment`].
    job: Job,
    resuming: bool,
    state: PendingState,
    early_status: Option<JobStatus>,
}

#[derive(Debug)]
pub struct WorkerState {
    cfg: WorkerConfig,
    timers: TimerSet<WorkerTimer>,
    conn: ConnState,
    worker_id: String,
    worker_load: f32,
    pending: Vec<Pending>,
    /// Requests parked until the next load sample, in arrival order.
    awaiting_load: VecDeque<Slot>,
    active: Vec<RunningJobInfo>,
    next_slot: u32,
    draining: bool,
    drain_reported: bool,
    previous_status: Option<WorkerStatus>,
    out: Vec<WorkerStateEffect>,
}

impl WorkerState {
    pub fn new(cfg: WorkerConfig) -> Self {
        WorkerState {
            cfg,
            timers: TimerSet::new(),
            conn: ConnState::Disconnected,
            worker_id: String::new(),
            worker_load: 0.0,
            pending: Vec::new(),
            awaiting_load: VecDeque::new(),
            active: Vec::new(),
            next_slot: 0,
            draining: false,
            drain_reported: false,
            previous_status: None,
            out: Vec::new(),
        }
    }


    pub fn config(&self) -> &WorkerConfig {
        &self.cfg
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }
    /// The server has answered the registration on the current connection.
    pub fn registered(&self) -> bool {
        self.conn == ConnState::Registered
    }
    pub fn draining(&self) -> bool {
        self.draining
    }
    pub fn worker_load(&self) -> f32 {
        self.worker_load
    }
    pub fn active_jobs(&self) -> &[RunningJobInfo] {
        &self.active
    }
    /// Requests waiting for a load sample do not reserve capacity.
    pub fn reserved_slots(&self) -> usize {
        self.pending.iter().filter(|p| p.state != PendingState::AwaitingLoad).count()
    }
    /// `None` means nothing is armed.
    pub fn next_deadline(&self) -> Option<Ts> {
        self.timers.next_deadline()
    }

    fn load_inputs(&self) -> LoadInputs {
        LoadInputs {
            worker_load: self.worker_load,
            active_jobs: self.active.len(),
            reserved_slots: self.reserved_slots(),
            load_threshold: self.cfg.load_threshold.get(self.cfg.devmode),
            num_idle_processes: self.cfg.num_idle_processes.get(self.cfg.devmode),
            draining: self.draining,
            simulation: self.cfg.simulation,
        }
    }


    /// Arms the 0.5 s load timer, which lives outside any connection.
    pub fn start(&mut self, now: Ts) -> Vec<WorkerStateEffect> {
        self.timers.arm(WorkerTimer::LoadSample, now.saturating_add(UPDATE_LOAD_INTERVAL));
        self.push(WorkerStateEffect::SampleLoad);
        self.take()
    }

    /// Idempotent; `None` or zero disables the deadline.
    pub fn drain(&mut self, now: Ts, timeout: Option<Duration>) -> Vec<WorkerStateEffect> {
        if self.draining {
            return Vec::new();
        }
        self.draining = true;
        log::info!(target: LOG, agent_name = self.cfg.agent_name.as_str(), id = self.worker_id.as_str(),
            timeout = timeout.map(|d| d.as_secs_f64()); "draining worker");
        // The draining status goes out immediately, before the wait.
        let msg = protocol::update_worker_draining(self.active.len() as u32);
        self.queue(msg);
        if let Some(d) = timeout.filter(|d| !d.is_zero()) {
            self.timers.arm(WorkerTimer::DrainDeadline, now.saturating_add(d));
        }
        self.check_drain_complete();
        self.take()
    }

    pub fn tick(&mut self, now: Ts) -> Vec<WorkerStateEffect> {
        while let Some(t) = self.timers.pop_due(now) {
            match t {
                WorkerTimer::LoadSample => {
                    self.timers
                        .arm(WorkerTimer::LoadSample, now.saturating_add(UPDATE_LOAD_INTERVAL));
                    self.push(WorkerStateEffect::SampleLoad);
                }
                WorkerTimer::StatusUpdate => {
                    self.timers
                        .arm(WorkerTimer::StatusUpdate, now.saturating_add(UPDATE_STATUS_INTERVAL));
                    self.send_worker_status();
                }
                WorkerTimer::DrainDeadline => {
                    self.drain_reported = true;
                    self.push(WorkerStateEffect::DrainTimedOut);
                }
                WorkerTimer::Assignment(slot) => self.on_assignment_timeout(slot),
            }
        }
        self.take()
    }

    pub fn handle(&mut self, now: Ts, ev: WorkerStateInput) -> Vec<WorkerStateEffect> {
        match ev {
            WorkerStateInput::Connecting => {
                self.conn = ConnState::Connecting;
                let msg = protocol::register(
                    self.cfg.server_type,
                    &self.cfg.agent_name,
                    &self.cfg.deployment,
                    &self.cfg.version,
                    &self.cfg.permissions,
                    &self.cfg.endpoints,
                    &self.cfg.instance_id,
                );
                // Sent directly, not queued: `register` must bypass the connecting-state drop.
                self.push(WorkerStateEffect::Send(msg));
            }
            WorkerStateInput::Registered(resp) => self.on_registered(now, resp),
            WorkerStateInput::Server(msg) => self.on_server(now, msg),
            WorkerStateInput::RequestDecision { slot, decision } => {
                self.on_decision(now, slot, decision)
            }
            WorkerStateInput::AvailabilitySent { job_id } => {
                if let Some(p) = self.pending.iter().find(|p| {
                    p.job.id == job_id && matches!(p.state, PendingState::AwaitingAssignment(_))
                }) {
                    self.timers.arm(
                        WorkerTimer::Assignment(p.slot),
                        now.saturating_add(ASSIGNMENT_TIMEOUT),
                    );
                }
            }
            WorkerStateInput::JobLaunched { slot, result } => self.on_launched(slot, result),
            WorkerStateInput::JobStatus { job_id, status } => self.on_job_status(job_id, status),
            WorkerStateInput::LoadSampled(load) => self.on_load(now, load),
            WorkerStateInput::Disconnected => {
                self.conn = ConnState::Disconnected;
                self.timers.cancel(WorkerTimer::StatusUpdate);
                // Decisions and assignments belong to the lost connection. Launches already
                // handed to the host must still complete and become active jobs.
                let stale: Vec<_> = self
                    .pending
                    .iter()
                    .filter(|p| !matches!(p.state, PendingState::Launching(_)))
                    .map(|p| p.slot)
                    .collect();
                for slot in stale {
                    self.drop_pending(slot);
                }
            }
        }
        self.check_drain_complete();
        self.take()
    }


    fn on_registered(&mut self, now: Ts, resp: RegisterWorkerResponse) {
        self.worker_id = resp.worker_id.clone();
        self.conn = ConnState::Registered;
        let (region, protocol) =
            resp.server_info.map_or((String::new(), 0), |s| (s.region, s.protocol));
        log::info!(target: LOG, agent_name = self.cfg.agent_name.as_str(), deployment = self.cfg.deployment.as_str(),
            id = resp.worker_id.as_str(), url = self.cfg.ws_url.as_str(), region = region.as_str(), protocol;
            "registered worker");
        self.push(WorkerStateEffect::Registered {
            worker_id: resp.worker_id,
            region,
            protocol,
            deployment: self.cfg.deployment.clone(),
            agent_name: self.cfg.agent_name.clone(),
        });
        self.timers.arm(WorkerTimer::StatusUpdate, now.saturating_add(UPDATE_STATUS_INTERVAL));
    // Report running jobs after registration so the server can migrate their ownership.
        if !self.active.is_empty() {
            let ids = self.active.iter().map(|j| j.job.id.clone()).collect();
            self.queue(protocol::migrate_jobs(ids));
        }
        self.send_worker_status();
    }

    fn on_server(&mut self, now: Ts, msg: ServerMessage) {
        use livekit_protocol::server_message::Message as M;
        match msg.message {
            Some(M::Availability(req)) => self.on_availability(req),
            Some(M::Assignment(a)) => self.on_assignment(a),
            Some(M::Termination(t)) => {
                // "safe to ignore" when we do not hold the job.
                if self.active.iter().any(|j| j.job.id == t.job_id) {
                    self.push(WorkerStateEffect::TerminateJob(t.job_id));
                }
            }
            Some(M::Register(r)) => self.on_registered(now, r),
            // The server is draining; reconnecting is the transport's job, so nothing to decide.
            Some(M::GoAway(_)) => {}
            Some(M::Pong(_)) | None => {}
        }
    }

    /// Parked until a fresh load sample ([`Self::on_load`]), so a burst is not judged on stale load.
    fn on_availability(&mut self, req: AvailabilityRequest) {
        let Some(job) = req.job else { return };
        let slot = self.alloc_slot();
        self.pending.push(Pending {
            slot,
            job,
            resuming: req.resuming,
            state: PendingState::AwaitingLoad,
            early_status: None,
        });
        self.awaiting_load.push_back(slot);
        self.push(WorkerStateEffect::SampleLoad);
    }

    fn on_load(&mut self, now: Ts, load: f32) {
        self.worker_load = load;
        if let Some(target) = self.load_inputs().target_idle_processes() {
            self.push(WorkerStateEffect::SetTargetIdleProcesses(target));
        }
        let parked: Vec<Slot> = self.awaiting_load.drain(..).collect();
        for slot in parked {
            self.resume_availability(now, slot);
        }
    }

    fn resume_availability(&mut self, _now: Ts, slot: Slot) {
        // This request is still `AwaitingLoad`, so it does not count itself.
        let available = self.load_inputs().is_available();

        let Some(p) = self.pending.iter_mut().find(|p| p.slot == slot) else { return };
        if !available {
            let job_id = p.job.id.clone();
            self.drop_pending(slot);
            // Rejected on load: `terminate` stays false.
            self.queue(protocol::availability_reject(&job_id, false));
            return;
        }
        p.state = PendingState::AwaitingUser;
        let request = JobRequest { job: protocol::job_info(&p.job), resuming: p.resuming };
        log::info!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = request.job.id.as_str(),
            dispatch_id = request.job.dispatch_id.as_str(), room = request.room_name(), resuming = request.resuming;
            "received job request");
        self.push(WorkerStateEffect::AskUser { slot, request });
    }

    fn on_decision(&mut self, _now: Ts, slot: Slot, decision: JobDecision) {
        let Some(p) = self.pending.iter().find(|p| p.slot == slot) else { return };
        let job_id = p.job.id.clone();
        match decision {
            JobDecision::Accept(args) => {
                let msg = protocol::availability_accept(&job_id, &self.cfg.agent_name, &args);
                if let Some(p) = self.pending.iter_mut().find(|p| p.slot == slot) {
                    p.state = PendingState::AwaitingAssignment(args);
                }
                self.queue(msg);
            }
            JobDecision::Reject { terminate } => {
                self.drop_pending(slot);
                self.queue(protocol::availability_reject(&job_id, terminate));
            }
            // Both are the "not answered" path: reject with `terminate=false`, unlike an explicit
            // `reject()`.
            JobDecision::Failed(err) => {
                log::error!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = job_id.as_str(),
                    error = err.to_string().as_str(); "job_request_fnc failed, rejecting the job");
                self.drop_pending(slot);
                self.queue(protocol::availability_reject(&job_id, false));
            }
            JobDecision::Unanswered => {
                log::warn!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = job_id.as_str();
                    "no answer was given inside the job_request_fnc, automatically rejecting the job");
                self.drop_pending(slot);
                self.queue(protocol::availability_reject(&job_id, false));
            }
        }
    }

    /// An unset or empty assignment URL falls back to the worker URL.
    fn on_assignment(&mut self, a: JobAssignment) {
        let job_id = a.job.as_ref().map_or(String::new(), |j| j.id.clone());
        let found = self.pending.iter().position(|p| {
            p.job.id == job_id && matches!(p.state, PendingState::AwaitingAssignment(_))
        });
        let Some(idx) = found else {
            log::warn!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = job_id.as_str();
                "received assignment for an unknown job");
            return;
        };
        let slot = self.pending[idx].slot;
        self.timers.cancel(WorkerTimer::Assignment(slot));
        let PendingState::AwaitingAssignment(args) = &self.pending[idx].state else {
            unreachable!("checked above")
        };
        let args = args.clone();
        let url = match a.url {
            Some(u) if !u.is_empty() => u,
            _ => self.cfg.ws_url.clone(),
        };
        let info = RunningJobInfo {
            accept_arguments: args,
            job: protocol::job_info(&self.pending[idx].job),
            url,
            token: a.token,
            worker_id: self.worker_id.clone(),
            fake_job: false,
        };
        self.pending[idx].state = PendingState::Launching(Box::new(info.clone()));
        self.push(WorkerStateEffect::LaunchJob { slot, info: Box::new(info) });
    }

    /// Drops the pending job and warns; nothing is sent to the server.
    fn on_assignment_timeout(&mut self, slot: Slot) {
        let Some(p) = self.pending.iter().find(|p| p.slot == slot) else { return };
        let job_id = p.job.id.clone();
        self.drop_pending(slot);
        log::warn!(target: LOG, agent_name = self.cfg.agent_name.as_str(); "assignment for job {job_id} timed out");
    }

    fn on_launched(&mut self, slot: Slot, result: Result<JobId, HostError>) {
        let Some(idx) = self.pending.iter().position(|p| p.slot == slot) else { return };
        let p = self.pending.remove(idx);
        self.timers.cancel(WorkerTimer::Assignment(slot));
        match result {
            Ok(_) => {
                if let PendingState::Launching(info) = p.state {
                    let job_id = info.job.id.clone();
                    if info.worker_id != self.worker_id && self.conn == ConnState::Registered {
                        self.queue(protocol::migrate_jobs(vec![job_id.clone()]));
                    }
                    self.active.push(*info);
                    if let Some(status) = p.early_status {
                        self.on_job_status(job_id, status);
                    }
                }
            }
            Err(e) => {
                log::error!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = p.job.id.as_str(),
                error = e.to_string().as_str(); "failed to launch job")
            }
        }
    }

    /// Completed jobs leave active bookkeeping.
    fn on_job_status(&mut self, job_id: JobId, status: JobStatus) {
        if let Some(pending) = self
            .pending
            .iter_mut()
            .find(|p| p.job.id == job_id && matches!(p.state, PendingState::Launching(_)))
        {
            pending.early_status = Some(status);
            return;
        }
        if !self.active.iter().any(|j| j.job.id == job_id) {
            log::warn!(target: LOG, agent_name = self.cfg.agent_name.as_str(), job_id = job_id.as_str();
                "job status for a job this worker does not have");
            return;
        }
        self.queue(protocol::update_job(&job_id, status));
        if matches!(status, JobStatus::JsSuccess | JobStatus::JsFailed) {
            self.active.retain(|j| j.job.id != job_id);
        }
    }

    fn send_worker_status(&mut self) {
        let job_cnt = self.active.len() as u32;
        if self.draining {
            self.queue(protocol::update_worker_draining(job_cnt));
            return;
        }
        let inputs = self.load_inputs();
        let status = inputs.worker_status();
        if self.previous_status != Some(status) {
            self.previous_status = Some(status);
            let full = status == WorkerStatus::Full;
            log::info!(target: LOG, agent_name = self.cfg.agent_name.as_str(), load = inputs.effective_load(),
                threshold = inputs.load_threshold;
                "{}", if full { "worker is at full capacity, marking as unavailable" } else { "worker is below capacity, marking as available" });
        }
        self.queue(protocol::update_worker(status, self.worker_load, job_cnt));
    }


    fn alloc_slot(&mut self) -> Slot {
        let s = Slot(self.next_slot);
        self.next_slot = self.next_slot.wrapping_add(1);
        s
    }

    fn drop_pending(&mut self, slot: Slot) {
        self.pending.retain(|p| p.slot != slot);
        self.awaiting_load.retain(|s| *s != slot);
        self.timers.cancel(WorkerTimer::Assignment(slot));
    }

    fn push(&mut self, a: WorkerStateEffect) {
        self.out.push(a);
    }

    /// Periodic status and pings are suppressed until registration completes.
    fn queue(&mut self, msg: WorkerMessage) {
        use livekit_protocol::worker_message::Message as M;
        if self.conn == ConnState::Connecting
            && matches!(msg.message, Some(M::UpdateWorker(_)) | Some(M::Ping(_)))
        {
            return;
        }
        self.push(WorkerStateEffect::Send(msg));
    }

    fn check_drain_complete(&mut self) {
        if !self.draining || self.drain_reported {
            return;
        }
        if self.pending.is_empty() && self.active.is_empty() {
            self.drain_reported = true;
            self.timers.cancel(WorkerTimer::DrainDeadline);
            self.push(WorkerStateEffect::DrainFinished);
        }
    }

    fn take(&mut self) -> Vec<WorkerStateEffect> {
        std::mem::take(&mut self.out)
    }
}

