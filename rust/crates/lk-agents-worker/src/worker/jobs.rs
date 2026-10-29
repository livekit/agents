//! The worker's registration, job offers, assignments, launches and drain, decided on the
//! worker's loop as each event arrives.

use std::time::Duration;

use lk_agents_abi::{OwnedTask, SpawnerExt, Ts};
use lk_agents_net::task::TokioSpawner;

use super::{Completion, Outbox, Worker, WorkerNotification};
use crate::config::{ASSIGNMENT_TIMEOUT, UPDATE_LOAD_INTERVAL, UPDATE_STATUS_INTERVAL};
use crate::load::{LoadInputs, WorkerStatus};
use crate::protocol::{
    self, AvailabilityRequest, Job, JobAssignment, JobStatus, RegisterWorkerResponse,
    ServerMessage, WorkerMessage,
};
use crate::{AcceptArguments, HostError, JobDecision, JobId, JobRequest, RunningJobInfo};
use lk_agents_abi::worker::worker::{WorkerDrainFinished, WorkerDrainTimedOut, WorkerRegistered};

/// One offered job's place while it is decided, assigned and launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(pub u32);

/// `Assignment` is the only per-job timer, bounded by the reserved slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum WorkerTimer {
    /// `UPDATE_LOAD_INTERVAL`, 0.5 s. Lives for the whole worker, connected or not.
    LoadSample,
    /// `UPDATE_STATUS_INTERVAL`, 2.5 s. Per-connection: armed at register, cancelled on disconnect.
    StatusUpdate,
    DrainDeadline,
    /// `ASSIGNMENT_TIMEOUT`, 7.5 s, one per accepted-but-unassigned job.
    Assignment(Slot),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConnState {
    Disconnected,
    /// Registration is pending; periodic status messages are suppressed.
    Connecting,
    Registered,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PendingState {
    /// Waiting for a fresh load sample before checking capacity.
    AwaitingLoad,
    AwaitingUser,
    /// Accepted; waiting for `JobAssignment` under the 7.5 s timer.
    AwaitingAssignment(AcceptArguments),
    Launching(Box<RunningJobInfo>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Pending {
    pub(super) slot: Slot,
    pub(super) generation: u64,
    /// The job from the **`AvailabilityRequest`**. See [`Worker::on_assignment`].
    pub(super) job: Job,
    pub(super) resuming: bool,
    pub(super) state: PendingState,
    pub(super) early_status: Option<JobStatus>,
    /// When the request arrived, and when it was accepted.
    pub(super) received: Ts,
    pub(super) accepted: Ts,
}

// Provisional worker-wide admission/history bound; calibrated by the shared budget manifest.
pub(super) const DEFAULT_ASSIGNMENT_CAPACITY: usize = 4096;

/// Linear backoff with an immediate first retry, capped at ten seconds.
pub fn retry_delay(retry_count: u32) -> Duration {
    Duration::from_secs((u64::from(retry_count) * 2).min(10))
}

impl Worker {
    pub(super) fn load_inputs(&self) -> LoadInputs {
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

    /// Requests waiting for a load sample do not reserve capacity.
    pub(super) fn reserved_slots(&self) -> usize {
        self.pending.iter().filter(|p| p.state != PendingState::AwaitingLoad).count()
    }

    /// Idempotent; `None` or zero disables the deadline.
    pub(super) fn drain(&mut self, timeout: Option<Duration>, outbox: &mut Outbox) {
        if self.draining {
            return;
        }
        self.draining = true;
        self.diagnostic(
            log::Level::Info,
            format!("draining worker id={} timeout={timeout:?}", self.worker_id),
        );
        // The draining status goes out immediately, before the wait.
        self.queue(protocol::update_worker_draining(self.active.len() as u32), outbox);
        if let Some(d) = timeout.filter(|d| !d.is_zero()) {
            self.timers.arm(WorkerTimer::DrainDeadline, self.clock.now().saturating_add(d));
        }
        self.settle();
    }

    /// What is due now.
    pub(super) fn tick(&mut self, outbox: &mut Outbox) {
        let now = self.clock.now();
        while let Some(t) = self.timers.pop_due(now) {
            match t {
                WorkerTimer::LoadSample => {
                    self.timers
                        .arm(WorkerTimer::LoadSample, now.saturating_add(UPDATE_LOAD_INTERVAL));
                    self.sample_load();
                }
                WorkerTimer::StatusUpdate => {
                    self.timers
                        .arm(WorkerTimer::StatusUpdate, now.saturating_add(UPDATE_STATUS_INTERVAL));
                    self.send_worker_status(outbox);
                }
                WorkerTimer::DrainDeadline => {
                    self.drain_reported = true;
                    self.shared().drain_outcome = Some("timed_out".into());
                    self.notify(WorkerNotification::DrainTimedOut(WorkerDrainTimedOut {}));
                }
                WorkerTimer::Assignment(slot) => self.on_assignment_timeout(slot),
            }
        }
        self.check_rollover();
    }

    /// After each event: a drain that has nothing left finishes, a full history rolls over.
    pub(super) fn settle(&mut self) {
        if self.draining
            && !self.drain_reported
            && self.pending.is_empty()
            && self.active.is_empty()
        {
            self.drain_reported = true;
            self.timers.cancel(WorkerTimer::DrainDeadline);
            self.shared().drain_outcome = Some("finished".into());
            self.notify(WorkerNotification::DrainFinished(WorkerDrainFinished {}));
        }
        self.check_rollover();
    }

    /// A new connection: registration goes out first.
    pub(super) fn on_connecting(&mut self, outbox: &mut Outbox) {
        self.generation += 1;
        self.dispositions.clear();
        self.rolling_over = false;
        self.conn = ConnState::Connecting;
        let msg = protocol::register(
            &self.cfg.agent_name,
            &self.cfg.deployment,
            &self.cfg.version,
            &self.cfg.permissions,
            &self.cfg.endpoints,
            &self.cfg.instance_id,
        );
        // Sent directly, not queued: `register` must bypass the connecting-state drop.
        self.send(msg, None, outbox);
    }

    pub(super) fn on_registered(&mut self, resp: RegisterWorkerResponse, outbox: &mut Outbox) {
        if self.conn != ConnState::Connecting {
            return;
        }
        self.worker_id = resp.worker_id.clone();
        self.conn = ConnState::Registered;
        let (region, protocol) =
            resp.server_info.map_or((String::new(), 0), |s| (s.region, s.protocol));
        self.diagnostic(
            log::Level::Info,
            format!(
                "registered worker id={} deployment={} url={} region={region} protocol={protocol}",
                resp.worker_id, self.cfg.deployment, self.cfg.ws_url
            ),
        );
        self.notify(WorkerNotification::Registered(WorkerRegistered {
            worker_id: resp.worker_id,
            region,
            protocol,
            deployment: self.cfg.deployment.clone(),
            agent_name: self.cfg.agent_name.clone(),
        }));
        self.timers.arm(
            WorkerTimer::StatusUpdate,
            self.clock.now().saturating_add(UPDATE_STATUS_INTERVAL),
        );
        // Report running jobs after registration so the server can migrate their ownership.
        if !self.active.is_empty() {
            let ids = self.active.iter().map(|j| j.job.id.clone()).collect();
            self.queue(protocol::migrate_jobs(ids), outbox);
        }
        self.update_idle_target();
        self.send_worker_status(outbox);
    }

    pub(super) fn on_server(&mut self, msg: ServerMessage, outbox: &mut Outbox) {
        use livekit_protocol::server_message::Message as M;
        if self.conn != ConnState::Registered {
            return;
        }
        match msg.message {
            Some(M::Availability(req)) => self.on_availability(req, outbox),
            Some(M::Assignment(a)) => self.on_assignment(a, outbox),
            Some(M::Termination(t)) => {
                // "safe to ignore" when we do not hold the job.
                if self.active.iter().any(|j| j.job.id == t.job_id) {
                    self.terminate_job(t.job_id);
                }
            }
            Some(M::Register(_)) => {}
            // The server is draining; reconnecting is the transport's job, so nothing to decide.
            Some(M::GoAway(_)) => {}
            Some(M::Pong(_)) | None => {}
        }
    }

    /// The connection is gone. Per-connection timers stop; assignment timers keep running.
    pub(super) fn on_disconnected(&mut self) {
        self.conn = ConnState::Disconnected;
        self.timers.cancel(WorkerTimer::StatusUpdate);
        // Decisions and assignments belong to the lost connection. Launches already handed to
        // the host must still complete and become active jobs.
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

    /// Parked until a fresh load sample ([`Self::on_load`]), so a burst is not judged on stale
    /// load.
    fn on_availability(&mut self, req: AvailabilityRequest, outbox: &mut Outbox) {
        let Some(job) = req.job else { return };
        if self.pending.iter().any(|pending| pending.job.id == job.id)
            || self.active.iter().any(|active| active.job.id == job.id)
            || self.dispositions.contains_key(&job.id)
        {
            return;
        }
        if self.rolling_over || self.assignment_usage() >= self.assignment_capacity {
            self.queue(protocol::availability_reject(&job.id), outbox);
            return;
        }
        let slot = Slot(self.next_slot);
        self.next_slot = self.next_slot.wrapping_add(1);
        let now = self.clock.now();
        self.pending.push(Pending {
            slot,
            generation: self.generation,
            job,
            resuming: req.resuming,
            state: PendingState::AwaitingLoad,
            early_status: None,
            received: now,
            accepted: now,
        });
        self.awaiting_load.push_back(slot);
        self.sample_load();
    }

    fn update_idle_target(&mut self) {
        if let Some(target) = self.load_inputs().target_idle_processes() {
            self.shared().target_idle_processes = Some(target);
        }
    }

    pub(super) fn on_load(&mut self, load: f32, outbox: &mut Outbox) {
        self.worker_load = load;
        self.update_idle_target();
        let parked: Vec<Slot> = self.awaiting_load.drain(..).collect();
        for slot in parked {
            self.resume_availability(slot, outbox);
        }
    }

    fn resume_availability(&mut self, slot: Slot, outbox: &mut Outbox) {
        // This request is still `AwaitingLoad`, so it does not count itself.
        let available = self.load_inputs().is_available();
        let Some(p) = self.pending.iter_mut().find(|p| p.slot == slot) else { return };
        if !available {
            let job_id = p.job.id.clone();
            self.drop_pending(slot);
            // Rejected on load: `terminate` stays false.
            self.queue(protocol::availability_reject(&job_id), outbox);
            return;
        }
        p.state = PendingState::AwaitingUser;
        let request = JobRequest { job: protocol::job_info(&p.job), resuming: p.resuming };
        self.diagnostic(
            log::Level::Info,
            format!(
                "received job request job_id={} dispatch_id={} room={} resuming={}",
                request.job.id,
                request.job.dispatch_id,
                request.room_name(),
                request.resuming
            ),
        );
        self.ask_user(slot, request);
    }

    /// The host decides on its own task; dropping the task (a disconnect, a close) cancels it.
    fn ask_user(&mut self, slot: Slot, request: JobRequest) {
        let (host, generation) = (self.host.clone(), self.generation);
        let (decided, decision) = tokio::sync::oneshot::channel();
        let task: OwnedTask = TokioSpawner::current().spawn(async move {
            let _ = decided.send(host.on_job_request(request).await);
        });
        self.requests.insert(slot, task);
        self.operations.push(Box::pin(async move {
            match decision.await {
                Ok(decision) => Completion::Decision { generation, slot, decision },
                Err(_) => Completion::Done,
            }
        }));
    }

    pub(super) fn on_decision(
        &mut self,
        generation: u64,
        slot: Slot,
        decision: JobDecision,
        outbox: &mut Outbox,
    ) {
        self.requests.remove(&slot);
        if generation != self.generation {
            return;
        }
        let Some(p) = self.pending.iter().find(|p| p.slot == slot) else { return };
        if p.state != PendingState::AwaitingUser {
            return;
        }
        let job_id = p.job.id.clone();
        match decision {
            JobDecision::Accept(args) => {
                let arguments = protocol::accept_arguments(&self.cfg.agent_name, args.clone());
                let now = self.clock.now();
                if let Some(p) = self.pending.iter_mut().find(|p| p.slot == slot) {
                    p.state = PendingState::AwaitingAssignment(args);
                    p.accepted = now;
                }
                self.send(super::availability_message(&job_id, arguments), Some(job_id), outbox);
            }
            JobDecision::Reject => {
                self.drop_pending(slot);
                self.queue(protocol::availability_reject(&job_id), outbox);
            }
        }
    }

    /// The transport finished writing an acceptance; its assignment is waited for from now.
    pub(super) fn on_availability_sent(&mut self, job_id: JobId) {
        if let Some(p) = self
            .pending
            .iter()
            .find(|p| p.job.id == job_id && matches!(p.state, PendingState::AwaitingAssignment(_)))
        {
            self.timers.arm(
                WorkerTimer::Assignment(p.slot),
                self.clock.now().saturating_add(ASSIGNMENT_TIMEOUT),
            );
        }
    }

    /// An unset or empty assignment URL falls back to the worker URL.
    fn on_assignment(&mut self, a: JobAssignment, outbox: &mut Outbox) {
        let job_id = a.job.as_ref().map_or(String::new(), |j| j.id.clone());
        if let Some(status) = self.dispositions.get(&job_id).copied() {
            self.queue(protocol::update_job(&job_id, status), outbox);
            return;
        }
        let found = self.pending.iter().position(|p| {
            p.job.id == job_id && matches!(p.state, PendingState::AwaitingAssignment(_))
        });
        let Some(idx) = found else {
            if self
                .pending
                .iter()
                .any(|p| p.job.id == job_id && matches!(p.state, PendingState::Launching(_)))
                || self.active.iter().any(|active| active.job.id == job_id)
            {
                return;
            }
            if let Some(slot) = self
                .pending
                .iter()
                .find(|pending| pending.job.id == job_id)
                .map(|pending| pending.slot)
            {
                self.drop_pending(slot);
                self.requests.remove(&slot);
            }
            if self.assignment_usage() < self.assignment_capacity {
                self.dispositions.insert(job_id.clone(), JobStatus::JsFailed);
            }
            self.queue(protocol::update_job(&job_id, JobStatus::JsFailed), outbox);
            self.diagnostic(
                log::Level::Warn,
                format!("received assignment for an unknown job job_id={job_id}"),
            );
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
        let unix = |ts: Ts| self.wall.wall(ts);
        let p = &self.pending[idx];
        let info = RunningJobInfo {
            accept_arguments: args,
            job: protocol::job_info(&p.job),
            url,
            token: a.token,
            worker_id: self.worker_id.clone(),
            fake_job: false,
            received_at: unix(p.received),
            accepted_at: unix(p.accepted),
            assigned_at: unix(self.clock.now()),
        };
        self.pending[idx].state = PendingState::Launching(Box::new(info.clone()));
        let (host, generation) = (self.host.clone(), self.generation);
        self.operations.push(Box::pin(async move {
            Completion::Launched { generation, slot, result: host.launch_job(info).await }
        }));
    }

    /// Drops the pending job and warns; nothing is sent to the server.
    fn on_assignment_timeout(&mut self, slot: Slot) {
        let Some(p) = self.pending.iter().find(|p| p.slot == slot) else { return };
        let job_id = p.job.id.clone();
        self.drop_pending(slot);
        self.diagnostic(log::Level::Warn, format!("assignment for job {job_id} timed out"));
    }

    pub(super) fn on_launched(
        &mut self,
        generation: u64,
        slot: Slot,
        result: Result<JobId, HostError>,
        outbox: &mut Outbox,
    ) {
        let Some(idx) =
            self.pending.iter().position(|p| p.slot == slot && p.generation == generation)
        else {
            return;
        };
        let p = self.pending.remove(idx);
        self.timers.cancel(WorkerTimer::Assignment(slot));
        match result {
            Ok(_) => {
                self.dispositions.insert(p.job.id.clone(), JobStatus::JsRunning);
                if let PendingState::Launching(info) = p.state {
                    let job_id = info.job.id.clone();
                    if info.worker_id != self.worker_id && self.conn == ConnState::Registered {
                        self.queue(protocol::migrate_jobs(vec![job_id.clone()]), outbox);
                    }
                    self.active.push(*info);
                    if let Some(status) = p.early_status {
                        self.on_job_status(job_id, status, outbox);
                    }
                }
            }
            Err(e) => {
                self.dispositions.insert(p.job.id.clone(), JobStatus::JsFailed);
                self.queue(protocol::update_job(&p.job.id, JobStatus::JsFailed), outbox);
                self.diagnostic(
                    log::Level::Error,
                    format!("failed to launch job job_id={} error={e}", p.job.id),
                );
            }
        }
    }

    /// A launched job changed state; completed jobs leave active bookkeeping.
    pub(super) fn on_job_status(&mut self, job_id: JobId, status: JobStatus, outbox: &mut Outbox) {
        if let Some(pending) = self
            .pending
            .iter_mut()
            .find(|p| p.job.id == job_id && matches!(p.state, PendingState::Launching(_)))
        {
            pending.early_status = Some(status);
            return;
        }
        if !self.active.iter().any(|j| j.job.id == job_id) {
            self.diagnostic(
                log::Level::Warn,
                format!("job status for a job this worker does not have job_id={job_id}"),
            );
            return;
        }
        self.queue(protocol::update_job(&job_id, status), outbox);
        if matches!(status, JobStatus::JsSuccess | JobStatus::JsFailed) {
            self.dispositions.insert(job_id.clone(), status);
            self.active.retain(|j| j.job.id != job_id);
        }
    }

    fn terminate_job(&mut self, id: JobId) {
        let host = self.host.clone();
        self.operations.push(Box::pin(async move {
            if let Err(error) = host.terminate_job(id.clone()).await {
                crate::diagnostics::emit(
                    log::Level::Error,
                    format!("failed to terminate job job_id={id} error={error}"),
                );
            }
            Completion::Done
        }));
    }

    pub(super) fn send_worker_status(&mut self, outbox: &mut Outbox) {
        let job_cnt = self.active.len() as u32;
        if self.draining {
            self.queue(protocol::update_worker_draining(job_cnt), outbox);
            return;
        }
        let inputs = self.load_inputs();
        let status = inputs.worker_status();
        if self.previous_status != Some(status) {
            self.previous_status = Some(status);
            let full = status == WorkerStatus::Full;
            self.diagnostic(
                log::Level::Info,
                format!(
                    "{} load={} threshold={}",
                    if full {
                        "worker is at full capacity, marking as unavailable"
                    } else {
                        "worker is below capacity, marking as available"
                    },
                    inputs.effective_load(),
                    inputs.load_threshold
                ),
            );
        }
        self.queue(protocol::update_worker(status, self.worker_load, job_cnt), outbox);
    }

    fn drop_pending(&mut self, slot: Slot) {
        if let Some(pending) = self.pending.iter().find(|pending| pending.slot == slot) {
            self.dispositions.entry(pending.job.id.clone()).or_insert(JobStatus::JsFailed);
        }
        self.pending.retain(|p| p.slot != slot);
        self.awaiting_load.retain(|s| *s != slot);
        self.timers.cancel(WorkerTimer::Assignment(slot));
    }

    fn assignment_usage(&self) -> usize {
        self.dispositions.len()
            + self.pending.len()
            + self
                .active
                .iter()
                .filter(|active| !self.dispositions.contains_key(&active.job.id))
                .count()
    }

    /// A full assignment history is cleared by reconnecting once nothing is pending.
    fn check_rollover(&mut self) {
        if !self.draining
            && !self.rolling_over
            && self.conn == ConnState::Registered
            && self.pending.is_empty()
            && !self.dispositions.is_empty()
            && self.assignment_usage() >= self.assignment_capacity
        {
            self.rolling_over = true;
            self.rollover_requested = true;
        }
    }

    fn diagnostic(&self, level: log::Level, message: String) {
        crate::diagnostics::emit(level, format!("{message} agent_name={}", self.cfg.agent_name));
    }

    /// Periodic status and pings are suppressed until registration completes.
    pub(super) fn queue(&mut self, msg: WorkerMessage, outbox: &mut Outbox) {
        use livekit_protocol::worker_message::Message as M;
        if self.conn == ConnState::Connecting
            && matches!(msg.message, Some(M::UpdateWorker(_)) | Some(M::Ping(_)))
        {
            return;
        }
        self.send(msg, None, outbox);
    }

    fn send(&mut self, msg: WorkerMessage, accepted: Option<JobId>, outbox: &mut Outbox) {
        if let Err(error) = outbox.push(msg, accepted) {
            self.send_failure = Some(error);
        }
    }
}
