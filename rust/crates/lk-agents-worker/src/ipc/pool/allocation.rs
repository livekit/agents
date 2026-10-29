//! Warm process allocation, sans I/O: admission, process eligibility, joins and the close
//! deadline. The pool applies its effects.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Duration;

pub const MAX_ACQUIRE_ATTEMPTS: u32 = 3;
pub const MAX_LAUNCH_ATTEMPTS: u32 = 3;
pub const REFILL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcessId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ReservationId(pub(super) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolJoinOutcome {
    Reaped,
    NotLaunched,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProcessPhase {
    Creating,
    Initializing,
    Ready,
    Reserved,
    Running,
    Retiring,
    Cleaned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolAdmissionError {
    NotStarted,
    Draining,
    Closed,
    Full,
    DuplicateJob,
}

impl std::fmt::Display for PoolAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotStarted => "process pool has not started",
            Self::Draining => "process pool is draining",
            Self::Closed => "process pool is closed",
            Self::Full => "process pool pending job limit reached",
            Self::DuplicateJob => "job is already admitted to the process pool",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PoolFailure {
    AcquireExhausted { attempts: u32 },
    LaunchExhausted { attempts: u32 },
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReservationState {
    Waiting,
    Reserved(ProcessId),
    Completed,
    Failed(PoolFailure),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PoolLimits {
    pub(super) idle_processes: usize,
    pub(super) concurrent_initializations: usize,
    pub(super) pending_jobs: usize,
    pub(super) cleanup_timeout: Duration,
}

pub(super) enum PoolInput {
    Start,
    Admit { request: ReservationId, job_id: String },
    Created(ProcessId),
    Ready(ProcessId),
    SpawnFailed(ProcessId),
    LaunchSucceeded { request: ReservationId, process: ProcessId },
    LaunchFailed { request: ReservationId, process: ProcessId },
    Exited(ProcessId),
    ExitFailed(ProcessId),
    Cleaned(ProcessId),
    Closed(ProcessId),
    Joined { process: ProcessId, outcome: PoolJoinOutcome },
    JoinFailed(ProcessId),
    Cancel(ReservationId),
    Forget(ReservationId),
    TargetIdle(usize),
    Drain,
    Close,
    CleanupComplete,
    Wake,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum PoolEffect {
    Create(ProcessId),
    Close(ProcessId),
    Join(ProcessId),
    WatchExit(ProcessId),
    Released { process: ProcessId, outcome: PoolJoinOutcome },
    Refused { request: ReservationId, reason: PoolAdmissionError },
    CleanupIncomplete(PoolCleanupObservation),
    CloseComplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolCleanupObservation {
    pub deadline: Duration,
    pub processes: Vec<ProcessId>,
}

struct Process {
    phase: ProcessPhase,
    request: Option<ReservationId>,
    job_id: Option<(String, ReservationId)>,
    close_settled: bool,
    receipt: Option<PoolJoinOutcome>,
}

struct Reservation {
    job_id: String,
    state: ReservationState,
    acquire_attempts: u32,
    launch_attempts: u32,
}

/// The pool's admission and process eligibility, sans I/O: inputs and the time in, effects out.
pub(super) struct Allocation {
    limits: PoolLimits,
    target_idle: usize,
    started: bool,
    draining: bool,
    closed: bool,
    next_process: u64,
    processes: BTreeMap<ProcessId, Process>,
    initializing: BTreeSet<ProcessId>,
    ready: VecDeque<ProcessId>,
    pending: VecDeque<ReservationId>,
    requests: BTreeMap<ReservationId, Reservation>,
    jobs: BTreeMap<String, ReservationId>,
    refill_at: Option<Duration>,
    close_deadline: Option<Duration>,
    joining: BTreeSet<ProcessId>,
    retry_join_at: BTreeMap<ProcessId, Duration>,
    watching_exit: BTreeSet<ProcessId>,
    retry_exit_at: BTreeMap<ProcessId, Duration>,
}

impl Allocation {
    pub(super) fn new(limits: PoolLimits) -> Self {
        assert!(limits.concurrent_initializations > 0);
        assert!(limits.pending_jobs > 0);
        Self {
            target_idle: limits.idle_processes,
            limits,
            started: false,
            draining: false,
            closed: false,
            next_process: 0,
            processes: BTreeMap::new(),
            initializing: BTreeSet::new(),
            ready: VecDeque::new(),
            pending: VecDeque::new(),
            requests: BTreeMap::new(),
            jobs: BTreeMap::new(),
            refill_at: None,
            close_deadline: None,
            joining: BTreeSet::new(),
            retry_join_at: BTreeMap::new(),
            watching_exit: BTreeSet::new(),
            retry_exit_at: BTreeMap::new(),
        }
    }

    pub(super) fn phase(&self, id: ProcessId) -> ProcessPhase {
        self.processes.get(&id).map_or(ProcessPhase::Cleaned, |p| p.phase)
    }

    pub(super) fn reservation(&self, id: ReservationId) -> Option<ReservationState> {
        self.requests.get(&id).map(|request| request.state)
    }

    pub(super) fn waiting(&self) -> usize {
        self.pending.len()
    }

    pub(super) fn target_idle(&self) -> usize {
        self.target_idle
    }

    pub(super) fn ready(&self) -> usize {
        self.ready.len()
    }

    pub(super) fn is_started(&self) -> bool {
        self.started
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed
    }

    pub(super) fn has_processes(&self) -> bool {
        !self.processes.is_empty()
    }

    pub(super) fn has_jobs(&self) -> bool {
        !self.pending.is_empty()
            || self
                .processes
                .values()
                .any(|p| matches!(p.phase, ProcessPhase::Reserved | ProcessPhase::Running))
    }

    pub(super) fn wake_at(&self) -> Option<Duration> {
        self.refill_at
            .into_iter()
            .chain(self.close_deadline)
            .chain(self.retry_join_at.values().copied())
            .chain(self.retry_exit_at.values().copied())
            .min()
    }

    pub(super) fn close_deadline(&self) -> Option<Duration> {
        self.close_deadline
    }

    pub(super) fn handle(&mut self, input: PoolInput, now: Duration) -> Vec<PoolEffect> {
        let mut effects = Vec::new();
        if self.close_deadline.is_some_and(|deadline| now >= deadline) {
            effects.push(PoolEffect::CleanupIncomplete(PoolCleanupObservation {
                deadline: self.close_deadline.take().unwrap(),
                processes: self.processes.keys().copied().collect(),
            }));
        }
        if self.refill_at.is_some_and(|deadline| now >= deadline) {
            self.refill_at = None;
        }
        let due_joins: Vec<_> =
            self.retry_join_at.iter().filter_map(|(id, at)| (*at <= now).then_some(*id)).collect();
        for id in due_joins {
            self.retry_join_at.remove(&id);
            if self.processes.contains_key(&id) && self.joining.insert(id) {
                effects.push(PoolEffect::Join(id));
            }
        }
        let due_exits: Vec<_> =
            self.retry_exit_at.iter().filter_map(|(id, at)| (*at <= now).then_some(*id)).collect();
        for id in due_exits {
            self.retry_exit_at.remove(&id);
            if self.processes.contains_key(&id) && self.watching_exit.insert(id) {
                effects.push(PoolEffect::WatchExit(id));
            }
        }
        match input {
            PoolInput::Start => self.started = true,
            PoolInput::Admit { request, job_id } => {
                let refused = if self.closed {
                    Some(PoolAdmissionError::Closed)
                } else if self.draining {
                    Some(PoolAdmissionError::Draining)
                } else if !self.started {
                    Some(PoolAdmissionError::NotStarted)
                } else if self.jobs.contains_key(&job_id) || self.requests.contains_key(&request) {
                    Some(PoolAdmissionError::DuplicateJob)
                } else if self.requests.len() >= self.limits.pending_jobs {
                    Some(PoolAdmissionError::Full)
                } else {
                    None
                };
                if let Some(reason) = refused {
                    effects.push(PoolEffect::Refused { request, reason });
                } else {
                    self.jobs.insert(job_id.clone(), request);
                    self.requests.insert(
                        request,
                        Reservation {
                            job_id,
                            state: ReservationState::Waiting,
                            acquire_attempts: 0,
                            launch_attempts: 0,
                        },
                    );
                    self.pending.push_back(request);
                }
            }
            PoolInput::Created(id) => {
                if let Some(process) = self.processes.get_mut(&id) {
                    if process.phase == ProcessPhase::Creating {
                        process.phase = ProcessPhase::Initializing;
                    } else if process.phase == ProcessPhase::Retiring {
                        effects.push(PoolEffect::Close(id));
                    }
                }
            }
            PoolInput::Ready(id) => {
                if let Some(process) = self.processes.get_mut(&id)
                    && process.phase == ProcessPhase::Initializing
                {
                    process.phase = ProcessPhase::Ready;
                    process.request = None;
                    self.initializing.remove(&id);
                    self.ready.push_back(id);
                    self.joining.insert(id);
                    effects.push(PoolEffect::Join(id));
                    self.watching_exit.insert(id);
                    effects.push(PoolEffect::WatchExit(id));
                }
            }
            PoolInput::SpawnFailed(id) => {
                self.refill_at = Some(now + REFILL_INTERVAL);
                self.retire(id, &mut effects);
            }
            PoolInput::LaunchSucceeded { request, process } => {
                if self.reservation(request) == Some(ReservationState::Reserved(process)) {
                    let pending = self.requests.get_mut(&request).unwrap();
                    pending.state = ReservationState::Completed;
                    let entry = self.processes.get_mut(&process).unwrap();
                    entry.phase = ProcessPhase::Running;
                    entry.request = None;
                }
            }
            PoolInput::LaunchFailed { request, process } => {
                if self.reservation(request) == Some(ReservationState::Reserved(process)) {
                    self.retire(process, &mut effects);
                }
            }
            PoolInput::Exited(id) => {
                self.watching_exit.remove(&id);
                self.retry_exit_at.remove(&id);
                self.retire(id, &mut effects);
            }
            PoolInput::ExitFailed(id) => {
                self.watching_exit.remove(&id);
                if self.processes.contains_key(&id) {
                    self.retire(id, &mut effects);
                    self.retry_exit_at.insert(id, now + REFILL_INTERVAL);
                }
            }
            PoolInput::Cleaned(id) => {
                self.retire(id, &mut effects);
                effects.retain(|effect| *effect != PoolEffect::Close(id));
                self.initializing.remove(&id);
                self.joining.remove(&id);
                self.retry_join_at.remove(&id);
                self.watching_exit.remove(&id);
                self.retry_exit_at.remove(&id);
                self.processes.remove(&id);
            }
            PoolInput::Closed(id) => {
                if let Some(process) = self.processes.get_mut(&id) {
                    process.close_settled = true;
                    if let Some(outcome) = process.receipt {
                        self.release(id, outcome, &mut effects);
                    } else if !self.joining.contains(&id) && !self.retry_join_at.contains_key(&id) {
                        self.joining.insert(id);
                        effects.push(PoolEffect::Join(id));
                    }
                }
            }
            PoolInput::Joined { process: id, outcome } => {
                self.joining.remove(&id);
                self.retry_join_at.remove(&id);
                if let Some(process) = self.processes.get_mut(&id) {
                    process.receipt = Some(outcome);
                    self.retire(id, &mut effects);
                    if self.processes.get(&id).is_some_and(|process| process.close_settled) {
                        self.release(id, outcome, &mut effects);
                    }
                }
            }
            PoolInput::JoinFailed(id) => {
                self.joining.remove(&id);
                if self.processes.contains_key(&id) {
                    self.retire(id, &mut effects);
                    self.retry_join_at.insert(id, now + REFILL_INTERVAL);
                }
            }
            PoolInput::Cancel(id) => {
                if let Some(request) = self.requests.remove(&id) {
                    self.pending.retain(|pending| *pending != id);
                    if let ReservationState::Reserved(process) = request.state {
                        self.retire(process, &mut effects);
                    }
                    if request.state != ReservationState::Completed {
                        release_job(&mut self.jobs, &request.job_id, id);
                    }
                    for process in self.processes.values_mut() {
                        if process.request == Some(id) {
                            process.request = None;
                        }
                    }
                }
            }
            PoolInput::Forget(id) => {
                if let Some(request) = self.requests.get(&id)
                    && matches!(
                        request.state,
                        ReservationState::Completed | ReservationState::Failed(_)
                    )
                {
                    self.requests.remove(&id);
                }
            }
            PoolInput::TargetIdle(n) => self.target_idle = n.min(self.limits.idle_processes),
            PoolInput::Drain => {
                self.draining = true;
                self.refill_at = None;
            }
            PoolInput::Close => {
                if !self.closed {
                    self.close_deadline = Some(now + self.limits.cleanup_timeout);
                }
                self.closed = true;
                self.refill_at = None;
                self.pending.clear();
                for (id, request) in &mut self.requests {
                    if request.state != ReservationState::Completed {
                        release_job(&mut self.jobs, &request.job_id, *id);
                        request.state = ReservationState::Failed(PoolFailure::Closed);
                    }
                }
                let ids: Vec<_> = self.processes.keys().copied().collect();
                for id in ids {
                    self.retire(id, &mut effects);
                }
            }
            PoolInput::CleanupComplete => {
                if self.closed && !self.has_processes() {
                    self.close_deadline = None;
                    effects.push(PoolEffect::CloseComplete);
                }
            }
            PoolInput::Wake => {}
        }
        self.allocate(&mut effects);
        effects
    }

    fn release(&mut self, id: ProcessId, outcome: PoolJoinOutcome, effects: &mut Vec<PoolEffect>) {
        self.initializing.remove(&id);
        self.joining.remove(&id);
        self.retry_join_at.remove(&id);
        self.watching_exit.remove(&id);
        self.retry_exit_at.remove(&id);
        self.ready.retain(|ready| *ready != id);
        self.processes.remove(&id);
        effects.push(PoolEffect::Released { process: id, outcome });
    }

    fn retire(&mut self, id: ProcessId, effects: &mut Vec<PoolEffect>) {
        let Some(process) = self.processes.get_mut(&id) else { return };
        if process.phase == ProcessPhase::Retiring {
            return;
        }
        let reserved = process.phase == ProcessPhase::Reserved;
        process.phase = ProcessPhase::Retiring;
        self.ready.retain(|ready| *ready != id);
        if let Some(request_id) = process.request
            && let Some(request) = self.requests.get_mut(&request_id)
            && (request.state == ReservationState::Waiting
                || request.state == ReservationState::Reserved(id))
        {
            if reserved {
                request.launch_attempts += 1;
            }
            if request.launch_attempts >= MAX_LAUNCH_ATTEMPTS {
                request.state = ReservationState::Failed(PoolFailure::LaunchExhausted {
                    attempts: request.launch_attempts,
                });
                release_job(&mut self.jobs, &request.job_id, request_id);
            } else if request.acquire_attempts >= MAX_ACQUIRE_ATTEMPTS {
                request.state = ReservationState::Failed(PoolFailure::AcquireExhausted {
                    attempts: request.acquire_attempts,
                });
                release_job(&mut self.jobs, &request.job_id, request_id);
            } else {
                request.state = ReservationState::Waiting;
                if !self.pending.contains(&request_id) {
                    self.pending.push_back(request_id);
                }
            }
        } else if let Some((job_id, request_id)) = process.job_id.take() {
            release_job(&mut self.jobs, &job_id, request_id);
        }
        effects.push(PoolEffect::Close(id));
    }

    fn allocate(&mut self, effects: &mut Vec<PoolEffect>) {
        if !self.started || self.closed {
            return;
        }
        self.pending.retain(|id| {
            self.requests.get(id).is_some_and(|request| request.state == ReservationState::Waiting)
        });
        while !self.ready.is_empty() && !self.pending.is_empty() {
            let first = self.pending.front().unwrap();
            if self.processes.values().any(|process| {
                process.request == Some(*first) && process.phase == ProcessPhase::Retiring
            }) {
                // The head reservation must finish cleanup before any replacement is assigned.
                // Spawning for later requests here would repeatedly replace their ready processes
                // while the head remains blocked, starving the runtime and growing the pool.
                return;
            }
            let process_id = self.ready.pop_front().unwrap();
            let request_id = self.pending.pop_front().unwrap();
            let request = self.requests.get_mut(&request_id).unwrap();
            request.state = ReservationState::Reserved(process_id);
            request.acquire_attempts = 0;
            for (other_id, process) in &mut self.processes {
                if *other_id != process_id && process.request == Some(request_id) {
                    process.request = None;
                }
            }
            let process = self.processes.get_mut(&process_id).unwrap();
            process.phase = ProcessPhase::Reserved;
            process.request = Some(request_id);
            process.job_id = Some((request.job_id.clone(), request_id));
        }
        for request_id in self.pending.iter().copied().collect::<Vec<_>>() {
            if self.processes.values().any(|p| p.request == Some(request_id)) {
                continue;
            }
            let process_id = self.processes.iter().find_map(|(id, process)| {
                (self.initializing.contains(id)
                    && process.request.is_none()
                    && process.phase != ProcessPhase::Retiring)
                    .then_some(*id)
            });
            let process_id = match process_id {
                Some(id) => id,
                None if self.initializing.len() < self.limits.concurrent_initializations => {
                    self.create(effects)
                }
                None => break,
            };
            self.requests.get_mut(&request_id).unwrap().acquire_attempts += 1;
            self.processes.get_mut(&process_id).unwrap().request = Some(request_id);
        }
        if !self.draining && self.refill_at.is_none() {
            let unassigned = self
                .processes
                .values()
                .filter(|process| {
                    process.request.is_none()
                        && matches!(
                            process.phase,
                            ProcessPhase::Creating
                                | ProcessPhase::Initializing
                                | ProcessPhase::Ready
                        )
                })
                .count();
            for _ in unassigned..self.target_idle {
                if self.initializing.len() >= self.limits.concurrent_initializations {
                    break;
                }
                self.create(effects);
            }
        }
    }

    fn create(&mut self, effects: &mut Vec<PoolEffect>) -> ProcessId {
        let id = ProcessId(self.next_process);
        self.next_process += 1;
        self.processes.insert(
            id,
            Process {
                phase: ProcessPhase::Creating,
                request: None,
                job_id: None,
                close_settled: false,
                receipt: None,
            },
        );
        self.initializing.insert(id);
        effects.push(PoolEffect::Create(id));
        id
    }
}

fn release_job(jobs: &mut BTreeMap<String, ReservationId>, job: &str, request: ReservationId) {
    if jobs.get(job) == Some(&request) {
        jobs.remove(job);
    }
}

#[cfg(test)]
mod tests;
