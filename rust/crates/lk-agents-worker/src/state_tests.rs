//! Protocol and job lifecycle tests driven by a virtual clock, without I/O.

use std::time::Duration;

use lk_agents_abi::{Clock, Ts, VirtualClock};

use crate::HostError;
use crate::config::*;
use crate::protocol::*;
use crate::state::*;
use crate::{AcceptArguments, JobDecision, WorkerPermissions};

// rig

struct Rig {
    m: WorkerState,
    clock: VirtualClock,
}

fn cfg() -> WorkerConfig {
    WorkerConfig {
        agent_name: "my-agent".into(),
        deployment: "prod".into(),
        version: "1.0.0".into(),
        ws_url: "wss://project.livekit.cloud".into(),
        load_threshold: EnvOption::Fixed(DEFAULT_LOAD_THRESHOLD_PROD),
        num_idle_processes: EnvOption::Fixed(3),
        drain_timeout: Some(DRAIN_TIMEOUT),
        ..Default::default()
    }
}

impl Rig {
    fn new() -> Self {
        logs::take();
        Rig::with(cfg())
    }

    fn with(cfg: WorkerConfig) -> Self {
        let clock = VirtualClock::new();
        // Get off Ts::ZERO so "armed at now" is distinguishable from "never armed".
        clock.advance(Duration::from_secs(1));
        let mut m = WorkerState::new(cfg);
        m.start(clock.now());
        Rig { m, clock }
    }

    fn now(&self) -> Ts {
        self.clock.now()
    }

    fn advance(&mut self, d: Duration) -> Vec<WorkerStateEffect> {
        self.clock.advance(d);
        let now = self.clock.now();
        let effects = self.m.tick(now);
        logs::record(&effects);
        effects
    }

    fn ev(&mut self, e: WorkerStateInput) -> Vec<WorkerStateEffect> {
        let now = self.clock.now();
        let actions = self.m.handle(now, e);
        logs::record(&actions);
        // The default rig models an immediately successful transport write.
        for action in &actions {
            if let WorkerStateEffect::Send(message) = action
                && let Some(livekit_protocol::worker_message::Message::Availability(a)) =
                    &message.message
                && a.available
            {
                self.m.handle(
                    now,
                    WorkerStateInput::AvailabilitySent { generation: 0, job_id: a.job_id.clone() },
                );
            }
        }
        actions
    }

    /// Connect and register.
    fn register(&mut self) -> Vec<WorkerStateEffect> {
        self.ev(WorkerStateInput::Connecting { generation: 0 });
        self.ev(WorkerStateInput::Registered {
            generation: 0,
            response: RegisterWorkerResponse {
                worker_id: "W_1".into(),
                server_info: Some(ServerInfo {
                    region: "us-east".into(),
                    protocol: 15,
                    ..Default::default()
                }),
                ..Default::default()
            },
        })
    }

    /// Deliver an availability request and let the load sample resolve at `load`.
    fn offer(&mut self, job_id: &str, load: f32) -> Vec<WorkerStateEffect> {
        self.ev(WorkerStateInput::Server {
            generation: 0,
            message: ServerMessage {
                message: Some(livekit_protocol::server_message::Message::Availability(
                    AvailabilityRequest { job: Some(job(job_id)), resuming: false },
                )),
            },
        });
        self.ev(WorkerStateInput::LoadSampled(load))
    }

    /// Offer `job_id` at load 0 and accept it; the acceptance is written.
    fn accept(&mut self, job_id: &str) -> Slot {
        let slot = slot_of(&self.offer(job_id, 0.0));
        self.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
        slot
    }

    /// Accept, assign and launch `job_id`.
    fn launch(&mut self, job_id: &str) -> Slot {
        let slot = self.accept(job_id);
        self.ev(assignment(job_id, None));
        self.ev(WorkerStateInput::JobLaunched { generation: 0, slot, result: Ok(job_id.into()) });
        slot
    }
}

fn job(id: &str) -> Job {
    Job {
        id: id.into(),
        dispatch_id: format!("d-{id}"),
        agent_name: "my-agent".into(),
        room: Some(livekit_protocol::Room {
            sid: "RM_1".into(),
            name: "room-1".into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn sent(actions: &[WorkerStateEffect]) -> Vec<&WorkerMessage> {
    actions
        .iter()
        .filter_map(|a| match a {
            WorkerStateEffect::Send(m) => Some(m),
            _ => None,
        })
        .collect()
}

fn availability_out(actions: &[WorkerStateEffect]) -> Vec<&AvailabilityResponse> {
    sent(actions)
        .into_iter()
        .filter_map(|m| match &m.message {
            Some(livekit_protocol::worker_message::Message::Availability(a)) => Some(a),
            _ => None,
        })
        .collect()
}

fn statuses(actions: &[WorkerStateEffect]) -> Vec<&UpdateWorkerStatus> {
    sent(actions)
        .into_iter()
        .filter_map(|m| match &m.message {
            Some(livekit_protocol::worker_message::Message::UpdateWorker(u)) => Some(u),
            _ => None,
        })
        .collect()
}

mod logs {
    use super::WorkerStateEffect;
    use std::cell::RefCell;
    thread_local! { static LINES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) }; }
    pub fn record(effects: &[WorkerStateEffect]) {
        LINES.with_borrow_mut(|lines| {
            lines.extend(effects.iter().filter_map(|effect| match effect {
                WorkerStateEffect::Diagnostic { message, .. } => Some(message.clone()),
                _ => None,
            }));
        });
    }
    pub fn take() -> Vec<String> {
        LINES.with_borrow_mut(std::mem::take)
    }
}

fn logged(pat: &str) -> bool {
    logs::take().iter().any(|l| l.contains(pat))
}

fn slot_of(actions: &[WorkerStateEffect]) -> Slot {
    actions
        .iter()
        .find_map(|a| match a {
            WorkerStateEffect::AskUser { slot, .. } => Some(*slot),
            _ => None,
        })
        .expect("no AskUser action")
}

fn assignment(job_id: &str, url: Option<&str>) -> WorkerStateInput {
    WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Assignment(JobAssignment {
                job: Some(job(job_id)),
                url: url.map(str::to_string),
                token: "TOK".into(),
            })),
        },
    }
}

fn accept() -> JobDecision {
    JobDecision::Accept(AcceptArguments {
        identity: "agent-1".into(),
        name: "Agent".into(),
        metadata: "{}".into(),
        attributes: vec![],
    })
}

// the 7.5 s assignment timeout

// Catches: the assignment timeout drifting off 7.5 s in either direction.
#[test]
fn assignment_times_out_at_exactly_7_5s() {
    let mut r = Rig::new();
    r.register();
    r.accept("job-1");
    assert_eq!(r.m.reserved_slots(), 1);

    // One nanosecond short: nothing fires.
    let out = r.advance(ASSIGNMENT_TIMEOUT - Duration::from_nanos(1));
    assert!(!logged("timed out"), "timed out early: {out:?}");
    assert_eq!(r.m.reserved_slots(), 1);

    let out = r.advance(Duration::from_nanos(1));
    assert!(logged("assignment for job job-1 timed out"), "{out:?}");
    assert_eq!(r.m.reserved_slots(), 0, "the slot must be released on timeout");
    // Assignment timeout is local bookkeeping and emits no server message.
    assert!(sent(&out).is_empty(), "nothing goes on the wire on assignment timeout: {out:?}");
}

// Catches: the timer surviving the assignment, so a running job is abandoned 7.5 s later.
#[test]
fn assignment_arriving_in_time_cancels_the_timer() {
    let mut r = Rig::new();
    r.register();
    r.accept("job-1");
    r.clock.advance(ASSIGNMENT_TIMEOUT - Duration::from_millis(1));
    let out = r.ev(assignment("job-1", Some("wss://sfu-2.livekit.cloud")));
    assert_eq!(
        out.iter().filter(|effect| matches!(effect, WorkerStateEffect::LaunchJob { .. })).count(),
        1,
        "{out:?}"
    );

    let out = r.advance(Duration::from_secs(60));
    assert!(!logged("timed out"), "a launched job must not later time out: {out:?}");
}

// Catches: the dispatch timeline stamped at the wrong step (the request's arrival, its acceptance,
// the assignment), or stamped without a wall anchor.
#[test]
fn a_launched_job_carries_when_it_was_offered_accepted_and_assigned() {
    let mut r = Rig::new();
    r.m =
        WorkerState::new(cfg()).with_wall(lk_agents_abi::WallAnchor { wall: 1000.0, at: r.now() });
    r.m.start(r.now());
    r.register();
    let slot = slot_of(&r.offer("job-1", 0.0));
    r.clock.advance(Duration::from_secs(2));
    r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    r.clock.advance(Duration::from_secs(3));
    let out = r.ev(assignment("job-1", None));
    let info = out
        .iter()
        .find_map(|effect| match effect {
            WorkerStateEffect::LaunchJob { info, .. } => Some(info),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{out:?}"));
    assert_eq!((info.received_at, info.accepted_at, info.assigned_at), (1000.0, 1002.0, 1005.0));

    let mut r = Rig::new();
    r.register();
    r.accept("job-1");
    let out = r.ev(assignment("job-1", None));
    let info = out
        .iter()
        .find_map(|effect| match effect {
            WorkerStateEffect::LaunchJob { info, .. } => Some(info),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{out:?}"));
    assert_eq!((info.received_at, info.assigned_at), (0.0, 0.0), "no anchor, no timeline");
}

// Catches: building RunningJobInfo from the assignment's job instead of the availability request's.
#[test]
fn running_job_uses_the_availability_job_not_the_assignment_job() {
    let mut r = Rig::new();
    r.register();
    r.ev(WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Availability(
                AvailabilityRequest {
                    job: Some(Job { metadata: "from-availability".into(), ..job("job-1") }),
                    resuming: false,
                },
            )),
        },
    });
    let a = r.ev(WorkerStateInput::LoadSampled(0.0));
    let slot = slot_of(&a);
    r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    let out = r.ev(WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Assignment(JobAssignment {
                job: Some(Job { metadata: "from-assignment".into(), ..job("job-1") }),
                url: None,
                token: "TOK".into(),
            })),
        },
    });
    let info = out
        .iter()
        .find_map(|effect| match effect {
            WorkerStateEffect::LaunchJob { info, .. } => Some(info),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{out:?}"));
    assert_eq!(info.job.metadata, "from-availability");
}

// Catches: an explicitly empty assignment url not falling back to the worker url (proto3 unset is "").
#[test]
fn empty_or_missing_assignment_url_falls_back_to_the_worker_url() {
    for (url, want) in [
        (None, "wss://project.livekit.cloud"),
        (Some(""), "wss://project.livekit.cloud"),
        (Some("wss://sfu-9.livekit.cloud"), "wss://sfu-9.livekit.cloud"),
    ] {
        let mut r = Rig::new();
        r.register();
        r.accept("job-1");
        let out = r.ev(assignment("job-1", url));
        let info = out
            .iter()
            .find_map(|effect| match effect {
                WorkerStateEffect::LaunchJob { info, .. } => Some(info),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{out:?}"));
        assert_eq!(info.url, want, "url={url:?}");
    }
}

// Catches: launching a job we never accepted.
#[test]
fn assignment_for_an_unaccepted_job_is_reported_not_launched() {
    let mut r = Rig::new();
    r.register();
    let out = r.ev(assignment("ghost", None));
    assert!(
        logs::take().iter().any(|l| l.contains("unknown job") && l.contains("job_id=ghost")),
        "{out:?}"
    );
    assert!(!out.iter().any(|a| matches!(a, WorkerStateEffect::LaunchJob { .. })));

    // ..and the same is true while the user's request_fnc is still deciding.
    r.offer("job-1", 0.0);
    let out = r.ev(assignment("job-1", None));
    assert!(
        logs::take().iter().any(|l| l.contains("unknown job") && l.contains("job_id=job-1")),
        "{out:?}"
    );
}

// availability answers

// Catches: rejecting with terminate=true when request_fnc simply did not answer.
#[test]
fn unanswered_and_raising_request_fnc_reject_without_terminating() {
    for decision in [JobDecision::Unanswered, JobDecision::Failed("boom".into())] {
        let mut r = Rig::new();
        r.register();
        let slot = slot_of(&r.offer("job-1", 0.0));
        let out = r.ev(WorkerStateInput::RequestDecision {
            generation: 0,
            slot,
            decision: decision.clone(),
        });
        let resp = availability_out(&out);
        assert_eq!(resp.len(), 1, "{out:?}");
        assert!(!resp[0].available);
        assert!(!resp[0].terminate, "unanswered must not terminate the job: {decision:?}");
        assert_eq!(r.m.reserved_slots(), 0);
    }
}

// Catches: an explicit reject losing its terminate flag.
#[test]
fn explicit_reject_preserves_terminate() {
    let mut r = Rig::new();
    r.register();
    let slot = slot_of(&r.offer("job-1", 0.0));
    let out = r.ev(WorkerStateInput::RequestDecision {
        generation: 0,
        slot,
        decision: JobDecision::Reject { terminate: true },
    });
    let resp = availability_out(&out);
    assert!(!resp[0].available);
    assert!(resp[0].terminate);
}

// Catches: asking request_fnc when the worker is already over threshold.
#[test]
fn an_overloaded_worker_rejects_before_asking_the_user() {
    let mut r = Rig::new();
    r.register();
    let out = r.offer("job-1", 0.95);
    assert!(!out.iter().any(|a| matches!(a, WorkerStateEffect::AskUser { .. })), "{out:?}");
    let resp = availability_out(&out);
    assert!(!resp[0].available);
    assert!(!resp[0].terminate, "the capacity early-return leaves terminate unset");
    assert_eq!(r.m.reserved_slots(), 0);
}

// Catches: judging availability on the last 0.5 s tick's load instead of a fresh sample.
#[test]
fn availability_waits_for_a_fresh_load_sample() {
    let mut r = Rig::new();
    r.register();
    let out = r.ev(WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Availability(
                AvailabilityRequest { job: Some(job("job-1")), resuming: false },
            )),
        },
    });
    assert!(out.contains(&WorkerStateEffect::SampleLoad), "{out:?}");
    assert!(!out.iter().any(|a| matches!(a, WorkerStateEffect::AskUser { .. })), "{out:?}");
    // Now the sample lands, and *it* decides.
    let out = r.ev(WorkerStateInput::LoadSampled(0.99));
    assert!(!availability_out(&out)[0].available);
}

// Catches: two back-to-back requests both accepted because neither sees the other's reserved slot.
#[test]
fn a_second_request_sees_the_first_reserved_slot() {
    let mut r = Rig::new();
    r.register();
    // threshold 0.7, no jobs running, num_idle 3 -> each reserved slot is charged 0.2333.
    for job in ["job-1", "job-2"] {
        // 0.3 + 1*0.2333 = 0.533 < 0.7 -> the second is still available.
        let slot = slot_of(&r.offer(job, 0.3));
        r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    }
    assert_eq!(r.m.reserved_slots(), 2);
    // 0.3 + 2*0.2333 = 0.766 >= 0.7 -> the third is refused without asking the user.
    let out = r.offer("job-3", 0.3);
    assert!(!out.iter().any(|a| matches!(a, WorkerStateEffect::AskUser { .. })), "{out:?}");
    assert!(!availability_out(&out)[0].available);
}

// status reporting

// Catches: the 0.5 s load and 2.5 s status intervals being swapped or collapsed.
#[test]
fn load_samples_at_0_5s_and_status_reports_at_2_5s() {
    let mut r = Rig::new();
    r.register();
    let mut samples = 0;
    let mut reports = 0;
    for _ in 0..5 {
        let out = r.advance(UPDATE_LOAD_INTERVAL);
        samples += out.iter().filter(|a| **a == WorkerStateEffect::SampleLoad).count();
        reports += statuses(&out).len();
    }
    assert_eq!(samples, 5, "one load sample per 0.5s");
    assert_eq!(reports, 1, "exactly one status report over 2.5s");
}

// Catches: sending `update_worker` while the register round-trip is still in flight.
#[test]
fn update_worker_is_dropped_while_connecting() {
    let mut r = Rig::new();
    r.ev(WorkerStateInput::Connecting { generation: 0 });
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    assert!(statuses(&out).is_empty(), "{out:?}");
    // Once registered the same tick sends one.
    r.ev(WorkerStateInput::Registered {
        generation: 0,
        response: RegisterWorkerResponse {
            worker_id: "W_1".into(),
            server_info: None,
            ..Default::default()
        },
    });
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    assert_eq!(statuses(&out).len(), 1, "{out:?}");
}

// Catches: logging the capacity line on every tick instead of on a transition.
#[test]
fn capacity_change_is_reported_only_on_a_transition() {
    let mut r = Rig::new();
    r.register();
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    assert_eq!(
        logs::take().iter().filter(|l| l.contains("capacity")).count(),
        1,
        "first report always logs: {out:?}"
    );
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    assert!(!logged("capacity"), "unchanged status must not re-log: {out:?}");
    r.ev(WorkerStateInput::LoadSampled(0.99));
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    assert!(logged("at full capacity"), "{out:?}");
}

// Catches: reporting the effective load on the wire, double-counting reserved slots at the server.
#[test]
fn the_wire_carries_raw_worker_load_not_effective_load() {
    let mut r = Rig::new();
    r.register();
    let slot = slot_of(&r.offer("job-1", 0.3));
    r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    let s = statuses(&out);
    assert_eq!(s[0].load, 0.3, "reserved slots must not inflate the reported load");
}

// Catches: not reporting held jobs after a reconnect, so the server re-dispatches them.
#[test]
fn active_jobs_are_migrated_after_re_registration() {
    let mut r = Rig::new();
    r.register();
    r.launch("job-1");
    assert_eq!(r.m.active_jobs().len(), 1);

    r.ev(WorkerStateInput::Disconnected { generation: 0 });
    let out = r.register();
    let migrated: Vec<_> = sent(&out)
        .into_iter()
        .filter_map(|m| match &m.message {
            Some(livekit_protocol::worker_message::Message::MigrateJob(x)) => Some(x),
            _ => None,
        })
        .collect();
    assert_eq!(migrated.len(), 1, "{out:?}");
    assert_eq!(migrated[0].job_ids, vec!["job-1".to_string()]);
}

// Catches: keeping a finished job in `active_jobs`, so a drain never completes.
#[test]
fn a_terminal_job_status_removes_it_from_active_jobs() {
    let mut r = Rig::new();
    r.register();
    r.launch("job-1");
    let out =
        r.ev(WorkerStateInput::JobStatus { job_id: "job-1".into(), status: JobStatus::JsRunning });
    assert_eq!(sent(&out).len(), 1);
    assert_eq!(r.m.active_jobs().len(), 1, "JS_RUNNING keeps the job");
    r.ev(WorkerStateInput::JobStatus { job_id: "job-1".into(), status: JobStatus::JsSuccess });
    assert_eq!(r.m.active_jobs().len(), 0);
}

// Catches: forwarding a termination for a job we do not hold.
#[test]
fn termination_for_an_unknown_job_is_ignored() {
    let mut r = Rig::new();
    r.register();
    let out = r.ev(WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Termination(JobTermination {
                job_id: "nope".into(),
            })),
        },
    });
    assert!(out.is_empty(), "{out:?}");
}

// the drain sequence

// Catches: drain finishing while an accepted-but-unlaunched job is still in flight.
#[test]
fn drain_waits_for_the_in_flight_request_then_the_running_job() {
    let mut r = Rig::new();
    r.register();
    let slot = r.accept("job-1");

    let now = r.now();
    let out = r.m.drain(now, Some(Duration::from_secs(60)));
    // Reports WS_FULL immediately, before waiting for anything.
    let s = statuses(&out);
    assert_eq!(s.len(), 1, "{out:?}");
    assert_eq!(s[0].status, Some(livekit_protocol::WorkerStatus::WsFull as i32));
    assert!(
        !out.contains(&WorkerStateEffect::DrainFinished),
        "a pending request is still in flight"
    );

    let out = r.ev(assignment("job-1", None));
    assert!(!out.contains(&WorkerStateEffect::DrainFinished), "the job has not launched yet");

    let out =
        r.ev(WorkerStateInput::JobLaunched { generation: 0, slot, result: Ok("job-1".into()) });
    assert!(!out.contains(&WorkerStateEffect::DrainFinished), "the job is still running");

    let out =
        r.ev(WorkerStateInput::JobStatus { job_id: "job-1".into(), status: JobStatus::JsSuccess });
    assert!(out.contains(&WorkerStateEffect::DrainFinished), "{out:?}");
}

// Catches: a second drain() re-sending the status or re-arming the deadline.
#[test]
fn drain_is_idempotent() {
    let mut r = Rig::new();
    r.register();
    let now = r.now();
    let first = r.m.drain(now, Some(Duration::from_secs(60)));
    assert!(!first.is_empty());
    let second = r.m.drain(now, Some(Duration::from_secs(1)));
    assert!(second.is_empty(), "{second:?}");
}

// Catches: the drain deadline never firing, so a wedged job blocks shutdown forever.
#[test]
fn drain_times_out_at_its_deadline() {
    let mut r = Rig::new();
    r.register();
    r.launch("job-1");

    let now = r.now();
    r.m.drain(now, Some(Duration::from_secs(30)));
    let out = r.advance(Duration::from_secs(29));
    assert!(!out.contains(&WorkerStateEffect::DrainTimedOut), "{out:?}");
    let out = r.advance(Duration::from_secs(1));
    assert!(out.contains(&WorkerStateEffect::DrainTimedOut), "{out:?}");
}

// Catches: treating `timeout=0` as an immediate deadline; like None it means "wait forever".
#[test]
fn a_zero_or_absent_drain_timeout_arms_no_deadline() {
    for timeout in [None, Some(Duration::ZERO)] {
        let mut r = Rig::new();
        r.register();
        r.launch("job-1");
        let now = r.now();
        r.m.drain(now, timeout);
        let out = r.advance(Duration::from_secs(86_400));
        assert!(!out.contains(&WorkerStateEffect::DrainTimedOut), "timeout={timeout:?}: {out:?}");
    }
}

// Catches: a draining worker still accepting jobs.
#[test]
fn a_draining_worker_refuses_new_jobs_even_in_simulation() {
    let mut r = Rig::with(WorkerConfig { simulation: true, ..cfg() });
    r.register();
    let now = r.now();
    r.m.drain(now, None);
    let out = r.offer("job-1", 0.0);
    assert!(!out.iter().any(|a| matches!(a, WorkerStateEffect::AskUser { .. })), "{out:?}");
    assert!(!availability_out(&out)[0].available);
}

// Catches: the periodic status reverting to WS_AVAILABLE mid-drain because the load dropped.
#[test]
fn periodic_status_stays_full_while_draining() {
    let mut r = Rig::new();
    r.register();
    logs::take();
    let now = r.now();
    r.m.drain(now, None);
    r.ev(WorkerStateInput::LoadSampled(0.0));
    let out = r.advance(UPDATE_STATUS_INTERVAL);
    let s = statuses(&out);
    assert_eq!(s[0].status, Some(livekit_protocol::WorkerStatus::WsFull as i32));
    assert_eq!(s[0].load, 0.0, "the draining branch never sets load");
    assert!(!logged("capacity"), "and never logs a capacity transition: {out:?}");
}

// Catches: drain hanging forever on an idle worker because completion is only checked on events.
#[test]
fn drain_on_an_idle_worker_finishes_immediately() {
    let mut r = Rig::new();
    r.register();
    let now = r.now();
    let out = r.m.drain(now, Some(Duration::from_secs(60)));
    assert!(out.contains(&WorkerStateEffect::DrainFinished), "{out:?}");
}

// misc

// Catches: the register message losing a field the server routes on, or a permission dropped in
// `as_proto`.
#[test]
fn the_register_message_carries_the_worker_identity_and_permissions() {
    let permissions = WorkerPermissions {
        can_publish: false,
        can_publish_sources: vec![1, 2],
        hidden: true,
        ..WorkerPermissions::default()
    };
    let mut r = Rig::with(WorkerConfig { permissions, ..cfg() });
    let out = r.ev(WorkerStateInput::Connecting { generation: 0 });
    let m = sent(&out);
    let Some(livekit_protocol::worker_message::Message::Register(reg)) = &m[0].message else {
        panic!("{out:?}")
    };
    assert_eq!((reg.agent_name.as_str(), reg.deployment.as_str()), ("my-agent", "prod"));
    assert_eq!(reg.version, "1.0.0");
    assert_eq!(reg.r#type, JobType::JtRoom as i32);
    let p = reg.allowed_permissions.as_ref().expect("permissions on the register message");
    assert!(!p.can_publish && p.hidden && p.can_subscribe, "{p:?}");
    assert_eq!(p.can_publish_sources, vec![1, 2]);
}

// Catches: losing the worker id, which every launched job's RunningJobInfo carries.
#[test]
fn the_worker_id_from_registration_reaches_the_launched_job() {
    let mut r = Rig::new();
    let out = r.register();
    assert!(out.iter().any(|a| matches!(
        a,
        WorkerStateEffect::Registered { worker_id, region, protocol, deployment, agent_name } if worker_id == "W_1" && region == "us-east" && *protocol == 15 && deployment == "prod" && agent_name == "my-agent"
    )), "{out:?}");
    r.accept("job-1");
    let out = r.ev(assignment("job-1", None));
    let info = out
        .iter()
        .find_map(|effect| match effect {
            WorkerStateEffect::LaunchJob { info, .. } => Some(info),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{out:?}"));
    assert_eq!(info.worker_id, "W_1");
    assert_eq!(info.token, "TOK");
    assert!(!info.fake_job);
}

// Catches: the accept response losing the participant fields, which is how the SFU names the agent.
#[test]
fn the_accept_response_carries_the_participant_identity() {
    let mut r = Rig::new();
    r.register();
    let slot = slot_of(&r.offer("job-1", 0.0));
    let out = r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    let a = availability_out(&out);
    assert!(a[0].available);
    assert_eq!(a[0].job_id, "job-1");
    assert_eq!(a[0].participant_identity, "agent-1");
    assert_eq!(a[0].participant_name, "Agent");
    assert_eq!(a[0].participant_attributes[lk_agents_abi::ATTRIBUTE_AGENT_NAME], "my-agent");
}

// Catches: a failed launch leaving its slot reserved forever, which slowly starves the worker.
#[test]
fn a_failed_launch_releases_the_slot() {
    let mut r = Rig::new();
    r.register();
    let slot = r.accept("job-1");
    r.ev(assignment("job-1", None));
    r.ev(WorkerStateInput::JobLaunched {
        generation: 0,
        slot,
        result: Err(HostError::new("no free process")),
    });
    assert_eq!(r.m.reserved_slots(), 0);
    assert_eq!(r.m.active_jobs().len(), 0);
}

#[test]
fn assignment_budget_starts_after_acceptance_is_written() {
    let mut r = Rig::new();
    r.register();
    let slot = slot_of(&r.offer("job", 0.0));
    // Bypass the rig's automatic send completion to model a congested transport.
    r.m.handle(
        r.now(),
        WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() },
    );
    r.advance(ASSIGNMENT_TIMEOUT);
    assert_eq!(r.m.reserved_slots(), 1, "unsent acceptance must not consume the assignment budget");
    r.ev(WorkerStateInput::AvailabilitySent { generation: 0, job_id: "job".into() });
    r.advance(ASSIGNMENT_TIMEOUT);
    assert_eq!(r.m.reserved_slots(), 0);
}

#[test]
fn disconnected_decisions_are_ignored_and_launches_preserve_their_assignment() {
    let mut r = Rig::new();
    r.register();
    let stale = slot_of(&r.offer("stale", 0.0));
    let launching = slot_of(&r.offer("launching", 0.0));
    r.ev(WorkerStateInput::RequestDecision { generation: 0, slot: launching, decision: accept() });
    r.ev(assignment("launching", Some("wss://assigned")));
    r.ev(WorkerStateInput::Disconnected { generation: 0 });
    r.register();
    let out =
        r.ev(WorkerStateInput::RequestDecision { generation: 0, slot: stale, decision: accept() });
    assert!(availability_out(&out).is_empty(), "old connection's decision was replayed");
    r.ev(WorkerStateInput::JobLaunched {
        generation: 0,
        slot: launching,
        result: Ok("launching".into()),
    });
    let jobs = r.m.active_jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].url, "wss://assigned");
    assert_eq!(jobs[0].token, "TOK");
}

#[test]
fn worker_registration_vectors_duplicate_availability_launches_once() {
    let mut r = Rig::new();
    r.register();
    let first = slot_of(&r.offer("duplicate", 0.0));
    let duplicate = r.offer("duplicate", 0.0);
    let mut slots = vec![first];
    slots.extend(duplicate.iter().filter_map(|effect| match effect {
        WorkerStateEffect::AskUser { slot, .. } => Some(*slot),
        _ => None,
    }));
    for slot in slots {
        r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() });
    }
    let mut launches = r.ev(assignment("duplicate", None));
    launches.extend(r.ev(assignment("duplicate", None)));
    assert_eq!(
        launches
            .iter()
            .filter(|effect| matches!(effect, WorkerStateEffect::LaunchJob { .. }))
            .count(),
        1,
        "the same job must never launch twice: {launches:?}",
    );
}

#[test]
fn worker_registration_vectors_failed_and_unknown_assignments_report_failure() {
    let mut r = Rig::new();
    r.register();
    let unknown = r.ev(assignment("unknown", None));
    assert!(
        sent(&unknown).iter().any(|message| matches!(
            &message.message,
            Some(livekit_protocol::worker_message::Message::UpdateJob(update))
                if update.job_id == "unknown" && update.status == JobStatus::JsFailed as i32
        )),
        "unknown assignment must receive a failed disposition: {unknown:?}"
    );
    let slot = r.accept("failed");
    r.ev(assignment("failed", None));
    let failed = r.ev(WorkerStateInput::JobLaunched {
        generation: 0,
        slot,
        result: Err(HostError::new("launch failed")),
    });
    assert!(
        sent(&failed).iter().any(|message| matches!(
            &message.message,
            Some(livekit_protocol::worker_message::Message::UpdateJob(update))
                if update.job_id == "failed" && update.status == JobStatus::JsFailed as i32
        )),
        "launch failure must reach the server: {failed:?}"
    );
}

#[test]
fn worker_reconnect_supplied_time() {
    let mut retry = ConnectionRetry::new(3);
    let start = Ts::from_nanos(42);
    assert_eq!(retry.disconnected(start), Ok(start));
    assert_eq!(retry.disconnected(start), Ok(start.saturating_add(Duration::from_secs(2))));
    assert_eq!(retry.wake_at(), Some(start.saturating_add(Duration::from_secs(2))));
    assert_eq!(retry.disconnected(start), Ok(start.saturating_add(Duration::from_secs(4))));
    assert_eq!(retry.disconnected(start), Err(3));
    assert_eq!(retry.wake_at(), None);
    retry.registered();
    assert_eq!(retry.disconnected(start), Ok(start));
}

#[test]
fn worker_registration_vectors_stale_generation_cannot_register_or_accept() {
    let mut r = Rig::new();
    r.register();
    let old_slot = slot_of(&r.offer("stale", 0.0));
    r.ev(WorkerStateInput::Disconnected { generation: 0 });
    r.ev(WorkerStateInput::Connecting { generation: 1 });
    let stale = r.ev(WorkerStateInput::Registered {
        generation: 0,
        response: RegisterWorkerResponse { worker_id: "stale".into(), ..Default::default() },
    });
    assert!(!r.m.registered());
    assert!(stale.iter().all(|effect| !matches!(effect, WorkerStateEffect::Registered { .. })));
    r.ev(WorkerStateInput::Registered {
        generation: 1,
        response: RegisterWorkerResponse { worker_id: "current".into(), ..Default::default() },
    });
    let stale = r.ev(WorkerStateInput::RequestDecision {
        generation: 0,
        slot: old_slot,
        decision: accept(),
    });
    assert!(availability_out(&stale).is_empty());
    r.ev(WorkerStateInput::Disconnected { generation: 0 });
    assert!(r.m.registered());
    assert_eq!(r.m.worker_id(), "current");
}

#[test]
fn worker_assignment_deadline_precedes_simultaneous_assignment() {
    let mut r = Rig::new();
    r.register();
    r.accept("late");
    r.clock.advance(ASSIGNMENT_TIMEOUT);
    let effects = r.ev(assignment("late", None));
    assert!(effects.iter().all(|effect| !matches!(effect, WorkerStateEffect::LaunchJob { .. })));
    assert_eq!(r.m.reserved_slots(), 0);
}

#[test]
fn worker_registration_vectors_terminal_assignment_cannot_launch_again() {
    for success in [true, false] {
        let mut rig = Rig::new();
        rig.register();
        let slot = rig.accept("finished");
        rig.ev(assignment("finished", None));
        rig.ev(WorkerStateInput::JobLaunched {
            generation: 0,
            slot,
            result: if success {
                Ok("finished".into())
            } else {
                Err(HostError::new("launch failed"))
            },
        });
        if success {
            rig.ev(WorkerStateInput::JobStatus {
                job_id: "finished".into(),
                status: JobStatus::JsSuccess,
            });
        }
        let effects = rig.offer("finished", 0.0);
        assert!(
            effects.iter().all(|effect| !matches!(effect, WorkerStateEffect::AskUser { .. })),
            "terminal job must retain its disposition, success={success}: {effects:?}"
        );
        assert!(
            rig.ev(assignment("finished", None))
                .iter()
                .all(|effect| !matches!(effect, WorkerStateEffect::LaunchJob { .. }))
        );
    }
}

#[test]
fn bounded_assignment_history_rolls_over_after_pending_assignments_settle() {
    let mut rig = Rig::new();
    rig.m = WorkerState::with_assignment_capacity(cfg(), 2);
    rig.m.start(rig.now());
    rig.register();
    rig.launch("active");
    let pending = rig.accept("pending");
    let refused = rig.ev(WorkerStateInput::Server {
        generation: 0,
        message: ServerMessage {
            message: Some(livekit_protocol::server_message::Message::Availability(
                AvailabilityRequest { job: Some(job("overload")), resuming: false },
            )),
        },
    });
    assert!(
        availability_out(&refused)
            .iter()
            .any(|reply| reply.job_id == "overload" && !reply.available)
    );
    assert!(refused.iter().all(|effect| !matches!(
        effect,
        WorkerStateEffect::AskUser { .. } | WorkerStateEffect::RolloverConnection
    )));
    rig.ev(assignment("pending", None));
    let settled = rig.ev(WorkerStateInput::JobLaunched {
        generation: 0,
        slot: pending,
        result: Err(HostError::new("launch refused")),
    });
    assert_eq!(
        settled
            .iter()
            .filter(|effect| matches!(effect, WorkerStateEffect::RolloverConnection))
            .count(),
        1
    );
    let duplicate = rig.offer("pending", 0.0);
    assert!(duplicate.iter().all(|effect| !matches!(
        effect,
        WorkerStateEffect::AskUser { .. } | WorkerStateEffect::RolloverConnection
    )));
    rig.ev(WorkerStateInput::Disconnected { generation: 0 });
    rig.ev(WorkerStateInput::Connecting { generation: 1 });
    rig.ev(WorkerStateInput::Registered {
        generation: 1,
        response: RegisterWorkerResponse { worker_id: "next".into(), ..Default::default() },
    });
    for (job_id, expected_requests) in [("active", 0), ("pending", 1)] {
        rig.ev(WorkerStateInput::Server {
            generation: 1,
            message: ServerMessage {
                message: Some(livekit_protocol::server_message::Message::Availability(
                    AvailabilityRequest { job: Some(job(job_id)), resuming: true },
                )),
            },
        });
        let offered = rig.ev(WorkerStateInput::LoadSampled(0.0));
        assert_eq!(
            offered
                .iter()
                .filter(|effect| matches!(effect, WorkerStateEffect::AskUser { .. }))
                .count(),
            expected_requests,
            "active IDs survive rollover; historical IDs can be reassigned in the new generation: {offered:?}"
        );
    }
}

#[test]
fn an_early_assignment_fails_once_and_cannot_be_accepted_by_a_late_decision() {
    for awaiting_user in [false, true] {
        let mut r = Rig::new();
        r.register();
        r.ev(WorkerStateInput::Server {
            generation: 0,
            message: ServerMessage {
                message: Some(livekit_protocol::server_message::Message::Availability(
                    AvailabilityRequest { job: Some(job("early")), resuming: false },
                )),
            },
        });
        let slot = if awaiting_user {
            Some(slot_of(&r.ev(WorkerStateInput::LoadSampled(0.0))))
        } else {
            None
        };
        let failed = r.ev(assignment("early", None));
        assert!(sent(&failed).iter().any(|message| matches!(
            &message.message,
            Some(livekit_protocol::worker_message::Message::UpdateJob(update))
                if update.job_id == "early" && update.status == JobStatus::JsFailed as i32
        )));
        if let Some(slot) = slot {
            assert!(
                failed.contains(&WorkerStateEffect::CancelRequest { slot }),
                "a rejected assignment must cancel its pending decision"
            );
        }
        let late = match slot {
            Some(slot) => {
                r.ev(WorkerStateInput::RequestDecision { generation: 0, slot, decision: accept() })
            }
            None => r.ev(WorkerStateInput::LoadSampled(0.0)),
        };
        assert!(
            availability_out(&late).is_empty(),
            "a terminally failed job cannot be accepted: {late:?}"
        );
        assert!(
            !late.iter().any(|effect| matches!(effect, WorkerStateEffect::AskUser { .. })),
            "a terminally failed job cannot request a decision: {late:?}"
        );
        assert_eq!(r.m.reserved_slots(), 0);
        assert!(
            !r.ev(assignment("early", None))
                .iter()
                .any(|effect| matches!(effect, WorkerStateEffect::LaunchJob { .. }))
        );
    }
}
