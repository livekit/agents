//! The text deterministic simulation: production text runs (the AgentRun driver, TextSession)
//! and A2A turns (the runtime's `Turn`: receive, run, commit, drain) over a fake model and fake
//! host callbacks, on the production SQLite thread storage with faults under its connection,
//! on paused virtual time. `rules/text_*.rs` judge the finished [`TextRecording`].

mod drive;
mod fakes;

pub use drive::{Outcome, Scenario, TurnPlan, run};
pub use fakes::{Callback, Fault};

use crate::check::{Check, Failure};
use lk_agents_abi::storage::TaskState;

/// How long a client waits on a run before it cancels it.
pub const CLIENT_DEADLINE_MS: u64 = 120_000;
/// How long a cancelled run may take to report and its driver to exit: the core's cleanup
/// bound and slack for the work it cancels.
pub const EXIT_BOUND_MS: u64 = 15_000;

/// What one simulation observed, judged once it finished.
#[derive(Clone, Debug, Default)]
pub struct TextRecording {
    pub runs: Vec<RunRecord>,
    pub turns: Vec<TurnRecord>,
    /// The partition once every turn ended; `None` when the scenario sent no message.
    pub partition_key: Option<PartitionRecord>,
    pub messages: Vec<MessageRecord>,
    pub tasks: Vec<TaskRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunKind {
    /// `agent.run()`: the AgentRun driver.
    Direct,
    /// A TextSession request.
    Session,
    /// The AgentRun an A2A handler runs.
    Handler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelBy {
    /// The client chose to stop it.
    Client,
    /// The client gave up at [`CLIENT_DEADLINE_MS`].
    Deadline,
}

#[derive(Clone, Debug)]
pub struct RunRecord {
    pub id: u32,
    pub kind: RunKind,
    /// A fake of this run was told never to answer (it still honours cancellation).
    pub hung: bool,
    /// A fake of this run panicked.
    pub panicked: bool,
    pub cancelled: Option<(u64, CancelBy)>,
    /// When the run's final outcome reached the client.
    pub terminal_ms: Option<u64>,
    /// When its driver exited (its update stream closed).
    pub exited_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Client {
    Waits,
    /// Stops reading: the turn runs on, owned like the door owns it.
    Disconnects,
    /// Sends the A2A cancel for its task.
    Cancels,
    /// Drops the turn's future at an await.
    Aborts,
}

#[derive(Clone, Debug)]
pub struct TurnRecord {
    pub message_id: String,
    pub client: Client,
    /// The task status the turn answered, its error, or `None` when it never ended.
    pub answered: Option<Result<TaskState, String>>,
}

#[derive(Clone, Copy, Debug)]
pub struct PartitionRecord {
    /// Runs started and never ended.
    pub unended: u32,
    /// A lease still held once every turn ended and the lease's TTL passed.
    pub held: bool,
    /// The database was down, lost a write or a worker died.
    pub faulted: bool,
}

#[derive(Clone, Debug)]
pub struct MessageRecord {
    pub message_id: String,
    /// History items carrying the message's id.
    pub items: u32,
    /// Task rows the message created.
    pub tasks: u32,
}

#[derive(Clone, Debug)]
pub struct TaskRecord {
    pub task_id: String,
    pub status: TaskState,
    /// Its client disconnected, cancelled or aborted.
    pub left: bool,
}

/// One guarantee over the text recording.
pub trait TextRule {
    const CONTRACT: TextContract;
    fn on_complete(r: &TextRecording, check: &mut Check);
}

pub struct TextContract {
    pub id: &'static str,
    pub guarantee: &'static str,
    pub examples: &'static [fn() -> TextRecording],
    pub counterexamples: &'static [fn() -> TextRecording],
}

pub struct TextRegistered {
    pub contract: TextContract,
    pub on_complete: fn(&TextRecording, &mut Check),
}

impl TextRegistered {
    pub fn run(&self, recording: &TextRecording) -> Check {
        let mut check = Check::new(self.contract.id);
        (self.on_complete)(recording, &mut check);
        check
    }
}

pub mod rules {
    use super::{TextRegistered, TextRule};
    include!(concat!(env!("OUT_DIR"), "/text_rules.rs"));
}

/// Every text rule over `recording`: the failures, and each rule's expectation count.
pub fn check_all(recording: &TextRecording) -> (Vec<Failure>, Vec<(&'static str, u32)>) {
    let mut failures = Vec::new();
    let mut counts = Vec::new();
    for rule in rules::TEXT_RULES {
        let check = rule.run(recording);
        counts.push((rule.contract.id, check.expectations));
        failures.extend(check.failures);
    }
    (failures, counts)
}

/// A run record for rule examples.
pub fn run_record(kind: RunKind) -> RunRecord {
    RunRecord {
        id: 0,
        kind,
        hung: false,
        panicked: false,
        cancelled: None,
        terminal_ms: Some(1_000),
        exited_ms: Some(1_000),
    }
}
