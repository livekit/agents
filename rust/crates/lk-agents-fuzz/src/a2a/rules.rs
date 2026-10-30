//! The A2A contracts, each a function over a finished run's [`Record`] answering its violations,
//! each keyed by the kind of subject it concerns (`known_findings.tsv` keys open findings by
//! contract and subject kind). Each has a counterexample below, which it must reject.

use std::collections::BTreeMap;

use lk_agents_abi::a2a::metadata::{INTERRUPTED_KEY, MERGED_INTO_KEY, REASON_KEY};
use lk_agents_abi::storage::{StorageError, TaskState};
use lk_agents_core::a2a::Policy;

use super::{Record, Request};

/// One broken promise: the kind of subject it concerns, and what happened.
#[derive(Debug, Clone)]
pub struct Violation {
    pub subject: &'static str,
    pub text: String,
}

/// A violation of a named rule.
#[derive(Debug, Clone)]
pub struct Finding {
    pub contract: &'static str,
    pub violation: Violation,
}

impl Finding {
    /// Whether this is one of the open findings (`known_findings.tsv`).
    pub fn known(&self) -> bool {
        crate::check::KNOWN_FINDINGS
            .iter()
            .any(|k| k.contract == self.contract && k.subject == self.violation.subject)
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}]: {}", self.contract, self.violation.subject, self.violation.text)
    }
}

type Rule = fn(&Record) -> Vec<Violation>;

pub const RULES: [(&str, Rule); 10] = [
    ("message_executes_once", message_executes_once),
    ("one_question_pending", one_question_pending),
    ("commit_is_what_the_run_staged", commit_is_what_the_run_staged),
    ("request_answered_once", request_answered_once),
    ("answer_is_an_outcome", answer_is_an_outcome),
    ("terminal_task_never_changes", terminal_task_never_changes),
    ("no_failure_nobody_observed", no_failure_nobody_observed),
    ("one_task_per_message", one_task_per_message),
    ("received_message_is_drained", received_message_is_drained),
    ("cancel_ends_only_its_task", cancel_ends_only_its_task),
];

/// Every violation of every rule.
pub fn check(record: &Record) -> Vec<Finding> {
    RULES
        .iter()
        .flat_map(|(contract, rule)| {
            rule(record).into_iter().map(|violation| Finding { contract, violation })
        })
        .collect()
}

fn subject(policy: Policy) -> &'static str {
    match policy {
        Policy::Batch => "Inbox",
        Policy::Interrupt => "Interrupt",
        Policy::Queue => "Queue",
        Policy::Concurrent => "Concurrent",
        Policy::Reject => "Refusable",
    }
}

/// A request's subject: its policy's, or `Continuation` for an exclusive one naming a task.
fn subject_of(r: &Request) -> &'static str {
    match r.policy {
        Policy::Interrupt | Policy::Queue if r.continues.is_some() => "Continuation",
        policy => subject(policy),
    }
}

fn subject_of_message(record: &Record, message_id: &str) -> &'static str {
    record.requests.iter().find(|r| r.message_id == message_id).map_or("Queue", subject_of)
}

/// A message's turn commits at most once, whatever resends and faults happened, and every answer
/// names the message's own task or the one it was merged into.
pub fn message_executes_once(record: &Record) -> Vec<Violation> {
    let mut out: Vec<Violation> = record
        .ledger
        .executions
        .iter()
        .filter(|(_, runs)| runs.len() > 1)
        .map(|(message_id, runs)| Violation {
            subject: subject_of_message(record, message_id),
            text: format!("{message_id} committed by {runs:?}"),
        })
        .collect();
    for r in &record.requests {
        let Some(Ok(answer)) = &r.answer else { continue };
        let Some(task) =
            record.tasks.values().find(|t| t.message_id.as_deref() == Some(r.message_id.as_str()))
        else {
            continue;
        };
        let merged = task.data.metadata.get(MERGED_INTO_KEY).and_then(|v| v.as_str());
        // a client canceled the receipt while the turn ran: the commit left its row as the client
        // wrote it, and the continued task carries the answer
        let continued = r.continues.as_deref().filter(|_| task.status == TaskState::Canceled);
        let named = [answer.task.task_id.as_str(), answer.answered_by.as_str()];
        let ours = |n: &str| n == task.task_id || Some(n) == merged || Some(n) == continued;
        if !named.iter().any(|n| ours(n)) {
            out.push(Violation {
                subject: subject_of(r),
                text: format!(
                    "{} is {} ({:?}), answered by {named:?}",
                    r.message_id, task.task_id, task.status
                ),
            });
        }
    }
    out
}

/// At most one question waits on the lane: a reply settles the one waiting before its turn asks
/// another.
pub fn one_question_pending(record: &Record) -> Vec<Violation> {
    let pending = &record.ledger.most_pending;
    if pending.len() < 2 {
        return Vec::new();
    }
    let batched = pending.iter().any(|task_id| {
        let message = record.tasks.get(task_id).and_then(|t| t.message_id.as_deref());
        record
            .requests
            .iter()
            .any(|r| Some(r.message_id.as_str()) == message && r.policy == Policy::Batch)
    });
    vec![Violation {
        subject: if batched { "Inbox" } else { "Lane" },
        text: format!("{pending:?} were pending at once"),
    }]
}

/// A committed run's terminal rows end as its commit wrote them, unless a client canceled one: no
/// other turn overwrites them. A request its committed run answered reads its answer settled:
/// no write of the run's settlement was dropped on its way to the database. A commit leaves no
/// row `working` but one its handler answered `Working`: the run that held it ended. (A write
/// storage refuses fails the turn `Invalid`, which [`answer_is_an_outcome`] rejects.)
pub fn commit_is_what_the_run_staged(record: &Record) -> Vec<Violation> {
    let overwritten =
        record.ledger.staged_terminal.iter().filter_map(|(commit, task_id, staged)| {
            let now = record.tasks.get(task_id).map(|t| t.status);
            (now != Some(*staged) && now != Some(TaskState::Canceled)).then(|| Violation {
                subject: "Run",
                text: format!("commit {commit}: wrote {task_id} {staged:?}, it is {now:?}"),
            })
        });
    let unsettled = record.requests.iter().filter_map(|r| {
        let Some(Ok(answer)) = &r.answer else { return None };
        let task = &answer.task;
        let settled = task.terminal()
            || task.status == TaskState::InputRequired
            || record.ledger.answered_working.contains(&task.task_id);
        (answer.run_id.is_some() && !settled).then(|| Violation {
            subject: subject_of(r),
            text: format!(
                "{} answered by its run with {} {:?}",
                r.message_id, task.task_id, task.status
            ),
        })
    });
    let working = record.ledger.left_working.iter().filter_map(|(commit, task_id)| {
        (!record.ledger.answered_working.contains(task_id)).then(|| Violation {
            subject: "Run",
            text: format!("commit {commit}: left {task_id} working"),
        })
    });
    overwritten.chain(unsettled).chain(working).collect()
}

/// A request whose worker lived is answered: no task is watched or waited on forever.
pub fn request_answered_once(record: &Record) -> Vec<Violation> {
    record
        .requests
        .iter()
        .filter(|r| !r.killed && r.answer.is_none())
        .map(|r| Violation {
            subject: subject_of(r),
            text: format!("{} never answered by {:?}", r.message_id, record.now),
        })
        .collect()
}

/// An error answer is one the client can act on: retry (`Unavailable`, `Uncertain`, a lost
/// lease or a fenced conflict), a cancelled turn, `Rejected` only under `reject`, `NotFound` for
/// a thread that is gone, `Unauthorized` only while the database refused the key during the
/// request. A framework-internal `Invalid` never reaches the client.
pub fn answer_is_an_outcome(record: &Record) -> Vec<Violation> {
    record
        .requests
        .iter()
        .filter_map(|r| match &r.answer {
            Some(Err(e)) => {
                // a follow-up to a task that ended is refused: a new task in the same context
                let ended = r
                    .continues
                    .as_ref()
                    .and_then(|t| record.tasks.get(t))
                    .is_some_and(|t| t.terminal());
                let actionable = match e {
                    StorageError::Invalid(_) => ended,
                    StorageError::Rejected { .. } => r.policy == Policy::Reject,
                    StorageError::Unauthorized(_) => record
                        .ledger
                        .key_refused
                        .iter()
                        .any(|(from, until)| r.start <= *until && *from <= r.answered_at),
                    StorageError::Unavailable(_)
                    | StorageError::Uncertain(_)
                    | StorageError::LeaseLost(_)
                    | StorageError::VersionConflict { .. }
                    | StorageError::NotFound(_)
                    | StorageError::Cancelled(_) => true,
                    _ => false,
                };
                (!actionable).then(|| Violation {
                    subject: subject_of(r),
                    text: format!("{} answered {e:?}", r.message_id),
                })
            }
            _ => None,
        })
        .collect()
}

/// A terminal task never changes.
pub fn terminal_task_never_changes(record: &Record) -> Vec<Violation> {
    record
        .ledger
        .terminal_changes
        .iter()
        .map(|change| Violation { subject: "Task", text: change.clone() })
        .collect()
}

/// A commit that applied with its acknowledgment lost is never followed by the framework failing
/// its message's task (`livekit/reason` on a failure only the framework writes): it reports no
/// failure it did not observe. Nor is a task a committed handler answered `Working`: no worker
/// holds it, so none died holding it. A later turn's own outcome is that turn's.
pub fn no_failure_nobody_observed(record: &Record) -> Vec<Violation> {
    let failed = |task_id: &str| {
        let row = record.tasks.get(task_id).filter(|r| r.status == TaskState::Failed)?;
        let message = row.data.status_message.as_ref()?;
        message.metadata.as_ref()?.get(REASON_KEY)?.as_str()
    };
    let unacknowledged =
        record.ledger.applied_unacknowledged.iter().filter_map(|(task_id, staged)| {
            failed(task_id).map(|reason| Violation {
                subject: "LostAck",
                text: format!("{task_id} applied as {staged:?}, then failed {reason:?}"),
            })
        });
    let continued =
        |task_id: &str| record.requests.iter().any(|r| r.continues.as_deref() == Some(task_id));
    let working = record.ledger.answered_working.iter().filter_map(|task_id| {
        let reason = failed(task_id).filter(|_| !continued(task_id))?;
        Some(Violation {
            subject: "Working",
            text: format!("{task_id} answered Working, then failed {reason:?}"),
        })
    });
    unacknowledged.chain(working).collect()
}

/// Every answer to one message names the task its first send created.
pub fn one_task_per_message(record: &Record) -> Vec<Violation> {
    let mut named: BTreeMap<&str, (&str, Policy)> = BTreeMap::new();
    let mut out = Vec::new();
    for r in &record.requests {
        let Some(Ok(answer)) = &r.answer else { continue };
        let task = answer.task.task_id.as_str();
        match named.get(r.message_id.as_str()) {
            Some((first, _)) if *first != task => out.push(Violation {
                subject: subject(r.policy),
                text: format!("{} answered by {first} and {task}", r.message_id),
            }),
            _ => {
                named.insert(&r.message_id, (task, r.policy));
            }
        }
        let receipt = record
            .tasks
            .values()
            .find(|row| row.message_id.as_deref() == Some(r.message_id.as_str()));
        if let Some(receipt) = receipt
            && receipt.task_id != task
            && answer.answered_by != task
        {
            out.push(Violation {
                subject: subject(r.policy),
                text: format!(
                    "{} answered by {task}, its task is {}",
                    r.message_id, receipt.task_id
                ),
            });
        }
    }
    out
}

/// Every received inbox message is answered once the run settles, while its client still waits
/// on it: a client whose every request died or failed gave up, and its row is left to the next
/// holder of the lane.
pub fn received_message_is_drained(record: &Record) -> Vec<Violation> {
    let waiting = |message: &str| {
        record
            .requests
            .iter()
            .any(|r| r.message_id == message && !r.killed && !matches!(r.answer, Some(Err(_))))
    };
    let pending: Vec<&str> =
        record.unanswered_receipts.iter().map(String::as_str).filter(|m| waiting(m)).collect();
    if pending.is_empty() {
        Vec::new()
    } else {
        vec![Violation { subject: "Inbox", text: format!("{pending:?} never drained") }]
    }
}

/// A task ends `canceled` only by its own client's cancel or a newer message's interruption. A
/// reply's cancel leaves the task it continued waiting on its question, so the person can answer
/// again (docs/protocol/a2a.md#tasks).
pub fn cancel_ends_only_its_task(record: &Record) -> Vec<Violation> {
    let canceled_by_client = |task_id: &str| record.ledger.client_canceled.contains(task_id);
    let canceled = record.tasks.values().filter_map(|t| {
        let interrupted = t.data.metadata.contains_key(INTERRUPTED_KEY);
        (t.status == TaskState::Canceled && !interrupted && !canceled_by_client(&t.task_id)).then(
            || Violation {
                subject: subject_of_message(record, t.message_id.as_deref().unwrap_or_default()),
                text: format!("{} canceled, by no client nor a newer message", t.task_id),
            },
        )
    });
    let unasked = record.requests.iter().filter_map(|r| {
        let continued = record.tasks.get(r.continues.as_deref()?)?;
        let asked = record.ledger.asked.contains(&continued.task_id);
        let asks = continued.internal.as_ref().is_some_and(|i| i.ask.is_some());
        let waiting = continued.status == TaskState::InputRequired;
        (canceled_by_client(&r.task_id) && asked && waiting && !asks).then(|| Violation {
            subject: "Continuation",
            text: format!("{} canceled, {} lost its question", r.message_id, continued.task_id),
        })
    });
    canceled.chain(unasked).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run that keeps every rule, with an answer its run committed.
    fn record() -> Record {
        let committed = |r: &Record| {
            r.requests.iter().any(|q| matches!(&q.answer, Some(Ok(a)) if a.run_id.is_some()))
        };
        (0..64)
            .map(|seed| super::super::run(seed, 6))
            .find(|r| committed(r) && check(r).is_empty())
            .unwrap()
    }

    #[test]
    fn a_second_commit_of_a_message_is_rejected() {
        let mut r = record();
        r.ledger.executions.insert("m0".into(), [1, 2].into());
        assert!(!message_executes_once(&r).is_empty());
    }

    #[test]
    fn an_unanswered_live_request_is_rejected() {
        let mut r = record();
        let live = r.requests.iter_mut().find(|q| !q.killed).unwrap();
        live.answer = None;
        assert!(!request_answered_once(&r).is_empty());
    }

    #[test]
    fn an_internal_error_answered_is_rejected() {
        let mut r = record();
        let live = r.requests.iter_mut().find(|q| q.policy != Policy::Reject && !q.killed);
        if let Some(q) = live {
            q.answer = Some(Err(StorageError::Invalid("internal".into())));
            assert!(!answer_is_an_outcome(&r).is_empty());
        }
    }

    #[test]
    fn unauthorized_while_the_key_was_accepted_is_rejected() {
        let mut r = record();
        r.ledger.key_refused.clear();
        let q = r.requests.iter_mut().find(|q| !q.killed).unwrap();
        q.answer = Some(Err(StorageError::Unauthorized("the key was refused".into())));
        assert!(!answer_is_an_outcome(&r).is_empty());
    }

    #[test]
    fn a_changed_terminal_row_is_rejected() {
        let mut r = record();
        r.ledger.terminal_changes.push("AST_1 Completed -> Failed".into());
        assert!(!terminal_task_never_changes(&r).is_empty());
    }

    #[test]
    fn two_questions_pending_at_once_are_rejected() {
        let mut r = record();
        r.ledger.most_pending = vec!["AST_1".into(), "AST_2".into()];
        assert!(!one_question_pending(&r).is_empty());
    }

    #[test]
    fn an_answer_its_run_left_unsettled_is_rejected() {
        let mut r = record();
        let q =
            r.requests.iter_mut().find(|q| matches!(&q.answer, Some(Ok(a)) if a.run_id.is_some()));
        let Some(Ok(answer)) = &mut q.unwrap().answer else { unreachable!() };
        answer.task.status = TaskState::Working;
        assert!(!commit_is_what_the_run_staged(&r).is_empty());
    }

    #[test]
    fn a_staged_terminal_row_that_ended_otherwise_is_rejected() {
        let mut r = record();
        let (id, row) = r.tasks.iter_mut().next().unwrap();
        row.status = TaskState::Failed;
        let staged = (1, id.clone(), TaskState::Completed);
        r.ledger.staged_terminal.push(staged);
        assert!(!commit_is_what_the_run_staged(&r).is_empty());
    }

    #[test]
    fn a_handler_working_task_failed_by_the_framework_is_rejected() {
        let mut r = record();
        let (id, row) = r.tasks.iter_mut().next().unwrap();
        row.status = TaskState::Failed;
        row.data.status_message =
            Some(lk_agents_core::a2a::failed_message(id, "worker-lost", "m".into()));
        let id = id.clone();
        r.requests.retain(|q| q.continues.as_deref() != Some(id.as_str()));
        r.ledger.answered_working.insert(id);
        assert!(!no_failure_nobody_observed(&r).is_empty());
    }

    #[test]
    fn a_row_a_commit_left_working_is_rejected() {
        let mut r = record();
        r.ledger.left_working.push((1, "AST_1".into()));
        assert!(!commit_is_what_the_run_staged(&r).is_empty());
        // unless its handler answered it `Working`
        r.ledger.answered_working.insert("AST_1".into());
        assert!(commit_is_what_the_run_staged(&r).iter().all(|v| !v.text.contains("AST_1")));
    }

    #[test]
    fn a_task_canceled_by_no_client_is_rejected() {
        let mut r = record();
        let row = r.tasks.values_mut().next().unwrap();
        row.status = TaskState::Canceled;
        row.data.metadata.remove(INTERRUPTED_KEY);
        r.ledger.client_canceled.remove(&row.task_id);
        assert!(!cancel_ends_only_its_task(&r).is_empty());
    }

    #[test]
    fn a_continued_task_left_without_its_question_is_rejected() {
        let mut r = record();
        let row = r.tasks.values_mut().next().unwrap();
        row.status = TaskState::InputRequired;
        row.internal = None;
        let continued = Some(row.task_id.clone());
        let reply = r.requests.iter_mut().next().unwrap();
        r.ledger.asked.extend(continued.clone());
        reply.continues = continued;
        r.ledger.client_canceled.insert(reply.task_id.clone());
        assert!(!cancel_ends_only_its_task(&r).is_empty());
    }

    #[test]
    fn an_answer_naming_another_task_is_rejected() {
        let mut r = record();
        let q = r.requests.iter_mut().find(|q| matches!(q.answer, Some(Ok(_)))).unwrap();
        if let Some(Ok(answer)) = &mut q.answer {
            answer.task.task_id = "AST_other".into();
            answer.answered_by = "AST_other".into();
        }
        assert!(!message_executes_once(&r).is_empty());
    }

    #[test]
    fn a_failed_task_whose_commit_applied_is_rejected() {
        let mut r = record();
        let (id, row) = r.tasks.iter_mut().next().unwrap();
        row.status = TaskState::Failed;
        row.data.status_message =
            Some(lk_agents_core::a2a::failed_message(id, "worker-lost", "m".into()));
        let id = id.clone();
        r.ledger.applied_unacknowledged.insert(id, TaskState::Completed);
        assert!(!no_failure_nobody_observed(&r).is_empty());
    }
}
