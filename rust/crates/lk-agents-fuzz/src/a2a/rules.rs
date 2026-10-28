//! The A2A contracts, each a function over a finished run's [`Record`] answering its violations,
//! each keyed by the kind of subject it concerns (`known_findings.tsv` keys open findings by
//! contract and subject kind). Each has a counterexample below, which it must reject.

use std::collections::BTreeMap;

use lk_agents_abi::storage::{StorageError, TaskState};
use lk_agents_core::a2a::Policy;

use super::Record;

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

pub const RULES: [(&str, Rule); 7] = [
    ("message_executes_once", message_executes_once),
    ("request_answered_once", request_answered_once),
    ("answer_is_an_outcome", answer_is_an_outcome),
    ("terminal_task_never_changes", terminal_task_never_changes),
    ("no_failure_nobody_observed", no_failure_nobody_observed),
    ("one_task_per_message", one_task_per_message),
    ("received_message_is_drained", received_message_is_drained),
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

fn policy_of(record: &Record, message_id: &str) -> Policy {
    record.requests.iter().find(|r| r.message_id == message_id).map_or(Policy::Queue, |r| r.policy)
}

/// A message's turn commits at most once, whatever resends and faults happened.
pub fn message_executes_once(record: &Record) -> Vec<Violation> {
    record
        .port
        .ledger
        .executions
        .iter()
        .filter(|(_, runs)| runs.len() > 1)
        .map(|(message_id, runs)| Violation {
            subject: subject(policy_of(record, message_id)),
            text: format!("{message_id} committed by {runs:?}"),
        })
        .collect()
}

/// A request whose worker lived is answered: no task is watched or waited on forever.
pub fn request_answered_once(record: &Record) -> Vec<Violation> {
    record
        .requests
        .iter()
        .filter(|r| !r.killed && r.answer.is_none())
        .map(|r| Violation {
            subject: subject(r.policy),
            text: format!("{} never answered by {:?}", r.message_id, record.now),
        })
        .collect()
}

/// An error answer is one the client can act on: retry (`Unavailable`, `Uncertain`, a lost
/// lease or a fenced conflict), a cancelled turn, `Rejected` only under `reject`, `NotFound` for
/// a thread that is gone. A framework-internal `Invalid` never reaches the client.
pub fn answer_is_an_outcome(record: &Record) -> Vec<Violation> {
    record
        .requests
        .iter()
        .filter_map(|r| match &r.answer {
            Some(Err(e)) => {
                let actionable = match e {
                    StorageError::Rejected { .. } => r.policy == Policy::Reject,
                    StorageError::Unavailable(_)
                    | StorageError::Uncertain(_)
                    | StorageError::LeaseLost(_)
                    | StorageError::VersionConflict { .. }
                    | StorageError::NotFound(_)
                    | StorageError::Cancelled(_) => true,
                    _ => false,
                };
                (!actionable).then(|| Violation {
                    subject: subject(r.policy),
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
        .port
        .ledger
        .terminal_changes
        .iter()
        .map(|change| Violation { subject: "Task", text: change.clone() })
        .collect()
}

/// A commit that applied with its acknowledgment lost ends its task completed (unless a client
/// canceled it), never failed: the framework reports no failure it did not observe.
pub fn no_failure_nobody_observed(record: &Record) -> Vec<Violation> {
    record
        .port
        .ledger
        .applied_unacknowledged
        .iter()
        .filter_map(|task_id| {
            let row = record.port.tasks.get(task_id)?;
            (row.status == TaskState::Failed).then(|| Violation {
                subject: "LostAck",
                text: format!("{task_id} applied, then answered failed"),
            })
        })
        .collect()
}

/// Every answer to one message names the task its first send created.
pub fn one_task_per_message(record: &Record) -> Vec<Violation> {
    let mut named: BTreeMap<&str, (&str, Policy)> = BTreeMap::new();
    let mut out = Vec::new();
    for r in &record.requests {
        let Some(Ok(answer)) = &r.answer else { continue };
        let task = answer.task_id.as_str();
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
            .port
            .tasks
            .values()
            .find(|row| row.message_id.as_deref() == Some(r.message_id.as_str()));
        if let Some(receipt) = receipt
            && receipt.task_id != task
            && answer.answered_by.as_deref() != Some(task)
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

/// Every received inbox message whose task has not ended is drained once the run settles.
pub fn received_message_is_drained(record: &Record) -> Vec<Violation> {
    let pending = record.port.undrained();
    if pending > 0 {
        vec![Violation { subject: "Inbox", text: format!("{pending} inbox rows never drained") }]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> Record {
        (0..64).map(|seed| super::super::run(seed, 6)).find(|r| check(r).is_empty()).unwrap()
    }

    #[test]
    fn a_second_commit_of_a_message_is_rejected() {
        let mut r = record();
        r.port.ledger.executions.insert("m0".into(), ["a".into(), "b".into()].into());
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
    fn a_changed_terminal_row_is_rejected() {
        let mut r = record();
        r.port.ledger.terminal_changes.push("AST_1 Completed -> Failed".into());
        assert!(!terminal_task_never_changes(&r).is_empty());
    }

    #[test]
    fn a_failed_task_whose_commit_applied_is_rejected() {
        let mut r = record();
        let (id, row) = r.port.tasks.iter_mut().next().unwrap();
        row.status = TaskState::Failed;
        let id = id.clone();
        r.port.ledger.applied_unacknowledged.insert(id);
        assert!(!no_failure_nobody_observed(&r).is_empty());
    }
}
