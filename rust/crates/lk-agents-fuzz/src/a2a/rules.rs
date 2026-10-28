//! The A2A contracts, each a function over a finished run's [`Record`] answering its violations,
//! each keyed by the kind of subject it concerns (`known_findings.tsv` keys open findings by
//! contract and subject kind). Each has a counterexample below: a real run's record made to break
//! it, which it must reject.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_core::a2a::{Policy, channel_of, resolve_policy};
use lk_agents_storage::StorageError;
use lk_agents_storage::run::{INTERRUPTED_KEY, MERGED_INTO_KEY, TaskRow, TaskState};

use super::Record;
use super::door::FOLLOW_UP_REFUSED;

/// One broken promise: the kind of subject it concerns, and what happened.
#[derive(Debug, Clone)]
pub struct Violation {
    pub subject: &'static str,
    pub text: String,
}

fn v(subject: &'static str, text: String) -> Violation {
    Violation { subject, text }
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
        write!(f, "{} ({}): {}", self.contract, self.violation.subject, self.violation.text)
    }
}

type Rule = fn(&Record) -> Vec<Violation>;

/// Every rule, by name.
pub const RULES: [(&str, Rule); 12] = [
    ("message_executes_once", message_executes_once),
    ("request_answered_once", request_answered_once),
    ("terminal_task_never_changes", terminal_task_never_changes),
    ("cancel_answers_the_row", cancel_answers_the_row),
    ("commit_is_what_the_run_staged", commit_is_what_the_run_staged),
    ("only_interrupt_yields", only_interrupt_yields),
    ("one_question_pending", one_question_pending),
    ("listing_is_complete_and_bound", listing_is_complete_and_bound),
    ("committed_items_stay_once", committed_items_stay_once),
    ("received_message_is_drained", received_message_is_drained),
    ("refusal_only_while_held", refusal_only_while_held),
    ("answer_is_an_outcome", answer_is_an_outcome),
];

/// Every violation of every rule; a terminal row seen changing while the run went is
/// `terminal_task_never_changes`'s too.
pub fn check(record: &Record) -> Vec<Finding> {
    let online = record.online.iter().map(|text| Finding {
        contract: "terminal_task_never_changes",
        violation: v("Task", text.clone()),
    });
    let mut out: Vec<Finding> = online.collect();
    for (contract, rule) in RULES {
        out.extend(rule(record).into_iter().map(|violation| Finding { contract, violation }));
    }
    out
}

/// The runs whose commit applied: a staged run with its `run_ended` in the journal.
fn committed(record: &Record) -> impl Iterator<Item = &super::StagedRun> {
    let ended: BTreeSet<&str> = record
        .events
        .iter()
        .filter(|e| e.r#type == "run_ended")
        .map(|e| e.run_id.as_str())
        .collect();
    record.seen.staged.iter().filter(move |s| ended.contains(s.run_id.as_str()))
}

/// A message's subject kind, by its channel's policy: `Inbox` under `batch`, `Concurrent`,
/// `Refusable` under `reject`; under `interrupt` and `queue` a `Continuation` when it continued
/// a pending question (it named that task), else a `Message`.
fn subject(record: &Record, message: &str) -> &'static str {
    match record.policies.get(message).copied() {
        Some(Policy::Batch) => "Inbox",
        Some(Policy::Concurrent) => "Concurrent",
        Some(Policy::Reject) => "Refusable",
        _ if record.continued.contains(message) => "Continuation",
        _ => "Message",
    }
}

/// A message runs at most one committed turn, and every answer to it names its one task or the
/// task that task was merged into (a later message answered it).
pub fn message_executes_once(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut runs: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for s in committed(record) {
        for message in &s.messages {
            runs.entry(message).or_default().push(&s.run_id);
        }
    }
    for (message, runs) in runs.iter().filter(|(_, r)| r.len() > 1) {
        out.push(v(
            subject(record, message),
            format!("{message} committed {} turns: {runs:?}", runs.len()),
        ));
    }
    let tasks: BTreeMap<&str, &TaskRow> =
        record.tasks.iter().filter_map(|t| Some((t.message_id.as_deref()?, t))).collect();
    for r in &record.seen.responses {
        let Ok(result) = &r.answer else { continue };
        let Some(task) = tasks.get(r.message_id.as_str()) else {
            out.push(v(
                subject(record, &r.message_id),
                format!("{} was answered, no task holds it", r.message_id),
            ));
            continue;
        };
        let merged = task.data.metadata.get(MERGED_INTO_KEY).and_then(|v| v.as_str());
        let named = [result.task.task_id.as_str(), result.answered_by.as_str()];
        if !named.iter().any(|n| *n == task.task_id || Some(*n) == merged) {
            out.push(v(
                subject(record, &r.message_id),
                format!(
                    "{} is {}, request {} was answered by {named:?}",
                    r.message_id, task.task_id, r.request
                ),
            ));
        }
    }
    out
}

/// Every request gets one answer; one whose worker died gets at most the one it gave first.
pub fn request_answered_once(record: &Record) -> Vec<Violation> {
    let mut count: BTreeMap<usize, usize> = BTreeMap::new();
    for r in &record.seen.responses {
        *count.entry(r.request).or_default() += 1;
    }
    let lost: BTreeSet<usize> = record.seen.lost.iter().copied().collect();
    let mut out = Vec::new();
    for (id, message) in &record.requests {
        let n = count.get(id).copied().unwrap_or(0);
        if n > 1 || (n == 0 && !lost.contains(id)) {
            out.push(v("Request", format!("request {id} ({message}) answered {n} times")));
        }
    }
    out
}

/// A terminal task keeps its status: no cancel revives it, no commit overwrites it.
pub fn terminal_task_never_changes(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    for (task, log) in &record.seen.statuses {
        if let Some(i) = log.iter().position(|s| s.is_terminal())
            && i + 1 < log.len()
        {
            out.push(v("Task", format!("{task}: {log:?}")));
        }
    }
    out
}

/// `tasks/cancel` cancels a live task and answers a terminal one as it is.
pub fn cancel_answers_the_row(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    for c in &record.seen.cancels {
        let expected = if c.before.is_terminal() { c.before } else { TaskState::Canceled };
        if c.after != Some(expected) {
            out.push(v(
                "Task",
                format!("{}: {:?} canceled answered {:?}", c.task_id, c.before, c.after),
            ));
        }
    }
    out
}

/// A run whose commit applied leaves each terminal row it staged as staged, unless a client
/// canceled it first; a lost acknowledgment never ends as a failure the run did not observe
/// (decision 39), and an answer given before the resolution shows the task live or as it ended.
pub fn commit_is_what_the_run_staged(record: &Record) -> Vec<Violation> {
    let final_status: BTreeMap<&str, TaskState> =
        record.tasks.iter().map(|t| (t.task_id.as_str(), t.status)).collect();
    let mut out = Vec::new();
    for s in committed(record) {
        for (task, staged) in s.writes.iter().filter(|(_, s)| s.is_terminal()) {
            let now = final_status.get(task.as_str()).copied();
            if now != Some(*staged) && now != Some(TaskState::Canceled) {
                out.push(v(
                    if s.leased { "Run" } else { "Concurrent" },
                    format!("{}: staged {task} {staged:?}, it is {now:?}", s.run_id),
                ));
            }
        }
    }
    for r in record.seen.responses.iter().filter(|r| r.uncertain) {
        let Ok(result) = &r.answer else { continue };
        let task = &result.task;
        let now = final_status.get(task.task_id.as_str()).copied();
        if task.terminal() && now != Some(task.status) {
            out.push(v(
                "Run",
                format!("{}: answered {:?} unresolved, it is {now:?}", task.task_id, task.status),
            ));
        }
    }
    out
}

/// Only a turn under `interrupt` yields to a newer message: an interrupted task answered a
/// message of a channel whose policy is `interrupt` (`a2a` by default). The turn's message is the
/// task's last: a continued task keeps the question's message first (78f8da08f).
pub fn only_interrupt_yields(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    for t in &record.tasks {
        let interrupted = t.data.metadata.get(INTERRUPTED_KEY).and_then(|v| v.as_bool());
        let channel = t.data.history.last().map(channel_of);
        let policy = channel.as_deref().map(|c| resolve_policy(&record.policy, c));
        if interrupted == Some(true) && policy != Some(Policy::Interrupt) {
            out.push(v("Task", format!("{} was interrupted on channel {channel:?}", t.task_id)));
        }
    }
    for s in record.seen.staged.iter().filter(|s| s.yielded_to_waiter && !s.interrupt) {
        out.push(v("Task", format!("{} yielded to a waiter under queue", s.run_id)));
    }
    out
}

/// Every history item a committed run added is in the lane's history, once: no run's commit
/// (an exclusive one fenced by version, a concurrent one appending) loses or repeats another's.
pub fn committed_items_stay_once(record: &Record) -> Vec<Violation> {
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for item in &record.items {
        *count.entry(item.id.as_str()).or_default() += 1;
    }
    let mut out = Vec::new();
    for s in committed(record) {
        for item in &s.items {
            let n = count.get(item.as_str()).copied().unwrap_or(0);
            if n != 1 {
                out.push(v(
                    "Run",
                    format!("{}: its item {item} is in the history {n} times", s.run_id),
                ));
            }
        }
    }
    out
}

/// A message received into the inbox is drained: once every request is done, its task is no
/// longer `submitted`, unless the lane's last runner died with it pending.
pub fn received_message_is_drained(record: &Record) -> Vec<Violation> {
    if !record.seen.lost.is_empty() {
        return Vec::new();
    }
    let status: BTreeMap<&str, TaskState> =
        record.at_rest.iter().map(|t| (t.task_id.as_str(), t.status)).collect();
    let mut out = Vec::new();
    for (request, message) in &record.seen.received {
        let task = format!("AST_r{request}");
        if status.get(task.as_str()) == Some(&TaskState::Submitted) {
            out.push(v("Inbox", format!("{message} ({task}) stayed in the inbox undrained")));
        }
    }
    out
}

/// `reject` refuses a message only while another run holds the lane, and a refused message
/// leaves no task row.
pub fn refusal_only_while_held(record: &Record) -> Vec<Violation> {
    let rows: BTreeSet<&str> = record.tasks.iter().map(|t| t.task_id.as_str()).collect();
    let mut out = Vec::new();
    for r in &record.seen.refusals {
        if !r.held {
            out.push(v("Request", format!("request {} was refused on a free lane", r.request)));
        }
        if rows.contains(r.task_id.as_str()) {
            out.push(v(
                "Request",
                format!("request {} was refused and left {}", r.request, r.task_id),
            ));
        }
    }
    out
}

/// A turn settles the pending question before it may ask its own: one is pending at a time
/// (`concurrent` turns may overlap and are not counted). An `Inbox` lane is one where a drained
/// batch left a question.
pub fn one_question_pending(record: &Record) -> Vec<Violation> {
    let s = &record.seen;
    if s.most_pending < 2 {
        return Vec::new();
    }
    let message =
        |task: &String| record.tasks.iter().find(|t| t.task_id == *task)?.message_id.clone();
    let inbox = s
        .most_pending_tasks
        .iter()
        .filter_map(message)
        .any(|m| record.policies.get(&m) == Some(&Policy::Batch));
    let subject = if inbox { "Inbox" } else { "Lane" };
    vec![v(subject, format!("{:?} were pending at once", s.most_pending_tasks))]
}

/// A request is answered with the task's outcome or with an error the client can act on: the
/// door refused a follow-up to a terminal task, the lane refused it (`reject`), the database could not be reached or its commit stayed
/// unresolved or was rejected, the handler was cancelled. A framework-internal condition (a
/// lease another run holds, an invalid write) never reaches the client.
pub fn answer_is_an_outcome(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    for r in &record.seen.responses {
        let Err(e) = &r.answer else { continue };
        let actionable = match e {
            StorageError::Rejected { .. } => {
                record.policies.get(&r.message_id) == Some(&Policy::Reject)
            }
            StorageError::Unavailable(_)
            | StorageError::Uncertain(_)
            | StorageError::VersionConflict { .. }
            | StorageError::Cancelled(_) => true,
            StorageError::Invalid(m) => m.starts_with(FOLLOW_UP_REFUSED),
            _ => false,
        };
        if !actionable {
            out.push(v(
                subject(record, &r.message_id),
                format!("request {} ({}) was answered {e:?}", r.request, r.message_id),
            ));
        }
    }
    out
}

/// A listing returns no task twice, every task that matched its filters and did not change
/// while it paged, and refuses a page token presented under other filters.
pub fn listing_is_complete_and_bound(record: &Record) -> Vec<Violation> {
    let mut out = Vec::new();
    for (i, l) in record.seen.listings.iter().enumerate() {
        let listed: BTreeSet<&str> = l.listed.iter().map(String::as_str).collect();
        if listed.len() != l.listed.len() {
            out.push(v("Listing", format!("listing {i} returned a task twice: {:?}", l.listed)));
        }
        for (task, at, status) in &l.before {
            let unchanged = l.after.iter().any(|a| a.0 == *task && a.1 == *at);
            if unchanged && l.matches(*at, *status) && !listed.contains(task.as_str()) {
                out.push(v(
                    "Listing",
                    format!("listing {i} missed {task}, unchanged while it paged"),
                ));
            }
        }
        if l.foreign_token == Some(false) {
            out.push(v(
                "Listing",
                format!("listing {i}'s page token was accepted under other filters"),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2a::{Cancel, Refusal, Response, run};
    use crate::sched::Sampling;

    /// A run that keeps every rule and reached what the counterexamples break: answers,
    /// committed runs, a cancel and a listing with pages.
    fn reached() -> Record {
        (0..200)
            .map(|seed| run(seed, 8, Sampling::BasicPos))
            .find(|r| {
                check(r).is_empty()
                    && committed(r).count() > 1
                    && !r.seen.cancels.is_empty()
                    && r.seen.listings.iter().any(|l| l.listed.len() > 1)
            })
            .expect("a seed reaches committed runs, a cancel and a listing")
    }

    fn breaks(rule: Rule, mutate: impl FnOnce(&mut Record)) {
        let mut record = reached();
        assert!(rule(&record).is_empty(), "{:?}", rule(&record));
        mutate(&mut record);
        assert!(!rule(&record).is_empty(), "the counterexample passed");
    }

    #[test]
    fn a_second_committed_turn_or_another_task_for_one_message_is_rejected() {
        breaks(message_executes_once, |r| {
            let mut twice = committed(r).next().unwrap().clone();
            twice.run_id = format!("{}_again", twice.run_id);
            let ended = r.events.iter().find(|e| e.r#type == "run_ended").unwrap().clone();
            r.events.push(lk_agents_storage::schema::EventRow {
                run_id: twice.run_id.clone(),
                ..ended
            });
            r.seen.staged.push(twice);
        });
        breaks(message_executes_once, |r| {
            let mut other = r.seen.responses.iter().find(|x| x.answer.is_ok()).unwrap().clone();
            if let Ok(result) = &mut other.answer {
                result.answered_by = "AST_other".into();
                result.task.task_id = "AST_other".into();
            }
            r.seen.responses.push(other);
        });
    }

    #[test]
    fn a_request_answered_twice_or_never_is_rejected() {
        breaks(request_answered_once, |r| {
            let again = r.seen.responses[0].clone();
            r.seen.responses.push(again);
        });
        breaks(request_answered_once, |r| {
            let request = r.seen.responses.remove(0).request;
            r.seen.lost.retain(|l| *l != request);
        });
    }

    #[test]
    fn a_terminal_task_that_changes_is_rejected() {
        breaks(terminal_task_never_changes, |r| {
            let log =
                r.seen.statuses.values_mut().find(|l| l.last().unwrap().is_terminal()).unwrap();
            log.push(TaskState::Working);
        });
    }

    #[test]
    fn a_cancel_that_revives_or_misses_is_rejected() {
        breaks(cancel_answers_the_row, |r| {
            let task_id = "AST_x".to_owned();
            let c =
                Cancel { task_id, before: TaskState::Completed, after: Some(TaskState::Canceled) };
            r.seen.cancels.push(c);
        });
        breaks(cancel_answers_the_row, |r| {
            let task_id = "AST_x".to_owned();
            let c = Cancel { task_id, before: TaskState::Working, after: Some(TaskState::Working) };
            r.seen.cancels.push(c);
        });
    }

    #[test]
    fn a_committed_row_other_than_staged_is_rejected() {
        breaks(commit_is_what_the_run_staged, |r| {
            let run = committed(r).find(|s| s.writes.iter().any(|w| w.1.is_terminal())).unwrap();
            let (task, staged) = run.writes.iter().find(|w| w.1.is_terminal()).unwrap().clone();
            let other =
                if staged == TaskState::Failed { TaskState::Completed } else { TaskState::Failed };
            r.tasks.iter_mut().find(|t| t.task_id == task).unwrap().status = other;
        });
    }

    #[test]
    fn a_yield_under_queue_is_rejected() {
        breaks(only_interrupt_yields, |r| {
            let mut queued = r.seen.staged[0].clone();
            queued.interrupt = false;
            queued.yielded_to_waiter = true;
            r.seen.staged.push(queued);
        });
    }

    #[test]
    fn two_pending_questions_are_rejected() {
        breaks(one_question_pending, |r| {
            r.seen.most_pending = 2;
            r.seen.most_pending_tasks = vec!["AST_x".into(), "AST_y".into()];
        });
    }

    fn paged(r: &mut Record) -> &mut crate::a2a::list::Listing {
        r.seen.listings.iter_mut().find(|l| l.listed.len() > 1).unwrap()
    }

    #[test]
    fn a_listing_that_repeats_misses_or_takes_a_foreign_token_is_rejected() {
        breaks(listing_is_complete_and_bound, |r| {
            let l = paged(r);
            let first = l.listed[0].clone();
            l.listed.push(first);
        });
        breaks(listing_is_complete_and_bound, |r| {
            // a task that matched and never changed, left out
            let l = paged(r);
            let missed = l.listed.remove(0);
            let entry = (missed, l.updated_after + 1, l.status.unwrap_or(TaskState::Completed));
            l.before.retain(|b| b.0 != entry.0);
            l.after.retain(|a| a.0 != entry.0);
            l.before.push(entry.clone());
            l.after.push(entry);
        });
        breaks(listing_is_complete_and_bound, |r| r.seen.listings[0].foreign_token = Some(false));
    }

    #[test]
    fn a_committed_item_lost_or_repeated_is_rejected() {
        let item = |r: &Record| {
            let run = committed(r).find(|s| !s.items.is_empty()).unwrap();
            run.items[0].clone()
        };
        breaks(committed_items_stay_once, |r| {
            let lost = item(r);
            r.items.retain(|i| i.id != lost);
        });
        breaks(committed_items_stay_once, |r| {
            let again = item(r);
            let row = r.items.iter().find(|i| i.id == again).unwrap().clone();
            r.items.push(row);
        });
    }

    #[test]
    fn a_received_message_left_in_the_inbox_is_rejected() {
        breaks(received_message_is_drained, |r| {
            r.seen.received.push((99, "m99".into()));
            let mut row = r.at_rest[0].clone();
            (row.task_id, row.status) = ("AST_r99".into(), TaskState::Submitted);
            r.at_rest.push(row);
        });
    }

    #[test]
    fn a_refusal_on_a_free_lane_or_with_a_row_is_rejected() {
        breaks(refusal_only_while_held, |r| {
            r.seen.refusals.push(Refusal { request: 99, task_id: "AST_r99".into(), held: false });
        });
        breaks(refusal_only_while_held, |r| {
            let task_id = r.tasks[0].task_id.clone();
            r.seen.refusals.push(Refusal { request: 99, task_id, held: true });
        });
    }

    #[test]
    fn an_internal_error_answered_is_rejected() {
        breaks(answer_is_an_outcome, |r| {
            let message_id = r.seen.responses[0].message_id.clone();
            let answer = Err(StorageError::Rejected { running_task_id: None });
            r.seen.responses.push(Response { request: 99, message_id, answer, uncertain: false });
        });
    }
}
