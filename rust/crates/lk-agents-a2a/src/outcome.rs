//! The decisions of one turn, given what storage loaded and what the handler returned: which
//! task a message answers when it continues a pending question ([`answer_pending`]), whether a
//! checkpoint yields ([`checkpoint_stop`]), and what each task row and the history become
//! ([`settle`]). Ids and times are supplied by the caller.

use serde_json::{Map, Value};

use livekit_protocol::agent::chat_context::chat_item;
use livekit_protocol::agent::{ChatRole, FunctionCallOutput};
use lk_agents_storage::run::{
    INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, ItemRow, MERGED_INTO_KEY, REASON_KEY, RunOutcome,
    TaskPatch, TaskRow, TaskState, WaitOptions, agent_message, wait_patch,
};
use lk_agents_storage::schema::TaskData;
use lk_agents_storage::{StorageError, timestamp};

use crate::{
    HandlerOutput, INTERRUPTION_NEW_MESSAGE, Message, TurnResult, channel_of, chat_message,
};

/// The metadata of a `canceled` row the server interrupted, as the turn writes it.
pub fn interrupted_metadata() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(INTERRUPTED_KEY.into(), Value::Bool(true));
    m.insert(INTERRUPTION_REASON_KEY.into(), Value::String(INTERRUPTION_NEW_MESSAGE.into()));
    m
}

/// Why a turn yields at a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// A newer message waits for the lane under `interrupt`.
    Interrupted,
    /// A client canceled the task with `tasks/cancel`.
    ClientCanceled,
}

/// A checkpoint's decision from what the lane holds: under `interrupt`, another holder waiting
/// for the lane stops the turn; a canceled task stops it under every policy.
pub fn checkpoint_stop(
    interrupt: bool,
    waiter: Option<&str>,
    holder: &str,
    status: Option<TaskState>,
) -> Option<Stop> {
    if interrupt && waiter.is_some_and(|w| !w.is_empty() && w != holder) {
        Some(Stop::Interrupted)
    } else if status == Some(TaskState::Canceled) {
        Some(Stop::ClientCanceled)
    } else {
        None
    }
}

/// One task row a turn writes: its status and what changes.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskTransition {
    pub task_id: String,
    pub status: TaskState,
    pub patch: TaskPatch,
}

/// Which task a message answers once a pending question was settled.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingAnswer {
    /// The task that carries the reply.
    pub answer_task: String,
    /// The tool the answer is handed to before the model runs, when the question named one.
    pub answer_tool: Option<String>,
    pub transitions: Vec<TaskTransition>,
}

/// A conversation tool's pending `InputRequired`, if `pending` holds one: the last message is
/// the call's output (replacing the placeholder the run recorded, else inserted as item
/// `item_id`). Continuing the waiting task (`taskId`) makes it the answer, working again with the
/// message appended to its history, and merges the row the message came in under into it;
/// otherwise the waiting task completes merged into this one.
pub fn answer_pending(
    history: &mut Vec<ItemRow>,
    pending: Option<&TaskRow>,
    last: &Message,
    received_task: &str,
    now: f64,
    item_id: String,
) -> PendingAnswer {
    let mut out = PendingAnswer {
        answer_task: received_task.to_owned(),
        answer_tool: None,
        transitions: Vec::new(),
    };
    let nonempty = |s: &Option<String>| s.clone().filter(|s| !s.is_empty());
    let Some(pending) = pending else { return out };
    let Some(ask) = pending.internal.as_ref().and_then(|internal| internal.ask.as_ref()) else {
        return out;
    };
    out.answer_tool = nonempty(&ask.answer_tool);
    if let Some(call_id) = nonempty(&ask.call_id) {
        let placeholder = history.iter_mut().rev().find_map(|i| match i.kind_mut() {
            Some(chat_item::Item::FunctionCallOutput(o)) if o.call_id == call_id => Some(o),
            _ => None,
        });
        match placeholder {
            Some(o) => {
                o.output = last.text();
                o.created_at = Some(timestamp(now - 1e-6));
            }
            None => {
                let item = ItemRow::new(chat_item::Item::FunctionCallOutput(FunctionCallOutput {
                    id: item_id,
                    name: nonempty(&ask.tool).unwrap_or_default(),
                    call_id,
                    output: last.text(),
                    is_error: false,
                    created_at: Some(timestamp(now - 1e-6)),
                }));
                insert_by_created_at(history, item);
            }
        }
    }
    if last.task_id.as_deref() == Some(pending.task_id.as_str()) {
        out.answer_task = pending.task_id.clone();
        out.transitions.push(TaskTransition {
            task_id: received_task.to_owned(),
            status: TaskState::Completed,
            patch: TaskPatch {
                message_id: Some(last.message_id.clone()),
                history: Some(vec![last.to_a2a()]),
                metadata: Some(merged_into(&pending.task_id)),
                ..TaskPatch::default()
            },
        });
        // the continued task keeps its own message and history; the reply is appended
        let mut history = pending.data.history.clone();
        history.push(last.to_a2a());
        out.transitions.push(TaskTransition {
            task_id: pending.task_id.clone(),
            status: TaskState::Working,
            patch: TaskPatch { history: Some(history), ..TaskPatch::default() },
        });
    } else {
        out.transitions.push(TaskTransition {
            task_id: pending.task_id.clone(),
            status: TaskState::Completed,
            patch: TaskPatch { metadata: Some(merged_into(received_task)), ..TaskPatch::default() },
        });
    }
    out
}

/// The ids a settlement may mint, supplied by the caller.
#[derive(Debug, Clone)]
pub struct TurnIds {
    /// The assistant item of a text answer no run stored.
    pub assistant_item: String,
    /// The answer task's status message.
    pub status_message: String,
    pub artifact: String,
}

/// What a turn settled: every row it writes, how the run ends and the reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Settlement {
    pub transitions: Vec<TaskTransition>,
    pub outcome: RunOutcome,
    pub reason: Option<String>,
    /// Journal entries to append: `(type, payload)`.
    pub events: Vec<(String, Value)>,
    pub output: Option<Value>,
    pub interrupted: bool,
}

/// What the turn's inputs and the handler's output make of the history and the task rows.
pub struct Settle<'a> {
    pub batch: &'a [Message],
    /// One received task per message, in order.
    pub task_ids: &'a [String],
    pub answer_task: &'a str,
    /// The batch as one user message, stored when no run stored it.
    pub user_message: ItemRow,
    pub out: &'a HandlerOutput,
    pub stop: Option<Stop>,
    /// The answer row was already canceled by a client when the turn yielded.
    pub answer_canceled: bool,
    /// Unix seconds, after everything loaded.
    pub now: f64,
    pub ids: TurnIds,
}

/// The turn's end: the runs' new items (or the message and a text answer) into `history`, then
/// the rows: a client cancel leaves the answer row as the client wrote it; a raise fails it; an
/// interrupt cancels it; a question waits; otherwise it completes with the answer as an artifact.
/// Every earlier message of a batch completes merged into the answer.
pub fn settle(history: &mut Vec<ItemRow>, s: Settle<'_>) -> Result<Settlement, StorageError> {
    let out = s.out;
    let last = s.batch.last().ok_or_else(|| StorageError::Invalid("an empty batch".into()))?;
    if !out.new_items.is_empty() {
        for payload in &out.new_items {
            let kind = payload.item.clone().ok_or_else(|| {
                StorageError::Invalid("the handler returned an empty chat item".into())
            })?;
            let user = matches!(&kind, chat_item::Item::Message(m) if m.role() == ChatRole::User);
            let item = ItemRow::new(kind);
            let id = item.id.clone();
            if !history.iter().any(|i| i.id == id) {
                insert_by_created_at(history, item);
            }
            if user && let Some(stored) = history.iter_mut().find(|i| i.id == id) {
                // the A2A `messageId` it came from: the idempotency key on `items`
                stored.message_id = Some(last.message_id.clone());
            }
        }
    } else {
        let user = ItemRow { message_id: Some(last.message_id.clone()), ..s.user_message };
        insert_by_created_at(history, user);
        if let Some(text) = out.output.as_ref().and_then(Value::as_str)
            && !text.is_empty()
            && out.error.is_none()
            && !out.interrupted
        {
            let item = chat_message(
                ChatRole::Assistant,
                vec![text.to_owned()],
                s.now + 1e-6,
                s.ids.assistant_item,
            );
            insert_by_created_at(history, item);
        }
    }
    let mut settled = Settlement {
        transitions: Vec::new(),
        outcome: RunOutcome::Completed,
        reason: None,
        events: out.events.iter().map(|e| (e.r#type.clone(), e.payload.clone())).collect(),
        output: None,
        interrupted: s.stop.is_some() || out.interrupted,
    };
    if s.stop == Some(Stop::ClientCanceled) || (settled.interrupted && s.answer_canceled) {
        // the row is the client's; the history keeps what ran, the run says why it ended
        settled.outcome = RunOutcome::Canceled;
        settled.reason = Some("client-cancel".into());
        return Ok(settled);
    }
    let answer = s.answer_task.to_owned();
    let n = s.batch.len();
    for (tid, m) in s.task_ids[..n - 1].iter().zip(&s.batch[..n - 1]) {
        settled.transitions.push(TaskTransition {
            task_id: tid.clone(),
            status: TaskState::Completed,
            patch: TaskPatch {
                message_id: Some(m.message_id.clone()),
                history: Some(vec![m.to_a2a()]),
                metadata: Some(merged_into(&answer)),
                ..TaskPatch::default()
            },
        });
    }
    // a continued task (an answer to its pending question) keeps its own message and the
    // history `answer_pending` gave it: the message is the receipt row's
    let received = s.task_ids.last().is_some_and(|t| *t == answer);
    let base = match received {
        true => TaskPatch {
            message_id: Some(last.message_id.clone()),
            history: Some(s.batch.iter().map(Message::to_a2a).collect()),
            ..TaskPatch::default()
        },
        false => TaskPatch::default(),
    };
    if let Some(err) = &out.error {
        settled.transitions.push(TaskTransition {
            task_id: answer.clone(),
            status: TaskState::Failed,
            patch: TaskPatch {
                status_message: Some(agent_text(&answer, err, s.ids.status_message)),
                ..base
            },
        });
        settled.events.push(("error".into(), serde_json::json!({"error": err})));
        settled.outcome = RunOutcome::Failed;
        settled.reason = Some(err.clone());
        return Ok(settled);
    }
    if settled.interrupted {
        settled.transitions.push(TaskTransition {
            task_id: answer,
            status: TaskState::Canceled,
            patch: TaskPatch { metadata: Some(interrupted_metadata()), ..base },
        });
        settled.outcome = RunOutcome::Canceled;
        settled.reason = Some(INTERRUPTION_NEW_MESSAGE.into());
        return Ok(settled);
    }
    let metadata = Some(out.metadata.clone()).filter(|m| !m.is_empty());
    if let Some(question) = &out.input_required {
        let call = out.pending_call.as_ref();
        let wait = WaitOptions {
            tool: call.map(|c| c.name.clone()),
            call_id: call.map(|c| c.call_id.clone()),
            answer_tool: call.and_then(|c| c.answer_tool.clone()),
            ..WaitOptions::default()
        };
        let asked = wait_patch(&answer, question, wait, s.ids.status_message);
        settled.transitions.push(TaskTransition {
            task_id: answer,
            status: TaskState::InputRequired,
            patch: TaskPatch {
                metadata,
                ..TaskPatch { ask: asked.ask, status_message: asked.status_message, ..base }
            },
        });
        settled.outcome = RunOutcome::InputRequired;
        settled.output = out.output.clone();
        return Ok(settled);
    }
    let output = out.output.clone().filter(|o| !o.is_null());
    let text = output.as_ref().and_then(Value::as_str).filter(|t| !t.is_empty());
    let status_message = text.map(|t| agent_text(&answer, t, s.ids.status_message.clone()));
    let artifacts = output.iter().map(|o| artifact(o, s.ids.artifact.clone())).collect();
    settled.transitions.push(TaskTransition {
        task_id: answer,
        status: TaskState::Completed,
        patch: TaskPatch { artifacts: Some(artifacts), status_message, metadata, ..base },
    });
    settled.output = output;
    Ok(settled)
}

/// The metadata of a row answered by the task it was merged into.
pub fn merged_into(task_id: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(MERGED_INTO_KEY.into(), Value::String(task_id.to_owned()));
    m
}

/// An agent message `message_id` of `task_id` with one text part.
pub fn agent_text(task_id: &str, text: &str, message_id: String) -> a2a::Message {
    agent_message(vec![a2a::Part::text(text)], task_id, message_id)
}

/// The status message `message_id` of a row the worker failed without a run, with
/// `livekit/reason`.
pub fn failed_message(task_id: &str, reason: &str, message_id: String) -> a2a::Message {
    let mut msg = agent_text(task_id, &format!("the task failed: {reason}"), message_id);
    msg.metadata = Some([(REASON_KEY.to_owned(), Value::String(reason.to_owned()))].into());
    msg
}

/// The answer as artifact `artifact_id`: a text part for a string, a data part otherwise.
pub fn artifact(value: &Value, artifact_id: String) -> a2a::Artifact {
    let part = match value.as_str() {
        Some(text) => a2a::Part::text(text),
        None => a2a::Part::data(value.clone()),
    };
    a2a::Artifact {
        artifact_id,
        name: None,
        description: None,
        parts: vec![part],
        metadata: None,
        extensions: None,
    }
}

/// The first artifact's text, or its data.
pub fn artifact_text(record: &TaskRow) -> Option<Value> {
    let first = record.data.artifacts.as_ref()?.first()?;
    first.parts.iter().find_map(|p| match &p.content {
        a2a::PartContent::Text(t) => Some(Value::String(t.clone())),
        a2a::PartContent::Data(d) => Some(d.clone()),
        _ => None,
    })
}

/// The channel of the message that started the task.
pub fn channel_of_record(record: &TaskRow) -> Option<String> {
    record.data.history.first().map(channel_of)
}

/// The result of a task that was not run here, answered by the task it was merged into when it
/// was.
pub fn result_for(record: TaskRow, thread_id: &str, run_id: Option<&str>) -> TurnResult {
    let answered_by = record
        .data
        .metadata
        .get(MERGED_INTO_KEY)
        .and_then(Value::as_str)
        .unwrap_or(&record.task_id)
        .to_owned();
    TurnResult {
        output: artifact_text(&record),
        answered_by,
        task: record,
        thread_id: thread_id.to_owned(),
        run_id: run_id.map(str::to_owned),
    }
}

/// A failed record the worker could not write, at `updated_at` (Unix milliseconds), its status
/// message `message_id`.
pub fn unpersisted_failed(
    task_id: &str,
    lane: &str,
    msg: Option<&Message>,
    reason: &str,
    updated_at: i64,
    message_id: String,
) -> TaskRow {
    let mut metadata = Map::new();
    metadata.insert(REASON_KEY.into(), Value::String(reason.to_owned()));
    TaskRow {
        task_id: task_id.to_owned(),
        lane: lane.to_owned(),
        seq: 0,
        name: None,
        message_id: msg.map(|m| m.message_id.clone()),
        status: TaskState::Failed,
        updated_at,
        data: TaskData {
            history: msg.map(|m| vec![m.to_a2a()]).unwrap_or_default(),
            artifacts: None,
            metadata,
            status_message: Some(failed_message(task_id, reason, message_id)),
        },
        internal: None,
    }
}

/// `ChatContext.insert`: after the last item whose `created_at` is not later.
pub fn insert_by_created_at(history: &mut Vec<ItemRow>, item: ItemRow) {
    let created_at = item.created_at();
    let idx = history.iter().rposition(|i| i.created_at() <= created_at).map_or(0, |i| i + 1);
    history.insert(idx, item);
}

/// A failed task's reason for a storage error: the exception name the host raises it as, and its
/// message.
pub fn python_reason(e: &StorageError) -> String {
    match e {
        StorageError::ThreadNotFound(m) => format!("ThreadNotFound: {m}"),
        StorageError::VersionConflict { .. } => format!("VersionConflict: {e}"),
        StorageError::NotSerializable(..) => format!("NotSerializable: {e}"),
        StorageError::TooLarge { .. } => format!("TooLarge: {e}"),
        StorageError::Unavailable(m) => format!("StorageUnavailable: {m}"),
        StorageError::Invalid(m) => format!("ValueError: {m}"),
        StorageError::Cancelled(m) => {
            if m.is_empty() {
                "CancelledError".into()
            } else {
                format!("CancelledError: {m}")
            }
        }
        other => format!("StorageError: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HandlerOutput;
    use lk_agents_storage::db::TaskWrite;
    use lk_agents_storage::run::merge_write;
    use serde_json::json;

    fn msg(id: &str, text: &str) -> Message {
        let raw: a2a::Message = serde_json::from_value(
            json!({"messageId": id, "role": "ROLE_USER", "parts": [{"text": text}]}),
        )
        .unwrap();
        Message::from_a2a(&raw, 1.0, None).unwrap()
    }

    fn ids() -> TurnIds {
        TurnIds {
            assistant_item: "item_a".into(),
            status_message: "msg_s".into(),
            artifact: "art_1".into(),
        }
    }

    fn settled(out: HandlerOutput, stop: Option<Stop>, answer_canceled: bool) -> Settlement {
        let batch = [msg("m1", "where"), msg("m2", "are you")];
        let task_ids = ["AST_1".to_owned(), "AST_2".to_owned()];
        let mut history = Vec::new();
        settle(
            &mut history,
            Settle {
                batch: &batch,
                task_ids: &task_ids,
                answer_task: "AST_2",
                user_message: crate::batch_to_chat_message(&batch, 2.0, "item_u".into()),
                out: &out,
                stop,
                answer_canceled,
                now: 2.0,
                ids: ids(),
            },
        )
        .unwrap()
    }

    fn statuses(s: &Settlement) -> Vec<(&str, TaskState)> {
        s.transitions.iter().map(|t| (t.task_id.as_str(), t.status)).collect()
    }

    #[test]
    fn a_checkpoint_yields_to_a_waiter_under_interrupt_and_to_a_client_cancel_always() {
        let stop = |i, w, s| checkpoint_stop(i, w, "me", s);
        assert_eq!(stop(true, Some("other"), None), Some(Stop::Interrupted));
        assert_eq!(stop(true, Some("me"), None), None, "its own waiter mark");
        assert_eq!(stop(false, Some("other"), None), None, "queue does not yield to a waiter");
        assert_eq!(stop(false, None, Some(TaskState::Canceled)), Some(Stop::ClientCanceled));
        assert_eq!(stop(true, None, Some(TaskState::Working)), None);
    }

    #[test]
    fn a_turn_settles_each_row_by_how_the_handler_ended() {
        let text = |t: &str| HandlerOutput { output: Some(json!(t)), ..HandlerOutput::default() };
        let done = settled(text("here"), None, false);
        assert_eq!(
            statuses(&done),
            vec![("AST_1", TaskState::Completed), ("AST_2", TaskState::Completed)]
        );
        assert_eq!(done.transitions[0].patch.metadata, Some(merged_into("AST_2")));
        assert_eq!(done.outcome, RunOutcome::Completed);

        let raised = HandlerOutput { error: Some("boom".into()), ..HandlerOutput::default() };
        let failed = settled(raised, None, false);
        assert_eq!(failed.transitions.last().unwrap().status, TaskState::Failed);
        assert_eq!(failed.outcome, RunOutcome::Failed);

        let asked =
            HandlerOutput { input_required: Some("when?".into()), ..HandlerOutput::default() };
        let waiting = settled(asked, None, false);
        let last = waiting.transitions.last().unwrap();
        assert_eq!(last.status, TaskState::InputRequired);
        assert_eq!(last.patch.ask.as_ref().map(|a| a.question.as_str()), Some("when?"));

        let interrupted = settled(text("late"), Some(Stop::Interrupted), false);
        assert_eq!(interrupted.transitions.last().unwrap().status, TaskState::Canceled);
        // the client's cancel owns the row: the turn writes no row at all
        let canceled = settled(text("late"), Some(Stop::ClientCanceled), true);
        assert!(canceled.transitions.is_empty());
        assert_eq!(canceled.outcome, RunOutcome::Canceled);
    }
    #[test]
    fn a_plain_question_can_continue_its_existing_task() {
        let mut waiting =
            unpersisted_failed("AST_waiting", "desk", None, "", 0, "msg_question".into());
        waiting.status = TaskState::InputRequired;
        waiting.internal = Some(lk_agents_abi::storage::TaskInternal {
            ask: Some(lk_agents_abi::storage::Ask {
                question: "Your name?".into(),
                ..Default::default()
            }),
            streams: Vec::new(),
        });
        let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("Ada")]);
        wire.message_id = "answer".into();
        wire.task_id = Some("AST_waiting".into());
        let message = Message::from_a2a(&wire, 0.0, None).unwrap();
        let answer = answer_pending(
            &mut Vec::new(),
            Some(&waiting),
            &message,
            "AST_receipt",
            0.0,
            "item".into(),
        );
        assert_eq!(answer.answer_task, "AST_waiting");
        assert_eq!(answer.transitions[0].task_id, "AST_receipt");
    }

    /// Found by the A2A DST (A2A_SEED=8): a reply continuing a pending question by its `taskId`
    /// stamped its `messageId` on both the receipt and the continued task, and the thread's
    /// unique (lane, message_id) task index refused the whole commit.
    #[test]
    fn a_continuation_keeps_one_task_per_message_and_the_continued_history() {
        let mut waiting =
            unpersisted_failed("AST_waiting", "desk", None, "", 0, "msg_question".into());
        waiting.status = TaskState::InputRequired;
        waiting.message_id = Some("question".into());
        waiting.data.history = vec![msg("question", "book me a table").to_a2a()];
        waiting.internal = Some(lk_agents_abi::storage::TaskInternal {
            ask: Some(lk_agents_abi::storage::Ask {
                question: "For how many?".into(),
                ..Default::default()
            }),
            streams: Vec::new(),
        });
        let mut wire = a2a::Message::new(a2a::Role::User, vec![a2a::Part::text("two")]);
        wire.message_id = "answer".into();
        wire.task_id = Some("AST_waiting".into());
        let answer = Message::from_a2a(&wire, 1.0, None).unwrap();
        let receipt = TaskWrite {
            task_id: "AST_receipt".into(),
            status: TaskState::Working,
            message_id: Some("answer".into()),
            name: None,
            data: TaskData { history: vec![answer.to_a2a()], ..TaskData::default() },
            internal: None,
        };
        let mut rows = vec![TaskWrite::from(waiting.clone()), receipt];
        let mut write = |t: TaskTransition| {
            let i = rows.iter().position(|r| r.task_id == t.task_id).unwrap();
            rows[i] = merge_write(&t.task_id, Some(&rows[i]), t.status, t.patch).unwrap();
        };
        let mut history = Vec::new();
        let pending =
            answer_pending(&mut history, Some(&waiting), &answer, "AST_receipt", 2.0, "i".into());
        pending.transitions.into_iter().for_each(&mut write);
        let out = HandlerOutput { output: Some(json!("booked")), ..HandlerOutput::default() };
        let batch = [answer.clone()];
        let settled = settle(
            &mut history,
            Settle {
                batch: &batch,
                task_ids: &["AST_receipt".to_owned()],
                answer_task: &pending.answer_task,
                user_message: crate::batch_to_chat_message(&batch, 2.0, "item_u".into()),
                out: &out,
                stop: None,
                answer_canceled: false,
                now: 2.0,
                ids: ids(),
            },
        )
        .unwrap();
        settled.transitions.into_iter().for_each(&mut write);
        let ids: Vec<_> = rows.iter().map(|r| r.message_id.as_deref()).collect();
        assert_eq!(ids, [Some("question"), Some("answer")], "one task per messageId");
        let continued = &rows[0];
        assert_eq!(continued.status, TaskState::Completed);
        let said: Vec<&str> =
            continued.data.history.iter().map(|m| m.message_id.as_str()).collect();
        assert_eq!(said, ["question", "answer"], "the continued task's history accumulates");
    }
}
