//! One turn over a batch inside an open run: the `_turn` of `storage/policies.py`. It writes the
//! task rows of plan 5.5 for every outcome and commits nothing itself; the continuation of a task
//! waiting on a question (plan 7.4), the boundary and the host call are here, with the helpers
//! `_turn` reads.

use serde_json::{Map, Value};

use super::host::{Boundary, HandlerOutput, HostError, TurnContext};
use super::message::{INTERRUPTION_NEW_MESSAGE, Message, batch_to_chat_message, now_secs, truthy};
use super::{Ctx, LOOKUP_LIMIT, TASKS_IN_CONTEXT, Turn};
use super::{TurnResult, interrupted_metadata};
use crate::StorageError;
use crate::run::{
    Item, Lane, MERGED_INTO_KEY, REASON_KEY, Run, RunOutcome, TaskPatch, TaskRecord, TaskState,
    WaitOptions, agent_message, artifact, shortuuid,
};
use crate::schema::TaskData;

/// What a turn leaves for `finish`: the task that carries the reply, and the reply.
pub(super) struct Outcome {
    pub task_id: String,
    pub output: Option<Value>,
    #[allow(dead_code)]
    pub interrupted: bool,
}

impl Turn {
    /// One turn over a batch inside an open run; writes the task rows, commits nothing itself.
    pub(super) async fn turn(
        &self,
        ctx: &Ctx,
        run: &mut Run,
        batch: &[Message],
        task_ids: &[String],
        interrupt: bool,
    ) -> Result<Outcome, StorageError> {
        let last = batch.last().ok_or_else(|| StorageError::Invalid("an empty batch".into()))?;
        let answer_task = task_ids
            .last()
            .ok_or_else(|| StorageError::Invalid("a batch without task ids".into()))?
            .clone();
        let lane = &ctx.lane;

        // what this turn adds goes after everything loaded, whatever the messages' own times
        let after = run.history.iter().map(|i| created_at_of(&i.payload)).fold(0.0, f64::max);
        let mut now = now_secs();
        if after >= now {
            now = after + 2e-6;
        }

        // the task this batch continues, when it waits on a question: the batch is the answer,
        // recorded once as the person's message; the asking call keeps its question as its
        // output, and the tool the question named, if any, takes the answer before the model runs
        let answered = run.tasks.get(&answer_task).await?.filter(|r| r.ask().is_some());
        let answer_tool = answered
            .as_ref()
            .and_then(TaskRecord::ask)
            .and_then(|a| a.get("answer_tool"))
            .filter(|t| truthy(t))
            .map(text_of);
        // a continued task keeps its first messageId and adds the batch to its history
        let row_message_id = answered.is_none().then(|| last.message_id.clone());
        let mut history: Vec<Value> =
            answered.as_ref().map(|r| r.data.history.clone()).unwrap_or_default();
        history.extend(batch.iter().map(Message::to_a2a));

        let mut user_message = batch_to_chat_message(batch);
        user_message["created_at"] = Value::from(now);
        if run.state.is_none() {
            // `load_state`: a fresh lane starts from the door's default-constructed state
            run.state = self.opts.state_default.clone();
        }
        let boundary = Boundary::new(
            lane.thread().db().clone(),
            lane.name(),
            run.holder.as_str(),
            &answer_task,
            interrupt,
            run.lease().cloned(),
        );
        let turn_ctx = TurnContext {
            thread_id: self.thread.id().to_owned(),
            lane: lane.name().to_owned(),
            message: user_message.clone(),
            batch: batch.to_vec(),
            history: run.history.clone(),
            state: run.state.clone(),
            task_id: answer_task.clone(),
            tasks: run.tasks.recent(TASKS_IN_CONTEXT).await?,
            agent_stack: run.agents.refs().to_vec(),
            caller_history: ctx.caller_history.clone(),
            answer_tool,
            boundary: boundary.clone(),
        };
        let out: HandlerOutput = match run.guard(self.host.handle(turn_ctx)).await? {
            Ok(out) => out,
            Err(HostError::Storage(e)) => return Err(e),
            Err(HostError::Cancelled(reason)) => return Err(StorageError::Cancelled(reason)),
        };
        if let Some(err) = &out.error {
            log::error!(target: "livekit.agents", lane = lane.name(), taskId = answer_task.as_str(); "the a2a_session handler raised; the task fails: {err}");
        }
        if out.state.is_some() {
            run.state = out.state.clone();
        }
        if let Some(stack) = &out.agent_stack {
            run.agents.replace(stack.clone());
        }

        // what the runs added to the history; the message itself when no run was made
        if !out.new_items.is_empty() {
            for item in &out.new_items {
                let item = Item::new(item.clone())?;
                if !run.history.iter().any(|i| i.id == item.id) {
                    insert_by_created_at(&mut run.history, item);
                }
            }
            for item in &out.new_items {
                if item.get("type").and_then(Value::as_str) == Some("message")
                    && item.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(id) = item.get("id").and_then(Value::as_str)
                {
                    run.mark_message(id, &last.message_id);
                }
            }
        } else {
            let user = Item::new(user_message)?;
            run.mark_message(&user.id, &last.message_id);
            insert_by_created_at(&mut run.history, user);
            if let Some(text) = out.output.as_ref().and_then(Value::as_str)
                && !text.is_empty()
                && out.error.is_none()
                && !out.interrupted
            {
                let created_at = now_secs().max(now + 1e-6);
                insert_by_created_at(
                    &mut run.history,
                    Item::new(assistant_message_item(text, created_at))?,
                );
            }
        }
        run.usage.requests += out.usage.requests;
        run.usage.input_tokens += out.usage.input_tokens;
        run.usage.output_tokens += out.usage.output_tokens;
        run.usage.tool_calls += out.usage.tool_calls;
        for event in &out.events {
            run.events.append(&event.r#type, event.payload.clone());
        }

        run.answer_task = Some(answer_task.clone());
        let interrupted = boundary.tripped() || out.interrupted;
        let outcome = |output: Option<Value>, interrupted: bool| Outcome {
            task_id: answer_task.clone(),
            output,
            interrupted,
        };
        if boundary.client_canceled()
            || (interrupted
                && lane
                    .task(&answer_task)
                    .await?
                    .is_some_and(|row| row.status == TaskState::Canceled.as_str()))
        {
            // the row is the client's; the history keeps what ran, the run says why it ended
            run.outcome = RunOutcome::Canceled;
            run.reason = Some("client-cancel".into());
            return Ok(outcome(None, true));
        }
        for (tid, m) in task_ids[..task_ids.len() - 1].iter().zip(&batch[..batch.len() - 1]) {
            run.tasks
                .write(
                    tid,
                    TaskState::Completed,
                    TaskPatch {
                        message_id: Some(m.message_id.clone()),
                        history: Some(vec![m.to_a2a()]),
                        metadata: Some(merged_into(&answer_task)),
                        ..TaskPatch::default()
                    },
                )
                .await?;
        }
        if let Some(err) = &out.error {
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Failed,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        status_message: Some(agent_text(&answer_task, err)),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.events.append("error", serde_json::json!({"error": err}));
            run.outcome = RunOutcome::Failed;
            run.reason = Some(err.clone());
            return Ok(outcome(None, false));
        }
        if interrupted {
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Canceled,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        metadata: Some(interrupted_metadata()),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::Canceled;
            run.reason = Some(INTERRUPTION_NEW_MESSAGE.into());
            return Ok(outcome(None, true));
        }
        let metadata = Some(out.metadata.clone()).filter(|m| !m.is_empty());
        if let Some(question) = &out.input_required {
            let call = out.pending_call.as_ref();
            run.tasks
                .wait(
                    &answer_task,
                    question,
                    WaitOptions {
                        tool: call.map(|c| c.name.clone()),
                        call_id: call.map(|c| c.call_id.clone()),
                        answer_tool: call.and_then(|c| c.answer_tool.clone()),
                        ..WaitOptions::default()
                    },
                )
                .await?;
            run.tasks
                .write(
                    &answer_task,
                    TaskState::InputRequired,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        metadata,
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::InputRequired;
            return Ok(outcome(out.output.clone(), false));
        }
        let output = out.output.clone().filter(|o| !o.is_null());
        let text = output.as_ref().and_then(Value::as_str).filter(|t| !t.is_empty());
        let artifacts: Vec<Value> = output.iter().map(artifact).collect();
        run.tasks
            .write(
                &answer_task,
                TaskState::Completed,
                TaskPatch {
                    message_id: row_message_id,
                    history: Some(history),
                    artifacts: Some(artifacts),
                    status_message: text.map(|t| agent_text(&answer_task, t)),
                    metadata,
                    ..TaskPatch::default()
                },
            )
            .await?;
        Ok(outcome(output, false))
    }

    /// `_finish`: the task after the commit, and the reply. A session switch a tool recorded is
    /// reported on the result; the merge itself (`Thread.complete_switch` in
    /// `storage/resolve.py`) is the host's, after this commit.
    pub(super) async fn finish(
        &self,
        ctx: &Ctx,
        outcome: Outcome,
        task_id: &str,
        run: &mut Run,
    ) -> Result<TurnResult, StorageError> {
        let run_id = run.run_id.as_str();
        let committed = match run.committed_task.take() {
            Some(r) if r.id == outcome.task_id => Some(r),
            _ => ctx.lane.task(&outcome.task_id).await?,
        };
        let record = match committed {
            Some(r) => r,
            None => unpersisted_failed(task_id, ctx.lane.name(), None, "not-written"),
        };
        Ok(TurnResult {
            task: record,
            output: outcome.output,
            thread_id: self.thread.id().to_owned(),
            answered_by: outcome.task_id,
            run_id: Some(run_id.to_owned()),
        })
    }
}

fn merged_into(task_id: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(MERGED_INTO_KEY.into(), Value::String(task_id.to_owned()));
    m
}

fn text_part(text: &str) -> Value {
    serde_json::json!({"kind": "text", "text": text})
}

/// `_agent_text`: `{messageId, role: "agent", parts: [text], taskId}`.
pub fn agent_text(task_id: &str, text: &str) -> Value {
    agent_message(vec![text_part(text)], task_id)
}

/// `_failed_message`: the status message of a row the worker failed without a run, with
/// `livekit/reason`.
pub fn failed_message(task_id: &str, reason: &str) -> Value {
    let mut msg = agent_text(task_id, &format!("the task failed: {reason}"));
    msg["metadata"] = serde_json::json!({REASON_KEY: reason});
    msg
}

/// `_artifact_text`: the first artifact's text, or its data.
pub fn artifact_text(record: &TaskRecord) -> Option<Value> {
    let first = record.data.artifacts.as_ref()?.first()?;
    for p in first.get("parts")?.as_array()? {
        match p.get("kind").and_then(Value::as_str) {
            Some("text") => return p.get("text").cloned(),
            Some("data") => return p.get("data").cloned(),
            _ => {}
        }
    }
    None
}

/// `_channel_of_record`: the channel of the message that started the task.
pub fn channel_of_record(record: &TaskRecord) -> Option<String> {
    record.data.history.first().filter(|f| f.is_object()).map(super::message::channel_of)
}

/// `_task_by_message_id`: the task a `messageId` created, over the lane's last 500 rows.
pub async fn task_by_message_id(
    lane: &Lane,
    message_id: &str,
) -> Result<Option<TaskRecord>, StorageError> {
    Ok(lane
        .tasks(None, LOOKUP_LIMIT)
        .await?
        .into_iter()
        .find(|r| r.message_id.as_deref() == Some(message_id)))
}

/// `_result_for`: the result of a task that was not run here, answered by the task it was merged
/// into when it was.
pub fn result_for(
    record: Option<TaskRecord>,
    thread_id: &str,
    task_id: &str,
    run_id: Option<&str>,
) -> TurnResult {
    let record = record.unwrap_or_else(|| unpersisted_failed(task_id, "", None, "not-written"));
    let answered_by = record
        .data
        .metadata
        .get(MERGED_INTO_KEY)
        .and_then(Value::as_str)
        .unwrap_or(&record.id)
        .to_owned();
    TurnResult {
        output: artifact_text(&record),
        answered_by,
        task: record,
        thread_id: thread_id.to_owned(),
        run_id: run_id.map(str::to_owned),
    }
}

/// `_unpersisted_failed`: a failed record the worker could not write: it held no lease (plan 5.12
/// `lease-timeout`).
pub fn unpersisted_failed(
    task_id: &str,
    lane: &str,
    msg: Option<&Message>,
    reason: &str,
) -> TaskRecord {
    let mut metadata = Map::new();
    metadata.insert(REASON_KEY.into(), Value::String(reason.to_owned()));
    TaskRecord {
        id: task_id.to_owned(),
        lane: lane.to_owned(),
        seq: 0,
        name: None,
        message_id: msg.map(|m| m.message_id.clone()),
        status: TaskState::Failed.as_str().to_owned(),
        status_message: Some(failed_message(task_id, reason)),
        updated_at: now_secs() as i64,
        data: TaskData {
            history: msg.map(|m| vec![m.to_a2a()]).unwrap_or_default(),
            artifacts: None,
            metadata,
        },
        internal: None,
    }
}

/// `ChatContext.insert`: after the last item whose `created_at` is not later.
pub fn insert_by_created_at(history: &mut Vec<Item>, item: Item) {
    let created_at = created_at_of(&item.payload);
    let idx =
        history.iter().rposition(|i| created_at_of(&i.payload) <= created_at).map_or(0, |i| i + 1);
    history.insert(idx, item);
}

pub fn created_at_of(payload: &Value) -> f64 {
    payload.get("created_at").and_then(Value::as_f64).unwrap_or(0.0)
}

/// An assistant `ChatMessage` with one text, as `item_to_payload` dumps it.
pub fn assistant_message_item(text: &str, created_at: f64) -> Value {
    serde_json::json!({
        "id": shortuuid("item_"),
        "type": "message",
        "role": "assistant",
        "content": [text],
        "interrupted": false,
        "extra": {},
        "created_at": created_at,
    })
}

/// `_reason` of `storage/run.py` for a Rust error: the Python exception name the `_core` binding
/// raises it as, and its message.
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

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(id: &str, created_at: f64) -> Item {
        Item::new(json!({"id": id, "type": "message", "role": "user", "content": [], "created_at": created_at})).unwrap()
    }

    #[test]
    fn insertion_follows_created_at_after_equal_times() {
        let mut h = vec![item("a", 1.0), item("b", 2.0), item("c", 2.0)];
        insert_by_created_at(&mut h, item("d", 2.0));
        insert_by_created_at(&mut h, item("e", 0.5));
        insert_by_created_at(&mut h, item("f", 9.0));
        assert_eq!(
            h.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
            vec!["e", "a", "b", "c", "d", "f"]
        );
        assert_eq!(created_at_of(&json!({})), 0.0);
    }

    #[test]
    fn the_helpers_shape_as_the_python() {
        let mut f = failed_message("AST_1", "worker-lost");
        f["messageId"] = json!("msg_X");
        assert_eq!(
            f.to_string(),
            r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"the task failed: worker-lost"}],"taskId":"AST_1","metadata":{"livekit/reason":"worker-lost"}}"#
        );
        let mut a = assistant_message_item("hi", 4.0);
        a["id"] = json!("item_X");
        assert_eq!(
            a.to_string(),
            r#"{"id":"item_X","type":"message","role":"assistant","content":["hi"],"interrupted":false,"extra":{},"created_at":4.0}"#
        );
        let failed = unpersisted_failed("AST_2", "front-desk", None, "lease-timeout");
        assert_eq!(failed.data.metadata.get(REASON_KEY), Some(&json!("lease-timeout")));
        assert_eq!(failed.status, "failed");
        assert!(failed.data.history.is_empty());
        assert_eq!(artifact_text(&failed), None);
        let mut done = failed.clone();
        done.data.artifacts = Some(vec![
            json!({"artifactId": "art_X", "parts": [{"kind": "data", "data": {"n": 1}}]}),
        ]);
        assert_eq!(artifact_text(&done), Some(json!({"n": 1})));
        done.data.history = vec![json!({"messageId": "m", "metadata": {"livekit/channel": "sms"}})];
        assert_eq!(channel_of_record(&done).as_deref(), Some("sms"));
        let r = result_for(Some(done.clone()), "DB_1", "AST_2", Some("r1"));
        assert_eq!((r.answered_by.as_str(), r.run_id.as_deref()), ("AST_2", Some("r1")));
        done.data.metadata.insert(MERGED_INTO_KEY.into(), json!("AST_9"));
        assert_eq!(result_for(Some(done), "DB_1", "AST_2", None).answered_by, "AST_9");
        assert_eq!(result_for(None, "DB_1", "AST_3", None).task.status, "failed");
        assert_eq!(
            python_reason(&StorageError::Unavailable("lost".into())),
            "StorageUnavailable: lost"
        );
        assert_eq!(python_reason(&StorageError::Cancelled(String::new())), "CancelledError");
    }
}
