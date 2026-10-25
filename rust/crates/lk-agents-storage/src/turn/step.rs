//! One turn over a batch inside an open run: the `_turn` of `storage/policies.py`. It writes the
//! task rows of plan 5.5 for every outcome and commits nothing itself; the pending input-required
//! call (plan 7.4), the boundary and the host call are here, with the helpers `_turn` reads.

use serde_json::{Map, Value};

use super::host::{Boundary, HandlerOutput, HostError, TurnContext};
use super::message::{INTERRUPTION_NEW_MESSAGE, Message, batch_to_chat_message, now_secs, truthy};
use super::{Ctx, LOOKUP_LIMIT, TASKS_IN_CONTEXT, Turn};
use super::{TurnResult, interrupted_metadata};
use crate::StorageError;
use crate::run::{
    ANSWERED_WITH_KEY, BACKGROUND_KEY, CANCEL_REQUESTED_KEY, Item, KIND_BACKGROUND, KIND_KEY, Lane,
    MERGED_INTO_KEY, QUEUED_KEY, REASON_KEY, Run, RunOutcome, TaskPatch, TaskRecord, TaskState,
    VERBATIM_KEY, WaitOptions, agent_message, artifact, shortuuid,
};
use crate::schema::TaskData;

/// What a turn leaves for `finish`: the task that carries the reply, and the reply.
pub(super) struct Outcome {
    pub task_id: String,
    pub output: Option<Value>,
    /// The tasks answered by the same reply: their background work ended together.
    pub answered_with: Vec<String>,
    #[allow(dead_code)]
    pub interrupted: bool,
    /// The task whose cut-off question this turn took over: it ended merged into this one.
    pub merged: Option<String>,
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
        arrivals_after: Option<i64>,
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
        // the background work of the task finished: the turn is the task's, with no message
        let resumed = batch
            .iter()
            .all(|m| m.metadata.get(KIND_KEY).and_then(Value::as_str) == Some(KIND_BACKGROUND));
        // a question a newer message cut off: this message is its answer. This task takes the
        // question's call over, and the task that asked ends merged into it
        let taken = match (&answered, resumed) {
            (None, false) => run
                .tasks
                .waiting()
                .await?
                .into_iter()
                .find(|r| cut_off(r) && !task_ids.contains(&r.id)),
            _ => None,
        };
        let ask = answered.as_ref().or(taken.as_ref()).and_then(TaskRecord::ask);
        let answer_call_id = ask.and_then(|a| a.get("call_id")).filter(|c| truthy(c)).map(text_of);
        let answer_tool = ask.and_then(|a| a.get("answer_tool")).filter(|t| truthy(t)).map(text_of);
        let hands_off = answer_tool.is_some();
        let asked_by = answer_call_id.clone();
        let cut_off_ask = ask.filter(|a| a.get("heard").is_some()).map(|a| {
            serde_json::json!({
                "question": a.get("question").cloned().unwrap_or_default(),
                "heard": a.get("heard").cloned().unwrap_or_default(),
                "verbatim": a.get("verbatim").is_some_and(truthy),
            })
        });
        // a continued task keeps its first messageId and adds the batch to its history
        let row_message_id = match (&answered, resumed) {
            (None, false) => Some(last.message_id.clone()),
            _ => None,
        };
        let mut history: Vec<Value> = match (&answered, resumed) {
            (Some(row), _) => row.data.history.clone(),
            (None, true) => {
                run.tasks.get(&answer_task).await?.map(|r| r.data.history).unwrap_or_default()
            }
            (None, false) => Vec::new(),
        };
        if !resumed {
            history.extend(batch.iter().map(Message::to_a2a));
        }

        // the person has one question in front of them: another open question of the lane that
        // was asked; with none, the oldest that waits behind none is relayed when this turn ends
        let others: Vec<TaskRecord> = run
            .tasks
            .waiting()
            .await?
            .into_iter()
            .filter(|r| {
                r.id != answer_task
                    && !task_ids.contains(&r.id)
                    && taken.as_ref().is_none_or(|t| t.id != r.id)
            })
            .collect();
        let front_question = others.iter().find(|r| !queued(r)).map(|r| r.id.clone());
        let relay = if front_question.is_none() {
            others.iter().find(|r| queued(r)).map(|r| {
                let ask = r.ask().cloned().unwrap_or_default();
                serde_json::json!({
                    "task_id": r.id,
                    "question": ask.get("question").cloned().unwrap_or_default(),
                    "verbatim": ask.get("verbatim").is_some_and(truthy),
                })
            })
        } else {
            None
        };

        let mut user_message = batch_to_chat_message(batch);
        user_message["created_at"] = Value::from(now);
        if run.state.is_none() {
            // `load_state`: a fresh lane starts from the door's default-constructed state
            run.state = self.opts.state_default.clone();
        }
        let mut boundary = Boundary::new(
            lane.thread().db().clone(),
            lane.name(),
            run.holder.as_str(),
            &answer_task,
            interrupt,
            run.lease().cloned(),
            run.version,
        );
        if let Some(last) = arrivals_after {
            boundary = boundary.live(last);
        }
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
            answer_call_id,
            front_question,
            relay: relay.clone(),
            cut_off: cut_off_ask,
            live: arrivals_after.is_some(),
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
        } else if !resumed {
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

        // tasks resumed together, one of whose work asked: that task waits on the person, and
        // the reply answers the others
        let answer_task = match &out.asked_task {
            Some(asking)
                if resumed
                    && out.input_required.is_some()
                    && *asking != answer_task
                    && task_ids.contains(asking) =>
            {
                history = run.tasks.get(asking).await?.map(|r| r.data.history).unwrap_or_default();
                asking.clone()
            }
            _ => answer_task,
        };
        run.answer_task = Some(answer_task.clone());
        // a question cut off by a newer message is not an interrupted turn: it waits
        let interrupted = (boundary.tripped() || out.interrupted) && out.heard.is_none();
        // a turn stopped before it handed the answer to the question's tool leaves the question
        // to the next message
        let handed = !hands_off
            || out.new_items.iter().any(|i| {
                i.get("type").and_then(Value::as_str) == Some("function_call")
                    && i.pointer("/extra/lk.answers").and_then(Value::as_str)
                        == asked_by.as_deref()
            });
        let taken = taken.filter(|_| !interrupted || handed);
        if let Some(relayed) = &out.relayed
            && let Some(row) = others.iter().find(|r| &r.id == relayed && queued(r))
        {
            // the reply put this question to the person: it is the one in front now
            let mut ask = row.ask().and_then(Value::as_object).cloned().unwrap_or_default();
            ask.remove("queued");
            let mut metadata = Map::new();
            metadata.insert(QUEUED_KEY.into(), Value::Bool(false));
            run.tasks
                .write(
                    relayed,
                    TaskState::InputRequired,
                    TaskPatch {
                        ask: Some(Value::Object(ask)),
                        metadata: Some(metadata),
                        ..TaskPatch::default()
                    },
                )
                .await?;
        }
        // tasks whose work ended together: one reply answers each of them
        let answered_with: Vec<String> = if resumed {
            task_ids.iter().filter(|t| **t != answer_task).cloned().collect()
        } else {
            Vec::new()
        };
        let outcome = |output: Option<Value>, interrupted: bool| Outcome {
            task_id: answer_task.clone(),
            output,
            answered_with: answered_with.clone(),
            interrupted,
            merged: taken.as_ref().map(|t| t.id.clone()),
        };
        // the row as it is now: a client may have canceled it, or asked to, during the turn
        let row =
            lane.thread().db().task(lane.name(), &answer_task).await?.map(TaskRecord::from_row);
        let cancel_requested = row
            .as_ref()
            .is_some_and(|r| r.data.metadata.get(CANCEL_REQUESTED_KEY) == Some(&Value::Bool(true)));
        if boundary.client_canceled()
            || cancel_requested
            || (interrupted
                && row.as_ref().is_some_and(|r| r.status == TaskState::Canceled.as_str()))
        {
            // a client canceled the task: it ends `canceled` now that its tools stopped, with
            // what they did in the history; a row the client already ended is kept as it is
            let mut metadata = Map::new();
            metadata.insert(CANCEL_REQUESTED_KEY.into(), Value::Bool(false));
            if let Some(reason) = row.as_ref().and_then(|r| r.data.metadata.get(REASON_KEY)) {
                metadata.insert(REASON_KEY.into(), reason.clone());
            }
            let mut answer = None;
            if row.is_some_and(|row| !row.terminal()) {
                // the caller learns what ran: the tool outputs this turn recorded, else the
                // reply, never the progress line the task had before. It is the task's status
                // and its answer, which replaces whatever streamed before the cancel
                let ran: Vec<String> = out
                    .new_items
                    .iter()
                    .filter(|i| {
                        i.get("type").and_then(Value::as_str) == Some("function_call_output")
                    })
                    .map(|i| {
                        let name = i.get("name").and_then(Value::as_str).unwrap_or("a tool");
                        let output = i.get("output").map(text_of).unwrap_or_default();
                        if i.get("is_error").is_some_and(truthy) {
                            format!("{name} did not finish ({output})")
                        } else {
                            format!("{name} returned {output}")
                        }
                    })
                    .collect();
                let reply = out.output.as_ref().and_then(Value::as_str).map(str::trim);
                let said = match reply {
                    _ if !ran.is_empty() => format!("stopped after: {}", ran.join("; ")),
                    Some(reply) if !reply.is_empty() => reply.to_owned(),
                    _ => "stopped before any work ended".to_owned(),
                };
                run.tasks
                    .write(
                        &answer_task,
                        TaskState::Canceled,
                        TaskPatch {
                            message_id: row_message_id.clone(),
                            history: Some(history.clone()),
                            status_message: Some(agent_text(&answer_task, &said)),
                            artifacts: Some(vec![artifact(&Value::String(said.clone()))]),
                            metadata: Some(metadata),
                            ..TaskPatch::default()
                        },
                    )
                    .await?;
                answer = Some(Value::String(said));
            }
            run.outcome = RunOutcome::Canceled;
            run.reason = Some("client-cancel".into());
            return Ok(outcome(answer, true));
        }
        for tid in &answered_with {
            if let Some(waits) = out.waiting.iter().find(|w| &w.task_id == tid) {
                // its work asked too: the question waits behind the one in front
                let call = &waits.pending_call;
                run.tasks
                    .wait(
                        tid,
                        &waits.question,
                        WaitOptions {
                            tool: Some(call.name.clone()),
                            call_id: Some(call.call_id.clone()),
                            answer_tool: call.answer_tool.clone(),
                            queued: true,
                            verbatim: waits.verbatim,
                            ..WaitOptions::default()
                        },
                    )
                    .await?;
                let mut metadata = Map::new();
                metadata.insert(QUEUED_KEY.into(), Value::Bool(true));
                metadata.insert(BACKGROUND_KEY.into(), Value::Bool(false));
                run.tasks
                    .write(
                        tid,
                        TaskState::InputRequired,
                        TaskPatch { metadata: Some(metadata), ..TaskPatch::default() },
                    )
                    .await?;
                continue;
            }
            // the reply answers this task too; it is not merged, its work was its own. When
            // another of the tasks asked, the reply is what answers this one, not the question
            let reply = match (&out.asked_task, &out.input_required) {
                (Some(_), Some(_)) => out.output.clone(),
                (_, question) => question.clone().map(Value::String).or_else(|| out.output.clone()),
            };
            let text = reply.as_ref().and_then(Value::as_str).filter(|t| !t.is_empty());
            let mut metadata = Map::new();
            metadata.insert(ANSWERED_WITH_KEY.into(), Value::String(answer_task.clone()));
            metadata.insert(BACKGROUND_KEY.into(), Value::Bool(false));
            let (state, status_message) = match &out.error {
                Some(err) => (TaskState::Failed, Some(agent_text(tid, err))),
                None => (TaskState::Completed, text.map(|t| agent_text(tid, t))),
            };
            run.tasks
                .write(
                    tid,
                    state,
                    TaskPatch {
                        status_message,
                        artifacts: Some(
                            reply.iter().filter(|_| out.error.is_none()).map(artifact).collect(),
                        ),
                        metadata: Some(metadata),
                        ..TaskPatch::default()
                    },
                )
                .await?;
        }
        for (tid, m) in task_ids[..task_ids.len() - 1].iter().zip(&batch[..batch.len() - 1]) {
            if *tid == answer_task || resumed {
                continue;
            }
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
        if let Some(row) = &taken {
            // its question went to this task with its answer; what it showed is not repeated
            let mut metadata = merged_into(&answer_task);
            metadata.insert(QUEUED_KEY.into(), Value::Bool(false));
            run.tasks
                .write(
                    &row.id,
                    TaskState::Completed,
                    TaskPatch {
                        status_message: Some(Value::Null),
                        metadata: Some(metadata),
                        ..TaskPatch::default()
                    },
                )
                .await?;
        }
        if out.background && (out.input_required.is_none() || interrupted) {
            // the turn ended and a tool of the task still runs, also when a newer message
            // stopped the reply: the task waits for that work, which answers it. A turn that
            // asks waits on the person instead, and the work's end joins its question
            let mut metadata = out.metadata.clone();
            metadata.insert(BACKGROUND_KEY.into(), Value::Bool(true));
            let text = out.output.as_ref().and_then(Value::as_str).filter(|t| !t.is_empty());
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Working,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        status_message: text.map(|t| agent_text(&answer_task, t)),
                        metadata: Some(metadata),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            return Ok(outcome(out.output.clone(), interrupted));
        }
        if interrupted {
            // the part of the reply shown before it stopped is the task's answer, as streamed
            let shown = out
                .output
                .as_ref()
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
                .map(|t| Value::String(t.to_owned()));
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Canceled,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        artifacts: shown.as_ref().map(|s| vec![artifact(s)]),
                        metadata: Some(interrupted_metadata()),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::Canceled;
            run.reason = Some(INTERRUPTION_NEW_MESSAGE.into());
            return Ok(outcome(shown, true));
        }
        let mut metadata = out.metadata.clone();
        if let Some(question) = &out.input_required {
            let call = out.pending_call.as_ref();
            let verbatim =
                out.metadata.get(VERBATIM_KEY).and_then(Value::as_str) == Some(question.as_str());
            run.tasks
                .wait(
                    &answer_task,
                    question,
                    WaitOptions {
                        tool: call.map(|c| c.name.clone()),
                        call_id: call.map(|c| c.call_id.clone()),
                        answer_tool: call.and_then(|c| c.answer_tool.clone()),
                        queued: out.queued,
                        verbatim,
                        heard: out.heard.clone(),
                        ..WaitOptions::default()
                    },
                )
                .await?;
            metadata.insert(QUEUED_KEY.into(), Value::Bool(out.queued));
            let was_background = answered
                .as_ref()
                .is_some_and(|r| r.data.metadata.get(BACKGROUND_KEY) == Some(&Value::Bool(true)));
            if out.background || resumed || was_background {
                metadata.insert(BACKGROUND_KEY.into(), Value::Bool(out.background));
            }
            run.tasks
                .write(
                    &answer_task,
                    TaskState::InputRequired,
                    TaskPatch {
                        message_id: row_message_id.clone(),
                        history: Some(history),
                        metadata: Some(metadata),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::InputRequired;
            return Ok(outcome(out.output.clone(), false));
        }
        if answered.is_some() {
            // a question this task asked is no longer waiting
            metadata.insert(QUEUED_KEY.into(), Value::Bool(false));
        }
        let was_background = answered
            .as_ref()
            .is_some_and(|r| r.data.metadata.get(BACKGROUND_KEY) == Some(&Value::Bool(true)));
        if resumed || was_background {
            metadata.insert(BACKGROUND_KEY.into(), Value::Bool(false));
        }
        let metadata = Some(metadata).filter(|m| !m.is_empty());
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

/// An open question a newer message cut off: that message, or the next, is its answer.
fn cut_off(row: &TaskRecord) -> bool {
    row.ask().and_then(|a| a.get("heard")).is_some()
}

/// An open question that waits behind the one in front of the person.
fn queued(row: &TaskRecord) -> bool {
    row.ask().and_then(|a| a.get("queued")).is_some_and(truthy)
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
