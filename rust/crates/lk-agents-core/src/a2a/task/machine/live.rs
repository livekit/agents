use super::*;
use lk_agents_abi::agent::run::{Output, RunOutcome as AgentOutcome};
use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
use lk_agents_abi::text_session::LiveTurnReport;

#[derive(Debug, Default)]
pub(super) struct LiveSchedule {
    pub(super) turns: std::collections::BTreeMap<String, Turning>,
    pub(super) managed: bool,
    pub(super) report: Option<LiveTurnReport>,
    checkpoint_source: Option<String>,
    outputs: std::collections::VecDeque<SourceOutput>,
    opaque: OpaqueWrite,
    poll: Poll,
    cancelled: std::collections::BTreeSet<String>,
}

#[derive(Debug)]
enum PollNext {
    Foreground,
    Finalizing,
    Inbox(Vec<lk_agents_abi::storage::thread::RoutedInbox>),
}

#[derive(Debug, Default)]
enum Poll {
    #[default]
    None,
    Checking {
        sources: Vec<(String, String)>,
        next: PollNext,
    },
    Native {
        sources: Vec<(String, String)>,
        next: PollNext,
    },
    Closing {
        sources: std::collections::VecDeque<String>,
        next: PollNext,
    },
}

#[derive(Debug, Default)]
enum OpaqueWrite {
    #[default]
    None,
    Preparing {
        source: String,
        stage: Box<Stage>,
    },
    Committing {
        source: String,
    },
    Activating,
}

#[derive(Debug)]
struct SourceOutput {
    request_id: String,
    output: Box<HandlerOutput>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum LiveStep {
    Poll,
    CancelNative,
    CancelSource,
    DialoguePrepared,
    DialogueCommitted,
    Commit,
    Acknowledged,
    Open,
    Inbox,
    Wait,
    Finalize,
    FinalizeCheckpoint { finalizing: bool },
    Close,
}

impl TaskMachine {
    pub(super) fn managed_turn(
        &mut self,
        report: LiveTurnReport,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        self.live.managed = true;
        let Some(mut turn) = self.live.turns.get(&report.origin.request_id).cloned() else {
            return Err(Finish::Error(StorageError::Invalid(
                "a managed turn lost its original transport request".into(),
            )));
        };
        if report.origin.task_ids != [turn.answer_task.clone()] || self.live.report.is_some() {
            return Err(Finish::Error(StorageError::Invalid(
                "a managed turn has conflicting task ownership".into(),
            )));
        }
        turn.output = Some(managed_output(&report));
        self.turn = Some(turn);
        let final_result = !report.working;
        let request_id = report.origin.request_id.clone();
        self.live.report = Some(report);
        if final_result {
            // Native output is already accepted. Let the same original handler finish only
            // its explicit application/scope writes before the one fenced final commit.
            self.phase = Phase::Live(LiveStep::Finalize);
            Ok(Some(Operation::NextSource {
                request_id,
                should_yield: None,
                pause: PARTITION_POLL_PAUSE,
            }))
        } else {
            self.settle_turn(false, now)
        }
    }

    pub(super) fn live_checkpoint(
        &mut self,
        request_id: String,
        finalizing: bool,
    ) -> Result<Option<Operation>, Finish> {
        let task = self
            .live
            .turns
            .get(&request_id)
            .ok_or_else(|| {
                Finish::Error(StorageError::Invalid(
                    "a live checkpoint lost its original handler".into(),
                ))
            })?
            .answer_task
            .clone();
        self.live.checkpoint_source = Some(request_id);
        self.phase = Phase::Live(LiveStep::FinalizeCheckpoint { finalizing });
        Ok(Some(Operation::Checkpoint(vec![task])))
    }

    pub(super) fn live_output(
        &mut self,
        request_id: String,
        mut output: Box<HandlerOutput>,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        if let Some(report) = &self.live.report {
            if report.origin.request_id != request_id {
                if !self.live.turns.contains_key(&request_id)
                    || self.live.outputs.iter().any(|source| source.request_id == request_id)
                {
                    return Err(Finish::Error(StorageError::Invalid(
                        "a live result has no unique admitted source".into(),
                    )));
                }
                self.live.outputs.push_back(SourceOutput { request_id, output });
                return Ok(Some(Operation::NextSource {
                    request_id: report.origin.request_id.clone(),
                    should_yield: None,
                    pause: PARTITION_POLL_PAUSE,
                }));
            }
            let current = self.turn.as_mut().expect("finalizing managed transport turn");
            let accepted = current.output.as_mut().expect("native accepted result retained");
            // Native history/state is authoritative. This exact original handler contributes
            // only its explicit application/scope writes after the report was admitted.
            accepted.scope_writes = output.scope_writes;
            accepted.app_writes = output.app_writes;
            accepted.error = output.error;
            accepted.interrupted |= output.interrupted;
            return self.settle_turn(false, now);
        }
        let mut turn = self.live.turns.remove(&request_id).ok_or_else(|| {
            Finish::Error(StorageError::Invalid("a live result lost its original handler".into()))
        })?;
        if self.live.managed {
            output.state = None;
        }
        let stopped = turn.stop.is_some() || output.interrupted;
        turn.output = Some(*output);
        let task = turn.answer_task.clone();
        self.turn = Some(turn);
        if stopped {
            self.phase = Phase::Turn(Step::AnswerRow);
            Ok(Some(Operation::ReadTask(task)))
        } else {
            self.settle_turn(false, now)
        }
    }

    pub(super) fn poll_live_tasks(
        &mut self,
        rows: Vec<lk_agents_abi::storage::thread::RoutedInbox>,
        foreground: bool,
    ) -> Result<Option<Operation>, Finish> {
        self.start_live_poll(if foreground { PollNext::Foreground } else { PollNext::Inbox(rows) })
    }

    fn start_live_poll(&mut self, next: PollNext) -> Result<Option<Operation>, Finish> {
        let sources: Vec<_> = self
            .live
            .turns
            .iter()
            .map(|(source, turn)| (source.clone(), turn.answer_task.clone()))
            .collect();
        let tasks = sources.iter().map(|(_, task)| task.clone()).collect();
        self.live.poll = Poll::Checking { sources, next };
        self.phase = Phase::Live(LiveStep::Poll);
        Ok(Some(Operation::Checkpoint(tasks)))
    }

    fn resume_poll(&mut self, next: PollNext) -> Result<Option<Operation>, Finish> {
        match next {
            PollNext::Finalizing => {
                self.phase = Phase::Live(LiveStep::Finalize);
                let source = self
                    .live
                    .report
                    .as_ref()
                    .expect("finalizing native source")
                    .origin
                    .request_id
                    .clone();
                Ok(Some(Operation::NextSource {
                    request_id: source,
                    should_yield: None,
                    pause: PARTITION_POLL_PAUSE,
                }))
            }
            PollNext::Foreground => {
                self.phase = Phase::Turn(Step::Next);
                Ok(Some(Operation::NextAll { should_yield: None, pause: PARTITION_POLL_PAUSE }))
            }
            PollNext::Inbox(rows) => self.accept_live_inbox(rows),
        }
    }

    fn close_cancelled_sources(&mut self) -> Result<Option<Operation>, Finish> {
        let Poll::Closing { mut sources, next } = std::mem::take(&mut self.live.poll) else {
            return Err(Finish::Error(StorageError::Invalid(
                "no canceled sources are pending".into(),
            )));
        };
        if let Some(source) = sources.pop_front() {
            self.live.poll = Poll::Closing { sources, next };
            self.phase = Phase::Live(LiveStep::CancelSource);
            Ok(Some(Operation::CancelSource(source)))
        } else {
            self.resume_poll(next)
        }
    }

    fn accept_live_inbox(
        &mut self,
        rows: Vec<lk_agents_abi::storage::thread::RoutedInbox>,
    ) -> Result<Option<Operation>, Finish> {
        let rows = self.next_drain(rows);
        if rows.is_empty() {
            if self.live.turns.is_empty() {
                self.phase = Phase::Live(LiveStep::Close);
                return Ok(Some(Operation::Commit {
                    answer_task: None,
                    lease: LeaseAfterCommit::Keep,
                    drained_id: None,
                }));
            }
            return Ok(Some(self.wait_live()));
        }
        self.rows = rows;
        self.index = 0;
        self.batch.clear();
        self.task_ids.clear();
        Ok(Some(self.drain_pickup()))
    }

    pub(super) fn prepare_opaque_dialogue(
        &mut self,
        stage: Box<Stage>,
    ) -> Result<Option<Operation>, Finish> {
        let source = self
            .turn
            .as_ref()
            .and_then(|turn| turn.user_message.as_ref())
            .expect("an opened opaque source retains its canonical human item")
            .id
            .clone();
        let mut messages = Vec::new();
        for edit in &stage.edits {
            let HistoryEdit::Add { item, .. } = edit;
            let native =
                lk_agents_abi::llm::chat_context::protocol::decode_chat_item(item.payload.clone())
                    .map_err(|error| Finish::Error(StorageError::Invalid(error.to_string())))?;
            if let lk_agents_abi::ChatItem::Message(message) = native
                && matches!(
                    message.role,
                    lk_agents_abi::Role::User | lk_agents_abi::Role::Assistant
                )
            {
                messages.push(message);
            }
        }
        if !matches!(self.live.opaque, OpaqueWrite::None) {
            return Err(Finish::Error(StorageError::Invalid(
                "another opaque result is already committing".into(),
            )));
        }
        self.live.opaque = OpaqueWrite::Preparing { source: source.clone(), stage };
        self.phase = Phase::Live(LiveStep::DialoguePrepared);
        Ok(Some(Operation::PrepareDialogue { source, messages }))
    }

    pub(super) fn commit_live_turn(&mut self) -> Operation {
        self.turn.take();
        self.phase = Phase::Live(LiveStep::Commit);
        Operation::Commit {
            answer_task: None,
            lease: LeaseAfterCommit::Keep,
            drained_id: self.rows.last().map(|row| row.id),
        }
    }

    fn open_live_run(&mut self) -> Operation {
        self.phase = Phase::Live(LiveStep::Open);
        Operation::Open {
            run_id: self.mint("live"),
            task_id: self.task_id.clone(),
            channel: None,
            access: Access::Kept,
            pickup: TaskPickup::Leave,
        }
    }

    fn read_live_inbox(&mut self) -> Operation {
        self.phase = Phase::Live(LiveStep::Inbox);
        Operation::ReadInbox
    }

    fn wait_live(&mut self) -> Operation {
        self.phase = Phase::Live(LiveStep::Wait);
        Operation::LiveWait { pause: PARTITION_POLL_PAUSE }
    }

    pub(super) fn live_step(
        &mut self,
        step: LiveStep,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let operation = match (step, completion) {
            (LiveStep::DialoguePrepared, Completion::DialoguePrepared(state)) => {
                let OpaqueWrite::Preparing { source, mut stage } =
                    std::mem::take(&mut self.live.opaque)
                else {
                    return Err(Finish::Error(StorageError::Invalid(
                        "no opaque write was prepared".into(),
                    )));
                };
                stage.end.as_mut().expect("an opaque result has its owned run end").state =
                    Some(state);
                self.live.opaque = OpaqueWrite::Committing { source };
                self.phase = Phase::Turn(Step::Settle);
                Operation::Stage(stage)
            }
            (LiveStep::DialogueCommitted, Completion::Done) => {
                self.live.opaque = OpaqueWrite::None;
                self.open_live_run()
            }
            (LiveStep::Commit, Completion::Committed { run_id, .. }) => {
                self.run_id = Some(run_id);
                if let OpaqueWrite::Committing { source } = &self.live.opaque {
                    let source = source.clone();
                    self.live.opaque = OpaqueWrite::Activating;
                    self.phase = Phase::Live(LiveStep::DialogueCommitted);
                    return Ok(Some(Operation::CommitDialogue(source)));
                }
                if let Some(report) = &self.live.report {
                    let request_id = report.origin.request_id.clone();
                    let turn = report.turn;
                    self.phase = Phase::Live(LiveStep::Acknowledged);
                    Operation::AcknowledgeLive { request_id, turn }
                } else {
                    self.open_live_run()
                }
            }
            (LiveStep::Acknowledged, Completion::Done) => {
                if let Some(report) = self.live.report.take()
                    && !report.working
                {
                    self.live.turns.remove(&report.origin.request_id);
                }
                self.open_live_run()
            }
            (LiveStep::Open, Completion::Opened { closed: false, .. }) => {
                if let Some(source) = self.live.outputs.pop_front() {
                    return self.live_output(source.request_id, source.output, now);
                }
                self.read_live_inbox()
            }
            (LiveStep::Finalize, Completion::Inbox(_)) => {
                return self.start_live_poll(PollNext::Finalizing);
            }
            (LiveStep::Inbox | LiveStep::Wait, Completion::Inbox(rows)) => {
                return self.poll_live_tasks(rows, false);
            }
            (LiveStep::Poll, Completion::Checkpoint { statuses }) => {
                let Poll::Checking { sources, next } = std::mem::take(&mut self.live.poll) else {
                    return Err(Finish::Error(StorageError::Invalid(
                        "no live task poll is pending".into(),
                    )));
                };
                if sources.len() != statuses.len() {
                    return Err(Finish::Error(StorageError::Invalid(
                        "a live task poll lost its task identities".into(),
                    )));
                }
                let mut canceled = Vec::new();
                for ((source, task), state) in sources.into_iter().zip(statuses) {
                    if state == Some(TaskState::Canceled) {
                        if let Some(turn) = self.live.turns.get_mut(&source) {
                            turn.stop = Some(crate::a2a::Stop::ClientCanceled);
                        }
                        if let Some(turn) = &mut self.turn
                            && turn.user_message.as_ref().is_some_and(|item| item.id == source)
                        {
                            turn.stop = Some(crate::a2a::Stop::ClientCanceled);
                        }
                        if !self.live.cancelled.contains(&task) {
                            canceled.push((source, task));
                        }
                    }
                }
                if canceled.is_empty() {
                    return self.resume_poll(next);
                }
                let tasks: std::collections::BTreeSet<_> =
                    canceled.iter().map(|(_, task)| task.clone()).collect();
                self.live.cancelled.extend(tasks.iter().cloned());
                self.live.poll = Poll::Native { sources: canceled, next };
                self.phase = Phase::Live(LiveStep::CancelNative);
                Operation::CancelLive(tasks.into_iter().collect())
            }
            (LiveStep::CancelNative, Completion::CancelledLive(mut unowned)) => {
                let Poll::Native { sources, next } = std::mem::take(&mut self.live.poll) else {
                    return Err(Finish::Error(StorageError::Invalid(
                        "no native task cancellation is pending".into(),
                    )));
                };
                if unowned
                    .iter()
                    .any(|task| !sources.iter().any(|(_, requested)| requested == task))
                {
                    return Err(Finish::Error(StorageError::Invalid(
                        "cancellation receipt changed task identity".into(),
                    )));
                }
                if let Some(report) = &self.live.report
                    && !report.working
                    && sources.iter().any(|(source, _)| source == &report.origin.request_id)
                {
                    unowned.extend(report.origin.task_ids.iter().cloned());
                }
                self.live.poll = Poll::Closing {
                    sources: sources
                        .into_iter()
                        .filter_map(|(source, task)| unowned.contains(&task).then_some(source))
                        .collect(),
                    next,
                };
                return self.close_cancelled_sources();
            }
            (LiveStep::CancelSource, Completion::Done) => return self.close_cancelled_sources(),
            (LiveStep::Wait, Completion::Event(HandlerEvent::Managed(report))) => {
                return self.managed_turn(*report, now);
            }
            (
                LiveStep::Wait | LiveStep::Finalize,
                Completion::Event(HandlerEvent::LiveOutput { request_id, output }),
            ) => {
                return self.live_output(request_id, output, now);
            }
            (LiveStep::Wait, Completion::Event(HandlerEvent::LiveCheckpoint { request_id })) => {
                return self.live_checkpoint(request_id, false);
            }
            (
                LiveStep::Finalize,
                Completion::Event(HandlerEvent::LiveCheckpoint { request_id }),
            ) => {
                return self.live_checkpoint(request_id, true);
            }
            (LiveStep::Finalize, Completion::Event(HandlerEvent::Checkpoint)) => {
                let tasks = self
                    .live
                    .report
                    .as_ref()
                    .expect("finalizing managed turn")
                    .origin
                    .task_ids
                    .clone();
                self.live.checkpoint_source =
                    self.live.report.as_ref().map(|report| report.origin.request_id.clone());
                self.phase = Phase::Live(LiveStep::FinalizeCheckpoint { finalizing: true });
                Operation::Checkpoint(tasks)
            }
            (LiveStep::FinalizeCheckpoint { finalizing }, Completion::Checkpoint { statuses }) => {
                let request_id = self.live.checkpoint_source.take().ok_or_else(|| {
                    Finish::Error(StorageError::Invalid(
                        "a checkpoint lost its selected managed source".into(),
                    ))
                })?;
                self.phase =
                    Phase::Live(if finalizing { LiveStep::Finalize } else { LiveStep::Wait });
                Operation::NextSource {
                    request_id,
                    should_yield: Some(statuses.contains(&Some(TaskState::Canceled))),
                    pause: PARTITION_POLL_PAUSE,
                }
            }
            (LiveStep::Close, Completion::Committed { .. }) => {
                self.phase = Phase::Release;
                Operation::ReleaseDrained
            }
            (_, other) => return Err(self.unexpected(other)),
        };
        Ok(Some(operation))
    }
}

fn managed_output(report: &LiveTurnReport) -> HandlerOutput {
    let mut output = HandlerOutput::default();
    match report.result.as_ref() {
        Ok(turn) => {
            output.new_items = turn.result.new_messages.iter().map(encode_chat_item).collect();
            output.usage = turn.result.usage.clone();
            output.interrupted = turn.result.interrupted;
            output.pending_call = turn.result.pending_call.clone();
            match &turn.result.outcome {
                AgentOutcome::Completed { output: Output::Text(text) } => {
                    output.output = Some(Value::String(text.clone()))
                }
                AgentOutcome::Completed { output: Output::Validated(_) } => {
                    output.output = turn
                        .state
                        .frames
                        .get(&turn.state.active_frame_id)
                        .and_then(|frame| frame.output.clone());
                }
                AgentOutcome::InputRequired(question) => {
                    output.input_required = Some(question.question.clone())
                }
                AgentOutcome::NoOutput => {}
            }
        }
        Err(error) => output.error = Some(error.to_string()),
    }
    if report.working {
        output.status = Some(TaskState::Working);
    }
    output
}

#[cfg(test)]
#[path = "live_tests.rs"]
mod tests;
