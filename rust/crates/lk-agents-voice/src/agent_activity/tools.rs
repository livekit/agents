//! The driver's tool side: the running set, background updates (`ctx.update()`), the deferred
//! tool reply, and the fillers (`RunContext.with_filler`).

use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::closing::SESSION_CLOSING;
use crate::agent_session::effects::{Out, TurnTimer};
use crate::agent_session::{Driver, FillerSource, HookCall, HookId, ToolUpdate};
use crate::events::{SessionEvent, ToolEndStatus, ToolLifecycle, ToolReplyStatus};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::{ReplySource, TurnInstructions};
use crate::tool_executor::{
    Batch, HANDOFF_AFTER_UPDATE, Running, ToolAnswer, ToolEnd, ToolOutcome, answer, decide,
    default_reply_instructions, emit_ended, pair_for,
};
use crate::trace::{
    CoreStep, FillerEnd, GenerationStep, HandoffStep, HookEnd, ToolStep as TraceToolStep,
};
use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput, Ts};
use std::time::Duration;

impl Driver {
    // ------------------------------------------------------------------ the running set

    /// A hosted call enters the running set and is announced; the away window is held off.
    pub(crate) fn on_tool_started(
        &mut self,
        now: Ts,
        speech: SpeechId,
        step: u32,
        call: FunctionCall,
        hosted: bool,
        out: &mut Out,
    ) {
        out.trace(TraceToolStep::CallStarted {
            speech,
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            hosted,
        });
        if !hosted {
            return;
        }
        let closing = self.session.closing_past_edge();
        out.machine().append(self.session.states.on_tool_call_started(now, closing));
        // The call is the run's once it runs, whichever run is open when it is committed.
        if let Some(run) = self.session.runs.id()
            && self.session.runs.owns(crate::run_result::Owned::Speech(speech))
        {
            let item = Box::new(lk_agents_abi::ChatItem::FunctionCall(call.clone()));
            out.machine().emit(SessionEvent::RunItemAdded { run, item });
        }
        let output_id = self.session.ids.mint("item");
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
            call: call.clone(),
        }));
        self.activity.tools.start(call, output_id, speech, step);
    }

    /// A blocking call leaves the running set; a detached one stays until its `ToolUpdate::Final`.
    pub(crate) fn on_tool_ended(
        &mut self,
        now: Ts,
        speech: SpeechId,
        call_id: String,
        end: ToolEnd,
        message: Option<String>,
        out: &mut Out,
    ) {
        out.trace(TraceToolStep::CallEnded { speech, call_id: call_id.clone(), end });
        // Ending the wrong call leaves one the close would answer twice.
        if !self.activity.tools.end(speech, &call_id, end == ToolEnd::Detached) {
            return;
        }
        let status = match end {
            ToolEnd::Done => ToolEndStatus::Done,
            ToolEnd::Error => ToolEndStatus::Error,
            ToolEnd::Cancelled => ToolEndStatus::Cancelled,
            ToolEnd::Detached => {
                self.fillers_reset_dwell(now, &call_id, out);
                out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id: call_id.clone(),
                    call_id,
                    message: message.unwrap_or_default(),
                }));
                return;
            }
        };
        // The last tool to land under two listening participants restarts the away window (#6883).
        let closing = self.session.closing_past_edge();
        out.machine().append(self.session.states.on_tool_call_ended(now, closing));
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
            id: call_id.clone(),
            call_id,
            message,
            status,
        }));
    }

    /// Settle blocking calls whose reply task was forcefully stopped (closing or the watchdog),
    /// each end before its pair, in start order.
    pub(crate) fn end_live_tools(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let live = self.activity.tools.live(speech);
        if live.is_empty() {
            return;
        }
        let wall = self.step.wall();
        let mut pairs = Vec::with_capacity(live.len());
        for r in &live {
            let outcome = if self.session.closing.is_some() {
                ToolOutcome::Error(SESSION_CLOSING.to_owned())
            } else {
                ToolOutcome::Cancelled
            };
            let execution = answer(r.call(), outcome, wall);
            self.on_tool_ended(
                now,
                speech,
                r.call().call_id.clone(),
                execution.end,
                (!execution.output.output.is_empty()).then(|| execution.output.output.clone()),
                out,
            );
            pairs.push((r.step(), execution.call, execution.output));
        }
        for (step, call, output) in pairs {
            self.commit_item(
                ChatItem::FunctionCall(call),
                Source::ToolCall { speech, step },
                Visibility::Both,
                out,
            );
            self.commit_item(
                ChatItem::FunctionCallOutput(output),
                Source::ToolOutput { speech, step },
                Visibility::Both,
                out,
            );
        }
        self.publish_activity();
    }

    // ------------------------------------------------------------------ background updates

    pub(crate) fn on_tool_update(
        &mut self,
        now: Ts,
        call_id: String,
        update: ToolUpdate,
        out: &mut Out,
    ) {
        if !self.activity.tools.is_detached(&call_id) {
            out.trace(TraceToolStep::UpdateRefused { call_id });
            return;
        }
        let wall = self.step.wall();
        // `update()` restarts a pending filler dwell so it can't race the update (`reset_dwell`).
        self.fillers_reset_dwell(now, &call_id, out);
        match update {
            ToolUpdate::Progress { message, raw } => {
                let (n, running) =
                    self.activity.tools.next_update(&call_id).expect("detached, checked above");
                let id = format!("{call_id}_update_{n}");
                let pair = pair_for(&running, &id, ToolOutcome::Reply(message), wall, || {
                    self.session.ids.mint("item")
                });
                out.trace(TraceToolStep::Updated {
                    call_id: call_id.clone(),
                    id: id.clone(),
                    is_final: false,
                });
                out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id: id.clone(),
                    call_id,
                    message: raw,
                }));
                self.commit_pair(&running, id, pair, out);
            }
            ToolUpdate::Final { answer: raw } => {
                let id = format!("{call_id}_final");
                let running = self.activity.tools.take(&call_id).expect("detached, checked above");
                // The end is journalled before the away window moves (#6883).
                out.trace(TraceToolStep::Updated {
                    call_id: call_id.clone(),
                    id: id.clone(),
                    is_final: true,
                });
                let closing = self.session.closing_past_edge();
                out.machine().append(self.session.states.on_tool_call_ended(now, closing));
                // Silent returns voice nothing; an `Agent` or cancellation is refused.
                let outcome = match raw {
                    ToolAnswer::Returned { value: None, agents } if agents.is_empty() => None,
                    ToolAnswer::Raised(crate::tool_executor::ToolRaise::StopResponse) => None,
                    ToolAnswer::Returned { agents, .. } if !agents.is_empty() => {
                        out.trace(CoreStep::PortFailure(HANDOFF_AFTER_UPDATE));
                        let message = Some(HANDOFF_AFTER_UPDATE.to_owned());
                        emit_ended(out, id, call_id, message, ToolEndStatus::Error);
                        return;
                    }
                    ToolAnswer::Cancelled => {
                        emit_ended(out, id, call_id, None, ToolEndStatus::Cancelled);
                        return;
                    }
                    other => Some(decide(other)),
                };
                let Some(outcome) = outcome else {
                    emit_ended(out, id, call_id, None, ToolEndStatus::Done);
                    return;
                };
                let pair = pair_for(&running, &id, outcome, wall, || self.session.ids.mint("item"));
                let status =
                    if pair.1.is_error { ToolEndStatus::Error } else { ToolEndStatus::Done };
                emit_ended(out, id.clone(), call_id, Some(pair.1.output.clone()), status);
                self.commit_pair(&running, id, pair, out);
            }
        }
        self.try_deliver_tool_reply(now, out);
    }

    /// Commit the pair to the record at once, so an earlier reply still sees it, and batch it.
    fn commit_pair(
        &mut self,
        running: &Running,
        id: String,
        pair: (FunctionCall, FunctionCallOutput),
        out: &mut Out,
    ) {
        let (call, output) = pair;
        let source = Source::ToolUpdate { speech: running.speech(), step: running.step() };
        let items = vec![ChatItem::FunctionCall(call), ChatItem::FunctionCallOutput(output)];
        for item in &items {
            self.commit_item(item.clone(), source.clone(), Visibility::Both, out);
        }
        self.publish_activity();
        let agent = self.activity.agent.clone();
        self.activity.tools.queue_update(agent, items, id);
    }

    // ------------------------------------------------------------------ the deferred reply

    /// One reply for every pending update, at the first idle instant with no current speech. Runs
    /// before idle waiters are answered, so a `foreground()` waits behind it.
    pub(crate) fn try_deliver_tool_reply(&mut self, now: Ts, out: &mut Out) {
        if !self.activity.tools.reply_due() {
            return;
        }
        if self.session.closing_past_edge() || self.session.idle.held() || !self.is_idle() {
            return;
        }
        let batch = self.activity.tools.take_batch();
        // "if the update is still the tail, the agent hasn't spoken since".
        let last_pending = batch.last_item_id();
        let last_record =
            self.session.chat_ctx.agent_chat_ctx().items.last().map(|i| i.id().to_owned());
        let at_tail = last_pending.is_some() && last_pending == last_record;
        let call_ids = batch.update_ids();
        let id = self.activity.hooks.mint();
        let call = HookCall::RenderToolReply {
            id,
            agent: self.activity.agent.clone(),
            call_ids: call_ids.clone(),
            at_tail,
        };
        let wait = self.hook_wait();
        if self.issue_hook(now, call, Some(wait), out) {
            self.activity.tools.rendering(id, batch, at_tail);
        } else {
            let text = default_reply_instructions(&call_ids, at_tail);
            self.schedule_tool_reply(now, batch, at_tail, text, out);
        }
    }

    pub(crate) fn on_tool_reply_rendered(
        &mut self,
        now: Ts,
        hook: HookId,
        text: Option<String>,
        out: &mut Out,
    ) {
        let Some((batch, at_tail)) = self.activity.tools.rendered(hook) else { return };
        let call_ids = batch.update_ids();
        let text = text.unwrap_or_else(|| default_reply_instructions(&call_ids, at_tail));
        self.schedule_tool_reply(now, batch, at_tail, text, out);
    }

    fn schedule_tool_reply(
        &mut self,
        now: Ts,
        batch: Batch,
        at_tail: bool,
        text: String,
        out: &mut Out,
    ) {
        let update_ids = batch.update_ids();
        // A previous frame's updates are re-inserted for this request only.
        let foreign = batch.foreign_items(&self.activity.agent);
        let chat_ctx = (!foreign.is_empty()).then(|| {
            let mut ctx = self.session.chat_ctx.agent_chat_ctx();
            for item in foreign {
                if !ctx.items.iter().any(|i| i.id() == item.id()) {
                    ctx.items.push(item);
                }
            }
            ctx
        });
        let message_id = self.session.ids.mint("item");
        let handle = self.start_speech(
            now,
            SpeechSpec {
                turn_instructions: Some(TurnInstructions {
                    instructions: crate::generation::Instructions::plain(text),
                    message_id,
                }),
                // The summary must not start another round of tool calls.
                tool_choice: Some(lk_agents_abi::ToolChoice::Forbidden),
                chat_ctx,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled,
                )
            },
            out,
        );
        let speech = handle.id();
        out.trace(TraceToolStep::ReplyScheduled {
            speech,
            update_ids: update_ids.clone(),
            at_tail,
        });
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Reply {
            update_ids: update_ids.clone(),
            status: ToolReplyStatus::Scheduled,
            speech,
        }));
        self.activity.tools.replying(speech, update_ids);
    }

    /// A speech ended: if it was the deferred reply, its fate is reported and the next batch may go.
    pub(crate) fn tool_reply_done(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let Some(update_ids) = self.activity.tools.reply_done(speech) else { return };
        let status = match self.activity.speech_scheduler.handle(speech) {
            Some(h) if h.interrupted() => ToolReplyStatus::Interrupted,
            // The instruction permits an empty response when the agent already covered the result.
            Some(h) if h.chat_items().is_empty() => ToolReplyStatus::Skipped,
            Some(_) | None => ToolReplyStatus::Completed,
        };
        out.trace(TraceToolStep::ReplyDone { speech, status });
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Reply {
            update_ids,
            status,
            speech,
        }));
        self.try_deliver_tool_reply(now, out);
    }

    /// `SpeechCommand::FillerStart`: the scope opened. The dwell begins at the first idle instant.
    pub(crate) fn on_filler_start(
        &mut self,
        now: Ts,
        id: FillerId,
        call_id: String,
        speech: SpeechId,
        delay: Duration,
        interval: Option<Duration>,
        max_steps: Option<u32>,
        source: FillerSource,
        out: &mut Out,
    ) {
        let fresh =
            !self.activity.tools.has_filler(id) && self.activity.speech_scheduler.contains(speech);
        let sched = fresh.then(|| {
            FillerScheduler::new(delay.as_secs_f64(), interval.map(|d| d.as_secs_f64()), max_steps)
        });
        let Some(Ok(sched)) = sched else {
            out.trace(GenerationStep::FillerStopped { filler: id, cause: FillerEnd::Refused });
            return;
        };
        out.trace(GenerationStep::FillerStarted { filler: id, speech });
        self.activity.tools.open_filler(id, sched, call_id, speech, source);
        self.pump_fillers(now, out);
    }

    pub(crate) fn end_filler(&mut self, id: FillerId, cause: FillerEnd, out: &mut Out) {
        let Some((armed, asking)) = self.activity.tools.close_filler(id) else { return };
        if armed {
            out.machine().cancel(TurnTimer::Filler(id));
        }
        if let Some(hook) = asking
            && let Some(kind) = self.activity.hooks.finish(hook)
        {
            out.trace(HandoffStep::HookFinished { hook, kind, how: HookEnd::Abandoned });
        }
        out.trace(GenerationStep::FillerStopped { filler: id, cause });
    }

    /// A speech ended or was interrupted: its tools' fillers go with it.
    pub(crate) fn fillers_speech_ended(&mut self, speech: SpeechId, out: &mut Out) {
        for id in self.activity.tools.fillers_of_speech(speech) {
            self.end_filler(id, FillerEnd::SpeechEnded, out);
        }
    }

    /// `ctx.update()` on `call_id`: a dwell in flight restarts (`reset_dwell`).
    fn fillers_reset_dwell(&mut self, now: Ts, call_id: &str, out: &mut Out) {
        for id in self.activity.tools.fillers_of_call(call_id) {
            if let Some(action) = self.activity.tools.filler_busy(id, now) {
                self.perform_filler(now, id, action, out);
            }
        }
    }

    /// The dwell waits for idle and restarts when the agent or user starts speaking (the
    /// reference's `wait_for_idle()` then race). Once per wake.
    pub(crate) fn pump_fillers(&mut self, now: Ts, out: &mut Out) {
        if !self.activity.tools.has_fillers() {
            return;
        }
        let idle = self.is_idle();
        let busy = self.session.states.someone_speaking();
        for id in self.activity.tools.filler_ids() {
            if let Some(action) = self.activity.tools.filler_wake(id, idle, busy, now) {
                self.perform_filler(now, id, action, out);
            }
        }
    }

    /// The dwell or cooldown is over. A deadline in the same wake as a speaking edge counts as busy.
    pub(crate) fn on_filler_timer(&mut self, now: Ts, id: FillerId, out: &mut Out) {
        let busy = self.session.states.someone_speaking();
        let Some(action) = self.activity.tools.filler_timer(id, busy, now) else { return };
        self.perform_filler(now, id, action, out);
    }

    pub(crate) fn on_filler_source(
        &mut self,
        now: Ts,
        hook: HookId,
        text: Option<String>,
        spoke: bool,
        out: &mut Out,
    ) {
        let Some((id, step)) = self.activity.tools.filler_answered(hook) else { return };
        self.fire_filler(now, id, step, text, spoke, out);
    }

    fn fire_filler(
        &mut self,
        now: Ts,
        id: FillerId,
        step: u32,
        text: Option<String>,
        spoke: bool,
        out: &mut Out,
    ) {
        let speech = text.map(|text| {
            let handle = self.start_speech(
                now,
                SpeechSpec::new(
                    SpeechKind::Say,
                    ReplySource::Text(text),
                    self.session.config.effective.turn.interruption.enabled,
                ),
                out,
            );
            handle.id()
        });
        let produced = spoke || speech.is_some();
        out.trace(GenerationStep::FillerFired { filler: id, step, spoke: produced, speech });
        let Some(action) = self.activity.tools.filler_fired(id, now, produced) else {
            return;
        };
        self.perform_filler(now, id, action, out);
    }

    /// Perform one machine action: a wait is a timer, a fire a say or hook, done the end.
    fn perform_filler(&mut self, now: Ts, id: FillerId, action: FillerAction, out: &mut Out) {
        if !self.activity.tools.has_filler(id) {
            return;
        }
        match action {
            FillerAction::WaitForIdle | FillerAction::Dwell(_) | FillerAction::CoolDown(_) => {
                let until = match action {
                    FillerAction::Dwell(at) | FillerAction::CoolDown(at) => Some(at),
                    FillerAction::WaitForIdle | FillerAction::Fire { .. } | FillerAction::Done => {
                        None
                    }
                };
                let Some(timer) = self.activity.tools.filler_wait(id, until) else { return };
                if timer.cancel {
                    out.machine().cancel(TurnTimer::Filler(id));
                }
                if let Some(at) = timer.arm {
                    out.machine().arm(TurnTimer::Filler(id), at);
                }
            }
            FillerAction::Fire { step } => match self.activity.tools.filler_source(id) {
                Some(FillerSource::Text(text)) => {
                    self.fire_filler(now, id, step, Some(text), false, out)
                }
                Some(FillerSource::Host) => {
                    let hook = self.activity.hooks.mint();
                    let wait = self.hook_wait();
                    let call = HookCall::FillerSource { id: hook, filler: id, step };
                    if !self.issue_hook(now, call, Some(wait), out) {
                        self.fire_filler(now, id, step, None, false, out)
                    } else {
                        self.activity.tools.filler_asking(id, hook);
                    }
                }
                None => {}
            },
            FillerAction::Done => self.end_filler(id, FillerEnd::Done, out),
        }
    }
}
