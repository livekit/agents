//! The driver's tool side: the running set, background updates (`ctx.update()`), the deferred
//! tool reply, and the fillers (`RunContext.with_filler`).

use crate::agent::{Source, Visibility};
use crate::agent_activity::realtime::Request;
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::closing::SESSION_CLOSING;
use crate::agent_session::effects::{Effect, Out, Timer};
use crate::agent_session::{Answer, RequestId};
use crate::agent_session::{Driver, FillerSource, HookCall, HookId, ToolUpdate};
use crate::events::{DiagnosticEvent, SessionEvent, ToolEndStatus, ToolLifecycle, ToolReplyStatus};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::ReplySource;
use crate::speech_handle::SpeechHandle;
use crate::tool_executor::{
    Batch, CallId, Deferred, Delivery, FillerRun, HostedTool, MadeCall, Pending, Rendering,
    Running, ToolEnd, ToolExecution, ToolGo, ToolOutcome, ToolReply, answer,
    default_reply_instructions, emit_ended, pair_for,
};
use crate::trace::{FillerEnd, GenerationStep, HandoffStep, HookEnd, ToolStep as TraceToolStep};
use lk_agents_abi::agent::{ArgumentIdentity, ReadBackAnswer, ToolTemplate};
use lk_agents_abi::llm::tool_executor::WAITING_PREFIX;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput, Ts};
use lk_agents_core::agent_run::Event as RunEvent;
use lk_agents_core::tool_calls::{Admission, admit};
use std::time::Duration;

/// `_FillerScheduler`'s log when its `say` raised: the filler ends there.
const FILLER_STOPPED: &str =
    "filler stopped on an error, no further filler will play for this tool call";

impl Driver {
    // ------------------------------------------------------------------ dispatch

    /// The model made a call: the core admits it (arguments a template leaked into repaired, an
    /// `InputRequired(tool=)` answer in place of the model's arguments). A call `tool_choice="none"`
    /// forbade is recorded as made and never runs. Any other waits for its step's authorization
    /// to play (a call made while another speech plays, or by a preemptive generation, waits),
    /// then joins its reply's run and starts with the first move the run decides; hosted by the
    /// reply's agent unless it names a tool the request did not offer.
    pub(crate) fn on_tool_made(
        &mut self,
        now: Ts,
        (speech, step): (SpeechId, u32),
        made: MadeCall,
        go: RequestId,
        out: &mut Out,
    ) {
        let MadeCall { mut call, spec, forbidden, answering, agent } = made;
        let answering = answering.as_ref().map(|a| (a.tool.as_str(), a.said.as_str()));
        let admission = admit(&mut call, spec.as_slice(), forbidden, answering);
        let advertised = admission.offered();
        out.trace(TraceToolStep::Calls { speech, names: vec![call.name.clone()] });
        let Some(handle) = self.activity.speech_scheduler.handle(speech).cloned() else {
            return out.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        };
        if let Admission::Forbidden { .. } = admission {
            if advertised {
                handle.item_added(&ChatItem::FunctionCall(call.clone()));
            }
            if handle.scheduled() {
                self.on_tool_started(now, (speech, step), call.clone(), None, out);
            }
            self.activity.tools.dropped.entry((speech, step)).or_default().push(call.name);
            return out.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        }
        let agent = agent.filter(|_| admission == Admission::Offered);
        let call = Deferred { speech, step, id: None, call, advertised, agent, go };
        self.activity.tools.deferred.push(call);
        self.settle_deferred_calls(out);
    }

    /// Starts each waiting call whose step is authorized, in the model's order, once its run
    /// decided its first move, and drops unrun those whose speech was interrupted or done before
    /// the authorization. `true` when one was.
    pub(crate) fn settle_deferred_calls(&mut self, out: &mut Out) -> bool {
        let mut settled = false;
        let mut waiting = Vec::new();
        let now = self.step.now;
        for mut d in std::mem::take(&mut self.activity.tools.deferred) {
            // Behind an earlier call of the same speech still waiting: started in order.
            if waiting.iter().any(|w: &Deferred| w.speech == d.speech) {
                waiting.push(d);
                continue;
            }
            // Authorization last: a discarded preemptive generation, or a speech interrupted
            // while queued, drops the call unrun. A start already authorized runs, interrupted or
            // not, as the reference's executor starts the body right after
            // `tool_execution_started_cb`.
            let handle = self.activity.speech_scheduler.handle(d.speech).cloned();
            match handle {
                Some(h) if h.step_authorized(d.step) => {
                    // `_tool_execution_started_cb`: stamped with its start (history is ordered by
                    // `created_at`, and it was made before the step was authorized).
                    if d.id.is_none() {
                        d.call.created_at = self.step.wall();
                        d.id = self.reply_call_made(d.speech, d.call.clone(), out);
                    }
                    match d.id {
                        Some(id) if !self.reply_first_ready(d.speech, id) => {
                            waiting.push(d);
                            continue;
                        }
                        Some(id) => self.start_call(now, d, id, &h, out),
                        None => out.answer(Some(d.go), Answer::ToolGo(ToolGo::Gone)),
                    }
                }
                Some(h) if !h.interrupted() && !h.done() => {
                    waiting.push(d);
                    continue;
                }
                Some(_) | None => out.answer(Some(d.go), Answer::ToolGo(ToolGo::Gone)),
            }
            settled = true;
        }
        self.activity.tools.deferred = waiting;
        settled
    }

    /// The call is told to the speech and started with its first move.
    fn start_call(&mut self, now: Ts, d: Deferred, id: CallId, h: &SpeechHandle, out: &mut Out) {
        let Deferred { speech, step, call, advertised, agent, go, .. } = d;
        if advertised {
            h.item_added(&ChatItem::FunctionCall(call.clone()));
        }
        let first = Box::new(self.reply_call_start(speech, id));
        let host = self.on_tool_started(now, (speech, step), call.clone(), agent.as_ref(), out);
        let start = ToolGo::Start { call: id, item: Box::new(call), host, first };
        out.answer(Some(go), Answer::ToolGo(start));
    }

    // ------------------------------------------------------------------ the running set

    /// A started call is the run's; a hosted one enters the running set and is announced, and the
    /// away window is held off. The invocation its callbacks run in, when hosted.
    pub(crate) fn on_tool_started(
        &mut self,
        now: Ts,
        (speech, step): (SpeechId, u32),
        call: FunctionCall,
        agent: Option<&std::sync::Arc<dyn crate::tool_executor::AgentFrame>>,
        out: &mut Out,
    ) -> Option<HostedTool> {
        let handle = self.activity.speech_scheduler.handle(speech);
        let hosted = agent.zip(handle).map(|(agent, handle)| {
            let operation = self.activity.tools.mint_operation();
            HostedTool::of(agent, self.session.ids.session(), handle, operation, Some(&call))
        });
        out.trace(TraceToolStep::CallStarted {
            speech,
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            hosted: hosted.is_some(),
        });
        // The call is the run's once it starts, whichever run is open when it is committed: one
        // answered without running (an unknown tool) too, as its output is.
        if let Some(run) = self.session.runs.current()
            && self.session.runs.owns(crate::run_result::Owned::Speech(speech))
        {
            let item = Box::new(lk_agents_abi::ChatItem::FunctionCall(call.clone()));
            out.machine().emit(SessionEvent::RunItemAdded { run, item });
        }
        let running = hosted.clone()?;
        let closing = self.session.closing_past_edge();
        out.machine().append(self.session.states.on_tool_call_started(now, closing));
        let output_id = self.session.ids.mint("item");
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
            call: call.clone(),
        }));
        self.activity.tools.running.push(Running {
            call,
            output_id,
            speech,
            step,
            updates: 0,
            early: Vec::new(),
            hosted: running,
        });
        hosted
    }

    /// The host prepared a call's arguments, or refused them: the reply's run admits the call
    /// (approval on the validated identity, the tool's duplicate policy) or records the refusal.
    pub(crate) fn on_tool_prepared(
        &mut self,
        speech: SpeechId,
        call: CallId,
        identity: Result<ArgumentIdentity, lk_agents_core::tool_calls::PrepareFailure>,
        go: RequestId,
        out: &mut Out,
    ) {
        let event = match identity {
            // The call's task holds the prepared arguments: the resource is the call's.
            Ok(identity) => RunEvent::ToolPrepared { call, prepared: ResourceId(call.0), identity },
            Err(failure) => RunEvent::ToolPrepareFailed { call, failure },
        };
        self.reply_call_asks(speech, call, go, event, out);
    }

    /// A call's read-back answered: the reply's run settles the call on it.
    pub(crate) fn on_read_back(
        &mut self,
        speech: SpeechId,
        call: CallId,
        result: Result<ReadBackAnswer, String>,
        go: RequestId,
        out: &mut Out,
    ) {
        let event = RunEvent::ReadBackRendered { call, result };
        self.reply_call_asks(speech, call, go, event, out);
    }

    /// A blocking call leaves the running set; a detached one stays until its `ToolUpdate::Final`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn on_tool_ended(
        &mut self,
        now: Ts,
        (speech, call): (SpeechId, Option<CallId>),
        call_id: String,
        end: ToolEnd,
        message: Option<String>,
        outcome: ToolOutcome,
        out: &mut Out,
    ) {
        // The reply's run sees the body's answer as it arrives: a later duplicate is admitted
        // against what still runs. A call answered for a task that never did (aborted, or
        // stopped by the close) is not the run's to settle.
        match call {
            Some(call) => self.reply_event(speech, RunEvent::ToolSettled { call, outcome }, out),
            None => self.reply_call_dropped(speech),
        }
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
                self.open_expert(&call_id);
                self.fillers_reset_dwell(now, &call_id, out);
                out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id: call_id.clone(),
                    call_id: call_id.clone(),
                    message: message.unwrap_or_default(),
                }));
                let early = self
                    .activity
                    .tools
                    .detached(&call_id)
                    .map(|r| std::mem::take(&mut r.early))
                    .unwrap_or_default();
                for update in early {
                    self.on_tool_update(now, call_id.clone(), update, out);
                }
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
    /// each end before its pair, in start order. Their bodies are cancelled too: one may have
    /// detached at the host before the driver heard its answer, and would outlive its told end.
    pub(crate) fn end_live_tools(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let live: Vec<Running> = self.activity.tools.live(speech).cloned().collect();
        if live.is_empty() {
            return;
        }
        let wall = self.step.wall();
        let mut pairs = Vec::with_capacity(live.len());
        for r in live {
            let (call_id, hosted) = (r.call.call_id.clone(), r.hosted.clone());
            out.machine().push(Effect::CancelTool { call_id, hosted });
            let outcome = if self.session.closing.is_some() {
                ToolOutcome::ToolError(SESSION_CLOSING.to_owned())
            } else {
                ToolOutcome::Cancelled
            };
            let execution = answer(&r.call, outcome.clone().into(), wall);
            self.on_tool_ended(
                now,
                (speech, None),
                r.call.call_id.clone(),
                execution.end,
                (!execution.output.output.is_empty()).then(|| execution.output.output.clone()),
                outcome,
                out,
            );
            pairs.push((r.step, execution.call, execution.output));
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

    /// `_ToolExecutor.aclose()`: every body still running at the host, a detached one by now, is
    /// cancelled and its end told as its own cancellation would tell it.
    pub(crate) fn cancel_running_tools(&mut self, now: Ts, out: &mut Out) {
        let running = std::mem::take(&mut self.activity.tools.running);
        if running.is_empty() {
            return;
        }
        for r in running {
            let call_id = r.call.call_id.clone();
            let id = if r.detached() { format!("{call_id}_final") } else { call_id.clone() };
            out.machine().push(Effect::CancelTool { call_id: call_id.clone(), hosted: r.hosted });
            let closing = self.session.closing_past_edge();
            out.machine().append(self.session.states.on_tool_call_ended(now, closing));
            emit_ended(out, id, call_id, None, ToolEndStatus::Cancelled);
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
        if self.activity.tools.detached(&call_id).is_none() {
            match self.activity.tools.undetached(&call_id) {
                Some(running) => running.early.push(update),
                None => out.trace(TraceToolStep::UpdateRefused { call_id }),
            }
            return;
        }
        if self.on_expert_update(now, &call_id, &update, out) {
            return;
        }
        let wall = self.step.wall();
        // `update()` restarts a pending filler dwell so it can't race the update (`reset_dwell`).
        self.fillers_reset_dwell(now, &call_id, out);
        match update {
            ToolUpdate::Progress { message, raw } => {
                let r = self.activity.tools.detached(&call_id).expect("detached, checked above");
                let id = format!("{call_id}_update_{}", r.updates);
                r.updates += 1;
                let running = r.clone();
                let progress = ToolOutcome::Value {
                    text: message,
                    reply_required: true,
                    verbatim: false,
                    more: None,
                };
                let pair = pair_for(&running, &id, progress.into(), wall, || {
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
            ToolUpdate::Final { report } => {
                let id = format!("{call_id}_final");
                let tools = &mut self.activity.tools.running;
                let i = tools.iter().position(|r| r.call.call_id == call_id);
                let running = tools.remove(i.expect("detached, checked above"));
                // The end is journalled before the away window moves (#6883).
                out.trace(TraceToolStep::Updated {
                    call_id: call_id.clone(),
                    id: id.clone(),
                    is_final: true,
                });
                let closing = self.session.closing_past_edge();
                out.machine().append(self.session.states.on_tool_call_ended(now, closing));
                // A silent return voices nothing (`_execute_tool`'s `None` and `StopResponse`).
                match report.outcome {
                    ToolOutcome::Cancelled => {
                        emit_ended(out, id, call_id, None, ToolEndStatus::Cancelled);
                        return self.try_deliver_tool_reply(now, out);
                    }
                    ToolOutcome::Nothing | ToolOutcome::StopResponse => {
                        emit_ended(out, id, call_id, None, ToolEndStatus::Done);
                        return self.try_deliver_tool_reply(now, out);
                    }
                    _ => {}
                }
                let pair = pair_for(&running, &id, report, wall, || self.session.ids.mint("item"));
                let status = match pair.end {
                    ToolEnd::Error => ToolEndStatus::Error,
                    ToolEnd::Done | ToolEnd::Cancelled | ToolEnd::Detached => ToolEndStatus::Done,
                };
                emit_ended(out, id.clone(), call_id, Some(pair.output.output.clone()), status);
                self.commit_pair(&running, id, pair, out);
            }
        }
        self.try_deliver_tool_reply(now, out);
    }

    /// Commit the pair to the record at once, so an earlier reply still sees it, deliver what it
    /// owes the person, and batch it for the deferred reply when it asks for one
    /// (`_enqueue_reply`).
    fn commit_pair(&mut self, running: &Running, id: String, pair: ToolExecution, out: &mut Out) {
        let ToolExecution { call, output, delivery, .. } = pair;
        let reply_required = output.reply_required;
        let source = Source::ToolUpdate { speech: running.speech, step: running.step };
        let items = vec![ChatItem::FunctionCall(call), ChatItem::FunctionCallOutput(output)];
        for item in &items {
            self.commit_item(item.clone(), source.clone(), Visibility::Both, out);
        }
        self.publish_activity();
        if let Some(delivery) = delivery {
            self.deliver_outcome(self.step.now, running.speech, delivery, out);
        }
        if reply_required {
            let agent = self.activity.agent.clone();
            let mut hosted = running.hosted.clone();
            hosted.invocation.operation = self.activity.tools.mint_operation();
            self.activity.tools.pending.push(Pending { agent, items, update_id: id, hosted });
        }
    }

    // ------------------------------------------------------------------ deliveries

    /// What a call's outcome owes the person: a verbatim result said as written (`say()`), or an
    /// `InputRequired` asked (`_apply_transition`), its call answered by the person's next
    /// message, the question the output of a run ending on the tool's speech.
    pub(crate) fn deliver_outcome(
        &mut self,
        now: Ts,
        speech: SpeechId,
        delivery: Delivery,
        out: &mut Out,
    ) {
        match delivery {
            Delivery::Say(text) => {
                self.say_text(now, text, out);
            }
            Delivery::Complete(value) => self.complete_voice_task(now, speech, value, out),
            Delivery::Ask(ask) => {
                let output = crate::run_result::RunOutput::Ask {
                    question: ask.question.clone(),
                    verbatim: ask.verbatim,
                    answer_tool: ask.tool.clone(),
                };
                self.session.runs.outputs.insert(speech, output.clone());
                let asking = if ask.verbatim {
                    self.say_text(now, ask.question.clone(), out)
                } else {
                    let instructions = format!(
                        "{}{}",
                        lk_agents_abi::llm::tool_executor::ASK_IN_OWN_WORDS,
                        ask.question
                    );
                    self.reply_with_instructions(now, instructions, None, out)
                };
                // `InputRequired(tool=)`: the person's next turn is a call of that tool; asked in
                // the agent's own words, once that reply is done.
                if let Some(tool) = &ask.tool {
                    let after = if ask.verbatim { None } else { asking };
                    self.activity.tools.claim =
                        Some(crate::tool_executor::Claim { tool: tool.clone(), after });
                }
                if let Some(asking) = asking {
                    self.session.runs.outputs.insert(asking, output);
                }
            }
        }
    }

    /// The person's message answers every call an `InputRequired` left waiting, by the core's
    /// rule (the request planner applies the same one): written as those calls' outputs.
    pub(crate) fn answer_pending_input(&mut self, text: &str, out: &mut Out) {
        let waits = |o: &FunctionCallOutput| o.output.starts_with(WAITING_PREFIX);
        let mut waiting: Vec<ChatItem> = (self.session.chat_ctx.items())
            .filter(|i| matches!(i, ChatItem::FunctionCallOutput(o) if waits(o)))
            .cloned()
            .collect();
        let approval_answers = lk_agents_core::request::answer_pending_input(&mut waiting, text);
        self.replies_answered(&approval_answers, out);
        self.activity.tools.approval_answers = approval_answers;
        for item in waiting {
            self.replace_item(item, out);
        }
    }

    /// The tool that claimed the person's turn, once (`_consume_next_turn`); not yet while the
    /// question it waits behind is still being asked.
    pub(crate) fn consume_claim(&mut self) -> Option<String> {
        let claim = self.activity.tools.claim.as_ref()?;
        let asking = claim.after.is_some_and(|s| {
            self.activity.speech_scheduler.contains(s) && !self.activity.finished.contains(&s)
        });
        if asking {
            return None;
        }
        self.activity.tools.claim.take().map(|c| c.tool)
    }

    /// `say(text)` from inside the driver: the realtime model's when it says and there is no
    /// TTS, else a pipeline say. `None` when nothing was scheduled (no voice to speak it).
    pub(crate) fn say_text(&mut self, now: Ts, text: String, out: &mut Out) -> Option<SpeechId> {
        let realtime = self.realtime_says();
        if self.session.tts.is_none() && self.session.outputs.plays_audio() && !realtime {
            out.trace(crate::trace::SpeechStep::SayRefused {
                why: crate::agent_activity::speech::SAY_WITHOUT_TTS_REFUSED,
            });
            return None;
        }
        if realtime {
            let request = Request::Say { text, add_to_chat_ctx: true };
            return self.realtime_request(now, request, None, None, out);
        }
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        let spec = SpeechSpec::new(SpeechKind::Say, ReplySource::Text(text), interruptible);
        Some(self.start_speech(now, spec, out).id())
    }

    /// `generate_reply(instructions=, tool_choice="none")` from inside the driver, on `chat_ctx`
    /// when given; a realtime model answers on its own socket.
    pub(crate) fn reply_with_instructions(
        &mut self,
        now: Ts,
        text: String,
        chat_ctx: Option<lk_agents_abi::ChatContext>,
        out: &mut Out,
    ) -> Option<SpeechId> {
        if self.activity.rt_session.is_some() {
            let tool_choice = Some(lk_agents_abi::ToolChoice::Forbidden);
            return self.realtime_reply(now, None, Some(text), None, tool_choice, None, false, out);
        }
        let handle = self.start_speech(
            now,
            SpeechSpec {
                turn_instructions: Some(text),
                tool_choice: Some(lk_agents_abi::ToolChoice::Forbidden),
                chat_ctx,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled(),
                )
            },
            out,
        );
        Some(handle.id())
    }

    // ------------------------------------------------------------------ the deferred reply

    /// One reply for every pending update, at the first idle instant with no current speech. Runs
    /// before idle waiters are answered, so a `foreground()` waits behind it.
    pub(crate) fn try_deliver_tool_reply(&mut self, now: Ts, out: &mut Out) {
        let tools = &self.activity.tools;
        if tools.pending.is_empty() || tools.reply.is_some() {
            return;
        }
        if self.session.closing_past_edge() || self.session.idle.held() || !self.is_idle() {
            return;
        }
        let batch = Batch(std::mem::take(&mut self.activity.tools.pending));
        // "if the update is still the tail, the agent hasn't spoken since".
        let last_pending = batch.last_item_id();
        let last_record = self.session.chat_ctx.last_agent_item_id().map(str::to_owned);
        let at_tail = last_pending.is_some() && last_pending == last_record;
        // The instructions are the templates of the latest update's call.
        let Some(mut hosted) = batch.0.last().map(|p| p.hosted.clone()) else { return };
        let render = self.activity.tools.mint_operation();
        hosted.invocation.operation = render;
        let call_ids = batch.update_ids();
        let template = if at_tail {
            ToolTemplate::ReplyAtTail { call_ids }
        } else {
            ToolTemplate::ReplyMaybeCovered { call_ids }
        };
        self.activity.tools.reply = Some(ToolReply::Rendering { render, batch, at_tail });
        let rendering = Rendering::ToolReply { render };
        out.machine().push(Effect::RenderTemplate { rendering, template, hosted });
    }

    /// The deferred reply's instructions are rendered: it is scheduled, unless the floor stopped
    /// being free meanwhile, when its updates wait for the next idle instant.
    pub(crate) fn tool_reply_rendered(
        &mut self,
        now: Ts,
        render: u64,
        text: Option<String>,
        out: &mut Out,
    ) {
        let rendering =
            |r: &mut ToolReply| matches!(r, ToolReply::Rendering { render: n, .. } if *n == render);
        let Some(ToolReply::Rendering { batch, at_tail, .. }) =
            self.activity.tools.reply.take_if(rendering)
        else {
            return;
        };
        if self.session.closing_past_edge() || self.session.idle.held() || !self.is_idle() {
            self.activity.tools.pending.splice(0..0, batch.0);
            return;
        }
        let text = text.unwrap_or_else(|| default_reply_instructions(&batch.update_ids(), at_tail));
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
        // A previous agent's updates are re-inserted for this request only.
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
        // The summary must not start another round of tool calls.
        let instructions = text.clone();
        let Some(speech) = self.reply_with_instructions(now, text, chat_ctx, out) else { return };
        self.tool_reply_scheduled(speech, update_ids, at_tail, instructions, out);
    }

    fn tool_reply_scheduled(
        &mut self,
        speech: SpeechId,
        update_ids: Vec<String>,
        at_tail: bool,
        instructions: String,
        out: &mut Out,
    ) {
        out.trace(TraceToolStep::ReplyScheduled {
            speech,
            update_ids: update_ids.clone(),
            at_tail,
            instructions,
        });
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Reply {
            update_ids: update_ids.clone(),
            status: ToolReplyStatus::Scheduled,
            speech,
        }));
        self.activity.tools.reply = Some(ToolReply::Speaking { speech, update_ids });
    }

    /// A speech ended: if it was the deferred reply, its fate is reported and the next batch may go.
    pub(crate) fn tool_reply_done(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let speaking =
            |r: &mut ToolReply| matches!(r, ToolReply::Speaking { speech: s, .. } if *s == speech);
        let Some(ToolReply::Speaking { update_ids, .. }) =
            self.activity.tools.reply.take_if(speaking)
        else {
            return;
        };
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
        let fresh = !self.activity.tools.fillers.contains_key(&id)
            && self.activity.speech_scheduler.contains(speech);
        let sched = fresh.then(|| {
            FillerScheduler::new(delay.as_secs_f64(), interval.map(|d| d.as_secs_f64()), max_steps)
        });
        let Some(Ok(sched)) = sched else {
            out.trace(GenerationStep::FillerStopped { filler: id, cause: FillerEnd::Refused });
            return;
        };
        out.trace(GenerationStep::FillerStarted { filler: id, speech });
        let run =
            FillerRun { sched, call_id, speech, source, asking: None, armed: None, fired_at: None };
        self.activity.tools.fillers.insert(id, run);
        self.pump_fillers(now, out);
    }

    pub(crate) fn end_filler(&mut self, id: FillerId, cause: FillerEnd, out: &mut Out) {
        let Some(run) = self.activity.tools.fillers.remove(&id) else { return };
        if run.armed.is_some() {
            out.machine().cancel(Timer::Filler(id));
        }
        if let Some(hook) = run.asking
            && let Some(kind) = self.activity.hooks.finish(hook)
        {
            self.disarm(Timer::Hook(hook), out);
            out.trace(HandoffStep::HookFinished { hook, kind, how: HookEnd::Abandoned });
        }
        out.trace(GenerationStep::FillerStopped { filler: id, cause });
    }

    /// A speech ended or was interrupted: its tools' fillers go with it.
    pub(crate) fn fillers_speech_ended(&mut self, speech: SpeechId, out: &mut Out) {
        for id in self.activity.tools.fillers_of(|r| r.speech == speech) {
            self.end_filler(id, FillerEnd::SpeechEnded, out);
        }
    }

    /// `ctx.update()` on `call_id`: a dwell in flight restarts (`reset_dwell`).
    fn fillers_reset_dwell(&mut self, now: Ts, call_id: &str, out: &mut Out) {
        for id in self.activity.tools.fillers_of(|r| r.call_id == call_id) {
            if let Some(run) = self.activity.tools.fillers.get_mut(&id) {
                let action = run.sched.on_busy(now);
                self.perform_filler(now, id, action, out);
            }
        }
    }

    /// The dwell waits for idle and restarts when the agent or user starts speaking (the
    /// reference's `wait_for_idle()` then race). Once per wake.
    pub(crate) fn pump_fillers(&mut self, now: Ts, out: &mut Out) {
        if self.activity.tools.fillers.is_empty() {
            return;
        }
        let idle = self.is_idle();
        let busy = self.session.states.someone_speaking();
        for id in self.activity.tools.fillers_of(|_| true) {
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
        answered: Option<(Option<String>, bool)>,
        out: &mut Out,
    ) {
        let Some((id, step)) = self.activity.tools.filler_answered(hook) else { return };
        // The reference's loop dies on the callable's exception, which ends the filler.
        let Some((text, spoke)) = answered else {
            self.end_filler(id, FillerEnd::Refused, out);
            return;
        };
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
        let realtime = self.realtime_says();
        // The reference's `say` raises with no voice to speak the text (no TTS, no realtime model
        // that says) while audio plays, which ends the filler.
        if text.is_some()
            && self.session.tts.is_none()
            && self.session.outputs.plays_audio()
            && !realtime
        {
            out.machine().emit(SessionEvent::Diagnostic(
                DiagnosticEvent::error(FILLER_STOPPED)
                    .with("error", crate::agent_activity::speech::SAY_WITHOUT_TTS_REFUSED),
            ));
            self.end_filler(id, FillerEnd::Refused, out);
            return;
        }
        let (speech, interrupted) = match text {
            // Spoken by the model when it says and there is no TTS, as the host's `say()`.
            Some(text) if realtime => {
                let request = Request::Say { text, add_to_chat_ctx: true };
                (self.realtime_request(now, request, None, None, out), false)
            }
            Some(text) => {
                let handle = self.start_speech(
                    now,
                    SpeechSpec::new(
                        SpeechKind::Say,
                        ReplySource::Text(text),
                        self.session.config.effective.turn.interruption.enabled(),
                    ),
                    out,
                );
                (Some(handle.id()), handle.interrupted())
            }
            None => (None, false),
        };
        let produced = spoke || speech.is_some();
        out.trace(GenerationStep::FillerFired { filler: id, step, spoke: produced, speech });
        // A say refused while scheduling drains is interrupted at birth and leaves the session
        // idle, so the next fire would come in the same instant. The reference's `say` raises
        // there, which ends the filler.
        if interrupted {
            self.end_filler(id, FillerEnd::Refused, out);
            return;
        }
        let Some(run) = self.activity.tools.fillers.get_mut(&id) else { return };
        let action = run.sched.fired(now, produced);
        self.perform_filler(now, id, action, out);
    }

    /// Perform one machine action: a wait is a timer, a fire a say or hook, done the end.
    fn perform_filler(&mut self, now: Ts, id: FillerId, action: FillerAction, out: &mut Out) {
        if !self.activity.tools.fillers.contains_key(&id) {
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
                    out.machine().cancel(Timer::Filler(id));
                }
                if let Some(at) = timer.arm {
                    out.machine().arm(Timer::Filler(id), at);
                }
            }
            FillerAction::Fire { step } => {
                // A second fire in the instant of the last: its say finished at birth (no voice
                // to speak it), left the session idle at once, and `interval=0` with no dwell
                // came round again with no time passing, forever. The reference's loop spins
                // there; this ends the filler, as it ends one whose say is refused.
                let Some(run) = self.activity.tools.fillers.get_mut(&id) else { return };
                if run.fired_at.replace(now) == Some(now) {
                    self.end_filler(id, FillerEnd::Refused, out);
                    return;
                }
                match self.activity.tools.fillers.get(&id).map(|r| r.source.clone()) {
                    Some(FillerSource::Text(text)) => {
                        self.fire_filler(now, id, step, Some(text), false, out)
                    }
                    Some(FillerSource::Host(source)) => {
                        let hook = self.activity.hooks.mint();
                        let wait = self.hook_wait();
                        let call = HookCall::FillerSource { id: hook, filler: id, step, source };
                        // A callable no host can run never produces: like one that raises, it ends
                        // the filler. Taken as a decline, the next fire could come in this instant.
                        if self.issue_hook(now, call, Some(wait), out) {
                            if let Some(run) = self.activity.tools.fillers.get_mut(&id) {
                                run.asking = Some(hook);
                            }
                        } else {
                            self.end_filler(id, FillerEnd::Refused, out);
                        }
                    }
                    None => {}
                }
            }
            FillerAction::Done => self.end_filler(id, FillerEnd::Done, out),
        }
    }
}
