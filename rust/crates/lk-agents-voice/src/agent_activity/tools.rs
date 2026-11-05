//! The driver's tool side: the running set, background updates (`ctx.update()`), the deferred
//! tool reply, and the fillers (`RunContext.with_filler`).

use crate::agent::{Source, Visibility};
use crate::agent_activity::realtime::{RealtimeReplyRole, Request};
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::closing::SESSION_CLOSING;
use crate::agent_session::effects::{Timer, VoiceEffect};
use crate::agent_session::{Answer, RequestId, VoiceCommandOrigin};
use crate::agent_session::{FillerSource, HookCall, HookId, ToolUpdate, VoiceDriver};
use crate::events::{DiagnosticEvent, SessionEvent, ToolEndStatus, ToolLifecycle, ToolReplyStatus};
use crate::filler_scheduler::{FillerAction, FillerId, FillerScheduler};
use crate::generation::ReplySource;
use crate::tool_executor::{
    Batch, CallId, Deferred, Delivery, FillerRun, HostedTool, MadeCall, Pending, Rendering,
    Running, ToolEnd, ToolExecution, ToolGo, ToolOutcome, ToolReply, answer,
    default_reply_instructions, pair_for, tool_ended,
};
use crate::trace::{FillerEnd, GenerationStep, ToolStep as TraceToolStep};
use lk_agents_abi::agent::{ArgumentIdentity, ReadBackAnswer, ToolTemplate};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{ChatItem, FunctionCall, Ts};
use lk_agents_core::agent_run::AgentRunEvent;
use lk_agents_core::tool_calls::{Admission, admit};
use std::time::Duration;

/// `_FillerScheduler`'s log when its `say` raised: the filler ends there.
const FILLER_STOPPED: &str =
    "filler stopped on an error, no further filler will play for this tool call";

impl VoiceDriver {
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
    ) {
        let MadeCall { mut call, spec, forbidden, answering, agent } = made;
        let answering = answering.as_ref().map(|a| (a.tool.as_str(), a.said.as_str()));
        let admission = admit(&mut call, spec.as_slice(), forbidden, answering);
        let advertised = admission.offered();
        self.record_trace(TraceToolStep::Calls { speech, names: vec![call.name.clone()] });
        let Some(state) = self.activity.speech_scheduler.speech_mut(speech) else {
            return self.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        };
        if let Admission::Forbidden { .. } = admission {
            if advertised {
                state.item_added(&ChatItem::FunctionCall(call.clone()));
            }
            if state.scheduled() {
                self.on_tool_started(now, (speech, step), call.clone(), None, None);
            }
            self.activity.tools.dropped.entry((speech, step)).or_default().push(call.name);
            return self.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        }
        let agent = agent.filter(|_| admission == Admission::Offered);
        let call = Deferred { speech, step, id: None, call, advertised, agent, go };
        self.activity.tools.deferred.push(call);
        self.settle_deferred_calls();
    }

    /// Starts each waiting call whose step is authorized, in the model's order, once its run
    /// decided its first move, and drops unrun those whose speech was interrupted or done before
    /// the authorization. `true` when one was.
    pub(crate) fn settle_deferred_calls(&mut self) -> bool {
        let mut settled = false;
        let mut waiting = Vec::new();
        let now = self.step.now;
        for d in std::mem::take(&mut self.activity.tools.deferred) {
            // Behind an earlier call of the same speech still waiting: started in order.
            if waiting.iter().any(|w: &Deferred| w.speech == d.speech) {
                waiting.push(d);
                continue;
            }
            match self.try_start_deferred(now, d) {
                Some(d) => waiting.push(d),
                None => settled = true,
            }
        }
        self.activity.tools.deferred = waiting;
        settled
    }

    /// One waiting call, started or dropped unrun; returned when it keeps waiting.
    ///
    /// Authorization last: a discarded preemptive generation, or a speech interrupted while
    /// queued, drops the call unrun. A start already authorized runs, interrupted or not, as the
    /// reference's executor starts the body right after `tool_execution_started_cb`.
    fn try_start_deferred(&mut self, now: Ts, d: Deferred) -> Option<Deferred> {
        let state = self.activity.speech_scheduler.speech(d.speech);
        match state.map(|s| (s.step_authorized(d.step), s.interrupted() || s.done())) {
            Some((true, _)) => return self.start_authorized(now, d),
            Some((false, false)) => return Some(d),
            Some(_) | None => self.answer(Some(d.go), Answer::ToolGo(ToolGo::Gone)),
        }
        None
    }

    /// An authorized call starts once its run decided its first move.
    fn start_authorized(&mut self, now: Ts, mut d: Deferred) -> Option<Deferred> {
        // `_tool_execution_started_cb`: stamped with its start (history is ordered by
        // `created_at`, and it was made before the step was authorized).
        if d.id.is_none() {
            d.call.created_at = self.step.wall();
            d.id = self.reply_call_made(d.speech, d.call.clone());
        }
        match d.id {
            Some(id) if !self.activity.reply_first_ready(d.speech, id) => return Some(d),
            Some(id) => self.start_call(now, d, id),
            None => self.answer(Some(d.go), Answer::ToolGo(ToolGo::Gone)),
        }
        None
    }

    /// The call is told to the speech and started with its first move.
    fn start_call(&mut self, now: Ts, d: Deferred, id: CallId) {
        let Deferred { speech, step, call, advertised, agent, go, .. } = d;
        if advertised && let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
            state.item_added(&ChatItem::FunctionCall(call.clone()));
        }
        let first = Box::new(self.activity.reply_call_start(speech, id));
        let host =
            self.on_tool_started(now, (speech, step), call.clone(), agent.as_ref(), Some(id));
        let start =
            ToolGo::Start { call: id, item: Box::new(call), host: host.map(Box::new), first };
        self.answer(Some(go), Answer::ToolGo(start));
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
        run_call: Option<CallId>,
    ) -> Option<HostedTool> {
        let held = self.activity.speech_scheduler.contains(speech);
        let hosted = agent.filter(|_| held).map(|agent| {
            let operation = self.activity.tools.mint_operation();
            HostedTool::of(agent, self.session.ids.session(), speech, operation, Some(&call))
        });
        self.record_trace(TraceToolStep::CallStarted {
            speech,
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            hosted: hosted.is_some(),
        });
        if let Some(turn) = self.agent_turn(speech) {
            turn.function_call(&call);
        }
        // The call is the run's once it starts, whichever run is open when it is committed: one
        // answered without running (an unknown tool) too, as its output is.
        if let Some(run) = self.session.runs.current()
            && self.run_contains_speech(speech)
        {
            let item = Box::new(lk_agents_abi::ChatItem::FunctionCall(call.clone()));
            self.publish_event(SessionEvent::RunItemAdded { run, item });
        }
        let running = hosted.clone()?;
        let closing = self.session.closing_past_edge();
        {
            let updates = self.session.states.on_tool_call_started(now, closing);
            self.apply_user_state_updates(updates);
        };
        let output_id = self.session.ids.mint("item");
        let call_id = call.call_id.as_str();
        if let Some(turn) = self.agent_turn(speech) {
            turn.tool_started(call_id);
        }
        self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started {
            call: call.clone(),
        }));
        self.activity.tools.running.push(Running {
            call,
            run_call,
            output_id,
            identity: None,
            executing: false,
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
    ) {
        if let Ok(identity) = &identity
            && let Some(running) = self
                .activity
                .tools
                .running
                .iter_mut()
                .find(|running| running.speech == speech && running.run_call == Some(call))
        {
            running.identity = Some(identity.clone());
        }
        let event = match identity {
            // The call's task holds the prepared arguments: the resource is the call's.
            Ok(identity) => {
                AgentRunEvent::ToolPrepared { call, prepared: ResourceId(call.0), identity }
            }
            Err(failure) => AgentRunEvent::ToolPrepareFailed { call, failure },
        };
        self.reply_call_asks(speech, call, go, event);
    }

    /// A call's read-back answered: the reply's run settles the call on it.
    pub(crate) fn on_read_back(
        &mut self,
        speech: SpeechId,
        call: CallId,
        result: Result<ReadBackAnswer, String>,
        go: RequestId,
    ) {
        let event = AgentRunEvent::ReadBackRendered { call, result };
        self.reply_call_asks(speech, call, go, event);
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
    ) {
        // The reply's run sees the body's answer as it arrives: a later duplicate is admitted
        // against what still runs. A call answered for a task that never did (aborted, or
        // stopped by the close) is not the run's to settle.
        match call {
            Some(call) => self.reply_event(speech, AgentRunEvent::ToolSettled { call, outcome }),
            None => self.activity.reply_call_dropped(speech),
        }
        self.record_trace(TraceToolStep::CallEnded { speech, call_id: call_id.clone(), end });
        // Ending the wrong call leaves one the close would answer twice.
        if !self.activity.tools.end(speech, call, end == ToolEnd::Detached) {
            return;
        }
        let status = match end {
            ToolEnd::Done => ToolEndStatus::Done,
            ToolEnd::Error => ToolEndStatus::Error,
            ToolEnd::Cancelled => ToolEndStatus::Cancelled,
            ToolEnd::Detached => {
                self.open_expert(&call_id);
                self.fillers_reset_dwell(now, &call_id);
                self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id: call_id.clone(),
                    call_id: call_id.clone(),
                    message: message.unwrap_or_default(),
                }));
                let early = self
                    .activity
                    .tools
                    .running
                    .iter_mut()
                    .find(|running| running.speech == speech && running.call.call_id == call_id)
                    .map(|running| {
                        (
                            running.hosted.invocation.invocation_id.clone(),
                            running.hosted.invocation.operation,
                            std::mem::take(&mut running.early),
                        )
                    });
                if let Some((invocation_id, operation, early)) = early {
                    for update in early {
                        self.on_tool_update(
                            now,
                            invocation_id.clone(),
                            operation,
                            call_id.clone(),
                            update,
                        );
                    }
                }
                return;
            }
        };
        self.publish_tool_end(now, call_id, message, status);
    }

    /// The shared final lifecycle projection, after the exact live owner has been removed.
    /// Force-stopping a reply uses it without pretending that its host body reported back.
    fn publish_tool_end(
        &mut self,
        now: Ts,
        call_id: String,
        message: Option<String>,
        status: ToolEndStatus,
    ) {
        // The last tool to land under two listening participants restarts the away window (#6883).
        let closing = self.session.closing_past_edge();
        {
            let updates = self.session.states.on_tool_call_ended(now, closing);
            self.apply_user_state_updates(updates);
        };
        self.agent_turn_tool_ended(&call_id);
        self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
            id: call_id.clone(),
            call_id,
            message,
            status,
        }));
    }

    /// Settle blocking calls whose reply task was forcefully stopped (closing or the watchdog),
    /// each end before its pair, in start order. Their bodies are cancelled too: one may have
    /// detached at the host before the driver heard its answer, and would outlive its told end.
    pub(crate) fn end_live_tools(&mut self, now: Ts, speech: SpeechId) {
        let live: Vec<Running> = self.activity.tools.live(speech).cloned().collect();
        if live.is_empty() {
            return;
        }
        let wall = self.step.wall();
        let mut pairs = Vec::with_capacity(live.len());
        for r in live {
            let (call_id, hosted) = (r.call.call_id.clone(), r.hosted.clone());
            self.record_effect(VoiceEffect::CancelTool { call_id, hosted });
            let (outcome, status) = if self.session.closing.is_some() {
                (ToolOutcome::ToolError(SESSION_CLOSING.to_owned()), ToolEndStatus::Error)
            } else {
                (ToolOutcome::Cancelled, ToolEndStatus::Cancelled)
            };
            let execution = answer(&r.call, outcome.clone().into(), wall);
            self.activity.reply_call_dropped(speech);
            self.record_trace(TraceToolStep::CallEnded {
                speech,
                call_id: r.call.call_id.clone(),
                end: execution.end,
            });
            if self
                .activity
                .tools
                .take_update_owner(
                    &r.hosted.invocation.invocation_id,
                    r.hosted.invocation.operation,
                    &r.call.call_id,
                )
                .is_some()
            {
                self.publish_tool_end(
                    now,
                    r.call.call_id.clone(),
                    (!execution.output.output.is_empty()).then(|| execution.output.output.clone()),
                    status,
                );
            }
            pairs.push((r.step, execution.call, execution.output));
        }
        for (step, call, output) in pairs {
            self.commit_item(
                ChatItem::FunctionCall(call),
                Source::ToolCall { speech, step },
                Visibility::Both,
            );
            self.commit_item(
                ChatItem::FunctionCallOutput(output),
                Source::ToolOutput { speech, step },
                Visibility::Both,
            );
        }
        self.publish_activity();
    }

    /// `_ToolExecutor.aclose()`: every body still running at the host, a detached one by now, is
    /// cancelled and its end told as its own cancellation would tell it.
    pub(crate) fn cancel_running_tools(&mut self, now: Ts) {
        let running = std::mem::take(&mut self.activity.tools.running);
        if running.is_empty() {
            return;
        }
        for r in running {
            let call_id = r.call.call_id.clone();
            let id = if r.detached() { format!("{call_id}_final") } else { call_id.clone() };
            self.record_effect(VoiceEffect::CancelTool {
                call_id: call_id.clone(),
                hosted: r.hosted,
            });
            let closing = self.session.closing_past_edge();
            {
                let updates = self.session.states.on_tool_call_ended(now, closing);
                self.apply_user_state_updates(updates);
            };
            self.publish_event(tool_ended(id, call_id, None, ToolEndStatus::Cancelled));
        }
        self.publish_activity();
    }

    // ------------------------------------------------------------------ background updates

    pub(crate) fn on_tool_update(
        &mut self,
        now: Ts,
        invocation_id: String,
        operation: u64,
        call_id: String,
        update: ToolUpdate,
    ) {
        // Runtime delivery may be queued while this call settles and its ID is reused.
        // Resolve the authoritative owner once; no call-ID-only fallback is allowed.
        let Some(running) = self.activity.tools.update_owner(&invocation_id, operation, &call_id)
        else {
            self.record_trace(TraceToolStep::UpdateRefused { call_id });
            return;
        };
        if !running.detached() {
            running.early.push(update);
            return;
        }
        if self.on_expert_update(now, &invocation_id, operation, &call_id, &update) {
            return;
        }
        let wall = self.step.wall();
        // `update()` restarts a pending filler dwell so it can't race the update (`reset_dwell`).
        self.fillers_reset_dwell(now, &call_id);
        match update {
            ToolUpdate::Progress { message, raw } => {
                let r = self
                    .activity
                    .tools
                    .update_owner(&invocation_id, operation, &call_id)
                    .expect("exact owner checked above");
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
                self.record_trace(TraceToolStep::Updated {
                    call_id: call_id.clone(),
                    id: id.clone(),
                    is_final: false,
                });
                self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
                    id: id.clone(),
                    call_id,
                    message: raw,
                }));
                self.commit_pair(&running, id, pair);
            }
            ToolUpdate::Final { report } => {
                let id = format!("{call_id}_final");
                let running = self
                    .activity
                    .tools
                    .take_update_owner(&invocation_id, operation, &call_id)
                    .expect("exact owner checked above");
                // The end is journalled before the away window moves (#6883).
                self.record_trace(TraceToolStep::Updated {
                    call_id: call_id.clone(),
                    id: id.clone(),
                    is_final: true,
                });
                let closing = self.session.closing_past_edge();
                {
                    let updates = self.session.states.on_tool_call_ended(now, closing);
                    self.apply_user_state_updates(updates);
                };
                // A silent return voices nothing (`_execute_tool`'s `None` and `StopResponse`).
                match report.outcome {
                    ToolOutcome::Cancelled => {
                        self.publish_event(tool_ended(id, call_id, None, ToolEndStatus::Cancelled));
                        return self.try_deliver_tool_reply();
                    }
                    ToolOutcome::Nothing | ToolOutcome::StopResponse => {
                        self.publish_event(tool_ended(id, call_id, None, ToolEndStatus::Done));
                        return self.try_deliver_tool_reply();
                    }
                    _ => {}
                }
                let pair = pair_for(&running, &id, report, wall, || self.session.ids.mint("item"));
                let status = match pair.end {
                    ToolEnd::Error => ToolEndStatus::Error,
                    ToolEnd::Done | ToolEnd::Cancelled | ToolEnd::Detached => ToolEndStatus::Done,
                };
                self.publish_event(tool_ended(
                    id.clone(),
                    call_id,
                    Some(pair.output.output.clone()),
                    status,
                ));
                self.commit_pair(&running, id, pair);
            }
        }
        self.try_deliver_tool_reply();
    }

    /// Commit the pair to the record at once, so an earlier reply still sees it, deliver what it
    /// owes the person, and batch it for the deferred reply when it asks for one
    /// (`_enqueue_reply`).
    fn commit_pair(&mut self, running: &Running, id: String, pair: ToolExecution) {
        let ToolExecution { call, output, delivery, .. } = pair;
        let reply_required = output.reply_required;
        let source = Source::ToolUpdate { speech: running.speech, step: running.step };
        let items = vec![ChatItem::FunctionCall(call), ChatItem::FunctionCallOutput(output)];
        for item in &items {
            self.commit_item(item.clone(), source.clone(), Visibility::Both);
        }
        self.publish_activity();
        if let Some(delivery) = delivery {
            self.deliver_outcome(self.step.now, running.speech, delivery);
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
    pub(crate) fn deliver_outcome(&mut self, now: Ts, speech: SpeechId, delivery: Delivery) {
        match delivery {
            Delivery::Say(text) => {
                self.say_text(now, text, self.run_contains_speech(speech));
            }
            Delivery::Complete(value) => self.end_voice_task(now, speech, Ok(value)),
            Delivery::Fail(reason) => self.end_voice_task(now, speech, Err(reason)),
            Delivery::Ask(ask) => {
                let output = crate::run_result::RunOutput::Ask {
                    question: ask.question.clone(),
                    verbatim: ask.verbatim,
                    answer_tool: ask.tool.clone(),
                };
                self.session.runs.outputs.insert(speech, output.clone());
                let asking = if ask.verbatim {
                    self.say_text(now, ask.question.clone(), self.run_contains_speech(speech))
                } else {
                    let instructions = format!(
                        "{}{}",
                        lk_agents_abi::llm::tool_executor::ASK_IN_OWN_WORDS,
                        ask.question
                    );
                    self.reply_with_instructions(
                        now,
                        instructions,
                        None,
                        self.run_contains_speech(speech),
                    )
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

    /// Resolve the compatibility-selected question's call. Other correlated requirements remain
    /// waiting; retaining them is not permission to answer them with this person's message.
    pub(crate) fn answer_pending_input(&mut self, text: &str) {
        let mut history: Vec<ChatItem> = self.session.chat_ctx.items().cloned().collect();
        let approval_answers = lk_agents_core::request::answer_pending_input(&mut history, text);
        history.retain(|item| matches!(item, ChatItem::FunctionCallOutput(output) if approval_answers.contains(&output.call_id)));
        self.replies_answered(&approval_answers);
        self.activity.tools.approval_answers = approval_answers;
        self.replace_items(history);
    }

    /// `say(text)` from inside the driver: the realtime model's when it says and there is no
    /// TTS, else a pipeline say. `None` when nothing was scheduled (no voice to speak it).
    pub(crate) fn say_text(
        &mut self,
        now: Ts,
        text: String,
        participates_in_voice_run: bool,
    ) -> Option<SpeechId> {
        let realtime = self.realtime_says();
        if self.session.tts.is_none() && self.session.outputs.plays_audio() && !realtime {
            self.record_trace(crate::trace::SpeechStep::SayRefused {
                why: crate::agent_activity::speech::SAY_WITHOUT_TTS_REFUSED,
            });
            return None;
        }
        if realtime {
            let request = Request::Say { text, add_to_chat_ctx: true };
            return self.realtime_request(now, request, None, None, participates_in_voice_run);
        }
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        let spec = SpeechSpec {
            participates_in_voice_run,
            ..SpeechSpec::new(SpeechKind::Say, ReplySource::Text(text), interruptible)
        };
        Some(self.start_speech(now, spec))
    }

    /// `generate_reply(instructions=, tool_choice="none")` from inside the driver, on `chat_ctx`
    /// when given; a realtime model answers on its own socket.
    pub(crate) fn reply_with_instructions(
        &mut self,
        now: Ts,
        text: String,
        chat_ctx: Option<lk_agents_abi::ChatContext>,
        participates_in_voice_run: bool,
    ) -> Option<SpeechId> {
        if self.activity.rt_session.is_some() {
            let tool_choice = Some(lk_agents_abi::ToolChoice::Forbidden);
            let role = if participates_in_voice_run {
                RealtimeReplyRole::VoiceRun
            } else {
                RealtimeReplyRole::Internal
            };
            return self.realtime_reply(now, None, Some(text), None, tool_choice, None, role);
        }
        let speech = self.start_speech(
            now,
            SpeechSpec {
                participates_in_voice_run,
                turn_instructions: Some(text),
                tool_choice: Some(lk_agents_abi::ToolChoice::Forbidden),
                chat_ctx,
                ..SpeechSpec::new(
                    SpeechKind::GenerateReply,
                    ReplySource::Llm,
                    self.session.config.effective.turn.interruption.enabled(),
                )
            },
        );
        Some(speech)
    }

    // ------------------------------------------------------------------ the deferred reply

    /// One reply for every pending update, at the first idle instant with no current speech. Runs
    /// before idle waiters are answered, so a `foreground()` waits behind it.
    pub(crate) fn try_deliver_tool_reply(&mut self) {
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
        let turn = self.speech_turn(hosted.speech);
        self.record_effect(VoiceEffect::RenderToolHandlingTemplate {
            rendering,
            template,
            hosted,
            turn,
        });
    }

    /// The deferred reply's instructions are rendered: it is scheduled, unless the floor stopped
    /// being free meanwhile, when its updates wait for the next idle instant.
    pub(crate) fn tool_reply_rendered(&mut self, now: Ts, render: u64, text: Option<String>) {
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
        self.schedule_tool_reply(now, batch, at_tail, text);
    }

    fn schedule_tool_reply(&mut self, now: Ts, batch: Batch, at_tail: bool, text: String) {
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
        let Some(speech) = self.reply_with_instructions(now, text, chat_ctx, false) else { return };
        self.tool_reply_scheduled(speech, update_ids, at_tail, instructions);
    }

    /// A speech ended: if it was the deferred reply, its fate is reported and the next batch may go.
    pub(crate) fn tool_reply_done(&mut self, speech: SpeechId) {
        let speaking =
            |r: &mut ToolReply| matches!(r, ToolReply::Speaking { speech: s, .. } if *s == speech);
        let Some(ToolReply::Speaking { update_ids, .. }) =
            self.activity.tools.reply.take_if(speaking)
        else {
            return;
        };
        let status = match self.activity.speech_scheduler.speech(speech) {
            Some(s) if s.interrupted() => ToolReplyStatus::Interrupted,
            // The instruction permits an empty response when the agent already covered the result.
            Some(s) if s.item_ids().is_empty() => ToolReplyStatus::Skipped,
            Some(_) | None => ToolReplyStatus::Completed,
        };
        self.record_trace(TraceToolStep::ReplyDone { speech, status });
        self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Reply {
            update_ids,
            status,
            speech,
        }));
        self.try_deliver_tool_reply();
    }

    /// `SpeechCommand::FillerStart`: the scope opened. The dwell begins at the first idle instant.
    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the operation’s existing explicit arguments"
    )]
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
    ) {
        let fresh = !self.activity.tools.fillers.contains_key(&id)
            && self.activity.speech_scheduler.contains(speech);
        let sched = fresh.then(|| {
            FillerScheduler::new(delay.as_secs_f64(), interval.map(|d| d.as_secs_f64()), max_steps)
        });
        let Some(Ok(sched)) = sched else {
            self.record_trace(GenerationStep::FillerStopped {
                filler: id,
                cause: FillerEnd::Refused,
            });
            return;
        };
        self.record_trace(GenerationStep::FillerStarted { filler: id, speech });
        let run =
            FillerRun { sched, call_id, speech, source, asking: None, armed: None, fired_at: None };
        self.activity.tools.fillers.insert(id, run);
        self.pump_fillers(now);
    }

    pub(crate) fn end_filler(&mut self, id: FillerId, cause: FillerEnd) {
        let Some(run) = self.activity.tools.fillers.remove(&id) else { return };
        if run.armed.is_some() {
            self.clear_deadline(Timer::Filler(id));
        }
        if let Some(hook) = run.asking {
            self.abandon_hook(hook);
        }
        self.record_trace(GenerationStep::FillerStopped { filler: id, cause });
    }

    /// A speech ended or was interrupted: its tools' fillers go with it.
    pub(crate) fn fillers_speech_ended(&mut self, speech: SpeechId) {
        for id in self.activity.tools.fillers_of(|r| r.speech == speech) {
            self.end_filler(id, FillerEnd::SpeechEnded);
        }
    }

    /// `ctx.update()` on `call_id`: a dwell in flight restarts (`reset_dwell`).
    fn fillers_reset_dwell(&mut self, now: Ts, call_id: &str) {
        for id in self.activity.tools.fillers_of(|r| r.call_id == call_id) {
            if let Some(run) = self.activity.tools.fillers.get_mut(&id) {
                let action = run.sched.on_busy(now);
                self.perform_filler(now, id, action);
            }
        }
    }

    /// The dwell waits for idle and restarts when the agent or user starts speaking (the
    /// reference's `wait_for_idle()` then race). Once per wake.
    pub(crate) fn pump_fillers(&mut self, now: Ts) {
        if self.activity.tools.fillers.is_empty() {
            return;
        }
        let idle = self.is_idle();
        let busy = self.session.states.someone_speaking();
        for id in self.activity.tools.fillers_of(|_| true) {
            if let Some(action) = self.activity.tools.filler_wake(id, idle, busy, now) {
                self.perform_filler(now, id, action);
            }
        }
    }

    /// The dwell or cooldown is over. A deadline in the same wake as a speaking edge counts as busy.
    pub(crate) fn on_filler_timer(&mut self, now: Ts, id: FillerId) {
        let busy = self.session.states.someone_speaking();
        let Some(action) = self.activity.tools.filler_timer(id, busy, now) else { return };
        self.perform_filler(now, id, action);
    }

    pub(crate) fn on_filler_source(
        &mut self,
        now: Ts,
        hook: HookId,
        answered: Option<(Option<String>, bool)>,
    ) {
        let Some((id, step)) = self.activity.tools.filler_answered(hook) else { return };
        // The reference's loop dies on the callable's exception, which ends the filler.
        let Some((text, spoke)) = answered else {
            self.end_filler(id, FillerEnd::Refused);
            return;
        };
        self.fire_filler(now, id, step, text, spoke);
    }

    fn fire_filler(&mut self, now: Ts, id: FillerId, step: u32, text: Option<String>, spoke: bool) {
        let realtime = self.realtime_says();
        // The reference's `say` raises with no voice to speak the text (no TTS, no realtime model
        // that says) while audio plays, which ends the filler.
        if text.is_some()
            && self.session.tts.is_none()
            && self.session.outputs.plays_audio()
            && !realtime
        {
            self.publish_event(SessionEvent::Diagnostic(
                DiagnosticEvent::error(FILLER_STOPPED)
                    .with("error", crate::agent_activity::speech::SAY_WITHOUT_TTS_REFUSED),
            ));
            self.end_filler(id, FillerEnd::Refused);
            return;
        }
        let participates_in_voice_run = self
            .activity
            .tools
            .fillers
            .get(&id)
            .is_some_and(|filler| self.run_contains_speech(filler.speech));
        let (speech, interrupted) = match text {
            // Spoken by the model when it says and there is no TTS, as the host's `say()`.
            Some(text) if realtime => {
                let request = Request::Say { text, add_to_chat_ctx: true };
                (self.realtime_request(now, request, None, None, participates_in_voice_run), false)
            }
            Some(text) => {
                let speech = self.start_speech(
                    now,
                    SpeechSpec {
                        participates_in_voice_run,
                        ..SpeechSpec::new(
                            SpeechKind::Say,
                            ReplySource::Text(text),
                            self.session.config.effective.turn.interruption.enabled(),
                        )
                    },
                );
                // Released at birth (refused) reads as interrupted: it will never play.
                let state = self.activity.speech_scheduler.speech(speech);
                (Some(speech), state.is_none_or(|s| s.interrupted()))
            }
            None => (None, false),
        };
        let produced = spoke || speech.is_some();
        self.record_trace(GenerationStep::FillerFired {
            filler: id,
            step,
            spoke: produced,
            speech,
        });
        // A say refused while scheduling drains is interrupted at birth and leaves the session
        // idle, so the next fire would come in the same instant. The reference's `say` raises
        // there, which ends the filler.
        if interrupted {
            self.end_filler(id, FillerEnd::Refused);
            return;
        }
        let Some(run) = self.activity.tools.fillers.get_mut(&id) else { return };
        let action = run.sched.fired(now, produced);
        self.perform_filler(now, id, action);
    }

    /// Perform one machine action: a wait is a timer, a fire a say or hook, done the end.
    fn perform_filler(&mut self, now: Ts, id: FillerId, action: FillerAction) {
        if !self.activity.tools.fillers.contains_key(&id) {
            return;
        }
        match action {
            FillerAction::WaitForIdle => self.arm_filler_timer(id, None),
            FillerAction::Dwell(at) | FillerAction::CoolDown(at) => {
                self.arm_filler_timer(id, Some(at))
            }
            FillerAction::Fire { step } => self.fire_filler_step(now, id, step),
            FillerAction::Done => self.end_filler(id, FillerEnd::Done),
        }
    }

    /// A wait: until idle (`None`) or until `until`.
    fn arm_filler_timer(&mut self, id: FillerId, until: Option<Ts>) {
        let Some(timer) = self.activity.tools.filler_wait(id, until) else { return };
        if timer.cancel {
            self.clear_deadline(Timer::Filler(id));
        }
        if let Some(at) = timer.arm {
            self.set_deadline(Timer::Filler(id), at);
        }
    }

    fn fire_filler_step(&mut self, now: Ts, id: FillerId, step: u32) {
        // A second fire in the instant of the last: its say finished at birth (no voice
        // to speak it), left the session idle at once, and `interval=0` with no dwell
        // came round again with no time passing, forever. The reference's loop spins
        // there; this ends the filler, as it ends one whose say is refused.
        let Some(run) = self.activity.tools.fillers.get_mut(&id) else { return };
        if run.fired_at.replace(now) == Some(now) {
            self.end_filler(id, FillerEnd::Refused);
            return;
        }
        // A fixed text: nothing was spoken by a host callable.
        let spoke = false;
        match self.activity.tools.fillers.get(&id).map(|r| r.source.clone()) {
            Some(FillerSource::Text(text)) => self.fire_filler(now, id, step, Some(text), spoke),
            Some(FillerSource::Host(source)) => self.ask_filler_source(now, id, step, source),
            None => {}
        }
    }

    /// The host's callable gives the fire's text, through the `filler_source` hook.
    fn ask_filler_source(
        &mut self,
        now: Ts,
        id: FillerId,
        step: u32,
        source: Option<lk_agents_abi::HostRef>,
    ) {
        let hook = self.activity.hooks.mint();
        let wait = self.activity.hook_wait();
        let call = HookCall::FillerSource { id: hook, filler: id, step, source };
        // A callable no host can run never produces: like one that raises, it ends
        // the filler. Taken as a decline, the next fire could come in this instant.
        if let Some(run) = self.activity.tools.fillers.get_mut(&id) {
            run.asking = Some(hook);
        }
        self.issue_hook(now, call, Some(wait));
    }
}

impl super::AgentActivity {
    /// The tool step is on the record; it no longer holds the activity in `thinking`.
    pub(super) fn tool_step_committed(&mut self, speech: SpeechId) {
        self.speech_history.commit_tools(speech);
        self.set_tools_in_flight(speech, false);
    }

    /// The tool that claimed the person's turn, once (`_consume_next_turn`); not yet while the
    /// question it waits behind is still being asked.
    pub(crate) fn consume_claim(&mut self) -> Option<String> {
        let claim = self.tools.claim.as_ref()?;
        let asking = claim
            .after
            .is_some_and(|s| self.speech_scheduler.contains(s) && !self.speech_history.finished(s));
        if asking {
            return None;
        }
        self.tools.claim.take().map(|c| c.tool)
    }
}

impl VoiceDriver {
    /// Another live speech is in its tools. One awaiting a VoiceTask waits on the person, through
    /// the task: the agent listens, not thinks.
    pub(crate) fn tools_running_elsewhere(&self, speech: SpeechId) -> bool {
        let scheduling = &self.activity.speech_scheduler;
        let tasks = &self.activity.voice_tasks;
        self.activity
            .tools_in_flight
            .iter()
            .copied()
            .any(|id| id != speech && scheduling.contains(id) && !tasks.awaits(id))
    }

    /// The speech whose tool sent the command (`VoiceCommandOrigin::Tool`).
    pub(crate) fn tool_speech(&self, origin: &VoiceCommandOrigin) -> Option<SpeechId> {
        match origin {
            VoiceCommandOrigin::Tool { invocation_id, operation, call_id } => {
                self.activity.tools.speech_of(invocation_id, *operation, call_id)
            }
            VoiceCommandOrigin::Host
            | VoiceCommandOrigin::Hook
            | VoiceCommandOrigin::Amd
            | VoiceCommandOrigin::Ivr => None,
        }
    }
}

impl VoiceDriver {
    fn tool_reply_scheduled(
        &mut self,
        speech: SpeechId,
        update_ids: Vec<String>,
        at_tail: bool,
        instructions: String,
    ) {
        self.record_trace(TraceToolStep::ReplyScheduled {
            speech,
            update_ids: update_ids.clone(),
            at_tail,
            instructions,
        });
        self.publish_event(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Reply {
            update_ids: update_ids.clone(),
            status: ToolReplyStatus::Scheduled,
            speech,
        }));
        self.activity.tools.reply = Some(ToolReply::Speaking { speech, update_ids });
    }
}

#[cfg(test)]
#[path = "tool_update_tests.rs"]
pub(super) mod tool_update_tests;
