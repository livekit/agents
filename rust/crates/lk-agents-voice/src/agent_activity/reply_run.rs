//! One model reply's `AgentRun`: a coordinated invocation whose steps the reply task streams and
//! whose calls run as the floor lets them. The run decides each step (call admission, approval,
//! retries, output acceptance, when to ask or stop); the floor decides when calls start and what
//! an interruption, `max_tool_steps` or a `function_tools_executed` handler leave of a step.

use crate::agent::Source;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Effect, Out};
use crate::agent_session::{Answer, Driver, RequestId};
use crate::events::SessionEvent;
use crate::generation::{PlanSpec, PlannedReply};
use crate::run_result::RunOutput;
use crate::tool_executor::{
    APPROVAL_KEY, AgentFrame, CallId, HostedTool, NextReply, Rendering, StepEnd, StepNext, ToolGo,
    Toolset,
};
use crate::trace::ToolStep as TraceToolStep;
use lk_agents_abi::agent::run::{Output, RunError, RunOutcome, RunResult, ValidationId};
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, Retries};
use lk_agents_abi::{
    ChatItem, ChoiceDelta, FunctionCall, HostRef, LlmChunk, MetricsReport, Modality, Role,
    ToolChoice, ToolSpec,
};
use lk_agents_core::agent_run::{
    AgentRun, AgentRunContinuation, Effect as RunEffect, Event, Opening, OutputVerdict, RunInput,
    StepId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A model reply's run, and what the floor and the reply task wait on from it.
pub struct ReplyRun {
    run: AgentRun,
    /// The agent hosting the reply's callbacks.
    agent: Option<Arc<dyn AgentFrame>>,
    /// What the reply's next steps are planned with.
    plan: PlanSpec,
    /// A realtime reply: its server plans each step.
    realtime: bool,
    /// The model request of the step in flight, until its words are in.
    request: Option<lk_agents_core::agent_run::RequestId>,
    /// The step's calls whose task still awaits a move.
    calls: BTreeMap<CallId, ReplyCall>,
    /// The step made calls the run admitted.
    acting: bool,
    /// The floor dropped a call of the step unrun: the run cannot settle the step.
    cut: bool,
    /// The step the run settled, waiting for the floor's verdict.
    pub(super) settling: Option<StepId>,
    pub(super) settled_items: Option<Vec<ChatItem>>,
    pub(super) terminal_call: Option<String>,
    pub(super) tool_end: Option<(u32, StepEnd)>,
    /// The floor's verdict on the step, waiting for the run to settle it.
    verdict: Option<AgentRunContinuation>,
    reply_required: BTreeMap<String, bool>,
    /// The reply task waiting for its next step.
    waiter: Option<RequestId>,
    /// The reply's next step, waiting for the task to ask.
    next: Option<StepNext>,
    previous_user_metrics: Option<MetricsReport>,
    /// The values the agent accepted, by validation.
    accepted: BTreeMap<ValidationId, HostRef>,
}

impl ReplyRun {
    pub(super) fn call_item(&self, call: CallId) -> Option<&FunctionCall> {
        self.run.step_call(call).map(|(item, _)| item)
    }
}

impl std::fmt::Debug for ReplyRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplyRun")
            .field("request", &self.request)
            .field("settling", &self.settling)
            .field("verdict", &self.verdict)
            .finish_non_exhaustive()
    }
}

/// One call of the step, from its admission until its task heard its last move.
#[derive(Default)]
struct ReplyCall {
    /// Its task runs.
    started: bool,
    /// The move the run decided, not yet told to the task.
    next: Option<ToolGo>,
    /// The task awaiting its next move.
    waiting: Option<RequestId>,
}

/// A move after which the call's task asks nothing more.
fn last(go: &ToolGo) -> bool {
    matches!(go, ToolGo::Execute { .. } | ToolGo::Settled { .. } | ToolGo::Gone)
}

impl Driver {
    /// The run of a model reply about to start as `speech`: defined by the frame's agent over
    /// the tools `toolset` offers (every function retains its installed policy
    /// metadata) and the budgets `retries`. `task`: a `VoiceTask`'s agent holds the floor. The first step's assistant message id and
    /// the tools it offers, or `None` when the definition cannot run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_reply_run(
        &mut self,
        speech: SpeechId,
        invocation_id: String,
        toolset: &Toolset,
        retries: Retries,
        plan: &PlanSpec,
        (realtime, task): (bool, bool),
        out: &mut Out,
    ) -> Option<(String, Vec<ToolSpec>)> {
        let definition = match self.activity.reply_definition(toolset, retries, task) {
            Ok(definition) => definition,
            Err(message) => {
                out.trace(crate::trace::CoreStep::PortFailure(message));
                return None;
            }
        };
        // Approval reads every agent's calls; the person's latest answers reach the run as they
        // come (the message this reply answers is committed once the speech is scheduled).
        let history: Vec<ChatItem> = self.session.chat_ctx.items().cloned().collect();
        let approval_answers = self.activity.tools.approval_answers.clone();
        let input = RunInput {
            invocation_id,
            opening: Opening::Coordinated { approval_answers },
            history,
            answer_tool: None,
            delegated: false,
            mode: lk_agents_core::tool_outcome::Mode::Voice,
            llm: None,
            wall: self.step.anchor(),
            cleanup_timeout: lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
        };
        let mut run = match AgentRun::new(definition, input) {
            Ok(run) => run,
            // The host validated the agent: an invalid one only reaches here from a handwritten
            // frame. The reply speaks, and makes no call.
            Err(_) => {
                out.trace(crate::trace::CoreStep::PortFailure(
                    "the reply's agent definition is invalid: its calls are not run",
                ));
                return None;
            }
        };
        let mut startup = run.handle(Event::Start, self.step.now);
        // Extract only the initial plan needed by speech construction. Every remaining effect
        // still goes through the ordinary owner after the reply record is installed.
        let first = startup
            .iter()
            .position(|effect| matches!(effect, RunEffect::PlanRequest { .. }))
            .map(|at| startup.remove(at));
        let (request, message_id, tools) = match first {
            Some(RunEffect::PlanRequest { request, message_id, tools, .. }) => {
                (Some(request), Some(message_id), tools)
            }
            None => (None, None, Vec::new()),
            Some(_) => unreachable!("selected a planning effect"),
        };
        // The request offers the reply's tools (provider tools too) and what the run adds.
        let mut offered = plan.tools.clone();
        let added: Vec<ToolSpec> =
            tools.into_iter().filter(|t| offered.iter().all(|o| o.name != t.name)).collect();
        offered.extend(added);
        let mut plan = plan.clone();
        plan.tools = offered.clone();
        plan.record = None;
        plan.new_message = None;
        plan.turn_instructions = None;
        plan.on_enter = false;
        plan.running_tool_calls = Vec::new();
        let reply = ReplyRun {
            run,
            agent: toolset.agent.clone(),
            plan,
            realtime,
            request,
            calls: BTreeMap::new(),
            acting: false,
            cut: false,
            settling: None,
            settled_items: None,
            terminal_call: None,
            tool_end: None,
            verdict: None,
            reply_required: BTreeMap::new(),
            waiter: None,
            next: None,
            previous_user_metrics: None,
            accepted: BTreeMap::new(),
        };
        self.activity.tools.replies.insert(speech, reply);
        self.reply_effects(speech, startup, out);
        message_id.map(|message_id| (message_id, offered))
    }

    /// The person's message answered these read-backs: every live reply's step that has not
    /// called yet grants with them.
    pub(crate) fn replies_answered(&mut self, answers: &BTreeSet<String>, out: &mut Out) {
        let speeches: Vec<SpeechId> = self.activity.tools.replies.keys().copied().collect();
        for speech in speeches {
            let approval_answers = answers.clone();
            self.reply_event(speech, Event::Answered { approval_answers }, out);
        }
    }

    /// A call another speech records (a read-back put, a grant spent) is what every other live
    /// reply's approval reads next.
    pub(crate) fn reply_recorded(&mut self, item: &ChatItem, source: &Source, out: &mut Out) {
        let ChatItem::FunctionCall(call) = item else { return };
        if !call.extra.contains_key(APPROVAL_KEY) {
            return;
        }
        let from = source.speech();
        let speeches: Vec<SpeechId> =
            self.activity.tools.replies.keys().copied().filter(|s| Some(*s) != from).collect();
        for speech in speeches {
            self.reply_event(speech, Event::Recorded { item: Box::new(item.clone()) }, out);
        }
    }

    /// One event for `speech`'s run, and what it decides.
    pub(crate) fn reply_event(&mut self, speech: SpeechId, event: Event, out: &mut Out) {
        let running = self.activity.running_tool_executions(speech);
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        let fx = reply.run.handle_with_running_tools(event, self.step.now, &running);
        self.reply_effects(speech, fx, out);
    }

    fn reply_effects(&mut self, speech: SpeechId, fx: Vec<RunEffect>, out: &mut Out) {
        for effect in fx {
            match effect {
                RunEffect::PlanRequest { request, message_id, pending, tools, tools_forbidden } => {
                    self.reply_planned(
                        speech,
                        request,
                        message_id,
                        pending,
                        tools,
                        tools_forbidden,
                        out,
                    )
                }
                RunEffect::HandleToolResults { step, items, terminal_call, .. } => {
                    if let Some(reply) = self.activity.tools.replies.get_mut(&speech) {
                        reply.settling = Some(step);
                        reply.settled_items = Some(items);
                        reply.terminal_call = terminal_call;
                    }
                    self.handle_reply_tool_results(speech, out);
                }
                RunEffect::CancelToolExecution { execution } => {
                    let running = self
                        .activity
                        .tools
                        .running
                        .iter()
                        .find(|r| r.hosted.invocation.operation == execution)
                        .cloned();
                    if let Some(running) = running {
                        if !running.detached()
                            && let Some(reply) =
                                self.activity.tools.replies.get_mut(&running.speech)
                        {
                            reply.run.tool_cancellation_requested(&running.call.call_id);
                        }
                        out.machine().push(Effect::CancelTool {
                            call_id: running.call.call_id,
                            hosted: running.hosted,
                        });
                    }
                }
                RunEffect::OutputValidate { validation, candidate } => {
                    self.validate_reply_output(speech, validation, candidate, out)
                }
                RunEffect::OutputValidateCancel { validation } => {
                    out.machine().push(Effect::CancelOutputValidation { speech, validation });
                }
                RunEffect::LlmRequestCancel { request } => {
                    // Keep the producer's explicit stream-close path. An unresponsive source
                    // is bounded by the existing speech interruption watchdog.
                    out.trace(crate::trace::GenerationStep::RequestCancelled {
                        speech,
                        request: request.0,
                    });
                    let _ = self.interrupt_speech(
                        self.step.now,
                        speech,
                        crate::trace::InterruptCause::RunCancelled,
                        true,
                        out,
                    );
                }
                RunEffect::ToolPrepare { call, arguments, .. } => {
                    self.activity.reply_move(speech, call, ToolGo::Prepare { call, arguments })
                }
                RunEffect::ReadBackRender { call, .. } => {
                    self.activity.reply_move(speech, call, ToolGo::ReadBack { call })
                }
                RunEffect::TemplateRender { call, template } => {
                    self.render_refusal(speech, call, template, out)
                }
                RunEffect::ToolExecute { call, function_call, .. } => {
                    self.session.usage.tool_started();
                    self.reply_executes(speech, &function_call, out);
                    let item = Box::new(function_call);
                    self.activity.reply_move(speech, call, ToolGo::Execute { item });
                }
                RunEffect::ToolCancel { call } => self.activity.reply_cancels(speech, call, out),
                RunEffect::Finished(result) => self.reply_finished(speech, result, out),
                // As a text run reports it: the work is still owned and named, the reply goes on.
                RunEffect::IncompleteCleanup { pending } => log::error!(
                    target: "livekit.agents",
                    "a reply's cleanup missed its deadline: {}",
                    pending.join(", ")
                ),
                // The voice lifecycle tells the calls' progress; prepared arguments are the call
                // tasks', released when told they will not run; the session renders instructions
                // and streams the requests.
                RunEffect::ToolArgumentsRelease { .. }
                | RunEffect::Emit(_)
                | RunEffect::ArgumentsRepaired { .. }
                | RunEffect::HostCheckpoint { .. }
                | RunEffect::HostCheckpointCancel { .. }
                | RunEffect::InstructionsRender { .. }
                | RunEffect::InstructionsCancel { .. }
                | RunEffect::LlmRequestStart { .. } => {}
            }
        }
        self.reply_settled_calls(speech, out);
        self.activity.reply_answer(speech, out);
    }

    // ------------------------------------------------------------------ calls

    /// A call of the reply's step is authorized to start: the run admits it as the floor
    /// stamped it, with the person's answers as they stand. Its id in the run, or `None` when the
    /// speech has no run to take it.
    pub(crate) fn reply_call_made(
        &mut self,
        speech: SpeechId,
        call: FunctionCall,
        out: &mut Out,
    ) -> Option<CallId> {
        let running = self.activity.running_tool_executions(speech);
        let reply = self.activity.tools.replies.get_mut(&speech)?;
        let request = reply.request?;
        let call_id = call.call_id.clone();
        reply.acting = true;
        let fx = reply.run.handle_with_running_tools(
            Event::CallMade { request, call },
            self.step.now,
            &running,
        );
        let id = reply.run.step_call_id(&call_id);
        if let Some(id) = id {
            reply.calls.entry(id).or_default();
        }
        self.reply_effects(speech, fx, out);
        id
    }

    /// The call's task reports `event` and awaits its next move (`go`).
    pub(crate) fn reply_call_asks(
        &mut self,
        speech: SpeechId,
        call: CallId,
        go: RequestId,
        event: Event,
        out: &mut Out,
    ) {
        let entry = (self.activity.tools.replies.get_mut(&speech))
            .and_then(|reply| reply.calls.get_mut(&call));
        let Some(entry) = entry else {
            return out.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        };
        entry.waiting = Some(go);
        self.reply_event(speech, event, out);
    }

    /// The run admitted a call's body: a cut call is committed as it ran, its grant spent, and
    /// the run, told the call as it started, is told it again with its approval marks.
    fn reply_executes(&mut self, speech: SpeechId, call: &FunctionCall, out: &mut Out) {
        let running = self.activity.tools.running.iter_mut();
        if let Some(r) =
            running.filter(|r| r.speech == speech).find(|r| r.call.call_id == call.call_id)
        {
            let created_at = r.call.created_at;
            r.call = FunctionCall { created_at, ..call.clone() };
            r.executing = true;
        }
        let call_id = call.call_id.clone();
        out.trace(TraceToolStep::CallAdmitted { speech, call_id });
        self.reply_marked(speech, call, out);
    }

    fn reply_marked(&mut self, speech: SpeechId, call: &FunctionCall, out: &mut Out) {
        if !call.extra.contains_key(APPROVAL_KEY) {
            return;
        }
        if let Some(run) = self.session.runs.current()
            && self.session.runs.owns(crate::run_result::Owned::Speech(speech))
        {
            let item = Box::new(ChatItem::FunctionCall(call.clone()));
            out.machine().emit(SessionEvent::RunItemAdded { run, item });
        }
    }

    /// Each call the run settled without its body is told so; every decided move reaches the
    /// started task awaiting it.
    fn reply_settled_calls(&mut self, speech: SpeechId, out: &mut Out) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        let mut marked = Vec::new();
        let mut told = Vec::new();
        for (id, entry) in &mut reply.calls {
            if entry.next.is_none()
                && let Some((item, Some(settled))) = reply.run.step_call(*id)
            {
                marked.push(item.clone());
                let (item, outcome) = (Box::new(item.clone()), settled.outcome.clone());
                entry.next = Some(ToolGo::Settled { item, outcome });
            }
            let (true, Some(go)) = (entry.started, entry.waiting) else { continue };
            let Some(next) = entry.next.take() else { continue };
            entry.waiting = None;
            told.push((*id, go, next));
        }
        for (id, go, next) in told {
            if last(&next) {
                reply.calls.remove(&id);
            }
            out.answer(Some(go), Answer::ToolGo(next));
        }
        for call in marked {
            self.reply_marked(speech, &call, out);
        }
    }

    // ------------------------------------------------------------------ steps

    /// The words of `speech`'s step are in: the run reads them as the model's response.
    pub(crate) fn reply_spoken(&mut self, speech: SpeechId, text: Option<String>, out: &mut Out) {
        let Some(request) =
            self.activity.tools.replies.get_mut(&speech).and_then(|r| r.request.take())
        else {
            return;
        };
        if let Some(text) = text.filter(|t| !t.is_empty()) {
            let delta = ChoiceDelta {
                role: Some(Role::Assistant),
                content: Some(text),
                tool_calls: Vec::new(),
                extra: None,
            };
            let chunk = LlmChunk {
                id: String::new(),
                delta: Some(delta),
                usage: None,
                flush: false,
                node_text: false,
            };
            self.reply_event(speech, Event::LlmChunk { request, chunk }, out);
        }
        self.reply_event(speech, Event::LlmEnded { request }, out);
    }

    /// The reply task reports its step's calls over and, with `answer`, waits for its next
    /// step. `true` when the step made calls the floor decides on.
    pub(crate) fn reply_step_ended(
        &mut self,
        speech: SpeechId,
        answer: Option<RequestId>,
        out: &mut Out,
    ) -> bool {
        if !self.activity.tools.replies.contains_key(&speech) {
            if answer.is_some() {
                out.answer(answer, Answer::ToolStep(StepNext::Done));
            }
            return false;
        }
        // A step that never spoke (cut before its words, or failed) is over for the run too.
        self.reply_spoken(speech, None, out);
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return false };
        reply.waiter = answer;
        let acting = reply.acting;
        self.activity.reply_answer(speech, out);
        acting
    }

    /// Voice constrains the shared outcome after canonical results and host edits are handled.
    pub(crate) fn reply_tool_results_handled(
        &mut self,
        speech: SpeechId,
        continuation: AgentRunContinuation,
        reply_required: BTreeMap<String, bool>,
        previous_user_metrics: Option<MetricsReport>,
        out: &mut Out,
    ) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        reply.previous_user_metrics = previous_user_metrics;
        reply.verdict = Some(continuation);
        reply.reply_required = reply_required;
        self.reply_verdict(speech, out);
        self.activity.reply_answer(speech, out);
    }

    fn reply_verdict(&mut self, speech: SpeechId, out: &mut Out) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        let (Some(step), Some(continuation)) = (reply.settling, reply.verdict) else { return };
        reply.settling = None;
        reply.verdict = None;
        let reply_required = std::mem::take(&mut reply.reply_required);
        self.reply_event(
            speech,
            Event::ToolResultsHandled { step, continuation, reply_required },
            out,
        );
    }

    /// The run's next step, planned: on the session's record as published, with what the run
    /// recorded since its last step (an output's feedback) for this request only.
    #[allow(clippy::too_many_arguments)]
    fn reply_planned(
        &mut self,
        speech: SpeechId,
        request: lk_agents_core::agent_run::RequestId,
        message_id: String,
        pending: Vec<ChatItem>,
        tools: Vec<ToolSpec>,
        tools_forbidden: bool,
        out: &mut Out,
    ) {
        let cut = (self.activity.speech_scheduler.speech(speech))
            .is_none_or(|s| s.interrupted() || s.done());
        let record = self.session.chat_ctx.agent_chat_ctx();
        let instructions = self.activity.instructions.render(Some(Modality::Audio));
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        reply.request = Some(request);
        reply.acting = false;
        reply.cut = false;
        reply.calls.clear();
        if cut {
            reply.next = Some(StepNext::Interrupted);
            return;
        }
        let tool_choice = if tools_forbidden { ToolChoice::Forbidden } else { ToolChoice::Auto };
        // A realtime server holds its own replies and the outputs pushed with the commit; what
        // else the run recorded since is pushed before the server is asked.
        let unknown: Vec<ChatItem> = (pending.iter())
            .filter(|i| !record.items.iter().any(|r| r.id() == i.id()))
            .filter(|i| !matches!(i, ChatItem::Message(m) if m.role == Role::Assistant))
            .cloned()
            .collect();
        let mut plan = reply.plan.clone();
        let added: Vec<ToolSpec> =
            tools.into_iter().filter(|t| plan.tools.iter().all(|o| o.name != t.name)).collect();
        plan.tools.extend(added);
        let wall = self.step.wall();
        let next = NextReply {
            planned: PlannedReply {
                request: plan.plan_pending(&record, &instructions, &pending, wall),
                tool_choice: Some(tool_choice),
            },
            message_id,
            previous_user_metrics: reply.previous_user_metrics.take(),
        };
        reply.next = Some(StepNext::Reply(Box::new(next)));
        if reply.realtime && !self.activity.realtime_state.ended() {
            for item in unknown {
                let push = lk_agents_abi::llm::realtime::PushItem {
                    item: Box::new(item),
                    reply_required: false,
                };
                self.activity.rt_send(lk_agents_abi::RealtimeCommand::PushItem(push), out);
            }
        }
    }

    /// The run settled: what it accepted, asked or failed with is the output of a session run
    /// ending on the speech, and the reply task's step is its last.
    fn reply_finished(
        &mut self,
        speech: SpeechId,
        result: Result<RunResult, RunError>,
        out: &mut Out,
    ) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        let mut interrupted = false;
        let output = match result {
            Ok(RunResult { outcome, interrupted: cut, .. }) => {
                interrupted = cut;
                match outcome {
                    RunOutcome::Completed { output: Output::Validated(v) } => {
                        reply.accepted.remove(&v).map(RunOutput::Value)
                    }
                    // A validated text: the agent's accepted value, the last validation's.
                    RunOutcome::Completed { output: Output::Text(_) } => {
                        reply.accepted.pop_last().map(|(_, v)| RunOutput::Value(v))
                    }
                    RunOutcome::NoOutput => None,
                    RunOutcome::InputRequired { question, verbatim, answer_tool } => {
                        Some(RunOutput::Ask { question, verbatim, answer_tool })
                    }
                }
            }
            Err(
                error @ (RunError::ToolRetriesExceeded { .. }
                | RunError::OutputRetriesExceeded { .. }
                | RunError::Host(_)),
            ) => Some(RunOutput::Failed(error)),
            // The session reports a model's failure on the speech; a cancelled run is the
            // person's cut.
            Err(_) => None,
        };
        reply.next = Some(if interrupted { StepNext::Interrupted } else { StepNext::Done });
        // Every call still waiting is over with the run.
        for entry in reply.calls.values_mut() {
            if let Some(go) = entry.waiting.take() {
                out.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
            }
        }
        reply.calls.clear();
        match output {
            Some(RunOutput::Failed(error)) => self.run_failed(speech, error, out),
            // A question a tool asked was delivered with its output: it stays the run's.
            Some(output @ RunOutput::Ask { .. }) => {
                self.session.runs.outputs.insert(speech, output);
            }
            Some(output) => {
                self.session.runs.outputs.insert(speech, output);
            }
            None => {}
        }
    }

    // ------------------------------------------------------------------ output

    /// An output candidate, validated by the reply's agent.
    fn validate_reply_output(
        &mut self,
        speech: SpeechId,
        validation: ValidationId,
        candidate: lk_agents_abi::agent::OutputCandidate,
        out: &mut Out,
    ) {
        let agent = self.activity.tools.replies.get(&speech).and_then(|r| r.agent.clone());
        let held = self.activity.speech_scheduler.contains(speech);
        let Some(agent) = agent.filter(|_| held) else {
            // No agent to validate it: refused as the host would refuse it.
            let verdict =
                OutputVerdict::Failed("the reply has no agent to validate its output".into());
            return self.reply_event(speech, Event::OutputValidated { validation, verdict }, out);
        };
        let operation = self.activity.tools.mint_operation();
        let hosted = HostedTool::of(&agent, self.session.ids.session(), speech, operation, None);
        out.machine().push(Effect::ValidateOutput { speech, validation, candidate, hosted });
    }

    /// The reply's agent answered an output candidate.
    pub(crate) fn on_output_validated(
        &mut self,
        speech: SpeechId,
        validation: ValidationId,
        verdict: OutputVerdict,
        value: Option<HostRef>,
        out: &mut Out,
    ) {
        if let (OutputVerdict::Accepted { .. }, Some(value), Some(reply)) =
            (&verdict, value, self.activity.tools.replies.get_mut(&speech))
        {
            reply.accepted.insert(validation, value);
        }
        self.reply_event(speech, Event::OutputValidated { validation, verdict }, out);
    }

    // ------------------------------------------------------------------ templates

    /// A duplicate call's refusal, rendered by the reply's agent with the call's templates.
    fn render_refusal(
        &mut self,
        speech: SpeechId,
        call: CallId,
        template: lk_agents_abi::agent::ToolTemplate,
        out: &mut Out,
    ) {
        let reply = self.activity.tools.replies.get(&speech);
        let agent = reply.and_then(|r| r.agent.clone());
        let item = reply.and_then(|r| r.run.step_call(call)).map(|(item, _)| item.clone());
        let held = self.activity.speech_scheduler.contains(speech);
        let Some(agent) = agent.filter(|_| held) else {
            return self.reply_event(speech, Event::TemplateRendered { call, text: None }, out);
        };
        let operation = self.activity.tools.mint_operation();
        let session = self.session.ids.session();
        let hosted = HostedTool::of(&agent, session, speech, operation, item.as_ref());
        let rendering = Rendering::Refusal { speech, call };
        out.machine().push(Effect::RenderTemplate { rendering, template, hosted });
    }

    /// The host rendered a template: a refusal answers its call, the deferred reply's
    /// instructions schedule it.
    pub(crate) fn on_template_rendered(
        &mut self,
        now: lk_agents_abi::Ts,
        rendering: Rendering,
        text: Option<String>,
        out: &mut Out,
    ) {
        match rendering {
            Rendering::Refusal { speech, call } => {
                self.reply_event(speech, Event::TemplateRendered { call, text }, out)
            }
            Rendering::ToolReply { render } => self.tool_reply_rendered(now, render, text, out),
        }
    }
}

impl super::AgentActivity {
    /// Facts from existing live bodies, including detached bodies and concurrent replies.
    /// This reply's blocking calls already belong to AgentRun's current call table.
    fn running_tool_executions(
        &self,
        speech: SpeechId,
    ) -> Vec<lk_agents_core::tool_calls::RunningToolExecution> {
        self.tools
            .running
            .iter()
            .filter(|running| running.executing && (running.speech != speech || running.detached()))
            .filter_map(|running| {
                Some(lk_agents_core::tool_calls::RunningToolExecution {
                    execution: running.hosted.invocation.operation,
                    call: running.call.clone(),
                    identity: running.identity.clone()?,
                })
            })
            .collect()
    }
    /// The reply's agent as its run reads it: each offered function tool with its policies. A
    /// `VoiceTask`'s agent completes through its tools' validated `Completed`: its replies are
    /// conversation, with no output to accept.
    fn reply_definition(
        &self,
        toolset: &Toolset,
        retries: Retries,
        task: bool,
    ) -> Result<AgentDefinition, &'static str> {
        let frame = toolset.agent.as_ref().map(|agent| agent.definition());
        let tools = offered_tool_infos(&toolset.specs, frame.map_or(&[], |frame| &frame.tools))?;
        Ok(AgentDefinition {
            name: frame.map_or_else(|| self.agent.clone(), |d| d.name.clone()),
            // The session plans each request: the run renders no instructions.
            instructions: String::new(),
            dynamic_instructions: false,
            tools,
            output: match frame {
                Some(d) if !task => d.output.clone(),
                Some(_) | None => OutputSpec::Text { validated: false },
            },
            retries,
        })
    }

    /// The run decided the first move of a call waiting to start.
    pub(crate) fn reply_first_ready(&self, speech: SpeechId, call: CallId) -> bool {
        let reply = self.tools.replies.get(&speech);
        reply.and_then(|r| r.calls.get(&call)).is_some_and(|c| c.next.is_some())
    }

    /// The call's task starts: its first move.
    pub(crate) fn reply_call_start(&mut self, speech: SpeechId, call: CallId) -> ToolGo {
        let Some(reply) = self.tools.replies.get_mut(&speech) else { return ToolGo::Gone };
        let Some(entry) = reply.calls.get_mut(&call) else { return ToolGo::Gone };
        let first = entry.next.take().unwrap_or(ToolGo::Gone);
        entry.started = true;
        if last(&first) {
            reply.calls.remove(&call);
        }
        first
    }

    /// A call of the reply was dropped unrun: its run will not settle the step.
    pub(crate) fn reply_call_dropped(&mut self, speech: SpeechId) {
        if let Some(reply) = self.tools.replies.get_mut(&speech) {
            reply.cut = true;
        }
    }

    fn reply_move(&mut self, speech: SpeechId, call: CallId, go: ToolGo) {
        let Some(reply) = self.tools.replies.get_mut(&speech) else { return };
        if let Some(entry) = reply.calls.get_mut(&call) {
            entry.next = Some(go);
        }
    }

    /// A duplicate the tool replaces: its running body is cancelled.
    fn reply_cancels(&mut self, speech: SpeechId, call: CallId, out: &mut Out) {
        let Some(reply) = self.tools.replies.get(&speech) else { return };
        let Some((item, _)) = reply.run.step_call(call) else { return };
        let call_id = item.call_id.clone();
        let running = self.tools.running.iter();
        if let Some(r) = running.filter(|r| r.speech == speech).find(|r| r.call.call_id == call_id)
        {
            out.machine().push(Effect::CancelTool { call_id, hosted: r.hosted.clone() });
        }
    }

    /// Answers the reply task's wait once the run decided its next step.
    fn reply_answer(&mut self, speech: SpeechId, out: &mut Out) {
        let Some(reply) = self.tools.replies.get_mut(&speech) else { return };
        if reply.waiter.is_none() || reply.next.is_none() {
            return;
        }
        let (waiter, next) = (reply.waiter.take(), reply.next.take());
        out.answer(waiter, Answer::ToolStep(next.expect("checked above")));
    }

    /// The speech left the loop: its run goes, and a call task still waiting on it is gone.
    pub(crate) fn reply_released(&mut self, speech: SpeechId, out: &mut Out) {
        let Some(reply) = self.tools.replies.remove(&speech) else { return };
        for go in reply.calls.into_values().filter_map(|c| c.waiting) {
            out.answer(Some(go), Answer::ToolGo(ToolGo::Gone));
        }
        if let Some(waiter) = reply.waiter {
            out.answer(Some(waiter), Answer::ToolStep(StepNext::Done));
        }
    }
}

impl Driver {
    /// Shared invocation cleanup deadlines remain owned by AgentRun, composed into the
    /// voice driver's deadline rather than copied into a second timer registry.
    pub(crate) fn reply_wake_at(&self) -> Option<lk_agents_abi::Ts> {
        self.activity.tools.replies.values().filter_map(|reply| reply.run.wake_at()).min()
    }

    pub(crate) fn wake_reply_runs(&mut self, now: lk_agents_abi::Ts, out: &mut Out) {
        let due: Vec<SpeechId> = self
            .activity
            .tools
            .replies
            .iter()
            .filter(|(_, reply)| reply.run.wake_at().is_some_and(|at| at <= now))
            .map(|(speech, _)| *speech)
            .collect();
        for speech in due {
            self.reply_event(speech, Event::Wake, out);
        }
    }
}

/// Offered schemas can differ from setup, but policy metadata must come from the installed
/// agent definition. Missing metadata is not permission to disable approvals or duplicates.
fn offered_tool_infos(
    offered: &[ToolSpec],
    known: &[FunctionToolInfo],
) -> Result<Vec<FunctionToolInfo>, &'static str> {
    offered
        .iter()
        .filter(|spec| spec.provider_tool.is_none())
        .map(|spec| {
            let info = known
                .iter()
                .find(|info| info.spec.name == spec.name)
                .ok_or("an advertised function tool has no installed policy metadata")?;
            Ok(FunctionToolInfo { spec: spec.clone(), ..info.clone() })
        })
        .collect()
}

#[cfg(test)]
#[path = "reply_run_tests.rs"]
mod tests;
