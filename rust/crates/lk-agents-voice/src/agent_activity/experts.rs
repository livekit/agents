//! The driver's side of the frontend expert event queue. A call of a configured remote tool that
//! detaches is a delegation: its body's later updates are its expert's progress, an
//! `InputRequired` it reports is the expert's question, and its end is the expert's outcome,
//! which resolves the originating call in place. The person's reply goes back to the body.

use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::{PreemptiveDiscard, SpeechKind};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::effects::{Effect, Out};
use crate::agent_session::{Driver, ToolUpdate};
use crate::audio_recognition::EndOfTurnInfo;
use crate::events::{SessionEvent, ToolEndStatus, ToolLifecycle};
use crate::experts::{ExpertTool, Floor, Presentation};
use crate::generation::ReplySource;
use crate::tool_executor::{Delivery, ToolEnd, ToolOutcome, answer};
use lk_agents_abi::llm::tool_executor::ASK_IN_OWN_WORDS;
use lk_agents_abi::{ChatItem, Ts};
use lk_agents_core::tts::markup::ChatMessageExt;

/// How a presentation-only reply is told to pass on an expert's progress.
const PRESENT_PROGRESS: &str = "Tell the person how their request with the expert is going, \
     briefly and in your own words, adding nothing it does not say: ";

impl Driver {
    /// `call_id` detached: a call of one of the active agent's configured remote tools is now a
    /// delegation the session resolves by its expert's outcome.
    pub(crate) fn open_expert(&mut self, call_id: &str) {
        let running = self.activity.tools.running.iter().find(|r| r.call.call_id == call_id);
        let Some(name) = running.map(|r| r.call.name.clone()) else { return };
        if let Some(tool) = self.activity.expert_tools().into_iter().find(|t| t.name == name) {
            self.session.experts.open(call_id, tool);
        }
    }

    /// A delegated call's update is its expert's event: `false` when the call is not one.
    pub(crate) fn on_expert_update(
        &mut self,
        now: Ts,
        call_id: &str,
        update: &ToolUpdate,
        out: &mut Out,
    ) -> bool {
        if !self.session.experts.owns(call_id) {
            return false;
        }
        match update {
            ToolUpdate::Progress { raw, .. } => {
                let Some(running) = self.activity.tools.detached(call_id) else { return true };
                running.updates += 1;
                let occurrence = running.updates.into();
                self.session.experts.progress(call_id, occurrence, raw.clone());
            }
            ToolUpdate::Final { report } => match &report.outcome {
                ToolOutcome::InputRequired { question, verbatim, .. } => {
                    let request_id = self.session.ids.mint("ask");
                    self.session.experts.ask(call_id, &request_id, question.clone(), *verbatim);
                }
                outcome => {
                    self.activity.tools.running.retain(|r| r.call.call_id != call_id);
                    let closing = self.session.closing_past_edge();
                    out.machine().append(self.session.states.on_tool_call_ended(now, closing));
                    self.session.experts.outcome(call_id, outcome.clone());
                }
            },
        }
        self.try_present_experts(now, out);
        true
    }

    /// The active agent changed: a question it cannot interact with stops taking replies.
    pub(crate) fn observe_experts(&mut self) {
        let tools = self.activity.expert_tools();
        let task = self.activity.voice_tasks.running();
        self.session.experts.observe(Floor { tools: &tools, free: false, task });
    }

    /// Present queued expert events while the floor is free: each presentation takes the floor,
    /// so the next waits for the following idle instant. Runs after the deferred tool reply.
    pub(crate) fn try_present_experts(&mut self, now: Ts, out: &mut Out) {
        let tools = self.activity.expert_tools();
        loop {
            let free = self.activity.tools.reply.is_none()
                && !self.session.closing_past_edge()
                && !self.session.idle.held()
                && self.is_idle();
            let floor = Floor { tools: &tools, free, task: self.activity.voice_tasks.running() };
            let Some(presentation) = self.session.experts.next(floor) else { return };
            match presentation {
                Presentation::Progress { text, .. } => {
                    let instructions = format!("{PRESENT_PROGRESS}{text}");
                    self.reply_with_instructions(now, instructions, None, out);
                }
                Presentation::Question { question, verbatim: true, .. } => {
                    self.say_text(now, question, out);
                }
                Presentation::Question { question, verbatim: false, .. } => {
                    let instructions = format!("{ASK_IN_OWN_WORDS}{question}");
                    self.reply_with_instructions(now, instructions, None, out);
                }
                Presentation::Outcome { call_id, outcome } => {
                    self.resolve_expert_call(now, &call_id, outcome, out)
                }
            }
        }
    }

    /// The outcome becomes the originating call's output in place of its pending one, and the
    /// caller's next model invocation presents it with the full history.
    fn resolve_expert_call(&mut self, now: Ts, call_id: &str, outcome: ToolOutcome, out: &mut Out) {
        let (mut call, mut pending) = (None, None);
        for item in self.session.chat_ctx.items() {
            match item {
                ChatItem::FunctionCall(c) if c.call_id == call_id => call = Some(c.clone()),
                ChatItem::FunctionCallOutput(o) if o.call_id == call_id => {
                    pending = Some(o.id.clone())
                }
                _ => {}
            }
        }
        // The application replaced the record meanwhile: nothing is left to resolve.
        let (Some(call), Some(pending)) = (call, pending) else { return };
        let mut execution = answer(&call, outcome.into(), self.step.wall());
        let status = match execution.end {
            ToolEnd::Error => ToolEndStatus::Error,
            ToolEnd::Cancelled => ToolEndStatus::Cancelled,
            ToolEnd::Done | ToolEnd::Detached => ToolEndStatus::Done,
        };
        out.machine().emit(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
            id: call_id.to_owned(),
            call_id: call_id.to_owned(),
            message: Some(execution.output.output.clone()),
            status,
        }));
        let reply_required = execution.output.reply_required;
        execution.output.id = pending;
        self.replace_items([ChatItem::FunctionCallOutput(execution.output)], out);
        self.publish_activity();
        if let Some(Delivery::Say(text)) = execution.delivery {
            self.say_text(now, text, out);
        } else if reply_required {
            self.reply_with_full_history(now, out);
        }
    }

    fn reply_with_full_history(&mut self, now: Ts, out: &mut Out) {
        if self.activity.rt_session.is_some() {
            self.realtime_reply(now, None, None, None, None, None, false, out);
            return;
        }
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        let spec = SpeechSpec::new(SpeechKind::GenerateReply, ReplySource::Llm, interruptible);
        self.start_speech(now, spec, out);
    }

    /// A pipeline user turn that answers the expert question owning the person's reply: committed
    /// as said and forwarded to exactly that request's delegated body, with no model reply and no
    /// tool selection.
    pub(crate) fn forward_expert_reply(
        &mut self,
        now: Ts,
        info: &EndOfTurnInfo,
        user_text: Option<&str>,
        out: &mut Out,
    ) -> bool {
        let tools = self.activity.expert_tools();
        let floor = Floor { tools: &tools, free: true, task: self.activity.voice_tasks.running() };
        let Some(request) = self.session.experts.collect(floor) else { return false };
        let running =
            self.activity.tools.running.iter().find(|r| r.call.call_id == request.call_id);
        // The body no longer runs: the turn is an ordinary one.
        let Some(hosted) = running.map(|r| r.hosted.clone()) else { return false };
        if self.session.experts.answer(&request).is_err() {
            return false;
        }
        if self.activity.speech_scheduler.preemptive_generation().is_some() {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        }
        let user = self.user_turn_item(info, user_text);
        let answer = user.text_content().unwrap_or_default();
        let source = Source::UserTurn { turn: info.turn_id };
        self.commit_item(ChatItem::Message(user), source, Visibility::Both, out);
        out.machine().push(Effect::AnswerExpert { hosted, answer });
        true
    }
}

impl super::AgentActivity {
    /// The expert tools the active agent exposes, by configured identity.
    fn expert_tools(&self) -> Vec<ExpertTool> {
        let Some(agent) = &self.toolset.agent else { return Vec::new() };
        let tools = &agent.definition().tools;
        tools
            .iter()
            .filter_map(|t| Some(ExpertTool::new(t.remote.as_deref()?, &t.spec.name)))
            .collect()
    }
}
