//! Apply AgentRun's canonical tool results to voice history and host callbacks.
//! Voice supplies floor/interruption constraints; it does not select the tool outcome again.

use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{Driver, HookCall, HookId, RequestId};
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::tool_executor::{Delivery, Handling, StepEnd, ToolsExecutedEvent};
use crate::trace::{CoreStep, ToolStep};
use lk_agents_abi::{ChatItem, Ts};
use lk_agents_core::agent_run::AgentRunContinuation;
use std::collections::BTreeMap;

impl Driver {
    /// The executor reports completion and voice facts, not a competing step decision.
    pub(crate) fn on_tools_finished(
        &mut self,
        _now: Ts,
        (speech, step): (SpeechId, u32),
        end: StepEnd,
        answer: Option<RequestId>,
        out: &mut Out,
    ) {
        for name in self.activity.tools.dropped.remove(&(speech, step)).unwrap_or_default() {
            let diagnostic = DiagnosticEvent::error(crate::events::diagnostics::TOOL_CHOICE_NONE)
                .with("function", name)
                .with("speech_id", speech.0.to_string());
            out.machine().emit(SessionEvent::Diagnostic(diagnostic));
        }
        if !self.reply_step_ended(speech, answer, out) {
            return;
        }
        if let Some(reply) = self.activity.tools.replies.get_mut(&speech) {
            reply.tool_end = Some((step, end));
        }
        self.handle_reply_tool_results(speech, out);
    }

    /// Both executor completion and the shared outcome must arrive before results are applied.
    pub(super) fn handle_reply_tool_results(&mut self, speech: SpeechId, out: &mut Out) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        if reply.settling.is_none() || reply.settled_items.is_none() || reply.tool_end.is_none() {
            return;
        }
        let (step, end) = reply.tool_end.take().expect("checked above");
        let items = reply.settled_items.take().expect("checked above");
        let terminal_call = reply.terminal_call.take();
        let mut event = ToolsExecutedEvent {
            function_calls: Vec::new(),
            function_call_outputs: Vec::new(),
            handoff_required: false,
        };
        for item in &items {
            match item {
                ChatItem::FunctionCall(call) => event.function_calls.push(call.clone()),
                ChatItem::FunctionCallOutput(output) => {
                    event.function_call_outputs.push(output.clone());
                }
                _ => {}
            }
        }
        let handling = Handling { speech, step, end, items, terminal_call, event };
        if !handling.event.function_calls.is_empty() && self.activity.hook_bound.is_some() {
            let id = self.activity.hooks.mint();
            let call = HookCall::FunctionToolsExecuted {
                id,
                speech,
                event: Box::new(handling.event.clone()),
            };
            let wait = self.activity.hook_wait();
            self.activity.tools.handling.insert(id, handling);
            self.issue_hook(self.step.now, call, Some(wait), out);
            return;
        }
        self.finish_tool_results(handling, false, out);
    }

    pub(crate) fn on_tools_executed_handled(
        &mut self,
        hook: HookId,
        left: Option<Vec<bool>>,
        out: &mut Out,
    ) {
        let Some(mut handling) = self.activity.tools.handling.remove(&hook) else { return };
        if let Some(reply_required) = left {
            if reply_required.len() == handling.event.function_call_outputs.len() {
                for (output, required) in
                    handling.event.function_call_outputs.iter_mut().zip(reply_required)
                {
                    output.reply_required = required;
                }
            } else {
                out.trace(CoreStep::PortFailure(
                    "tool handler output count does not match its calls",
                ));
            }
        }
        self.finish_tool_results(handling, true, out);
    }

    fn finish_tool_results(&mut self, handling: Handling, hosted: bool, out: &mut Out) {
        let Handling { speech, step, end, mut items, terminal_call, mut event } = handling;
        // Host tool-executed handlers may have awaited while the user cut this speech.
        // Their original executor snapshot must not resurrect a question or claim afterward.
        let interrupted = end.interrupted_before_tools
            || end.interrupted
            || self
                .activity
                .speech_scheduler
                .speech(speech)
                .is_none_or(|state| state.interrupted() || state.done());
        // A host may edit only reply_required. Apply the identical patch to both histories.
        let reply_required: BTreeMap<String, bool> = event
            .function_call_outputs
            .iter_mut()
            .map(|output| {
                // A validated VoiceTask transition ends its live interaction rather than
                // requesting another model reply. Apply that voice constraint to the same
                // canonical output patch sent back to AgentRun.
                let completes_task = end.executions.iter().any(|execution| {
                    execution.call.call_id == output.call_id
                        && matches!(
                            execution.delivery,
                            Some(Delivery::Complete(_) | Delivery::Fail(_))
                        )
                });
                if interrupted || completes_task {
                    output.reply_required = false;
                }
                (output.call_id.clone(), output.reply_required)
            })
            .collect();
        for item in &mut items {
            if let ChatItem::FunctionCallOutput(output) = item
                && let Some(required) = reply_required.get(&output.call_id)
            {
                output.reply_required = *required;
            }
        }
        if let Some(rt) = &self.activity.rt_session
            && !self.activity.realtime_state.ended()
        {
            let auto = rt.capabilities.auto_tool_reply_generation;
            for item in items.iter().filter(|item| matches!(item, ChatItem::FunctionCallOutput(_)))
            {
                self.activity.rt_send(
                    lk_agents_abi::RealtimeCommand::PushItem(
                        lk_agents_abi::llm::realtime::PushItem {
                            item: Box::new(item.clone()),
                            reply_required: !auto,
                        },
                    ),
                    out,
                );
            }
        }
        let committed = items.len();
        for item in items {
            let source = if matches!(item, ChatItem::FunctionCall(_)) {
                Source::ToolCall { speech, step }
            } else {
                Source::ToolOutput { speech, step }
            };
            if matches!(item, ChatItem::FunctionCall(_))
                && let Some(state) = self.activity.speech_scheduler.speech_mut(speech)
                && !state.item_ids().iter().any(|id| id == item.id())
            {
                state.item_added(&item);
            }
            self.commit_item(item, source, Visibility::Both, out);
        }
        self.activity.tool_step_committed(speech);
        self.publish_activity();
        let completes_task = end.executions.iter().any(|execution| {
            matches!(execution.delivery, Some(Delivery::Complete(_) | Delivery::Fail(_)))
        });
        for execution in &end.executions {
            let Some(delivery) = execution.delivery.clone() else { continue };
            // Only core's selected terminal question can reach the person. Other delivery
            // forms retain their own domain contract (including VoiceTask completion).
            if matches!(delivery, Delivery::Ask(_))
                && (interrupted || terminal_call.as_deref() != Some(&execution.call.call_id))
            {
                continue;
            }
            self.deliver_outcome(self.step.now, speech, delivery, out);
        }
        // Speech step numbering starts at one.
        let max_steps_reached = step > end.max_tool_steps;
        if max_steps_reached {
            out.machine().emit(SessionEvent::Diagnostic(
                DiagnosticEvent::warning(crate::events::diagnostics::MAX_TOOL_STEPS)
                    .with("speech_id", speech.0.to_string()),
            ));
            out.trace(CoreStep::PortFailure(
                "max_tool_steps reached: the reply is forced to tool_choice=none",
            ));
        }
        out.trace(ToolStep::Finished { speech, committed });
        if !hosted && !event.function_calls.is_empty() {
            out.machine().emit(SessionEvent::FunctionToolsExecuted(Box::new(event)));
        }
        let draining = self.activity.speech_scheduler.is_paused()
            || self
                .activity
                .speech_agent(speech)
                .is_some_and(|agent| *agent != self.activity.agent);
        let continuation = if interrupted {
            AgentRunContinuation::Interrupted
        } else if completes_task || self.session.closing_drained() {
            AgentRunContinuation::Stop
        } else {
            AgentRunContinuation::Continue { tools_forbidden: draining || max_steps_reached }
        };
        let metrics = end.forwarded_text_empty.then_some(end.user_metrics).flatten();
        self.reply_tool_results_handled(speech, continuation, reply_required, metrics, out);
    }
}

#[cfg(test)]
#[path = "tool_results_tests.rs"]
mod tests;
