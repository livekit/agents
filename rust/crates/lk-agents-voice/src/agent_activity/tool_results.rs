//! Apply AgentRun's canonical tool results to voice history and host callbacks.
//! Voice supplies floor/interruption constraints; it does not select the tool outcome again.

use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{HookCall, HookId, RequestId, VoiceDriver};
use crate::events::{DiagnosticEvent, SessionEvent};
use crate::tool_executor::{Delivery, Handling, StepEnd, ToolsExecutedEvent};
use crate::trace::{CoreStep, ToolStep};
use lk_agents_abi::{ChatItem, Ts};
use lk_agents_core::agent_run::AgentRunContinuation;
use std::collections::BTreeMap;

impl VoiceDriver {
    /// The executor reports completion and voice facts, not a competing step decision.
    pub(crate) fn on_tools_finished(
        &mut self,
        _now: Ts,
        (speech, step): (SpeechId, u32),
        end: StepEnd,
        answer: Option<RequestId>,
    ) {
        for name in self.activity.tools.dropped.remove(&(speech, step)).unwrap_or_default() {
            let diagnostic = DiagnosticEvent::error(crate::events::diagnostics::TOOL_CHOICE_NONE)
                .with("function", name)
                .with("speech_id", speech.0.to_string());
            self.publish_event(SessionEvent::Diagnostic(diagnostic));
        }
        if !self.reply_step_ended(speech, answer) {
            return;
        }
        if let Some(reply) = self.activity.tools.replies.get_mut(&speech) {
            reply.tool_end = Some((step, end));
        }
        self.handle_reply_tool_results(speech);
    }

    /// Both executor completion and the shared outcome must arrive before results are applied.
    pub(super) fn handle_reply_tool_results(&mut self, speech: SpeechId) {
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
            self.issue_hook(self.step.now, call, Some(wait));
            return;
        }
        self.finish_tool_results(handling, false);
    }

    pub(crate) fn on_tools_executed_handled(&mut self, hook: HookId, left: Option<Vec<bool>>) {
        let Some(mut handling) = self.activity.tools.handling.remove(&hook) else { return };
        if let Some(reply_required) = left {
            if reply_required.len() == handling.event.function_call_outputs.len() {
                for (output, required) in
                    handling.event.function_call_outputs.iter_mut().zip(reply_required)
                {
                    output.reply_required = required;
                }
            } else {
                self.record_trace(CoreStep::PortFailure(
                    "tool handler output count does not match its calls",
                ));
            }
        }
        self.finish_tool_results(handling, true);
    }

    fn finish_tool_results(&mut self, handling: Handling, hosted: bool) {
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
                self.rt_send(lk_agents_abi::RealtimeCommand::PushItem(
                    lk_agents_abi::llm::realtime::PushItem {
                        item: Box::new(item.clone()),
                        reply_required: !auto,
                    },
                ));
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
            self.commit_item(item, source, Visibility::Both);
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
            self.deliver_outcome(self.step.now, speech, delivery);
        }
        // Speech step numbering starts at one.
        let max_steps_reached = step > end.max_tool_steps;
        if max_steps_reached {
            self.publish_event(SessionEvent::Diagnostic(
                DiagnosticEvent::warning(crate::events::diagnostics::MAX_TOOL_STEPS)
                    .with("speech_id", speech.0.to_string()),
            ));
            self.record_trace(CoreStep::PortFailure(
                "max_tool_steps reached: the reply is forced to tool_choice=none",
            ));
        }
        self.record_trace(ToolStep::Finished { speech, committed });
        if !hosted && !event.function_calls.is_empty() {
            self.publish_event(SessionEvent::FunctionToolsExecuted(Box::new(event)));
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
        self.reply_tool_results_handled(speech, continuation, reply_required, metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_activity::scheduling::SpeechKind;
    use crate::speech_handle::SpeechState;
    use crate::tool_executor::{ToolOutcome, answer};
    use crate::trace::InterruptCause;
    use lk_agents_abi::{FunctionCall, FunctionCallOutput};
    use std::time::UNIX_EPOCH;

    #[test]
    fn interrupt_while_tool_handler_waits_suppresses_the_old_question_and_claim() {
        let mut driver = crate::agent_session::tests::driver();
        let speech = SpeechId(41);
        assert!(driver.schedule_speech(
            Ts::ZERO,
            speech,
            SpeechState::new(true),
            SpeechKind::GenerateReply,
            true,
            true,
            true,
        ));
        let call = FunctionCall {
            id: "flight-item".into(),
            call_id: "flight-call".into(),
            name: "book_flight".into(),
            arguments: "{}".into(),
            created_at: UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        let output = FunctionCallOutput::new(
            "flight-output",
            "flight-call",
            "Which date?",
            false,
            UNIX_EPOCH,
        );
        let execution = answer(
            &call,
            ToolOutcome::InputRequired {
                question: "Which date?".into(),
                verbatim: true,
                tool: Some("book_flight".into()),
            }
            .into(),
            UNIX_EPOCH,
        );
        let hook = HookId(7);
        driver.activity.tools.handling.insert(
            hook,
            Handling {
                speech,
                step: 1,
                // Both executor reads happened before the host handler started awaiting.
                end: StepEnd {
                    executions: vec![execution],
                    interrupted_before_tools: false,
                    interrupted: false,
                    forwarded_text_empty: true,
                    max_tool_steps: 5,
                    user_metrics: None,
                },
                items: vec![call.clone().into(), output.clone().into()],
                terminal_call: Some(call.call_id.clone()),
                event: ToolsExecutedEvent {
                    function_calls: vec![call],
                    function_call_outputs: vec![output],
                    handoff_required: false,
                },
            },
        );
        // The person interrupts while function_tools_executed is still running at the host.
        let _ = driver.interrupt_speech(Ts::ZERO, speech, InterruptCause::Host, true).unwrap();
        assert!(driver.activity.speech_scheduler.speech(speech).unwrap().interrupted());
        driver.on_tools_executed_handled(hook, None);
        assert!(driver.activity.tools.claim.is_none(), "the old question cannot claim a new turn");
        assert!(
            !driver.session.runs.outputs.contains_key(&speech),
            "no old question was delivered"
        );
        assert_eq!(
            driver.activity.speech_scheduler.speech_ids().count(),
            1,
            "no question speech was created"
        );
        let output = driver
            .session
            .chat_ctx
            .items()
            .find_map(|item| match item {
                ChatItem::FunctionCallOutput(output) => Some(output),
                _ => None,
            })
            .expect("the result remains recorded even though its question is suppressed");
        assert!(!output.reply_required);
    }
}
