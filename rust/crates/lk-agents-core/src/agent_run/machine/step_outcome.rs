//! A settled step's outcome: its calls and outputs recorded in model order, and whether the
//! invocation continues, asks, stops or fails.

use lk_agents_abi::agent::OutputSpec;
use lk_agents_abi::agent::run::{Output, RunError};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ChatItem, FunctionCall, FunctionCallOutput, Ts};

use super::{AgentRun, AgentRunEffect, Ask, Phase, StepId, ValidationId};
use crate::tool_calls::Recorded;
use crate::tool_outcome;

impl AgentRun {
    /// Once every call of the step settled: record the calls and outputs in model order, then
    /// continue, ask, or settle.
    pub(super) fn step_settled(&mut self, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        let settled =
            matches!(&self.phase, Phase::Acting(a) if a.finals.is_empty() && a.calls.all_settled());
        if !settled {
            return;
        }
        let Phase::Acting(acting) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            unreachable!()
        };
        let recorded = acting.calls.finish();
        // Retry budgets: consecutive failures per tool, any success resets.
        let structured = matches!(self.agent.output, OutputSpec::Structured { .. });
        let is_final = |r: &&Recorded| structured && r.item.name == crate::agent::FINAL_RESULT_TOOL;
        let output_rejected = (recorded.iter().filter(is_final))
            .any(|r| matches!(r.settled.outcome, ToolOutcome::ToolError(_)));
        let tools = recorded.iter().filter(|r| !is_final(r));
        let mut exceeded = self.tool_retries.step(
            tools.map(|r| (r.item.name.as_str(), r.settled.retry)),
            &self.agent.tools,
            self.agent.retries.tools,
        );
        let accepted = acting.accepted.first_key_value().map(|(c, v)| (*c, *v));
        if output_rejected && accepted.is_none() {
            self.output_retries += 1;
            if self.output_retries > self.agent.retries.output {
                exceeded.get_or_insert(RunError::OutputRetriesExceeded {
                    retries: self.agent.retries.output,
                });
            }
        }
        // Keep the singular compatibility projection in model order, while retaining every
        // correlated input requirement independently. No unselected question is answered.
        let mut terminal: Option<Terminal> = None;
        let mut terminal_call = None;
        let mut input_requirements = Vec::new();
        let mut replies = false;
        let mut calls = Vec::with_capacity(recorded.len());
        let mut outputs = Vec::with_capacity(recorded.len());
        for Recorded { id, item, settled } in &recorded {
            let (id, mut item) = (*id, item.clone());
            let outcome = &settled.outcome;
            let shape = tool_outcome::shape(outcome, self.input.mode);
            let text = tool_outcome::delivered(outcome).unwrap_or(shape.text);
            let reply_required = shape.reply_required;
            if let Some(said) = tool_outcome::verbatim(outcome) {
                self.verbatim.push(said.to_owned());
            }
            let this = match outcome {
                _ if accepted.is_some_and(|(call, _)| call == id) => Some(Terminal::Output),
                ToolOutcome::InputRequired { question, verbatim, tool } => {
                    // False distinguishes retained-but-unselected requests from legacy history
                    // which predates explicit correlation metadata.
                    item.extra.insert(
                        crate::request::INPUT_REQUESTED_KEY.into(),
                        serde_json::Value::Bool(false),
                    );
                    let requirement = input_requirement(&self.input.invocation_id, &item, outcome)
                        .expect("an InputRequired outcome retains its call identity");
                    if terminal.is_none() {
                        self.pending_call = Some(requirement.call.clone());
                    }
                    input_requirements.push(requirement);
                    Some(Terminal::Ask(Ask {
                        question: question.clone(),
                        verbatim: *verbatim,
                        answer_tool: tool.clone(),
                    }))
                }
                ToolOutcome::StopResponse => Some(Terminal::Stop),
                _ if tool_outcome::verbatim(outcome).is_some() && !shape.reply_required => {
                    Some(Terminal::Stop)
                }
                _ => None,
            };
            if terminal.is_none() {
                if exceeded.is_none() && matches!(this, Some(Terminal::Ask(_))) {
                    item.extra.insert(
                        crate::request::INPUT_REQUESTED_KEY.into(),
                        serde_json::Value::Bool(true),
                    );
                }
                if this.is_some() {
                    terminal_call = Some(item.call_id.clone());
                }
                terminal = this;
            }
            replies |= reply_required;
            let mut output = FunctionCallOutput::new(
                format!("{}_out", item.id),
                item.call_id.clone(),
                text,
                shape.is_error,
                self.stamp(now),
            );
            output.name = item.name.clone();
            output.reply_required = reply_required;
            calls.push(ChatItem::FunctionCall(item));
            outputs.push(ChatItem::FunctionCallOutput(output));
        }
        self.input_requirements.extend(input_requirements.iter().cloned());
        // Calls retain their dispatch stamps; outputs are stamped when the step settles.
        // One canonical order agrees with chronological voice/storage projections without
        // rewriting those stamps: all calls, then all outputs, each in model order.
        let items: Vec<ChatItem> = calls.into_iter().chain(outputs).collect();
        let editable_continuation = exceeded.is_none() && terminal.is_none();
        let next = match (exceeded, terminal) {
            (Some(error), _) => Next::Fail(error),
            (None, Some(Terminal::Output)) => {
                let (_, validation) = accepted.expect("an accepted candidate is the terminal");
                Next::Output(validation)
            }
            (None, Some(Terminal::Ask(ask))) => Next::Ask(ask),
            (None, Some(Terminal::Stop)) => Next::Stop,
            // A step whose every output asks for no reply ends the turn there.
            (None, None) if !replies => Next::Stop,
            (None, None) => Next::Continue,
        };
        if !self.input.opening.coordinated() {
            self.history.extend(items);
            return self.proceed(next, now, fx);
        }
        let step = StepId(self.mint());
        let continues = matches!(next, Next::Continue);
        if matches!(next, Next::Fail(_)) {
            terminal_call = None;
        }
        fx.push(AgentRunEffect::HandleToolResults {
            step,
            input_requirements,
            items: items.clone(),
            continues,
            terminal_call,
        });
        self.history.extend(items);
        self.phase = Phase::Settling { step, next, calls: recorded, editable_continuation };
    }

    pub(super) fn proceed(&mut self, next: Next, now: Ts, fx: &mut Vec<AgentRunEffect>) {
        match next {
            Next::Fail(error) => self.finish(Err(error), fx),
            Next::Output(validation) => self.complete(Output::Validated(validation), fx),
            Next::Ask(ask) => self.ask_from_tool(ask, now, fx),
            Next::Stop => self.end_without_output(fx),
            Next::Continue => self.next_step(now, fx),
        }
    }
}

/// One reported question, whether its step settles normally or is aborted with siblings live.
/// This is metadata only: neither collection nor cleanup chooses its presentation order.
pub(super) fn input_requirement(
    invocation_id: &str,
    item: &FunctionCall,
    outcome: &ToolOutcome,
) -> Option<lk_agents_abi::agent::run::RunInputRequirement> {
    let ToolOutcome::InputRequired { question, verbatim, tool } = outcome else { return None };
    Some(lk_agents_abi::agent::run::RunInputRequirement {
        invocation_id: invocation_id.to_owned(),
        call: lk_agents_abi::a2a::turn::PendingCall {
            name: item.name.clone(),
            call_id: item.call_id.clone(),
            answer_tool: tool.clone(),
        },
        question: question.clone(),
        verbatim: *verbatim,
    })
}

/// What a settled step leads to.
pub(super) enum Next {
    Fail(RunError),
    Output(ValidationId),
    Ask(Ask),
    Stop,
    Continue,
}

/// What a step ends the invocation with.
enum Terminal {
    Output,
    Ask(Ask),
    Stop,
}
