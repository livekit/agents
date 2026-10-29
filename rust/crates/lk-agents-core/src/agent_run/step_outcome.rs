//! A settled step's outcome: its calls and outputs recorded in model order, and whether the
//! invocation continues, asks, stops or fails.

use lk_agents_abi::agent::OutputSpec;
use lk_agents_abi::agent::run::{Output, RunError};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ChatItem, FunctionCallOutput, Ts};

use super::{AgentRun, Ask, Effect, Phase, StepId, ValidationId};
use crate::tool_calls::Recorded;
use crate::tool_outcome;

impl AgentRun {
    /// Once every call of the step settled: record the calls and outputs in model order, then
    /// continue, ask, or settle.
    pub(super) fn step_settled(&mut self, now: Ts, fx: &mut Vec<Effect>) {
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
        // The first terminal outcome in model order decides; the others are recorded as answered.
        let mut terminal: Option<Terminal> = None;
        let mut replies = false;
        let mut items = Vec::with_capacity(recorded.len() * 2);
        for Recorded { id, item, settled } in &recorded {
            let (id, item) = (*id, item.clone());
            let outcome = &settled.outcome;
            let shape = tool_outcome::shape(outcome, self.input.mode);
            let mut text = tool_outcome::delivered(outcome).unwrap_or(shape.text);
            let mut reply_required = shape.reply_required;
            if let Some(said) = tool_outcome::verbatim(outcome) {
                self.verbatim.push(said.to_owned());
            }
            let this = match outcome {
                _ if accepted.is_some_and(|(call, _)| call == id) => Some(Terminal::Output),
                ToolOutcome::InputRequired { question, verbatim, tool } => {
                    if terminal.is_some() {
                        // Not the question asked: its text is recorded as answered.
                        text = question.clone();
                    }
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
                terminal = this;
            } else if matches!(outcome, ToolOutcome::InputRequired { .. }) {
                reply_required = false;
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
            items.push(ChatItem::FunctionCall(item));
            items.push(ChatItem::FunctionCallOutput(output));
        }
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
        fx.push(Effect::StepSettled { step, items: items.clone(), continues });
        self.history.extend(items);
        self.phase = Phase::Settling { step, next, calls: recorded };
    }

    pub(super) fn proceed(&mut self, next: Next, now: Ts, fx: &mut Vec<Effect>) {
        match next {
            Next::Fail(error) => self.finish(Err(error), fx),
            Next::Output(validation) => self.complete(Output::Validated(validation), fx),
            Next::Ask(ask) => self.ask_from_tool(ask, now, fx),
            Next::Stop => self.complete(Output::Nothing, fx),
            Next::Continue => self.next_step(now, fx),
        }
    }
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
