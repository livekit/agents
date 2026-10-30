//! Cancellation: what cancelled work is still owed, and when the invocation settles anyway.

use lk_agents_abi::Ts;
use lk_agents_abi::agent::run::RunError;
use std::collections::BTreeSet;

use super::{Acting, AgentRun, CallId, CheckpointId, Effect, Event, Phase, RenderId, RequestId};
use super::{ValidationId, end_text};

/// Cancelled work whose settlement is still owed.
#[derive(Default)]
pub(super) struct Owed {
    checkpoint: Option<CheckpointId>,
    request: Option<RequestId>,
    render: Option<RenderId>,
    calls: BTreeSet<CallId>,
    validations: BTreeSet<ValidationId>,
}

impl Owed {
    fn is_empty(&self) -> bool {
        self.checkpoint.is_none()
            && self.request.is_none()
            && self.render.is_none()
            && self.calls.is_empty()
            && self.validations.is_empty()
    }

    fn describe(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(self.checkpoint.map(|c| format!("managed checkpoint {}", c.0)));
        out.extend(self.request.map(|r| format!("llm request {}", r.0)));
        out.extend(self.render.map(|r| format!("instructions render {}", r.0)));
        out.extend(self.calls.iter().map(|c| format!("tool call {}", c.0)));
        out.extend(self.validations.iter().map(|v| format!("output validation {}", v.0)));
        out
    }
}

impl Acting {
    /// Cancels every call and validation still in flight; their settlement is owed.
    pub(super) fn cancel(mut self, owed: &mut Owed, fx: &mut Vec<Effect>) {
        owed.calls.extend(self.calls.cancel(fx));
        for validation in self.finals.into_keys() {
            fx.push(Effect::OutputValidateCancel { validation });
            owed.validations.insert(validation);
        }
    }
}

impl AgentRun {
    /// Cancels everything in flight; settles once it all reported back, or at the deadline.
    pub(super) fn abort(&mut self, error: RunError, now: Ts, fx: &mut Vec<Effect>) {
        let mut owed = Owed::default();
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Idle => {}
            Phase::Checkpoint(pending) => {
                fx.push(Effect::HostCheckpointCancel { checkpoint: pending.id });
                owed.checkpoint = Some(pending.id);
            }
            Phase::Done => {
                self.phase = Phase::Done;
                return;
            }
            phase @ Phase::Aborting { .. } => {
                self.phase = phase;
                return;
            }
            Phase::Rendering(render) => {
                fx.push(Effect::InstructionsCancel { render });
                owed.render = Some(render);
            }
            Phase::Requesting(step, acting) => {
                end_text(&step, true, fx);
                fx.push(Effect::LlmRequestCancel { request: step.request });
                owed.request = Some(step.request);
                acting.cancel(&mut owed, fx);
            }
            Phase::Acting(acting) => acting.cancel(&mut owed, fx),
            Phase::ValidatingText { validation, .. } => {
                fx.push(Effect::OutputValidateCancel { validation });
                owed.validations.insert(validation);
            }
            Phase::Settling { .. } => {}
        }
        if owed.is_empty() {
            return self.finish(Err(error), fx);
        }
        let deadline = now.saturating_add(self.input.cleanup_timeout);
        self.phase = Phase::Aborting { error, deadline, owed };
    }

    /// An event for work cancelled (or finished) earlier: it settles what is owed, and a resource
    /// it returns is released rather than adopted.
    pub(super) fn settle_owed(&mut self, event: Event, fx: &mut Vec<Effect>) {
        if let Event::ToolPrepared { prepared, .. } = &event {
            fx.push(Effect::ToolArgumentsRelease { prepared: *prepared });
        }
        let Phase::Aborting { owed, .. } = &mut self.phase else { return };
        match event {
            Event::HostCheckpointDone { checkpoint, .. } if owed.checkpoint == Some(checkpoint) => {
                owed.checkpoint = None;
            }
            Event::InstructionsRendered { render, .. } if owed.render == Some(render) => {
                owed.render = None
            }
            Event::LlmChunk { request, chunk } if owed.request == Some(request) => {
                if let Some(usage) = chunk.usage.as_deref() {
                    self.ledger.report(request.0, usage.into());
                }
            }
            Event::LlmEnded { request } if owed.request == Some(request) => {
                owed.request = None;
                self.ledger.finish(request.0, true);
            }
            Event::LlmFailed { request, .. } if owed.request == Some(request) => {
                owed.request = None;
                self.ledger.finish(request.0, false);
            }
            Event::ToolPrepared { call, .. }
            | Event::ToolPrepareFailed { call, .. }
            | Event::ReadBackRendered { call, .. }
            | Event::ToolSettled { call, .. }
            | Event::ToolLaunchFailed { call, .. } => {
                owed.calls.remove(&call);
            }
            Event::OutputValidated { validation, .. } => {
                owed.validations.remove(&validation);
            }
            _ => {}
        }
        if owed.is_empty() {
            let Phase::Aborting { error, .. } = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                unreachable!()
            };
            self.finish(Err(error), fx);
        }
    }

    pub(super) fn wake(&mut self, now: Ts, fx: &mut Vec<Effect>) {
        let due = matches!(&self.phase, Phase::Aborting { deadline, .. } if *deadline <= now);
        if !due {
            return;
        }
        let Phase::Aborting { error, owed, .. } = std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            unreachable!()
        };
        fx.push(Effect::IncompleteCleanup { pending: owed.describe() });
        // Its cleanup is still outstanding: the usage stays unsettled.
        fx.push(Effect::Finished(Err(error)));
        self.phase = Phase::Done;
    }
}
