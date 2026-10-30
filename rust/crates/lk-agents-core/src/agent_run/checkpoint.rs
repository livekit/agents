//! Managed-turn checkpoints gate model requests and accepted result delivery.

use super::{AgentRun, CheckpointId, Effect, Phase, Purpose, RunError, RunResult};
use lk_agents_abi::Ts;

pub(super) struct Pending {
    pub(super) id: CheckpointId,
    next: Next,
}

enum Next {
    Request { purpose: Purpose, dynamic: Option<String> },
    Finish(Box<RunResult>),
}

impl AgentRun {
    /// The runtime supplies a managed host checkpoint capability; pure callers can omit it.
    pub fn with_checkpoints(mut self) -> Self {
        self.checkpoints = true;
        self
    }

    pub(super) fn checkpoint_request(
        &mut self,
        purpose: Purpose,
        dynamic: Option<String>,
        effects: &mut Vec<Effect>,
    ) {
        self.checkpoint(Next::Request { purpose, dynamic }, effects);
    }

    pub(super) fn checkpoint_finish(&mut self, result: RunResult, effects: &mut Vec<Effect>) {
        self.checkpoint(Next::Finish(Box::new(result)), effects);
    }

    fn checkpoint(&mut self, next: Next, effects: &mut Vec<Effect>) {
        let id = CheckpointId(self.mint());
        let list_tools = matches!(&next, Next::Request { purpose: Purpose::Turn, .. });
        self.phase = Phase::Checkpoint(Pending { id, next });
        effects.push(Effect::HostCheckpoint { checkpoint: id, list_tools });
    }

    pub(super) fn checkpoint_done(
        &mut self,
        checkpoint: CheckpointId,
        result: Result<bool, String>,
        tools: Option<Vec<lk_agents_abi::agent::FunctionToolInfo>>,
        now: Ts,
        effects: &mut Vec<Effect>,
    ) {
        if !matches!(&self.phase, Phase::Checkpoint(pending) if pending.id == checkpoint) {
            return;
        }
        let Phase::Checkpoint(pending) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            unreachable!()
        };
        match result {
            Ok(false) => match pending.next {
                Next::Request { purpose, dynamic } => {
                    if let Some(tools) = tools {
                        self.agent.tools = tools;
                        if let Err(error) = crate::agent::validate(&self.agent) {
                            return self
                                .finish_now(Err(RunError::Host(error.to_string())), effects);
                        }
                    }
                    self.start_request(purpose, dynamic, now, effects)
                }
                Next::Finish(result) => self.finish_now(Ok(*result), effects),
            },
            Ok(true) => self.complete_interrupted(effects),
            Err(error) => self.finish_now(Err(RunError::Host(error)), effects),
        }
    }
}
