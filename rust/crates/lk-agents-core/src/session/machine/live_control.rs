use super::*;
use crate::session::live::Observation;
use lk_agents_abi::storage::{StorageError, Unapplied};

impl Session {
    pub fn unowned_live_tasks(&self, tasks: &[String]) -> Vec<String> {
        tasks
            .iter()
            .filter(|task| self.live.as_ref().is_none_or(|live| !live.owns_task(task)))
            .cloned()
            .collect()
    }

    /// An explicit task row cancellation affects only that source's local bodies. Canceling
    /// a RemoteAgent body closes its observer; no downstream business cancellation is inferred.
    pub fn cancel_live_tasks(&mut self, tasks: &[String], now: Ts) -> Vec<SessionEffect> {
        let mut effects = Vec::new();
        let current = self.phase.invocation().map(|invocation| invocation.id.clone());
        let Some(live) = &mut self.live else { return effects };
        for (invocation_id, call) in live.cancel_tasks(tasks, current.as_deref()) {
            effects.push(SessionEffect::Run {
                invocation_id,
                effect: agent_run::AgentRunEffect::ToolCancel { call },
            });
        }
        if live.current_cancelled() {
            match &mut self.phase {
                Phase::Running(_) => {
                    self.run_event(agent_run::AgentRunEvent::Cancel, now, &mut effects)
                }
                Phase::Releasing(
                    _,
                    InvocationRelease::Settled(result, ..) | InvocationRelease::Recovered(result),
                ) => {
                    result.interrupted = true;
                    result.outcome = RunOutcome::NoOutput;
                }
                _ => {}
            }
        }
        self.next_live(now, &mut effects);
        effects
    }

    pub fn live_dialogue_pending(&self) -> bool {
        self.live.as_ref().is_some_and(|live| live.observation.is_some())
    }

    /// The existing owner lends a prospective state for the SAME fenced opaque-turn write.
    /// No actual dialogue is admitted until its exact write acknowledgment activates it.
    pub fn prepare_live_dialogue(
        &mut self,
        source: String,
        messages: Vec<ChatMessage>,
    ) -> Result<SessionSnapshot, StorageError> {
        if self.closing || self.persistence_failure.is_some() || source.is_empty() {
            return Err(StorageError::Invalid("the conversation cannot prepare dialogue".into()));
        }
        let mut preview = self.state.clone();
        let frame = preview.active_frame_id.clone();
        if let Some(invocation) = self.phase.invocation() {
            preview.frames.get_mut(&invocation.frame_id).expect("invoked frame exists").history =
                invocation.run.progress_history();
        }
        observe(&mut preview, &frame, &messages)?;
        let live = self.live.as_mut().ok_or_else(|| {
            StorageError::Invalid("the conversation has no live endpoint owner".into())
        })?;
        if let Some(pending) = &live.observation {
            if pending.source == source && pending.messages == messages {
                return Ok(preview);
            }
            return Err(StorageError::Unavailable(Unapplied::local(
                "another dialogue write is pending",
            )));
        }
        live.observation = Some(Observation { source, frame, messages });
        Ok(preview)
    }

    /// Only the owning A2A commit's ACK admits these words. A failed/closed owner discards the
    /// barrier; native transitions cannot publish a snapshot across this pending atomic write.
    pub fn commit_live_dialogue(
        &mut self,
        source: &str,
        now: Ts,
    ) -> Result<Vec<SessionEffect>, StorageError> {
        let pending = self
            .live
            .as_mut()
            .and_then(|live| live.observation.take())
            .ok_or_else(|| StorageError::Invalid("no dialogue write is pending".into()))?;
        if pending.source != source {
            self.live.as_mut().expect("pending live owner").observation = Some(pending);
            return Err(StorageError::Invalid("the dialogue acknowledgment changed source".into()));
        }
        observe(&mut self.state, &pending.frame, &pending.messages)?;
        if let Some(invocation) = self.phase.invocation_mut() {
            invocation.run.observe_context(&pending.messages);
        }
        if let Phase::Releasing(
            _,
            InvocationRelease::Settled(result, ..) | InvocationRelease::Recovered(result),
        ) = &mut self.phase
        {
            for message in &pending.messages {
                if !result.all_messages.iter().any(|item| item.id() == message.id) {
                    result.all_messages.push(message.clone().into());
                }
            }
        }
        let mut effects = Vec::new();
        self.next_live(now, &mut effects);
        Ok(effects)
    }
}

fn observe(
    state: &mut SessionSnapshot,
    frame: &FrameId,
    messages: &[ChatMessage],
) -> Result<(), StorageError> {
    for message in messages {
        if message.id.is_empty() || !matches!(message.role, Role::User | Role::Assistant) {
            return Err(StorageError::Invalid(
                "opaque context contains a non-dialogue item".into(),
            ));
        }
        if state
            .chat_history
            .iter()
            .find(|seen| seen.id == message.id)
            .is_some_and(|seen| seen.role != message.role || seen.content != message.content)
        {
            return Err(StorageError::Invalid("opaque dialogue changed canonical content".into()));
        }
    }
    let frame = state
        .frames
        .get_mut(frame)
        .ok_or_else(|| StorageError::Invalid("opaque dialogue lost its owning frame".into()))?;
    for message in messages {
        if !frame.history.iter().any(|item| item.id() == message.id) {
            frame.history.push(message.clone().into());
        }
        record_delivered(frame, message.clone());
        if !state.chat_history.iter().any(|seen| seen.id == message.id) {
            state.chat_history.push(message.clone());
        }
    }
    Ok(())
}
