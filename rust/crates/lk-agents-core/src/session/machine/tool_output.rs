use super::*;
use crate::agent_run::ToolOutputId;
use lk_agents_abi::agent::run::SayOutcome;

pub(in crate::session) struct Pending {
    frame: FrameId,
    call: agent_run::CallId,
    stream: bool,
    phase: OutputPhase,
}

enum OutputPhase {
    Delivering(Box<ChatMessage>),
    Committing,
}

fn settled(invocation_id: String, output: ToolOutputId, outcome: SayOutcome) -> SessionEffect {
    SessionEffect::Run {
        invocation_id,
        effect: agent_run::AgentRunEffect::ToolOutputSettled { output, outcome },
    }
}

impl Session {
    pub(super) fn output_pending_for_frame(&self, frame: &FrameId) -> bool {
        self.tool_outputs.values().any(|pending| &pending.frame == frame)
    }

    pub(super) fn request_tool_output(
        &mut self,
        invocation_id: String,
        output: ToolOutputId,
        call: Option<agent_run::CallId>,
        text: String,
        stream: bool,
        now: Ts,
    ) -> Vec<SessionEffect> {
        let key = (invocation_id.clone(), output);
        if self.tool_outputs.contains_key(&key) {
            return Vec::new();
        }
        let planned = (|| {
            if self.closing || matches!(self.phase, Phase::Done) {
                return Err(SayOutcome::Interrupted);
            }
            if self.persistence_failure.is_some() {
                return Err(SayOutcome::Failed);
            }
            // A lost tool operation or body has no live source to speak for.
            let call = call.ok_or(SayOutcome::Failed)?;
            if let Some(invocation) =
                self.phase.invocation_mut().filter(|invocation| invocation.id == invocation_id)
                && let Some(message) = invocation.run.plan_tool_output(call, text.clone(), now)
            {
                return Ok((invocation.frame_id.clone(), call, message));
            }
            let source = self
                .live
                .as_ref()
                .and_then(|live| live.output_source(&(invocation_id.clone(), call)))
                .map(|(frame, _)| frame.clone())
                .ok_or(SayOutcome::Failed)?;
            let id = self.mint("say");
            let at = self.stamp(now);
            Ok((source, call, ChatMessage::text(id, Role::Assistant, text, at)))
        })();
        match planned {
            Ok((frame, call, message)) => {
                self.tool_outputs.insert(
                    key,
                    Pending {
                        frame: frame.clone(),
                        call,
                        stream,
                        phase: OutputPhase::Delivering(Box::new(message.clone())),
                    },
                );
                let mut effects = vec![SessionEffect::Run {
                    invocation_id: invocation_id.clone(),
                    effect: agent_run::AgentRunEffect::DeliverToolOutput {
                        output,
                        message: Box::new(message.clone()),
                    },
                }];
                if stream {
                    self.emit(
                        frame,
                        invocation_id,
                        lk_agents_abi::agent::run::StreamEvent::TextDelta {
                            message_id: message.id.clone(),
                            delta: message.raw_text_content().unwrap_or_default(),
                        },
                        &mut effects,
                    );
                }
                effects
            }
            Err(outcome) => vec![settled(invocation_id, output, outcome)],
        }
    }

    pub(super) fn delivered_tool_output(
        &mut self,
        invocation_id: String,
        output: ToolOutputId,
        outcome: SayOutcome,
        now: Ts,
    ) -> Vec<SessionEffect> {
        let key = (invocation_id.clone(), output);
        let Some(mut pending) = self.tool_outputs.remove(&key) else {
            return Vec::new();
        };
        let OutputPhase::Delivering(message) = pending.phase else {
            self.tool_outputs.insert(key, pending);
            return Vec::new();
        };
        let mut effects = Vec::new();
        if pending.stream {
            self.emit(
                pending.frame.clone(),
                invocation_id.clone(),
                lk_agents_abi::agent::run::StreamEvent::TextEnd {
                    message_id: message.id.clone(),
                    interrupted: outcome != SayOutcome::Delivered,
                },
                &mut effects,
            );
        }
        let recorded = (|| {
            if outcome != SayOutcome::Delivered {
                return Err(outcome);
            }
            // Words the owner can no longer record were cut off by its close, or lost.
            if self.closing || matches!(self.phase, Phase::Done) {
                return Err(SayOutcome::Interrupted);
            }
            if self.persistence_failure.is_some() {
                return Err(SayOutcome::Failed);
            }
            let frame = self.state.frames.get_mut(&pending.frame).ok_or(SayOutcome::Failed)?;
            if !frame.history.iter().any(|item| item.id() == message.id) {
                frame.history.push(ChatItem::Message(*message.clone()));
            }
            record_delivered(frame, *message.clone());
            if !self.state.chat_history.iter().any(|known| known.id == message.id) {
                self.state.chat_history.push(*message.clone());
            }
            if let Some(invocation) =
                self.phase.invocation_mut().filter(|current| current.frame_id == pending.frame)
            {
                if invocation.id == invocation_id {
                    invocation.run.record_tool_output(*message.clone());
                } else {
                    invocation.run.observe_context(std::slice::from_ref(message.as_ref()));
                }
            }
            if let Phase::Releasing(
                invocation,
                InvocationRelease::Settled(result, ..) | InvocationRelease::Recovered(result),
            ) = &mut self.phase
                && invocation.frame_id == pending.frame
            {
                if !result.all_messages.iter().any(|item| item.id() == message.id) {
                    result.all_messages.push(ChatItem::Message(*message.clone()));
                }
                if invocation.id == invocation_id
                    && !result.new_messages.iter().any(|item| item.id() == message.id)
                {
                    result.new_messages.push(ChatItem::Message(*message.clone()));
                }
            }
            Ok(())
        })();
        if let Err(outcome) = recorded {
            effects.push(settled(invocation_id, output, outcome));
            return effects;
        }
        let call = pending.call;
        pending.phase = OutputPhase::Committing;
        self.tool_outputs.insert(key.clone(), pending);
        // Exact Working progress may release its own current model step after actual handoff;
        // the original native body and output remain owned independently of the current turn.
        if self.is_current(&invocation_id)
            && self.live.as_ref().is_some_and(|live| {
                live.output_source(&(invocation_id.clone(), call)).is_some()
                    && !live.detached(&(invocation_id.clone(), call))
            })
        {
            self.run_event(
                agent_run::AgentRunEvent::ToolSettled {
                    call,
                    report: lk_agents_abi::llm::tool_executor::ToolOutcome::Update {
                        message: "The tool provided an exact progress update.".into(),
                        raw: String::new(),
                        reply_required: false,
                    }
                    .into(),
                },
                now,
                &mut effects,
            );
        }
        if self.recording {
            effects.insert(
                0,
                SessionEffect::ToolOutputCheckpoint {
                    invocation_id,
                    output,
                    state: Box::new(self.snapshot()),
                },
            );
        } else {
            self.tool_outputs.remove(&key);
            effects.insert(0, settled(invocation_id, output, SayOutcome::Delivered));
        }
        effects
    }

    pub(super) fn committed_tool_output(
        &mut self,
        invocation_id: String,
        output: ToolOutputId,
        result: Result<(), lk_agents_abi::storage::StorageError>,
        now: Ts,
    ) -> Vec<SessionEffect> {
        let key = (invocation_id.clone(), output);
        if !self
            .tool_outputs
            .get(&key)
            .is_some_and(|pending| matches!(pending.phase, OutputPhase::Committing))
        {
            return Vec::new();
        }
        self.tool_outputs.remove(&key);
        match result {
            Ok(()) => vec![settled(invocation_id, output, SayOutcome::Delivered)],
            // Delivered words whose record could not be checkpointed are not a safe delivery.
            Err(error) => {
                let mut effects = vec![settled(invocation_id, output, SayOutcome::Failed)];
                effects.extend(self.handle(SessionEvent::PersistenceFailed(error), now));
                effects
            }
        }
    }
}
