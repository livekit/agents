//! The host event stream: what the driver hands its host, in order, and each item's wire record.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::commands::{Command, HookId};
use crate::agent_session::driver::Tx;
use crate::events::SessionEvent;
use crate::filler_scheduler::FillerId;
use lk_agents_abi::voice::events::HostWire;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookKind {
    OnEnter,
    OnExit,
    OnUserTurnCompleted,
    DescribeAgent,
    RenderToolReply,
    FillerSource,
    FunctionToolsExecuted,
}

impl HookKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HookKind::OnEnter => "on_enter",
            HookKind::OnExit => "on_exit",
            HookKind::OnUserTurnCompleted => "on_user_turn_completed",
            HookKind::DescribeAgent => "describe_agent",
            HookKind::RenderToolReply => "render_tool_reply",
            HookKind::FillerSource => "filler_source",
            HookKind::FunctionToolsExecuted => "function_tools_executed",
        }
    }
}

/// Answered by [`Command::HookDone`]; what the hook does arrives tagged `Origin::Hook`.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum HookCall {
    /// Nothing waits on it.
    OnEnter { id: HookId, agent: String },
    /// **Before** scheduling pauses, so a goodbye can still be spoken. The drain waits for it,
    /// bounded.
    OnExit { id: HookId, agent: String },
    /// The decided turn parks until answer or bound.
    OnUserTurnCompleted {
        id: HookId,
        turn: crate::audio_recognition::TurnId,
        new_transcript: String,
        transcript_confidence: f64,
    },
    /// Answered with `HookOutcome::Described`.
    DescribeAgent { id: HookId, agent: lk_agents_abi::HostRef, speech: SpeechId },
    /// Unanswered, the reference's defaults are used.
    RenderToolReply { id: HookId, agent: String, call_ids: Vec<String>, at_tail: bool },
    FillerSource { id: HookId, filler: FillerId, step: u32 },
    /// `function_tools_executed`, emitted by the host; answered with `HookOutcome::FunctionToolsExecuted`.
    /// Unanswered, the event stands as issued.
    FunctionToolsExecuted {
        id: HookId,
        speech: SpeechId,
        event: Box<lk_agents_abi::voice::events::ToolsExecutedEvent>,
    },
}

impl HookCall {
    pub fn id(&self) -> HookId {
        match self {
            HookCall::OnEnter { id, .. }
            | HookCall::OnExit { id, .. }
            | HookCall::OnUserTurnCompleted { id, .. }
            | HookCall::DescribeAgent { id, .. }
            | HookCall::RenderToolReply { id, .. }
            | HookCall::FillerSource { id, .. }
            | HookCall::FunctionToolsExecuted { id, .. } => *id,
        }
    }

    pub fn kind(&self) -> HookKind {
        match self {
            HookCall::OnEnter { .. } => HookKind::OnEnter,
            HookCall::OnExit { .. } => HookKind::OnExit,
            HookCall::OnUserTurnCompleted { .. } => HookKind::OnUserTurnCompleted,
            HookCall::DescribeAgent { .. } => HookKind::DescribeAgent,
            HookCall::RenderToolReply { .. } => HookKind::RenderToolReply,
            HookCall::FillerSource { .. } => HookKind::FillerSource,
            HookCall::FunctionToolsExecuted { .. } => HookKind::FunctionToolsExecuted,
        }
    }
}

/// A hook request asks the host to run user code and answer; it is not just an event to log.
#[derive(Debug, lk_agents_abi::Vocabulary)]
pub enum HostEvent {
    Session(SessionEvent),
    Room(crate::room_io::RoomIoEvent),
    /// On this stream, so its order against the other events is a fact.
    Hook(HookCall),
}

impl HostEvent {
    /// `anchor` turns monotonic stamps into epoch seconds; `api` rides with each speech handle.
    pub fn wire(&self, anchor: lk_agents_abi::WallAnchor, api: &Tx<Command>) -> HostWire {
        match self {
            HostEvent::Session(ev) => ev.wire(anchor, api),
            HostEvent::Room(ev) => match ev {
                crate::room_io::RoomIoEvent::ParticipantLinked(e) => {
                    HostWire::ParticipantLinked(e.clone())
                }
                crate::room_io::RoomIoEvent::ParticipantUnlinked(e) => {
                    HostWire::ParticipantUnlinked(e.clone())
                }
                crate::room_io::RoomIoEvent::TextInput(e) => HostWire::TextInput(e.clone()),
                crate::room_io::RoomIoEvent::DeleteRoomRequested(e) => {
                    HostWire::DeleteRoomRequested(e.clone())
                }
            },
            HostEvent::Hook(call) => call.wire(),
        }
    }
}

impl HookCall {
    pub fn wire(&self) -> HostWire {
        use lk_agents_abi::voice::events as w;
        let hook = self.kind().as_str().to_owned();
        let id = self.id().0;
        match self {
            HookCall::OnEnter { agent, .. } => {
                HostWire::HookOnEnter(w::OnEnterHookWire { hook, id, agent: agent.clone() })
            }
            HookCall::OnExit { agent, .. } => {
                HostWire::HookOnExit(w::OnExitHookWire { hook, id, agent: agent.clone() })
            }
            HookCall::OnUserTurnCompleted { turn, new_transcript, transcript_confidence, .. } => {
                HostWire::HookOnUserTurnCompleted(w::OnUserTurnCompletedHookWire {
                    hook,
                    id,
                    turn: turn.0,
                    new_transcript: new_transcript.clone(),
                    transcript_confidence: *transcript_confidence,
                })
            }
            HookCall::FillerSource { filler, step, .. } => {
                HostWire::HookFillerSource(w::FillerSourceHookWire {
                    hook,
                    id,
                    filler: filler.0,
                    step: *step,
                })
            }
            HookCall::RenderToolReply { agent, call_ids, at_tail, .. } => {
                HostWire::HookRenderToolReply(w::RenderToolReplyHookWire {
                    hook,
                    id,
                    agent: agent.clone(),
                    call_ids: call_ids.clone(),
                    at_tail: *at_tail,
                })
            }
            HookCall::FunctionToolsExecuted { event, .. } => {
                use lk_agents_abi::ChatItem;
                HostWire::HookToolsExecuted(w::ToolsExecutedHookWire {
                    hook,
                    id,
                    function_calls: event.function_calls.iter().cloned().map(ChatItem::FunctionCall).collect(),
                    function_call_outputs: event
                        .function_call_outputs
                        .iter()
                        .cloned()
                        .map(ChatItem::FunctionCallOutput)
                        .collect(),
                    handoff_required: event.handoff_required,
                })
            }
            HookCall::DescribeAgent { agent, speech, .. } => {
                HostWire::HookDescribeAgent(w::DescribeAgentHookWire {
                    hook,
                    id,
                    speech_id: crate::speech_handle::SpeechWireId(*speech).to_string(),
                    agent: agent.clone(),
                })
            }
        }
    }
}

impl SessionEvent {
    pub fn wire(&self, anchor: lk_agents_abi::WallAnchor, api: &Tx<Command>) -> HostWire {
        use crate::events::{RunFailure, RunOutcome};
        use crate::speech_handle::{HostSpeech, SpeechWireId};
        use lk_agents_abi::voice::events as w;
        let speech_id = |id: SpeechId| SpeechWireId(id).to_string();
        // `created_at` is the host's stamp (`#[binding(skip)]`): inert here.
        let host_stamp = 0.0;
        match self {
            SessionEvent::UserInputTranscribed(t) => HostWire::UserTranscript(w::UserInputTranscribed {
                transcript: t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default(),
                is_final: t.is_final,
                item_id: t.item_id.clone(),
                speaker_id: None,
                language: None,
                created_at: host_stamp,
            }),
            SessionEvent::StartOfSpeech(e) => HostWire::StartOfSpeech(*e),
            SessionEvent::EndOfSpeech(e) => HostWire::EndOfSpeech(*e),
            SessionEvent::SpeechCreated { speech, kind, user_initiated } => {
                HostWire::SpeechCreated(w::SpeechCreated {
                    user_initiated: *user_initiated,
                    source: kind.as_source_str().to_owned(),
                    speech_handle: HostSpeech::object(speech, api),
                    created_at: host_stamp,
                })
            }
            SessionEvent::Diagnostic(d) => HostWire::Diagnostic(d.wire()),
            SessionEvent::FunctionToolsExecuted(e) => HostWire::FunctionToolsExecuted((**e).clone()),
            SessionEvent::UserTurnCompleted(e) => {
                HostWire::UserTurnCompleted(w::UserTurnCompletedWire {
                    turn: e.turn.0,
                    new_transcript: e.new_transcript.clone(),
                    transcript_confidence: e.transcript_confidence,
                })
            }
            SessionEvent::EotPrediction(e) => HostWire::EotPrediction(w::EotPrediction {
                probability: e.probability,
                threshold: e.threshold,
                inference_duration: e.inference_duration,
                delay: e.delay.as_secs_f64(),
                created_at: host_stamp,
            }),
            SessionEvent::AgentBackchannelOpportunity(e) => {
                HostWire::AgentBackchannelOpportunity(e.clone())
            }
            SessionEvent::PreemptiveGeneration(e) => HostWire::PreemptiveGeneration(e.clone()),
            SessionEvent::UserTurnExceeded(e) => HostWire::UserTurnExceeded(e.clone()),
            // Wire kind `transcription_timeout`, model discriminant `user_transcription_timeout`.
            // `turn_start` is monotonic, hence the anchor.
            SessionEvent::UserTranscriptionTimeout(e) => {
                HostWire::TranscriptionTimeout(w::UserTranscriptionTimeout {
                    speech_duration: e.speech_duration.as_secs_f64(),
                    vad_speech_started_at: anchor.seconds(e.turn_start),
                    created_at: host_stamp,
                })
            }
            SessionEvent::BackchannelConfirmed => {
                HostWire::BackchannelConfirmed(w::BackchannelConfirmedWire {})
            }
            SessionEvent::OverlapInterruption(e) => HostWire::OverlapInterruption(*e),
            SessionEvent::SpanStarted(e) => HostWire::SpanStarted(w::SpanStartedWire {
                span: e.span,
                start_time: e.start_time.map(|t| anchor.seconds(t)),
            }),
            SessionEvent::SpanEnded(e) => HostWire::SpanEnded(w::SpanEndedWire {
                span: e.span,
                end_time: e.end_time.map(|t| anchor.seconds(t)),
            }),
            SessionEvent::AgentStateChanged { old, new } => {
                HostWire::AgentStateChanged(w::AgentStateChanged {
                    old_state: *old,
                    new_state: *new,
                    created_at: host_stamp,
                })
            }
            // Back-dated to the speech boundary; anchored into a wall float.
            SessionEvent::UserStateChanged { old, new, at } => {
                HostWire::UserStateChanged(w::UserStateChanged {
                    old_state: *old,
                    new_state: *new,
                    created_at: anchor.seconds(*at),
                })
            }
            SessionEvent::AgentFalseInterruption { resumed } => {
                HostWire::AgentFalseInterruption(w::AgentFalseInterruption {
                    resumed: *resumed,
                    created_at: host_stamp,
                    message: (),
                    extra_instructions: None,
                })
            }
            SessionEvent::AmdResult { verdict } => HostWire::AmdResult(verdict.clone()),
            SessionEvent::ConversationItemAdded { item, agent, visibility, run, .. } => {
                HostWire::ItemAdded(w::ItemAddedWire {
                    item: (**item).clone(),
                    agent: agent.clone(),
                    visibility: visibility.as_str().to_owned(),
                    run: run.map(|r| r.0),
                })
            }
            // The host moves these ids between its context projections; no item crosses.
            SessionEvent::ContextCarried { from, to, items } => {
                HostWire::ContextCarried(w::ContextCarriedWire {
                    from_agent: from.clone(),
                    to_agent: to.clone(),
                    items: items.clone(),
                })
            }
            SessionEvent::RunStarted { run } => {
                HostWire::RunStarted(w::RunStartedWire { run: run.0 })
            }
            SessionEvent::RunItemAdded { run, item } => {
                HostWire::RunItemAdded(w::RunItemAddedWire { run: run.0, item: (**item).clone() })
            }
            SessionEvent::RunRefused { cause } => {
                HostWire::RunRefused(w::RunRefusedWire { cause: cause.as_str().to_owned() })
            }
            // What `RunResult._mark_done` reads; the binding resolves the refs off its side table.
            SessionEvent::RunSettled { run, outcome, speech } => {
                let (final_output, error) = match outcome {
                    RunOutcome::Completed { final_output } => (final_output.clone(), None),
                    RunOutcome::Failed(RunFailure::Output(err)) => (None, Some(err.clone())),
                    RunOutcome::Failed(RunFailure::Speech) | RunOutcome::Closed => (None, None),
                };
                HostWire::RunSettled(w::RunSettledWire {
                    run: run.0,
                    outcome: outcome.as_str().to_owned(),
                    final_output,
                    error,
                    speech_handle: speech.as_ref().map(|h| HostSpeech::object(h, api)),
                })
            }
            SessionEvent::SpeechDone { speech, outcome } => {
                HostWire::SpeechDone(w::SpeechDoneWire {
                    speech_id: speech_id(*speech),
                    outcome: outcome.as_str().to_owned(),
                })
            }
            SessionEvent::FloorChanged { holder } => {
                HostWire::FloorChanged(w::FloorChangedWire { speech_id: holder.map(speech_id) })
            }
            SessionEvent::Close { reason, error } => HostWire::Close(w::CloseWire {
                reason: *reason,
                has_error: error.is_some(),
                error: error.clone(),
            }),
            SessionEvent::MetricsCollected { metrics, speech } => {
                HostWire::MetricsCollected(w::MetricsCollectedWire {
                    metrics: (**metrics).clone(),
                    speech_id: speech.map(speech_id),
                })
            }
            SessionEvent::Error(e) => HostWire::Error(w::SessionErrorWire {
                source: e.source.as_str().to_owned(),
                recoverable: e.recoverable,
                label: match e.error {
                    lk_agents_abi::errors::ProviderError::Host { .. } => "host",
                    _ => "engine",
                }
                .to_owned(),
                error: e.error.clone(),
                speech_id: e.speech.map(speech_id),
            }),
            SessionEvent::ToolExecutionUpdated(update) => {
                HostWire::ToolExecutionUpdated(w::ToolExecutionUpdated {
                    update: update.wire(),
                    created_at: host_stamp,
                })
            }
        }
    }
}
