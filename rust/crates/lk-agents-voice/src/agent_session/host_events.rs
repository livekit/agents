//! The host event stream: what the driver hands its host, in order, and each item's wire record.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::commands::HookId;
use crate::events::{ErrorEvent, SessionEvent};
use crate::filler_scheduler::FillerId;
use lk_agents_abi::voice::events::HostWire;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookKind {
    OnEnter,
    OnExit,
    OnUserTurnCompleted,
    OnUserTurnExceeded,
    FillerSource,
    FunctionToolsExecuted,
}

impl HookKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HookKind::OnEnter => "on_enter",
            HookKind::OnExit => "on_exit",
            HookKind::OnUserTurnCompleted => "on_user_turn_completed",
            HookKind::OnUserTurnExceeded => "on_user_turn_exceeded",
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
    /// The decided turn parks until answered. `chat_ctx` is a copy of `agent`'s own context
    /// (`Agent.chat_ctx`) for the hook to edit for this reply.
    OnUserTurnCompleted {
        id: HookId,
        agent: String,
        chat_ctx: Vec<lk_agents_abi::ChatItem>,
        turn: crate::audio_recognition::TurnId,
        new_transcript: String,
        transcript_confidence: f64,
    },
    /// The user turn passed its limit with no reply on the way. Nothing waits on it; a turn that
    /// passes the limit again while it runs is not handed over.
    OnUserTurnExceeded {
        id: HookId,
        agent: String,
        event: lk_agents_abi::voice::events::UserTurnExceededEvent,
    },
    FillerSource {
        id: HookId,
        filler: FillerId,
        step: u32,
        /// The host's callable.
        source: Option<lk_agents_abi::HostRef>,
    },
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
            | HookCall::OnUserTurnExceeded { id, .. }
            | HookCall::FillerSource { id, .. }
            | HookCall::FunctionToolsExecuted { id, .. } => *id,
        }
    }

    pub fn kind(&self) -> HookKind {
        match self {
            HookCall::OnEnter { .. } => HookKind::OnEnter,
            HookCall::OnExit { .. } => HookKind::OnExit,
            HookCall::OnUserTurnCompleted { .. } => HookKind::OnUserTurnCompleted,
            HookCall::OnUserTurnExceeded { .. } => HookKind::OnUserTurnExceeded,
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
    /// Nothing will name the speech again: the host's object for it is let go once every event
    /// before this one has been delivered.
    SpeechRetired(SpeechId),
}

impl HostEvent {
    /// `anchor` turns monotonic stamps into epoch seconds; `speeches` hands back the host's object
    /// for a speech an event names. A hook call has no wire: the host's hooks answer it
    /// (`SessionHooks`).
    pub fn wire(&self, anchor: lk_agents_abi::WallAnchor, speeches: Speeches) -> Option<HostWire> {
        Some(match self {
            HostEvent::Session(ev) => ev.wire(anchor, speeches),
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
                crate::room_io::RoomIoEvent::KeyPressed(_) => return None,
            },
            HostEvent::Hook(_) | HostEvent::SpeechRetired(_) => return None,
        })
    }
}

/// The host's object for a speech the runtime holds.
pub type Speeches<'a> =
    &'a dyn Fn(SpeechId) -> Arc<dyn lk_agents_abi::voice::speech_handle::Speech>;

impl SessionEvent {
    pub fn wire(&self, anchor: lk_agents_abi::WallAnchor, speeches: Speeches) -> HostWire {
        use crate::events::{RunFailure, RunOutcome};
        use lk_agents_abi::metrics::base::AgentMetrics;
        use lk_agents_abi::voice::events as w;
        let speech_id = |id: SpeechId| id.to_string();
        // `created_at` is the host's stamp (`#[binding(skip)]`): inert here.
        let host_stamp = 0.0;
        match self {
            SessionEvent::HistoryReloaded { items } => {
                HostWire::HistoryReloaded(w::HistoryReloadedWire { items: items.clone() })
            }
            // `alternatives[0]`'s text, speaker and language; an unnamed language is none.
            SessionEvent::UserInputTranscribed(t) => {
                let best = t.ev.best_alternative();
                HostWire::UserTranscript(w::UserInputTranscribed {
                    transcript: best.map(|a| a.text.clone()).unwrap_or_default(),
                    is_final: t.is_final,
                    item_id: t.item_id.clone(),
                    speaker_id: best.and_then(|a| a.speaker_id.clone()),
                    language: best
                        .map(|a| a.language.to_string())
                        .filter(|language| !language.is_empty()),
                    created_at: host_stamp,
                })
            }
            SessionEvent::StartOfSpeech(e) => HostWire::StartOfSpeech(*e),
            SessionEvent::EndOfSpeech(e) => HostWire::EndOfSpeech(*e),
            SessionEvent::SpeechCreated { speech, kind, user_initiated } => {
                HostWire::SpeechCreated(w::SpeechCreated {
                    user_initiated: *user_initiated,
                    source: kind.as_source_str().to_owned(),
                    speech_handle: speeches(*speech),
                })
            }
            SessionEvent::Diagnostic(d) => HostWire::Diagnostic(d.wire()),
            SessionEvent::FunctionToolsExecuted(e) => {
                HostWire::FunctionToolsExecuted((**e).clone())
            }
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
            SessionEvent::SpanStarted(e) => HostWire::SpanStarted((**e).clone()),
            SessionEvent::SpanEnded(e) => HostWire::SpanEnded((**e).clone()),
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
                })
            }
            SessionEvent::AmdResult { verdict } => HostWire::AmdResult(verdict.clone()),
            SessionEvent::ConversationItemAdded { item, agent, visibility, run, .. } => {
                HostWire::ItemAdded(w::ItemAddedWire {
                    item: (**item).clone(),
                    agent: agent.clone(),
                    visibility: visibility.as_str().to_owned(),
                    run: run.as_ref().and_then(|r| r.host.clone()),
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
                HostWire::RunStarted(w::RunStartedWire { run: run.host.clone() })
            }
            SessionEvent::RunItemAdded { run, item } => {
                HostWire::RunItemAdded(w::RunItemAddedWire {
                    run: run.host.clone(),
                    item: (**item).clone(),
                })
            }
            SessionEvent::RunRefused { cause, host } => HostWire::RunRefused(w::RunRefusedWire {
                run: host.clone(),
                cause: cause.as_str().to_owned(),
            }),
            // What `RunResult._mark_done` reads; the binding resolves the refs off its side table.
            SessionEvent::RunSettled { run, outcome, speech } => {
                let (final_output, run_error) = match outcome {
                    RunOutcome::Completed { final_output } => (final_output.clone(), None),
                    RunOutcome::Failed(RunFailure::Run(err)) => (None, Some(err.clone())),
                    RunOutcome::InputRequired { .. }
                    | RunOutcome::Failed(RunFailure::Speech)
                    | RunOutcome::Closed => (None, None),
                };
                let question = match outcome {
                    RunOutcome::InputRequired { question, verbatim, answer_tool } => {
                        Some(w::RunQuestion {
                            question: question.clone(),
                            verbatim: *verbatim,
                            answer_tool: answer_tool.clone(),
                        })
                    }
                    _ => None,
                };
                HostWire::RunSettled(w::RunSettledWire {
                    run: run.host.clone(),
                    outcome: outcome.as_str().to_owned(),
                    final_output,
                    question,
                    run_error,
                    speech_handle: speech.map(speeches),
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
                error: error.as_ref().map(ErrorEvent::wire),
            }),
            SessionEvent::OverlappingSpeech(event) => {
                HostWire::OverlappingSpeech(w::OverlappingSpeechWire { event: (**event).clone() })
            }
            SessionEvent::MetricsCollected { metrics, speech } => {
                let leg = matches!(**metrics, AgentMetrics::Vad(_)).then(|| "vad".to_owned());
                HostWire::MetricsCollected(w::MetricsCollectedWire {
                    metrics: (**metrics).clone(),
                    speech_id: speech.map(speech_id),
                    leg,
                })
            }
            SessionEvent::Error(e) => HostWire::Error(e.wire()),
            SessionEvent::ToolExecutionUpdated(update) => {
                HostWire::ToolExecutionUpdated(w::ToolExecutionUpdated {
                    update: update.wire(),
                })
            }
        }
    }
}

impl ErrorEvent {
    /// `error`'s payload, and `close`'s for the error that closed the session.
    fn wire(&self) -> lk_agents_abi::voice::events::SessionErrorWire {
        lk_agents_abi::voice::events::SessionErrorWire {
            source: self.source.as_str().to_owned(),
            recoverable: self.recoverable,
            label: match self.error {
                lk_agents_abi::errors::ProviderError::Host { .. } => "host",
                _ => "engine",
            }
            .to_owned(),
            error: self.error.clone(),
            speech_id: self.speech.map(|id| id.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{ErrorSource, UserTranscriptEvent};
    use lk_agents_abi::voice::events::CloseReason;
    use lk_agents_abi::{SpeechData, SpeechEvent, SpeechEventType};

    fn anchor() -> lk_agents_abi::WallAnchor {
        lk_agents_abi::WallAnchor { wall: 0.0, at: lk_agents_abi::Ts::ZERO }
    }

    // Catches: `user_input_transcribed` naming no speaker and no language whatever the STT said
    // (the reference reads both off `alternatives[0]`), as the e2e suite found on LiveKit Cloud.
    #[test]
    fn a_transcript_names_its_speaker_and_language() {
        let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
        let mut alt = SpeechData::new("en", "hello");
        alt.speaker_id = Some("S1".into());
        ev.alternatives = vec![alt];
        let t = UserTranscriptEvent { ev, is_final: true, speaking: None, item_id: None };
        let HostWire::UserTranscript(w) =
            SessionEvent::UserInputTranscribed(Box::new(t)).wire(anchor(), &|_| unreachable!())
        else {
            panic!("not a transcript")
        };
        assert_eq!((w.speaker_id.as_deref(), w.language.as_deref()), (Some("S1"), Some("en")));
        assert_eq!(w.item_id, None, "the reference's STT path names no item");
    }

    // Catches: `close.error` always `None`, though the reference closes with the error record of
    // the unrecoverable `error` event.
    #[test]
    fn an_error_close_carries_its_error() {
        let error = ErrorEvent {
            source: ErrorSource::Llm,
            recoverable: false,
            error: lk_agents_abi::errors::ApiError::new("boom").into(),
            speech: None,
        };
        let close = SessionEvent::Close { reason: CloseReason::Error, error: Some(error) };
        let HostWire::Close(w) = close.wire(anchor(), &|_| unreachable!()) else { panic!("not a close") };
        let e = w.error.expect("the close's error");
        assert_eq!((e.source.as_str(), e.recoverable), ("llm_error", false));
    }
}
