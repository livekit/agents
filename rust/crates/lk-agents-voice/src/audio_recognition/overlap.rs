//! `OverlapJudge`: the adaptive interruption channel's verdicts on speech over the agent.

use super::{InterruptionSignal, OverlapVerdict};
use crate::agent_activity::pause::AudioInterruptTrigger;
use crate::agent_session::effects::{AudioInterruptGuard, Decision, Effects};
use crate::events::{OverlapInterruptionEvent, SessionEvent};
use lk_agents_abi::Ts;
use lk_agents_abi::turn::detector::ReqId;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum In {
    /// Over agent speech, with the detector live (`_on_start_of_overlap_speech`).
    Started {
        started_at: Ts,
        speech_duration: Duration,
    },
    /// The overlap audio ended: the user stopped (`agent_ended: false`) or the agent did (`true`).
    Ended {
        ended_at: Ts,
        agent_ended: bool,
    },
    /// The detector ruled on `req` (`None`: the open overlap).
    Verdict {
        req: Option<ReqId>,
        verdict: OverlapVerdict,
        user_speaking: bool,
    },
    Boundary {
        armed: bool,
    },
    /// Any user onset: clears the prior backchannel verdict.
    UserOnset {
        agent_speaking: bool,
    },
    /// The question dies with the speech it was about.
    AgentSpeechEnded,
    /// The turn committed or was dropped: turn-scoped interruption state resets once per logical turn.
    TurnClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum OverlapRefusal {
    NoOpenOverlap,
    /// The verdict names an overlap a newer one superseded. `open` is the one still awaiting.
    Superseded {
        open: ReqId,
    },
    /// A backchannel verdict inside the boundary cooldown: terminal for its overlap, not trusted.
    InsideBoundary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlapJudge {
    /// `_interruption_detection_enabled`: whether verdicts set `interruption_detected`.
    detection_enabled: bool,
    judging: Option<ReqId>,
    seq: u64,
    /// The overlap audio is ongoing (`_overlap_open`).
    open: bool,
    /// `_overlap_in_current_turn`; can be true while none is open.
    in_turn: bool,
    backchannel_over_agent: bool,
    interruption_detected: bool,
    boundary_armed: bool,
    /// InterruptByAudioActivity-in is the detector's until `Release`; outlives `judging`.
    guard_taken: bool,
}

impl OverlapJudge {
    pub fn new(detection_enabled: bool) -> Self {
        OverlapJudge {
            detection_enabled,
            judging: None,
            seq: 0,
            open: false,
            in_turn: false,
            backchannel_over_agent: false,
            interruption_detected: false,
            boundary_armed: false,
            guard_taken: false,
        }
    }

    /// `_restore_interruption_by_audio_activity`, once per takeover: back to the session's default,
    /// not `true`.
    fn release_guard(&mut self) -> Effects {
        let mut eff = Effects::new();
        if std::mem::take(&mut self.guard_taken) {
            eff.decide(Decision::AudioInterruptGuard(AudioInterruptGuard::Restore));
        }
        eff
    }

    pub fn open(&self) -> Option<ReqId> {
        self.judging
    }

    /// This turn's speech overlapped agent speech and the detector called it a backchannel.
    pub fn backchannel_over_agent(&self) -> bool {
        self.backchannel_over_agent
    }

    /// The detector's last word on this turn was "interruption".
    pub fn interruption_detected(&self) -> bool {
        self.interruption_detected
    }

    pub fn boundary_armed(&self) -> bool {
        self.boundary_armed
    }

    /// One input, its effects in order. A verdict not applied comes back, for the journal.
    pub fn step(&mut self, input: In) -> (Option<OverlapRefusal>, Effects) {
        let mut eff = Effects::new();
        match input {
            In::Started { started_at, speech_duration } => {
                self.in_turn = true;
                self.open = true;
                // Supersedes any overlap still awaiting a verdict.
                self.seq += 1;
                self.judging = Some(ReqId(self.seq));
                self.guard_taken = true;
                // The detector owns interruption during the overlap, or the VAD would interrupt
                // before the verdict.
                eff.decide(Decision::AudioInterruptGuard(AudioInterruptGuard::Set(false)));
                eff.signal(InterruptionSignal::OverlapStarted { started_at, speech_duration });
            }
            In::Ended { ended_at, agent_ended } => {
                // The overlap ends once; a call can arrive with it already closed.
                if !self.open {
                    return (None, eff);
                }
                self.open = false;
                eff.signal(InterruptionSignal::OverlapEnded { ended_at, agent_ended });
                // The user stopping doesn't close the question; the agent stopping does.
                if agent_ended {
                    self.judging = None;
                    eff.append(self.release_guard());
                }
            }
            In::Verdict { req, verdict, user_speaking } => {
                let Some(live) = self.judging else {
                    return (Some(OverlapRefusal::NoOpenOverlap), eff);
                };
                if req.is_some_and(|r| r != live) {
                    return (Some(OverlapRefusal::Superseded { open: live }), eff);
                }
                // Every verdict is terminal for its overlap, including one the cooldown ignores.
                self.judging = None;
                self.open = false;

                if self.boundary_armed && !verdict.is_interruption {
                    // "falling back to vad".
                    eff.append(self.release_guard());
                    return (Some(OverlapRefusal::InsideBoundary), eff);
                }
                // `on_interruption`: the guard goes back before the interruption. A backchannel verdict
                // keeps it until the agent speech ends, or the VAD would cut what was just cleared.
                if verdict.is_interruption {
                    eff.append(self.release_guard());
                }
                if self.detection_enabled {
                    self.interruption_detected = verdict.is_interruption;
                }
                // "only honor the verdict while this turn's overlap is unresolved".
                if self.in_turn && !verdict.agent_ended {
                    self.backchannel_over_agent = !verdict.is_interruption;
                    // "clear the backchannel audio, but only between segments".
                    if !verdict.is_interruption && !user_speaking {
                        eff.emit(SessionEvent::BackchannelConfirmed);
                    }
                }
                if verdict.is_interruption {
                    let until = verdict.ignore_user_transcript_until;
                    eff.emit(SessionEvent::OverlapInterruption(OverlapInterruptionEvent {
                        ignore_user_transcript_until: until,
                    }));
                    // Permission to *ask*: `decide_interrupt` still rules.
                    eff.decide(Decision::InterruptByAudioActivity(
                        AudioInterruptTrigger::DetectorVerdict {
                            ignore_user_transcript_until: until,
                        },
                    ));
                }
            }
            In::Boundary { armed } => self.boundary_armed = armed,
            In::UserOnset { agent_speaking } => {
                // "every speech onset clears the prior backchannel verdict; an overlap re-derives
                // it"
                self.backchannel_over_agent = false;
                if !agent_speaking {
                    self.in_turn = false;
                }
                self.interruption_detected = false;
            }
            In::AgentSpeechEnded => {
                // A later verdict must not stop what plays next; the guard goes back with the
                // speech.
                self.judging = None;
                self.open = false;
                eff.append(self.release_guard());
            }
            In::TurnClosed => {
                // "reset turn-scoped interruption state once per logical turn (commit or drop)"
                self.backchannel_over_agent = false;
                self.in_turn = false;
            }
        }
        (None, eff)
    }
}
