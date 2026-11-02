//! User-turn recognition domain: configuration, commit guards and diagnostic records.
//! [`machine::AudioRecognition`] owns live turn state and transitions; [`effects`] declares
//! its ordered work, [`observation`] records facts, and [`overlap`] supplies overlap judgments.

use crate::agent_activity::false_interruption::count_min_words;
use crate::agent_session::effects::Timer;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{ChatContext, EotPrediction, Ts};
use std::time::Duration;

pub mod effects;
pub mod machine;
pub mod observation;
pub mod overlap;

pub use machine::AudioRecognition;

/// Raw VAD silence before the streaming detector is asked.
pub const MIN_SILENCE_DURATION: Duration = Duration::from_millis(200);

/// A decided turn parked on the end-of-turn hook until the host answers or the bound passes.
pub(super) struct PendingTurn {
    pub(super) info: EndOfTurnInfo,
    /// `on_user_turn_completed_delay` is measured from here.
    pub(super) issued_at: Ts,
    /// `None` when no host runs hooks.
    pub(super) hook: Option<crate::agent_session::HookId>,
}

impl PendingTurn {
    pub(super) fn finished(mut self, now: Ts) -> EndOfTurnInfo {
        self.info.metrics.on_user_turn_completed_delay =
            Some(now.saturating_duration_since(self.issued_at));
        self.info
    }
}

/// The hook's edits as one request applies them. `StopResponse` never reaches here.
#[derive(Debug, Default)]
pub(super) struct TurnEdits {
    pub(super) instructions: Option<String>,
    pub(super) chat_ctx: Option<ChatContext>,
    pub(super) user_text: Option<String>,
}

pub(super) struct PendingCommit {
    pub(super) skip_reply: bool,
    pub(super) call: Option<crate::agent_session::RequestId>,
    pub(super) close: bool,
}

/// Everything `on_end_of_turn` reads before deciding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommitInputs<'a> {
    /// `self._scheduling_paused`.
    pub scheduling_paused: bool,
    /// `self._session._closing`; read only inside the paused branch.
    pub closing: bool,
    /// `self.stt is not None`.
    pub stt_enabled: bool,
    /// `self._turn_detection != "manual"`.
    pub manual_turn_detection: bool,
    pub current_speech: Option<CommitSpeech>,
    /// `interruption["min_words"]`. `0` disables the guard.
    pub min_interruption_words: usize,
    /// `info.new_transcript`.
    pub new_transcript: &'a str,
    /// `isinstance(self.llm, llm.RealtimeModel)`.
    pub realtime_llm: bool,
    /// `self._rt_turn_detection_enabled`.
    pub rt_turn_detection: bool,
    /// `self._interruption_detection_enabled`.
    pub interruption_detection: bool,
    /// `info.backchannel_over_agent`.
    pub backchannel_over_agent: bool,
    /// `info.skip_reply`, after AMD has had its say.
    pub skip_reply: bool,
    /// Started by a final with no onset since the previous commit: a late final.
    pub late: bool,
}

/// The three guard-relevant facts about `self._current_speech`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitSpeech {
    pub allow_interruptions: bool,
    pub interrupted: bool,
}

/// Which branch of `on_end_of_turn` ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum UserTurnCommitVerdict {
    Reply,
    /// Scheduling paused or new turns blocked. When closing, the transcript is still appended so
    /// the user's last words are kept.
    Dropped {
        append_user_message: bool,
    },
    /// Fewer than `min_words` words. The transcript is retained.
    TooFewWords,
    /// A realtime backchannel: transcript kept, the model's buffered audio cleared.
    RealtimeBackchannel,
    /// A turn with no words that asks for no reply: consumed, with no item.
    Empty,
    /// A final that landed after its turn committed, with no new onset. Commits like `Reply`.
    LateTranscript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserTurnCommitDecision {
    pub verdict: UserTurnCommitVerdict,
    /// The reference's return value.
    pub committed: bool,
    /// `self._cancel_false_interruption_timer()`, except where the reply task returns early.
    pub cancel_false_interruption: bool,
}

/// `on_end_of_turn`'s guards, in the reference's order. Pure.
pub fn decide_commit(i: &CommitInputs<'_>) -> UserTurnCommitDecision {
    if i.scheduling_paused {
        return UserTurnCommitDecision {
            verdict: UserTurnCommitVerdict::Dropped { append_user_message: i.closing },
            committed: true,
            cancel_false_interruption: false,
        };
    }

    // "avoid interruption if the new_transcript is too short"; every conjunct matters.
    if i.stt_enabled
        && !i.manual_turn_detection
        && i.min_interruption_words > 0
        && i.current_speech.is_some_and(|s| s.allow_interruptions && !s.interrupted)
        && count_min_words(i.new_transcript) < i.min_interruption_words
    {
        return UserTurnCommitDecision {
            verdict: UserTurnCommitVerdict::TooFewWords,
            committed: false,
            cancel_false_interruption: false,
        };
    }

    // An unjudged overlap commits (realtime, no STT): interrupting is recoverable, dropping a turn
    // isn't.
    if !i.stt_enabled
        && !i.manual_turn_detection
        && i.realtime_llm
        && !i.rt_turn_detection
        && i.interruption_detection
        && i.backchannel_over_agent
    {
        return UserTurnCommitDecision {
            verdict: UserTurnCommitVerdict::RealtimeBackchannel,
            committed: false,
            cancel_false_interruption: false,
        };
    }

    // `_user_turn_completed_impl`: a turn that asks for no reply keeps no item without words.
    if i.skip_reply && i.new_transcript.is_empty() {
        return UserTurnCommitDecision {
            verdict: UserTurnCommitVerdict::Empty,
            committed: true,
            cancel_false_interruption: false,
        };
    }

    UserTurnCommitDecision {
        verdict: if i.late {
            UserTurnCommitVerdict::LateTranscript
        } else {
            UserTurnCommitVerdict::Reply
        },
        committed: true,
        cancel_false_interruption: !i.skip_reply && !i.rt_turn_detection,
    }
}

impl UserTurnCommitVerdict {
    pub fn replies(self) -> bool {
        match self {
            UserTurnCommitVerdict::Reply | UserTurnCommitVerdict::LateTranscript => true,
            UserTurnCommitVerdict::Dropped { .. }
            | UserTurnCommitVerdict::TooFewWords
            | UserTurnCommitVerdict::RealtimeBackchannel
            | UserTurnCommitVerdict::Empty => false,
        }
    }
}

/// `TurnDetectionMode`, as `AudioRecognition._turn_detection_mode` holds it: resolved once by
/// [`crate::agent_activity::options::resolve_turn_detection_mode`]; changed only by
/// [`AudioRecognition::set_turn_detection_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum TurnDetectionMode {
    /// `"vad"`: a VAD end of speech starts the countdown; STT finals feed it.
    Vad,
    /// `"stt"`: the STT's own end of speech starts the countdown (`"vad"` with no VAD falls here
    /// too).
    Stt,
    /// `"manual"`: nothing but [`TurnCommand::Commit`](crate::agent_session::TurnCommand::Commit)
    /// ends a turn.
    Manual,
    /// `"realtime_llm"`: the realtime model ends turns; server VAD and transcripts feed
    /// [`AudioRecognition::on_server`].
    RealtimeLlm,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Detector {
    /// Every turn commits after `min_delay`.
    None,
    /// `_StreamingTurnDetector`: inference starts on VAD silence, bounded by `prediction_timeout`.
    Streaming { prediction_timeout: Duration },
}

/// `session.options.turn_handling["user_turn_limit"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UserTurnLimit {
    pub max_words: Option<usize>,
    pub max_duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnConfig {
    /// `self._stt is not None`.
    pub stt_enabled: bool,
    /// `self._vad is not None`.
    pub vad_enabled: bool,
    /// `self._using_default_vad`: a default VAD is not trusted for timing.
    pub using_default_vad: bool,
    /// `_turn_detection_mode`.
    pub turn_detection_mode: TurnDetectionMode,
    /// `_interruption_enabled` (an adaptive interruption detector and a VAD), which is also
    /// `_interruption_detection_enabled`, the gate on `_on_end_of_speech`'s `interruption=`.
    pub interruption_enabled: bool,
    /// `session.options.transcription_timeout`. `None` disables it (#6182).
    pub transcription_timeout: Option<Duration>,
    pub user_turn_limit: UserTurnLimit,
    /// `interruption["backchannel_boundary"]`, normalised to a start/end pair.
    pub backchannel_boundary: Option<(Duration, Duration)>,
    pub detector: Detector,
}

/// `_UserTurnTracker`: words and text since the agent last spoke (longer than one turn).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UserTurnTracker {
    pub words: usize,
    pub transcript: String,
    pub started_at: Option<Ts>,
    /// `UserTurnExceeded` was emitted for this accumulation; it doesn't repeat until reset.
    pub exceeded: bool,
}

/// `_EndOfTurnMetrics`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EndOfTurnMetrics {
    pub started_speaking_at: Option<Ts>,
    pub stopped_speaking_at: Option<Ts>,
    pub transcription_delay: Option<Duration>,
    pub end_of_turn_delay: Option<Duration>,
    /// How long `on_user_turn_completed` ran; `None` until it answers and without host hooks.
    pub on_user_turn_completed_delay: Option<Duration>,
}

/// `_compute_end_of_turn_metrics`, including its refusal to guess: an anchor that predates the
/// turn start yields `None` rather than an inflated delay (issue #6093).
pub fn compute_end_of_turn_metrics(
    speech_start_time: Option<Ts>,
    last_speaking_time: Option<Ts>,
    last_final_transcript_time: Option<Ts>,
    now: Ts,
) -> EndOfTurnMetrics {
    let (Some(start), Some(stopped), Some(transcribed)) =
        (speech_start_time, last_speaking_time, last_final_transcript_time)
    else {
        return EndOfTurnMetrics::default();
    };
    // stale/out-of-order anchor: stopping to speak cannot predate the turn start
    if stopped < start {
        return EndOfTurnMetrics::default();
    }
    EndOfTurnMetrics {
        started_speaking_at: Some(start),
        stopped_speaking_at: Some(stopped),
        transcription_delay: Some(transcribed.saturating_duration_since(stopped)),
        end_of_turn_delay: Some(now.saturating_duration_since(stopped)),
        on_user_turn_completed_delay: None,
    }
}

/// One committed turn, strictly increasing; a verdict for a turn the machine has left is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnId(pub u64);

/// The turn anchors copied before waiting: "copy the last_speaking_time before awaiting (the
/// value can change)" (`._run_eou_detection`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TurnSnapshot {
    pub last_speaking_time: Option<Ts>,
    pub last_final_transcript_time: Option<Ts>,
    pub speech_start_time: Option<Ts>,
}

/// What one countdown was started with; the delays are read at start (`min_delay` keeps moving).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
    pub snapshot: TurnSnapshot,
    pub skip_reply: bool,
    pub min_delay: Duration,
    pub max_delay: Duration,
    /// Started by a late final for a turn that already ended.
    pub late: bool,
    /// `_user_turn_committed` as the countdown carries it: `true` only when the owner declared the
    /// end (STT end of speech, a commit). A VAD end or a final carries `false`.
    pub declared: bool,
}

/// The streaming detector's answer, mirroring `self._turn_detector_prediction_fut`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prediction {
    /// Nothing requested, or the request was cancelled by new speech.
    Absent,
    Pending(ReqId),
    /// Landed before the turn ended: the countdown resolves without waiting and arms no timeout.
    Ready(EotPrediction),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Phase {
    /// Nothing heard since the last turn was consumed.
    Idle,
    /// `declared`: the owner already declared the end once and the user resumed.
    Speaking { since: Ts, prediction: Prediction, declared: bool },
    /// No countdown could start: the owner hasn't declared the end, or no words yet.
    Ended { anchor: Ts, prediction: Prediction, declared: bool },
    /// Waiting on the detector until `deadline`, the prediction timeout.
    Awaiting { run: Run, req: ReqId, deadline: Ts },
    /// Serving the endpointing delay; `prediction` is reused if a final restarts the countdown.
    Delaying {
        run: Run,
        fire_at: Ts,
        delay: Duration,
        prediction: Option<EotPrediction>,
        threshold: Option<f64>,
    },
    /// A manual commit serving its delay. Authoritative: a VAD onset does not cancel it.
    ManualCommitted { run: Run, fire_at: Ts, speaking: bool },
    /// Handed to the driver as [`effects::AudioRecognitionEffect::EndOfTurn`]; terminal until the verdict for `turn_id`.
    Committed { turn_id: TurnId, speaking: bool },
}

impl Phase {
    fn timer(&self) -> Option<(Timer, Ts)> {
        match self {
            Phase::Idle
            | Phase::Speaking { .. }
            | Phase::Ended { .. }
            | Phase::Committed { .. } => None,
            Phase::Awaiting { deadline, .. } => Some((Timer::EotPredictionTimeout, *deadline)),
            Phase::Delaying { fire_at, .. } | Phase::ManualCommitted { fire_at, .. } => {
                Some((Timer::EndpointingDelay, *fire_at))
            }
        }
    }

    /// The endpointing delay being served: its run, and whether the commit finds the user speaking.
    fn delay(&self) -> Option<(Run, bool)> {
        match *self {
            Phase::Delaying { run, .. } => Some((run, false)),
            Phase::ManualCommitted { run, speaking, .. } => Some((run, speaking)),
            _ => None,
        }
    }

    /// `_end_of_turn_task is not None and not done()`: a countdown is in flight.
    pub fn is_running(&self) -> bool {
        match self {
            Phase::Awaiting { .. } | Phase::Delaying { .. } | Phase::ManualCommitted { .. } => true,
            Phase::Idle
            | Phase::Speaking { .. }
            | Phase::Ended { .. }
            | Phase::Committed { .. } => false,
        }
    }
}

/// `_EndOfTurnInfo`: the machine's half; the driver adds the peer half and runs [`decide_commit`].
#[derive(Debug, Clone, PartialEq)]
pub struct EndOfTurnInfo {
    pub turn_id: TurnId,
    pub owner: TurnDetectionMode,
    /// `self._stt is not None`, as the machine believes it.
    pub stt_enabled: bool,
    /// A reply was already triggered and should be skipped after end-of-turn detection.
    pub skip_reply: bool,
    pub new_transcript: String,
    pub transcript_confidence: f64,
    pub metrics: EndOfTurnMetrics,
    /// The speech overlapped the agent and was classified a backchannel.
    pub backchannel_over_agent: bool,
    /// A late final for a turn that already ended.
    pub late: bool,
    /// The countdown's anchors: "only reset if there is no new speech" compares against them.
    pub snapshot: TurnSnapshot,
}

pub use lk_agents_abi::turn::interruption::{InterruptionSignal, OverlapVerdict};

pub use lk_agents_abi::WallAnchor;

/// The realtime model's own turn taking, as this machine hears it
/// ([`TurnDetectionMode::RealtimeLlm`]).
#[derive(Debug, Clone, PartialEq)]
pub enum ServerSpeech {
    /// `input_speech_started`.
    Started,
    /// `input_speech_stopped`.
    Stopped,
    /// `input_audio_transcription`: the server's transcript of the user's words, and the id of
    /// the item it transcribes (`UserInputTranscribedEvent.item_id`).
    Transcription { text: String, is_final: bool, item_id: String },
}
