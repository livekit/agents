use crate::agent_session::effects::Timer;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{ChatContext, EotPrediction, Ts};
use lk_agents_core::tokenize::spoken_words;
use std::time::Duration;

pub mod effects;
pub mod machine;
pub mod observation;
pub mod overlap;

pub use effects::{EouDetected, EouPredicted};
pub use machine::AudioRecognition;

/// Raw VAD silence before the streaming detector is asked.
pub const MIN_SILENCE_DURATION: Duration = Duration::from_millis(200);

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

/// Everything the end-of-turn commit reads before deciding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommitInputs<'a> {
    pub scheduling_paused: bool,
    /// Read only inside the paused branch.
    pub closing: bool,
    pub stt_enabled: bool,
    pub manual_turn_detection: bool,
    pub current_speech: Option<CommitSpeech>,
    /// `0` disables the guard.
    pub min_interruption_words: usize,
    pub new_transcript: &'a str,
    pub realtime_llm: bool,
    pub rt_turn_detection: bool,
    pub interruption_detection: bool,
    pub backchannel_over_agent: bool,
    /// After AMD has had its say.
    pub skip_reply: bool,
    /// Started by a final with no onset since the previous commit: a late final.
    pub late: bool,
}

/// The guard-relevant facts about the current speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitSpeech {
    pub allow_interruptions: bool,
    pub interrupted: bool,
}

/// Which branch of the end-of-turn commit ran.
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
    pub committed: bool,
    pub cancel_false_interruption: bool,
}

/// The end-of-turn commit guards, in order.
pub fn decide_commit(i: &CommitInputs<'_>) -> UserTurnCommitDecision {
    if i.scheduling_paused {
        return UserTurnCommitDecision {
            verdict: UserTurnCommitVerdict::Dropped { append_user_message: i.closing },
            committed: true,
            cancel_false_interruption: false,
        };
    }

    if i.stt_enabled
        && !i.manual_turn_detection
        && i.min_interruption_words > 0
        && i.current_speech.is_some_and(|s| s.allow_interruptions && !s.interrupted)
        && spoken_words(i.new_transcript) < i.min_interruption_words
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

    // A turn that asks for no reply keeps no item without words.
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

/// Resolved once by [`crate::agent_session::options::resolve_turn_detection_mode`]; changed only
/// by [`AudioRecognition::set_turn_detection_mode`].
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
    /// Inference starts on VAD silence, bounded by `prediction_timeout`.
    Streaming { prediction_timeout: Duration },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UserTurnLimit {
    pub max_words: Option<usize>,
    pub max_duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnConfig {
    pub stt_enabled: bool,
    pub vad_enabled: bool,
    /// A default VAD is not trusted for timing.
    pub using_default_vad: bool,
    pub turn_detection_mode: TurnDetectionMode,
    /// An adaptive interruption detector and a VAD are both present.
    pub interruption_enabled: bool,
    /// `None` disables it.
    pub transcription_timeout: Option<Duration>,
    pub user_turn_limit: UserTurnLimit,
    /// Normalised to a start/end pair.
    pub backchannel_boundary: Option<(Duration, Duration)>,
    pub detector: Detector,
}

/// Words and text since the agent last spoke (longer than one turn).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UserTurnTracker {
    pub words: usize,
    pub transcript: String,
    pub started_at: Option<Ts>,
    /// `UserTurnExceeded` was emitted for this accumulation; it doesn't repeat until reset.
    pub exceeded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EndOfTurnMetrics {
    pub started_speaking_at: Option<Ts>,
    pub stopped_speaking_at: Option<Ts>,
    pub transcription_delay: Option<Duration>,
    pub end_of_turn_delay: Option<Duration>,
    /// How long `on_user_turn_completed` ran; `None` until it answers and without host hooks.
    pub on_user_turn_completed_delay: Option<Duration>,
}

/// An anchor that predates the turn start yields no metrics rather than an inflated delay.
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

/// The turn anchors copied before waiting, since the live values keep changing.
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
    /// `true` only when the owner declared the end (STT end of speech, a commit). A VAD end or a
    /// final carries `false`.
    pub declared: bool,
}

/// The streaming detector's answer.
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

    /// A countdown is in flight.
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

/// The machine's half; the driver adds the peer half and runs [`decide_commit`].
#[derive(Debug, Clone, PartialEq)]
pub struct EndOfTurnInfo {
    pub turn_id: TurnId,
    pub owner: TurnDetectionMode,
    /// As the machine believes it.
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
    /// The countdown's anchors: a reset happens only if no new speech arrived since them.
    pub snapshot: TurnSnapshot,
}

pub use lk_agents_abi::turn::interruption::{InterruptionSignal, OverlapVerdict};

pub use lk_agents_abi::WallAnchor;

/// The realtime model's own turn taking, as this machine hears it
/// ([`TurnDetectionMode::RealtimeLlm`]).
#[derive(Debug, Clone, PartialEq)]
pub enum ServerSpeech {
    Started,
    Stopped,
    /// The server's transcript of the user's words, and the id of the item it transcribes.
    Transcription {
        text: String,
        is_final: bool,
        item_id: String,
    },
}
