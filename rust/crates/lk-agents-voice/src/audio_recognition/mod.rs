//! `voice/audio_recognition.py`: `AudioRecognition`, the user's turn folded from the STT and VAD
//! streams, and the types it decides with.

use crate::agent_activity::false_interruption::count_min_words;
use crate::agent_activity::options::{DetectorKind, LegSet, SessionConfig};
use crate::agent_session::effects::{Effects, TurnTimer};
use crate::audio_recognition::overlap::OverlapJudge;
use crate::endpointing::Endpointing;
use crate::events::{AgentBackchannelOpportunityEvent, SessionEvent, UserTranscriptEvent};
use lk_agents_abi::tokenize::language::{LanguageCode, ThresholdOptions};
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{ChatContext, EotPrediction, SpeechEvent, Ts};
use std::collections::VecDeque;
use std::time::Duration;

mod end_of_turn;
pub mod overlap;
pub mod primary_speaker;
mod speech;

const FRESH_FINAL: Duration = Duration::from_millis(500);

/// Raw VAD silence before the streaming detector is asked (`base.MIN_SILENCE_DURATION_MS`).
pub const MIN_SILENCE_DURATION: Duration = Duration::from_millis(200);

/// `audio_recognition.MIN_LANGUAGE_DETECTION_LENGTH`, in **characters** (Python's `len()` on a
/// `str`).
const MIN_LANGUAGE_DETECTION_LENGTH: usize = 5;

/// Codes that say nothing about the language; they never overwrite a real detection.
const NON_SPECIFIC_LANGUAGE_CODES: [&str; 2] = ["auto", "multi"];

/// `voice/audio_recognition.py`'s `AudioRecognition`: the STT and VAD streams' events folded into
/// the user's turn, the end-of-turn countdown and its detector, the adaptive interruption channel,
/// and a host `commit_user_turn()` waiting for its transcript.
pub struct AudioRecognition {
    /// `_turn_detector`; `None` commits turns on silence alone.
    eot: Option<std::sync::Arc<lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>>>,
    /// The interruption detector's port; `None` is VAD-only interruption.
    interruption:
        Option<std::sync::Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>,
    /// `_commit_user_turn_atask` waiting for the final transcript.
    pending_commit: Option<PendingCommit>,
    cfg: TurnConfig,
    phase: Phase,
    endpointing: Endpointing,
    thresholds: ThresholdOptions,
    /// Pairs wall and monotonic clocks so server onsets map without reading a clock.
    pub(crate) wall_anchor: Option<WallAnchor>,

    // --- transcript accumulation -------------------------------------------------------------
    audio_transcript: String,
    interim_transcript: String,
    preflight_transcript: String,
    final_confidences: Vec<f64>,
    last_language: Option<LanguageCode>,

    // --- turn anchors ------------------------------------------------------------------------
    last_final_transcript_time: Option<Ts>,
    last_speaking_time: Option<Ts>,
    speech_start_time: Option<Ts>,
    /// `_user_turn_start`; also stands for "the `user_turn` span is open".
    user_turn_start: Option<Ts>,

    // --- independent facts -------------------------------------------------------------------
    vad_speech_started: bool,
    /// Written only by `_on_start_of_agent_speech`, from the driver's first-frame sequencer.
    agent_speaking: bool,
    agent_speech_started_at: Option<Ts>,

    // --- the hold-flush buffer ---------------------------------------------------------------
    /// `_ignore_user_transcript_until`.
    ignore_user_transcript_until: Option<Ts>,
    transcript_buffer: VecDeque<SpeechEvent>,
    /// `_STTPipeline.input_started_at`, owned by the driver: the STT stream's anchor, stamped on
    /// the first frame and reset on stream swap; every hold-window test is relative to it.
    pub(crate) input_started_at: Option<Ts>,
    /// `_adaptive_interruption_active`: the channel is open. Unlike the static config flag,
    /// teardown clears this one.
    interruption_active: bool,

    // --- adaptive interruption ---------------------------------------------------------------
    overlap: OverlapJudge,

    // --- user turn limit ---------------------------------------------------------------------
    turn_tracker: UserTurnTracker,
    turn_limit_timer_armed: bool,

    // --- transcription timeout ---------------------------------------------------------------
    turn_speech_duration: Duration,
    turn_transcript_received: bool,
    transcription_timeout_armed: bool,

    // --- the close ---------------------------------------------------------------------------
    /// `on_closing` ran: late words still accumulate for the trailing commit, but nothing arms a
    /// timer, decides a turn or reads the VAD.
    closed: bool,

    // --- ids and dedup -----------------------------------------------------------------------
    /// The phase timer being handled: `set_phase` must not cancel it (it is no longer armed).
    fired: Option<TurnTimer>,
    eot_seq: u64,
    turn_seq: u64,
    /// `self._last_emitted_prediction`: one prediction is announced once, compared by value.
    last_emitted: Option<EotPrediction>,
    /// A backchannel opportunity whose countdown commits in the same cycle, held for the driver's
    /// verdict: released on a refusal or `Dropped`, dropped on a commit that replies.
    deferred_opportunity: Option<AgentBackchannelOpportunityEvent>,
}

impl AudioRecognition {
    pub fn new(cfg: &SessionConfig, legs: &LegSet, thresholds: ThresholdOptions) -> Self {
        let i = &cfg.turn.interruption;
        let turn = TurnConfig {
            stt_enabled: legs.stt,
            vad_enabled: legs.vad,
            using_default_vad: legs.using_default_vad,
            turn_detection_mode: cfg.turn.owner,
            // Without an adaptive detector, transcripts over agent speech reach the turn at once.
            interruption_enabled: i.detection_enabled,
            transcription_timeout: cfg.timers.transcription_timeout,
            user_turn_limit: UserTurnLimit {
                max_words: cfg.turn.user_turn_limit.max_words.map(|w| w as usize),
                max_duration: cfg.turn.user_turn_limit.max_duration,
            },
            backchannel_boundary: i.backchannel_boundary,
            detector: match legs.detector {
                None => Detector::None,
                Some(DetectorKind::Batch) => Detector::Batch,
                Some(DetectorKind::Streaming) => {
                    Detector::Streaming { prediction_timeout: cfg.turn.endpointing.max_delay }
                }
            },
        };
        AudioRecognition {
            eot: None,
            interruption: None,
            pending_commit: None,
            overlap: OverlapJudge::new(turn.interruption_enabled),
            cfg: turn,
            phase: Phase::Idle,
            endpointing: Endpointing::new(&cfg.turn.endpointing),
            thresholds,
            wall_anchor: None,
            audio_transcript: String::new(),
            interim_transcript: String::new(),
            preflight_transcript: String::new(),
            final_confidences: Vec::new(),
            last_language: None,
            last_final_transcript_time: None,
            last_speaking_time: None,
            speech_start_time: None,
            user_turn_start: None,
            vad_speech_started: false,
            agent_speaking: false,
            agent_speech_started_at: None,
            ignore_user_transcript_until: None,
            transcript_buffer: VecDeque::new(),
            input_started_at: None,
            interruption_active: false,
            turn_tracker: UserTurnTracker::default(),
            turn_limit_timer_armed: false,
            turn_speech_duration: Duration::ZERO,
            turn_transcript_received: false,
            transcription_timeout_armed: false,
            closed: false,
            fired: None,
            eot_seq: 0,
            turn_seq: 0,
            last_emitted: None,
            deferred_opportunity: None,
        }
    }

    // --- reads: data out, host never touched ---------------------------------------------------

    /// `_turn_detector`.
    pub(crate) fn eot(
        &self,
    ) -> Option<&std::sync::Arc<lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>>> {
        self.eot.as_ref()
    }

    /// The interruption detector's port.
    pub(crate) fn interruption(
        &self,
    ) -> Option<&std::sync::Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>>
    {
        self.interruption.as_ref()
    }

    /// A host `commit_user_turn()` waits for its transcript.
    pub(crate) fn commit_pending(&self) -> bool {
        self.pending_commit.is_some()
    }

    /// When the waiting commit's `transcript_timeout` runs out.
    pub(crate) fn commit_deadline(&self) -> Option<Ts> {
        self.pending_commit.as_ref().map(|c| c.deadline)
    }

    pub(crate) fn take_pending_commit(&mut self) -> Option<PendingCommit> {
        self.pending_commit.take()
    }

    pub(crate) fn set_pending_commit(&mut self, commit: PendingCommit) {
        self.pending_commit = Some(commit);
    }

    /// A final within the last 0.5 s: a commit takes the turn at once, no flush, no wait.
    pub fn final_is_fresh(&self, now: Ts) -> bool {
        self.last_final_transcript_time
            .is_some_and(|at| now.saturating_duration_since(at) <= FRESH_FINAL)
    }

    pub fn phase(&self) -> &Phase {
        &self.phase
    }

    /// `_current_transcript`: this turn's transcript, interim included (the interruption `min_words`
    /// guard reads this). An empty interim adds no separator.
    pub fn current_transcript(&self) -> String {
        if self.interim_transcript.is_empty() {
            self.audio_transcript.clone()
        } else {
            format!("{} {}", self.audio_transcript, self.interim_transcript).trim().to_owned()
        }
    }

    /// `_end_of_turn_task is not None and not done()`. `Committed` is not open.
    pub fn turn_open(&self) -> bool {
        self.phase.is_running()
    }

    /// `self._speaking`, i.e. `not self._user_silence_ev.is_set()`. Read off the phase.
    pub fn user_speaking(&self) -> bool {
        match &self.phase {
            Phase::Speaking { .. } => true,
            Phase::ManualCommitted { speaking, .. } | Phase::Committed { speaking, .. } => {
                *speaking
            }
            Phase::Idle | Phase::Ended { .. } | Phase::Awaiting { .. } | Phase::Delaying { .. } => {
                false
            }
        }
    }

    pub fn last_speaking_time(&self) -> Option<Ts> {
        self.last_speaking_time
    }

    /// `_last_final_transcript_time`: `commit_user_turn` skips its flush within 0.5 s of it.
    pub fn last_final_transcript_time(&self) -> Option<Ts> {
        self.last_final_transcript_time
    }

    pub fn speech_start_time(&self) -> Option<Ts> {
        self.speech_start_time
    }

    pub fn open_overlap(&self) -> Option<ReqId> {
        self.overlap.open()
    }

    pub fn prediction_timeout(&self) -> Option<Duration> {
        match self.cfg.detector {
            Detector::Streaming { prediction_timeout } => Some(prediction_timeout),
            Detector::None | Detector::Batch => None,
        }
    }

    /// `_turn_detection`.
    pub fn turn_detection_mode(&self) -> TurnDetectionMode {
        self.cfg.turn_detection_mode
    }

    pub fn stt_enabled(&self) -> bool {
        self.cfg.stt_enabled
    }

    // --- driver-owned inputs -----------------------------------------------------------------

    /// The turn detector and the interruption port, as the session opened them.
    pub(crate) fn set_detectors(
        &mut self,
        eot: Option<std::sync::Arc<lk_agents_abi::EotLeg<dyn lk_agents_abi::EotDetector>>>,
        interruption: Option<
            std::sync::Arc<dyn crate::agent_activity::false_interruption::InterruptionPort>,
        >,
    ) {
        self.eot = eot;
        self.interruption = interruption;
    }

    /// `_adaptive_interruption_active`: the interruption channel exists and is not closed.
    pub fn set_interruption_active(&mut self, active: bool) {
        self.interruption_active = active;
    }

    /// `update_options(endpointing_opts=..)`: the estimator is replaced, learned floor and all.
    pub fn set_endpointing(&mut self, cfg: &crate::turn::EndpointingOptions) {
        self.endpointing = Endpointing::new(cfg);
    }

    /// `update_options(turn_detection=..)`. Only crossing the manual boundary retires the running
    /// evaluation (`._update_options`): under manual only `commit_user_turn()` may end a turn.
    pub fn set_turn_detection_mode(&mut self, now: Ts, mode: TurnDetectionMode) -> Effects {
        let mut eff = Effects::new();
        let previous = self.cfg.turn_detection_mode;
        if previous == mode {
            return eff;
        }
        self.cfg.turn_detection_mode = mode;
        if previous != TurnDetectionMode::Manual && mode != TurnDetectionMode::Manual {
            return eff;
        }
        eff.append(self.retire(now));
        eff
    }

    /// The mode is not re-resolved here; it arrives as
    /// [`AudioRecognition::set_turn_detection_mode`].
    pub fn set_legs(&mut self, stt: bool, vad: bool) {
        self.cfg.stt_enabled = stt;
        self.cfg.vad_enabled = vad;
    }
}

/// A decided turn parked on the end-of-turn hook until the host answers or the bound passes.
pub(super) struct PendingTurn {
    pub(super) info: EndOfTurnInfo,
    /// `on_user_turn_completed_delay` is measured from here.
    pub(super) issued_at: Ts,
    pub(super) deadline: Ts,
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
    pub(super) deadline: Ts,
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
pub enum CommitVerdict {
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
    /// A commit with no words: the turn is consumed and nothing is generated.
    Empty,
    /// A final that landed after its turn committed, with no new onset. Commits like `Reply`.
    LateTranscript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitDecision {
    pub verdict: CommitVerdict,
    /// The reference's return value.
    pub committed: bool,
    /// `self._cancel_false_interruption_timer()`, except where the reply task returns early.
    pub cancel_false_interruption: bool,
}

/// `on_end_of_turn`'s guards, in the reference's order. Pure.
pub fn decide_commit(i: &CommitInputs<'_>) -> CommitDecision {
    // No item without content. A realtime session with no STT has no transcript by design.
    if i.new_transcript.trim().is_empty() && (i.stt_enabled || !i.realtime_llm) {
        return CommitDecision {
            verdict: CommitVerdict::Empty,
            committed: true,
            cancel_false_interruption: false,
        };
    }

    if i.scheduling_paused {
        return CommitDecision {
            verdict: CommitVerdict::Dropped { append_user_message: i.closing },
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
        return CommitDecision {
            verdict: CommitVerdict::TooFewWords,
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
        return CommitDecision {
            verdict: CommitVerdict::RealtimeBackchannel,
            committed: false,
            cancel_false_interruption: false,
        };
    }

    CommitDecision {
        verdict: if i.late { CommitVerdict::LateTranscript } else { CommitVerdict::Reply },
        committed: true,
        cancel_false_interruption: !i.skip_reply && !i.rt_turn_detection,
    }
}

impl CommitVerdict {
    pub fn replies(self) -> bool {
        match self {
            CommitVerdict::Reply | CommitVerdict::LateTranscript => true,
            CommitVerdict::Dropped { .. }
            | CommitVerdict::TooFewWords
            | CommitVerdict::RealtimeBackchannel
            | CommitVerdict::Empty => false,
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
    /// `_TurnDetector`: the countdown asks and waits with no timeout.
    Batch,
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
    /// Waiting on the detector. `deadline` is `None` for [`Detector::Batch`].
    Awaiting { run: Run, req: ReqId, deadline: Option<Ts> },
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
    /// Handed to the driver as [`Decision::EndOfTurn`]; terminal until the verdict for `turn_id`.
    Committed { turn_id: TurnId, speaking: bool },
}

impl Phase {
    fn timer(&self) -> Option<(TurnTimer, Ts)> {
        match self {
            Phase::Idle
            | Phase::Speaking { .. }
            | Phase::Ended { .. }
            | Phase::Committed { .. } => None,
            Phase::Awaiting { deadline: Some(at), .. } => {
                Some((TurnTimer::EotPredictionTimeout, *at))
            }
            Phase::Awaiting { deadline: None, .. } => None,
            Phase::Delaying { fire_at, .. } | Phase::ManualCommitted { fire_at, .. } => {
                Some((TurnTimer::EndpointingDelay, *fire_at))
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
    /// `input_audio_transcription`: the server's transcript of the user's words.
    Transcription { text: String, is_final: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Start {
    /// A speech end or a final. `declared`: the STT's own end of speech, as opposed to a VAD end.
    Auto { declared: bool },
    /// `commit_user_turn` (`AudioRecognition._commit_user_turn`: `_user_turn_committed = True`).
    Manual { skip_reply: bool },
}

fn user_transcript(ev: &SpeechEvent, is_final: bool, speaking: Option<bool>) -> SessionEvent {
    SessionEvent::UserInputTranscribed(Box::new(UserTranscriptEvent {
        ev: ev.clone(),
        is_final,
        speaking,
        item_id: None,
    }))
}
