//! User-turn recognition state and transitions over supplied STT, VAD and detector observations.
//!
//! `AudioRecognition` owns transcript accumulation, endpointing, overlap integration, commit and
//! close state. Inputs enter through its methods; ordered `AudioRecognitionEffect` values leave
//! each transition. VoiceDriver executes those requests and returns detector/timer outcomes.
//! `end_of_turn` and `speech` are private implementation children of this same owner.

use super::{
    Detector, EndOfTurnInfo, InterruptionSignal, MIN_SILENCE_DURATION, OverlapVerdict,
    PendingCommit, Phase, Prediction, Run, ServerSpeech, TurnConfig, TurnDetectionMode, TurnId,
    TurnSnapshot, UserTurnCommitDecision, UserTurnCommitVerdict, UserTurnLimit, UserTurnTracker,
    WallAnchor, compute_end_of_turn_metrics,
};
use crate::agent_activity::options::{AgentSessionOptions, DetectorKind, SessionLegs};
use crate::agent_session::effects::Timer;
use crate::audio_recognition::effects::AudioRecognitionEffect;
use crate::audio_recognition::overlap::OverlapJudge;
use crate::endpointing::Endpointing;
use crate::events::{AgentBackchannelOpportunityEvent, SessionEvent, UserTranscriptEvent};
use lk_agents_abi::tokenize::language::{LanguageCode, ThresholdOptions, TurnDetectorModel};
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{EotPrediction, SpeechEvent, Ts};
use std::collections::VecDeque;
use std::time::Duration;

mod end_of_turn;
mod speech;

const FRESH_FINAL: Duration = Duration::from_millis(500);

// A short transcript never overwrites an established language (characters, as Python len).
const MIN_LANGUAGE_DETECTION_LENGTH: usize = 5;
const NON_SPECIFIC_LANGUAGE_CODES: [&str; 2] = ["auto", "multi"];

/// STT and VAD events folded into the user's turn, the end-of-turn countdown and its detector,
/// the adaptive interruption channel, and a host `commit_user_turn()` waiting for its transcript.
pub struct AudioRecognition {
    eot_provider: Option<lk_agents_abi::ProviderId>,
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
    fired: Option<Timer>,
    eot_seq: u64,
    turn_seq: u64,
    /// `self._last_emitted_prediction`: one prediction is announced once, compared by value.
    last_emitted: Option<EotPrediction>,
    /// A backchannel opportunity whose countdown commits in the same cycle, held for the driver's
    /// verdict: released on a refusal or `Dropped`, dropped on a commit that replies.
    deferred_opportunity: Option<AgentBackchannelOpportunityEvent>,
}

impl AudioRecognition {
    pub fn new(cfg: &AgentSessionOptions, legs: &SessionLegs) -> Self {
        let i = &cfg.turn.interruption;
        let turn = TurnConfig {
            stt_enabled: legs.stt,
            vad_enabled: legs.vad,
            using_default_vad: legs.using_default_vad,
            turn_detection_mode: cfg.turn.owner(),
            // Without an adaptive detector, transcripts over agent speech reach the turn at once.
            interruption_enabled: i.detection_enabled(),
            transcription_timeout: cfg.timers.transcription_timeout(),
            user_turn_limit: UserTurnLimit {
                max_words: cfg.turn.user_turn_limit.max_words().map(|w| w as usize),
                max_duration: cfg.turn.user_turn_limit.max_duration(),
            },
            backchannel_boundary: i.backchannel_boundary(),
            detector: match legs.detector {
                None => Detector::None,
                Some(DetectorKind::Streaming) => {
                    Detector::Streaming { prediction_timeout: cfg.turn.endpointing.max_delay() }
                }
            },
        };
        AudioRecognition {
            eot_provider: None,
            pending_commit: None,
            overlap: OverlapJudge::new(turn.interruption_enabled),
            cfg: turn,
            phase: Phase::Idle,
            endpointing: Endpointing::new(&cfg.turn.endpointing),
            thresholds: ThresholdOptions::new(TurnDetectorModel::V1Mini, None, None),
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

    pub(crate) fn eot_provider(&self) -> Option<&lk_agents_abi::ProviderId> {
        self.eot_provider.as_ref()
    }

    /// A host `commit_user_turn()` waits for its transcript.
    pub(crate) fn commit_pending(&self) -> bool {
        self.pending_commit.is_some()
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

    /// `_audio_transcript`: this turn's final words.
    pub fn audio_transcript(&self) -> &str {
        &self.audio_transcript
    }

    /// `_end_of_turn_task is not None and not done()`. `Committed` is not open.
    pub fn turn_open(&self) -> bool {
        self.phase.is_running()
    }

    /// `self._speaking`, i.e. `not self._user_silence_ev.is_set()`. Read off the phase.
    /// Speaking on the recogniser's word alone: no detector heard the start, so none will hear
    /// the end.
    pub fn speaking_on_recogniser(&self) -> bool {
        self.user_speaking() && !self.vad_speech_started
    }

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

    pub fn open_overlap(&self) -> Option<ReqId> {
        self.overlap.open()
    }

    pub fn prediction_timeout(&self) -> Option<Duration> {
        match self.cfg.detector {
            Detector::Streaming { prediction_timeout } => Some(prediction_timeout),
            Detector::None => None,
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

    pub(crate) fn set_eot_provider(&mut self, provider: Option<lk_agents_abi::ProviderId>) {
        self.eot_provider = provider;
    }

    /// The detector's own thresholds (`unlikely_threshold(language)`, ..), read by the driver
    /// when the session opens and with each answer, as the reference reads them once the
    /// prediction lands. `None` (a detector without a table) keeps what the machine has.
    pub(crate) fn set_thresholds(&mut self, thresholds: Option<ThresholdOptions>) {
        if let Some(thresholds) = thresholds {
            self.thresholds = thresholds;
        }
    }

    /// `_adaptive_interruption_active`: the interruption channel exists and is not closed.
    pub fn set_interruption_active(&mut self, active: bool) {
        self.interruption_active = active;
    }

    /// The endpointing delay a turn serves now.
    pub(crate) fn min_delay(&self) -> Duration {
        self.endpointing.min_delay()
    }

    /// `update_options(endpointing_opts=..)`: the estimator is replaced, learned floor and all.
    pub fn set_endpointing(&mut self, cfg: &crate::turn::EndpointingOptions) {
        self.endpointing = Endpointing::new(cfg);
    }

    /// `update_options(turn_detection=..)`. Only crossing the manual boundary retires the running
    /// evaluation (`._update_options`): under manual only `commit_user_turn()` may end a turn.
    #[must_use = "recognition effects must be applied before the next transition"]
    pub fn set_turn_detection_mode(
        &mut self,
        now: Ts,
        mode: TurnDetectionMode,
    ) -> Vec<AudioRecognitionEffect> {
        let mut eff = Vec::new();
        let previous = self.cfg.turn_detection_mode;
        if previous == mode {
            return eff;
        }
        self.cfg.turn_detection_mode = mode;
        if previous != TurnDetectionMode::Manual && mode != TurnDetectionMode::Manual {
            return eff;
        }
        eff.extend(self.retire(now));
        eff
    }

    /// The mode is not re-resolved here; it arrives as
    /// [`AudioRecognition::set_turn_detection_mode`].
    pub fn set_legs(&mut self, stt: bool, vad: bool) {
        self.cfg.stt_enabled = stt;
        self.cfg.vad_enabled = vad;
    }
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

impl AudioRecognition {
    /// Whether recognition is actively deciding a user turn, independent of session idle holds.
    pub(crate) fn is_deciding_turn(&self) -> bool {
        match self.phase() {
            Phase::Idle | Phase::Ended { .. } => false,
            Phase::Speaking { .. }
            | Phase::Awaiting { .. }
            | Phase::Delaying { .. }
            | Phase::ManualCommitted { .. } => true,
            Phase::Committed { speaking, .. } => *speaking,
        }
    }
}

impl AudioRecognition {
    /// The recognizer may still owe a final transcript while closing.
    pub(crate) fn transcript_pending(&self, now: Ts) -> bool {
        let turn = self;
        // A manual turn is committed whatever it holds: the reference's commit completes it even
        // with no words.
        let manual = turn.turn_detection_mode() == TurnDetectionMode::Manual;
        if manual || turn.user_speaking() || !turn.current_transcript().trim().is_empty() {
            return true;
        }
        match (turn.last_speaking_time(), turn.last_final_transcript_time()) {
            (Some(spoke), last_final) => {
                last_final.is_none_or(|f| f < spoke)
                    && now.saturating_duration_since(spoke) <= Duration::from_secs(1)
            }
            (None, _) => false,
        }
    }
}
