use lk_agents_abi::Ts;
use lk_agents_abi::Vocabulary;
pub use lk_agents_abi::voice::amd::{AMD_PROMPT, AmdCategory, AmdPrediction, AmdReason};
use std::time::Duration;

/// Above this, the greeting is long enough to be worth an LLM call.
pub const HUMAN_SPEECH_THRESHOLD: Duration = Duration::from_millis(2500);

/// How long a short, transcript-less greeting must stay silent before it is called a human.
pub const HUMAN_SILENCE_THRESHOLD: Duration = Duration::from_millis(500);

/// The post-speech silence window for everything else.
pub const MACHINE_SILENCE_THRESHOLD: Duration = Duration::from_millis(1500);

pub const NO_SPEECH_THRESHOLD: Duration = Duration::from_secs(10);

pub const DETECTION_TIMEOUT: Duration = Duration::from_secs(20);

/// Fallback for the synthetic end-of-turn backstop; see [`resolve_max_endpointing_delay`].
pub const MAX_ENDPOINTING_DELAY: Duration = Duration::from_secs(3);

/// Once spent, `postpone_termination` is no longer offered, forcing the model to commit.
pub const MAX_EXTENSIONS: u32 = 3;

/// Upper clamp on one `postpone_termination(seconds)` request.
pub const MAX_EXTENSION_SECS: f64 = 10.0;

pub use lk_agents_abi::voice::amd::{AmdConfigKwargs, TranscriptSource};

/// A host duration that `Duration::from_secs_f64` would abort on (negative or NaN), named by key.
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidDuration {
    pub key: &'static str,
    pub value: f64,
}

impl std::fmt::Display for InvalidDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} is not a valid duration in seconds (must be finite and >= 0)",
            self.key, self.value
        )
    }
}

impl std::error::Error for InvalidDuration {}

#[derive(Debug, Clone, PartialEq)]
pub struct AmdConfig {
    pub human_speech_threshold: Duration,
    pub human_silence_threshold: Duration,
    pub machine_silence_threshold: Duration,
    pub no_speech_threshold: Duration,
    pub detection_timeout: Duration,
    pub max_endpointing_delay: Duration,
    pub source: TranscriptSource,
    /// Once any speech was heard, the detection timeout no longer forces a verdict.
    pub wait_until_finished: bool,
    /// A `machine-*` verdict is followed by `Interrupt{force}` before authorization resumes.
    pub interrupt_on_machine: bool,
    /// A `machine-ivr` verdict hands the heard transcript to the IVR navigator.
    pub ivr_detection: bool,
}

impl Default for AmdConfig {
    fn default() -> Self {
        AmdConfig {
            human_speech_threshold: HUMAN_SPEECH_THRESHOLD,
            human_silence_threshold: HUMAN_SILENCE_THRESHOLD,
            machine_silence_threshold: MACHINE_SILENCE_THRESHOLD,
            no_speech_threshold: NO_SPEECH_THRESHOLD,
            detection_timeout: DETECTION_TIMEOUT,
            max_endpointing_delay: MAX_ENDPOINTING_DELAY,
            source: TranscriptSource::SessionStt,
            wait_until_finished: false,
            interrupt_on_machine: false,
            ivr_detection: false,
        }
    }
}

impl AmdConfig {
    /// The host's sparse config over the defaults. Every host duration is checked here: a panic
    /// across the FFI is an abort.
    pub fn resolve(kw: &AmdConfigKwargs) -> Result<AmdConfig, InvalidDuration> {
        let defaults = AmdConfig::default();
        let secs = |key: &'static str, v: Option<f64>, default: Duration| match v {
            None => Ok(default),
            Some(s) if s.is_finite() && s >= 0.0 => Ok(Duration::from_secs_f64(s)),
            Some(s) => Err(InvalidDuration { key, value: s }),
        };
        macro_rules! secs {
            ($f:ident) => {
                secs(stringify!($f), kw.$f, defaults.$f)?
            };
        }
        Ok(AmdConfig {
            human_speech_threshold: secs!(human_speech_threshold),
            human_silence_threshold: secs!(human_silence_threshold),
            machine_silence_threshold: secs!(machine_silence_threshold),
            no_speech_threshold: secs!(no_speech_threshold),
            detection_timeout: secs!(detection_timeout),
            max_endpointing_delay: secs!(max_endpointing_delay),
            source: kw.source.unwrap_or(defaults.source),
            wait_until_finished: kw.wait_until_finished,
            interrupt_on_machine: kw.interrupt_on_machine,
            ivr_detection: kw.ivr_detection,
        })
    }
}

/// The machine's four timers. One handle each; arming an armed timer replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Vocabulary)]
pub enum AmdTimer {
    Detection,
    /// Listening opened and nothing was heard.
    NoSpeech,
    /// The post-speech silence window: one timer, three meanings (`SilenceArm`).
    Silence,
    /// The synthetic end-of-turn backstop, for when the session's turn detector never reports one.
    Eot,
}

// --- events ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum Start {
    /// Arm or move the detection budget (moves, never stacks).
    DetectionTimer,
    /// Open the input gate and arm the no-speech timer. One-way: refused once a verdict is out.
    Listening,
}

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum Human {
    SpeechStarted,
    /// `silence` already observed since the utterance ended; deadlines are shortened by it.
    SpeechEnded {
        silence: Duration,
    },
    Transcript {
        text: String,
        source: TranscriptSource,
    },
    EndOfTurn,
}

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum Verdict {
    /// The LLM committed via `save_prediction`. An `uncertain` label is discarded.
    Llm { category: AmdCategory, transcript: String },
    /// `postpone_termination(seconds)`: the transcript is ambiguous and more audio is expected.
    Postpone { seconds: f64 },
    /// A fallback verdict from outside (`participant_missing`); the only ungated entry.
    Settle { category: AmdCategory, reason: AmdReason, speech_duration: Option<Duration> },
}

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum In {
    Start(Start),
    Human(Human),
    Verdict(Verdict),
    Timeout(AmdTimer),
    /// No more transcripts, and cancel whatever classification is in flight.
    EndInput,
    /// Wake waiters without a verdict, sweep timers, resume authorization. Idempotent.
    Close,
    /// `AMD.execute()` past the verdict and its IVR hand-off: a `machine-ivr` verdict's interrupt
    /// and resume, which wait for it. Idempotent; before a verdict, nothing.
    Execute,
}

// --- effects ---

/// The classification request; the driver cancels the in-flight one first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifyRequest {
    /// The accumulated transcript, space-joined and left-trimmed.
    pub transcript: String,
    /// Whether `postpone_termination` is offered. `false` once [`MAX_EXTENSIONS`] are spent.
    pub allow_postpone: bool,
}

/// Why an input was not applied; the state is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum Refusal {
    /// A human event outside the listening window.
    NotListening,
    InputEnded,
    /// A transcript from the STT this machine is not reading.
    WrongSource,
    SpeechOutOfOrder,
    /// `Start::Listening` after a verdict was released.
    AlreadyEmitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authorization {
    Pause,
    Resume,
}

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum Effect {
    Arm(AmdTimer, Ts),
    Cancel(AmdTimer),
    Prediction(AmdPrediction),
    RequestClassification(ClassifyRequest),
    CancelClassification,
    Refused(Refusal),
    /// `_activity._pause_authorization()` / `_resume_authorization()`, performed by the driver.
    Authorization(Authorization),
    /// `session.interrupt(force=True)`: a greeting being spoken over must be cut.
    Interrupt {
        force: bool,
    },
}

#[must_use = "the effects must be performed: an unperformed Arm is a deadline that never fires"]
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Out {
    pub effects: Vec<Effect>,
}

impl Out {
    fn push(&mut self, e: Effect) {
        self.effects.push(e);
    }
}

// --- the machine ---

/// What the one [`AmdTimer::Silence`] handle means; `ShortSpeech`'s duration is captured at arming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SilenceArm {
    ShortSpeech(Duration),
    LongSpeech,
    Postpone,
}

/// The data every phase carries. The phase itself is the [`Amd`] variant and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct Core {
    cfg: AmdConfig,
    silence_reached: bool,
    eot_reached: bool,
    /// Set by the first `SpeechStarted` only, so `speech_duration` spans the whole greeting.
    speech_started_at: Option<Ts>,
    speech_ended_at: Option<Ts>,
    speech_active: bool,
    /// When listening opened; a speech end with no seen start is measured from here.
    listening_since: Option<Ts>,
    verdict: Option<AmdPrediction>,
    transcript: String,
    extension_count: u32,
    input_open: bool,
    /// Whether this machine holds the session's authorization (resumed exactly once).
    held: bool,
    detection_armed: bool,
    no_speech_armed: bool,
    eot_armed: bool,
    silence: Option<SilenceArm>,
}

/// The variant is the phase.
#[derive(Debug, Clone, PartialEq)]
pub enum Amd {
    /// Constructed. Human events are refused (pre-answer audio must not count); `Settle` and the
    /// detection budget work.
    Dormant(Core),
    Listening(Core),
    /// A verdict was released. Terminal for input; `Start::Listening` is refused.
    Emitted(Core),
    /// `Close` ran. Terminal for everything.
    Closed(Core),
}

impl Amd {
    /// The machine as the driver attaches it: dormant and already holding authorization, as
    /// `AMD._run` pauses right after registering the classifier.
    pub fn attached(cfg: AmdConfig) -> (Self, Out) {
        let mut out = Out::default();
        let mut c = Core {
            cfg,
            silence_reached: false,
            eot_reached: false,
            speech_started_at: None,
            speech_ended_at: None,
            speech_active: false,
            listening_since: None,
            verdict: None,
            transcript: String::new(),
            extension_count: 0,
            input_open: true,
            held: false,
            detection_armed: false,
            no_speech_armed: false,
            eot_armed: false,
            silence: None,
        };
        c.hold(&mut out);
        (Amd::Dormant(c), out)
    }

    /// The audio gate: a query read at frame rate, not an effect.
    pub fn listening(&self) -> bool {
        matches!(self, Amd::Listening(_))
    }

    /// The table. Closed takes nothing; `Close` from any other phase closes.
    pub fn step(self, now: Ts, input: In) -> (Amd, Out) {
        let mut out = Out::default();
        let listening = self.listening();
        let next = match (self, input) {
            (Amd::Closed(c), _) => Amd::Closed(c),
            (Amd::Dormant(mut c) | Amd::Listening(mut c) | Amd::Emitted(mut c), In::Close) => {
                c.close(&mut out);
                Amd::Closed(c)
            }
            (Amd::Emitted(mut c), In::EndInput) => {
                c.end_input(&mut out);
                Amd::Emitted(c)
            }
            // Only a `machine-ivr` hand-off still holds here.
            (Amd::Emitted(mut c), In::Execute) => {
                c.finish(true, &mut out);
                Amd::Emitted(c)
            }
            (amd @ (Amd::Dormant(_) | Amd::Listening(_)), In::Execute) => amd,
            // ------------------------------------------------------------ Dormant, Listening
            (
                Amd::Dormant(mut c) | Amd::Listening(mut c),
                input @ (In::Verdict(_) | In::Timeout(_) | In::EndInput),
            ) => {
                let released = match input {
                    In::Verdict(v) => c.on_verdict(now, v, &mut out),
                    In::Timeout(t) => c.on_timer(now, t, &mut out),
                    _ => {
                        c.end_input(&mut out);
                        false
                    }
                };
                if released {
                    Amd::Emitted(c)
                } else if listening {
                    Amd::Listening(c)
                } else {
                    Amd::Dormant(c)
                }
            }
            // ---------------------------------------------------------------------- Dormant
            (Amd::Dormant(mut c), In::Start(Start::DetectionTimer)) => {
                c.hold(&mut out);
                c.arm_detection(now, &mut out);
                Amd::Dormant(c)
            }
            (Amd::Dormant(mut c), In::Start(Start::Listening)) => {
                c.hold(&mut out);
                c.open_listening(now, &mut out);
                Amd::Listening(c)
            }
            (Amd::Dormant(c), In::Human(_)) => {
                out.push(Effect::Refused(Refusal::NotListening));
                Amd::Dormant(c)
            }
            // -------------------------------------------------------------------- Listening
            (Amd::Listening(mut c), In::Start(Start::DetectionTimer)) => {
                c.arm_detection(now, &mut out);
                Amd::Listening(c)
            }
            (Amd::Listening(c), In::Start(Start::Listening)) => Amd::Listening(c),
            (Amd::Listening(mut c), In::Human(Human::SpeechStarted)) => {
                c.on_speech_started(now, &mut out);
                Amd::Listening(c)
            }
            (Amd::Listening(mut c), In::Human(Human::SpeechEnded { silence })) => {
                c.on_speech_ended(now, silence, &mut out);
                Amd::Listening(c)
            }
            (Amd::Listening(mut c), In::Human(Human::Transcript { text, source })) => {
                c.push_text(now, &text, source, &mut out);
                Amd::Listening(c)
            }
            (Amd::Listening(mut c), In::Human(Human::EndOfTurn)) => {
                let released = c.reach_eot(&mut out);
                if released { Amd::Emitted(c) } else { Amd::Listening(c) }
            }
            // ---------------------------------------------------------------------- Emitted
            (Amd::Emitted(c), In::Start(Start::DetectionTimer | Start::Listening)) => {
                out.push(Effect::Refused(Refusal::AlreadyEmitted));
                Amd::Emitted(c)
            }
            (Amd::Emitted(c), In::Human(_)) => {
                out.push(Effect::Refused(Refusal::NotListening));
                Amd::Emitted(c)
            }
            // A late LLM answer is stored but never re-released.
            (Amd::Emitted(mut c), In::Verdict(Verdict::Llm { category, transcript })) => {
                if category != AmdCategory::Uncertain {
                    c.verdict = Some(c.prediction(now, category, AmdReason::Llm, transcript));
                }
                Amd::Emitted(c)
            }
            (Amd::Emitted(mut c), In::Verdict(Verdict::Postpone { .. })) => {
                c.extension_count += 1;
                Amd::Emitted(c)
            }
            (Amd::Emitted(c), In::Verdict(Verdict::Settle { .. })) => Amd::Emitted(c),
            // Release leaves the silence timer armed; its fire lands here, inert.
            (Amd::Emitted(mut c), In::Timeout(t)) => {
                c.disarm(t);
                Amd::Emitted(c)
            }
        };
        (next, out)
    }
}

fn clamp_extension(seconds: f64) -> f64 {
    // Floored at zero (the reference has no lower bound; a negative fires at once). NaN -> 0.
    if seconds.is_nan() { 0.0 } else { seconds.clamp(0.0, MAX_EXTENSION_SECS) }
}

impl Core {
    fn speech_duration(&self, now: Ts) -> Duration {
        match self.speech_started_at {
            Some(start) => self.speech_ended_at.unwrap_or(now).saturating_duration_since(start),
            None => Duration::ZERO,
        }
    }

    fn delay(&self, now: Ts) -> Duration {
        self.speech_ended_at.map(|t| now.saturating_duration_since(t)).unwrap_or_default()
    }

    fn prediction(
        &self,
        now: Ts,
        category: AmdCategory,
        reason: AmdReason,
        transcript: String,
    ) -> AmdPrediction {
        AmdPrediction {
            speech_duration: self.speech_duration(now),
            category,
            reason,
            transcript,
            delay: self.delay(now),
        }
    }

    fn can_release(&self, verdict: &AmdPrediction) -> bool {
        self.silence_reached && (verdict.is_human() || self.eot_reached)
    }

    // --- authorization --------------------------------------------------------------------

    /// Pause authorization once, on the first `Start`, or a speech could play over the greeting.
    fn hold(&mut self, out: &mut Out) {
        if !self.held {
            self.held = true;
            out.push(Effect::Authorization(Authorization::Pause));
        }
    }

    fn release_hold(&mut self, out: &mut Out) {
        if self.held {
            self.held = false;
            out.push(Effect::Authorization(Authorization::Resume));
        }
    }

    // --- lifecycle ------------------------------------------------------------------------

    fn arm_detection(&mut self, now: Ts, out: &mut Out) {
        if self.detection_armed {
            out.push(Effect::Cancel(AmdTimer::Detection));
        }
        out.push(Effect::Arm(AmdTimer::Detection, now.saturating_add(self.cfg.detection_timeout)));
        self.detection_armed = true;
    }

    fn open_listening(&mut self, now: Ts, out: &mut Out) {
        if self.listening_since.is_none() {
            self.listening_since = Some(now);
        }
        if !self.no_speech_armed {
            out.push(Effect::Arm(
                AmdTimer::NoSpeech,
                now.saturating_add(self.cfg.no_speech_threshold),
            ));
            self.no_speech_armed = true;
        }
    }

    fn close(&mut self, out: &mut Out) {
        self.input_open = false;
        self.cancel_all(out);
        out.push(Effect::CancelClassification);
        // Unconditionally on the way out: a cancelled AMD must not leave the session mute.
        self.release_hold(out);
    }

    fn cancel_all(&mut self, out: &mut Out) {
        for t in [AmdTimer::NoSpeech, AmdTimer::Silence, AmdTimer::Detection, AmdTimer::Eot] {
            self.cancel(t, out);
        }
    }

    /// Forget `t` was armed (it fired, or is being cancelled); whether it was.
    fn disarm(&mut self, t: AmdTimer) -> bool {
        match t {
            AmdTimer::Detection => std::mem::take(&mut self.detection_armed),
            AmdTimer::NoSpeech => std::mem::take(&mut self.no_speech_armed),
            AmdTimer::Eot => std::mem::take(&mut self.eot_armed),
            AmdTimer::Silence => self.silence.take().is_some(),
        }
    }

    fn cancel(&mut self, t: AmdTimer, out: &mut Out) {
        if self.disarm(t) {
            out.push(Effect::Cancel(t));
        }
    }

    // --- speech ---------------------------------------------------------------------------

    fn on_speech_started(&mut self, now: Ts, out: &mut Out) {
        for t in [AmdTimer::Silence, AmdTimer::NoSpeech, AmdTimer::Eot] {
            self.cancel(t, out);
        }
        if self.speech_started_at.is_none() {
            self.speech_started_at = Some(now);
        }
        self.speech_active = true;
        // The user is talking again: both gates re-open.
        self.silence_reached = false;
        self.eot_reached = false;
    }

    fn on_speech_ended(&mut self, now: Ts, silence_duration: Duration, out: &mut Out) {
        let started = match self.speech_started_at {
            Some(started) => started,
            // Listening opened mid-utterance: the utterance began when listening did.
            None => {
                let Some(since) = self.listening_since else {
                    out.push(Effect::Refused(Refusal::SpeechOutOfOrder));
                    return;
                };
                self.on_speech_started(since, out);
                since
            }
        };
        let ended = now.saturating_sub(silence_duration);
        self.speech_ended_at = Some(ended);
        let speech_duration = ended.saturating_duration_since(started);
        // Before `arm_eot`, which refuses while speech is active.
        self.speech_active = false;
        let eot_delay = self.cfg.max_endpointing_delay.saturating_sub(silence_duration);
        self.arm_eot(now, Some(eot_delay), out);

        // A short greeting (`<=` threshold) with no transcript pre-bakes `human/short_greeting`;
        // anything else arms the machine-silence deadline. Floored at zero: the VAD reports
        // silence already elapsed.
        let (window, arm) =
            if speech_duration <= self.cfg.human_speech_threshold && self.transcript.is_empty() {
                (self.cfg.human_silence_threshold, SilenceArm::ShortSpeech(speech_duration))
            } else {
                (self.cfg.machine_silence_threshold, SilenceArm::LongSpeech)
            };
        self.cancel(AmdTimer::Silence, out);
        let fire_at = now.saturating_add(window.saturating_sub(silence_duration));
        out.push(Effect::Arm(AmdTimer::Silence, fire_at));
        self.silence = Some(arm);
    }

    // --- transcripts ----------------------------------------------------------------------

    fn push_text(&mut self, now: Ts, text: &str, source: TranscriptSource, out: &mut Out) {
        if !self.input_open {
            out.push(Effect::Refused(Refusal::InputEnded));
            return;
        }
        if source != self.cfg.source {
            out.push(Effect::Refused(Refusal::WrongSource));
            return;
        }

        // A final transcript restarts the end-of-turn backstop, even without a VAD end.
        self.arm_eot(now, None, out);

        // Words in a short greeting must be classified, not assumed human. The replacement keeps
        // the original anchor, so transcripts can't walk the deadline forward.
        if let Some(SilenceArm::ShortSpeech(_)) = self.silence {
            out.push(Effect::Cancel(AmdTimer::Silence));
            let anchor = self
                .speech_ended_at
                .unwrap_or(now)
                .saturating_add(self.cfg.machine_silence_threshold);
            out.push(Effect::Arm(AmdTimer::Silence, anchor.max(now)));
            self.silence = Some(SilenceArm::LongSpeech);
        }

        self.cancel(AmdTimer::NoSpeech, out);
        if !self.transcript.is_empty() {
            self.transcript.push(' ');
        }
        self.transcript.push_str(text);
        self.request_classification(out);
    }

    fn request_classification(&mut self, out: &mut Out) {
        out.push(Effect::RequestClassification(ClassifyRequest {
            transcript: self.transcript.clone(),
            allow_postpone: self.extension_count < MAX_EXTENSIONS,
        }));
    }

    fn end_input(&mut self, out: &mut Out) {
        if !self.input_open {
            return;
        }
        self.input_open = false;
        out.push(Effect::CancelClassification);
    }

    // --- end of turn ----------------------------------------------------------------------

    fn reach_eot(&mut self, out: &mut Out) -> bool {
        self.cancel(AmdTimer::Eot, out);
        self.eot_reached = true;
        // A greeting the VAD never bounded ends at the end of turn.
        if self.speech_ended_at.is_none() {
            self.silence_reached = true;
        }
        self.try_release(out)
    }

    /// Re-arming clears the gate: more words mean the greeting wasn't over.
    fn arm_eot(&mut self, now: Ts, delay: Option<Duration>, out: &mut Out) {
        if self.speech_active {
            return;
        }
        if self.eot_armed {
            out.push(Effect::Cancel(AmdTimer::Eot));
        }
        self.eot_reached = false;
        out.push(Effect::Arm(
            AmdTimer::Eot,
            now.saturating_add(delay.unwrap_or(self.cfg.max_endpointing_delay)),
        ));
        self.eot_armed = true;
    }

    // --- verdicts -------------------------------------------------------------------------

    fn on_verdict(&mut self, now: Ts, v: Verdict, out: &mut Out) -> bool {
        match v {
            Verdict::Llm { category, transcript } => {
                // Keep the verdict unset so the timeout and extensions can still do better.
                if category == AmdCategory::Uncertain {
                    return false;
                }
                self.verdict = Some(self.prediction(now, category, AmdReason::Llm, transcript));
                self.try_release(out)
            }
            Verdict::Postpone { seconds } => {
                let clamped = clamp_extension(seconds);
                // Counted either way, which guarantees termination.
                self.extension_count += 1;
                self.cancel(AmdTimer::Silence, out);
                out.push(Effect::Arm(
                    AmdTimer::Silence,
                    now.saturating_add(Duration::from_secs_f64(clamped)),
                ));
                self.silence = Some(SilenceArm::Postpone);
                false
            }
            Verdict::Settle { category, reason, speech_duration } => {
                self.settle(now, category, reason, speech_duration, out)
            }
        }
    }

    /// Commit a fallback verdict and try to release it. A zero `speech_duration` override falls
    /// through to the measured value (truthiness).
    fn settle(
        &mut self,
        now: Ts,
        category: AmdCategory,
        reason: AmdReason,
        speech_duration: Option<Duration>,
        out: &mut Out,
    ) -> bool {
        self.cancel(AmdTimer::Silence, out);
        self.silence_reached = true;

        // With no speech and no transcript there is nothing to wait for: release at once.
        let has_speech = self.speech_started_at.is_some() || !self.transcript.is_empty();
        if !(self.cfg.wait_until_finished && has_speech) {
            self.eot_reached = true;
        }

        if self.verdict.is_none() {
            let measured = self.speech_duration(now);
            let speech_duration = speech_duration.filter(|d| !d.is_zero()).unwrap_or(measured);
            self.verdict = Some(AmdPrediction {
                speech_duration,
                category,
                reason,
                // Deliberately empty on every fallback, even when one was accumulated.
                transcript: String::new(),
                delay: self.delay(now),
            });
        }
        self.try_release(out)
    }

    fn try_release(&mut self, out: &mut Out) -> bool {
        let Some(verdict) = self.verdict.clone() else { return false };
        if !self.can_release(&verdict) {
            return false;
        }

        for t in [AmdTimer::Detection, AmdTimer::NoSpeech, AmdTimer::Eot] {
            self.cancel(t, out);
        }
        // The silence timer is not cancelled (the reference cancels three of four); `Close` sweeps it.

        // A `machine-ivr` hand-off is `execute()`'s: its interrupt and resume wait for the
        // navigator (`In::Execute`), so a reply planned without the DTMF tool is cut, not played.
        let machine = verdict.is_machine();
        let ivr = verdict.category == AmdCategory::MachineIvr && self.cfg.ivr_detection;
        out.push(Effect::Prediction(verdict));
        if !ivr {
            self.finish(machine, out);
        }
        true
    }

    /// Interrupt, then resume, last and unconditional (the caller drives its own voicemail reply).
    /// Once: nothing after the hold is released.
    fn finish(&mut self, machine: bool, out: &mut Out) {
        if !self.held {
            return;
        }
        if machine && self.cfg.interrupt_on_machine {
            out.push(Effect::Interrupt { force: true });
        }
        self.release_hold(out);
    }

    // --- time -----------------------------------------------------------------------------

    fn on_timer(&mut self, now: Ts, timer: AmdTimer, out: &mut Out) -> bool {
        match timer {
            AmdTimer::Detection => {
                self.disarm(timer);
                self.settle(now, AmdCategory::Uncertain, AmdReason::DetectionTimeout, None, out)
            }
            AmdTimer::NoSpeech => {
                self.disarm(timer);
                self.settle(now, AmdCategory::Uncertain, AmdReason::NoSpeechTimeout, None, out)
            }
            AmdTimer::Eot => {
                self.disarm(timer);
                self.reach_eot(out)
            }
            // `take()` first: the arm has fired, so nothing downstream may cancel it again.
            AmdTimer::Silence => match self.silence.take() {
                Some(SilenceArm::ShortSpeech(d)) => {
                    self.settle(now, AmdCategory::Human, AmdReason::ShortGreeting, Some(d), out)
                }
                Some(SilenceArm::LongSpeech) => {
                    self.silence_reached = true;
                    self.try_release(out)
                }
                // The extension window expired: flip the gate and re-request; once extensions are
                // spent `allow_postpone` is false, forcing a commit.
                Some(SilenceArm::Postpone) => {
                    self.silence_reached = true;
                    if self.input_open {
                        self.request_classification(out);
                    }
                    self.try_release(out)
                }
                None => false,
            },
        }
    }
}

// --- option resolution ---

/// An explicit option, else the activity's value, else the default; by presence, not value.
pub fn resolve_max_endpointing_delay(
    provided: Option<Duration>,
    activity: Option<Duration>,
    default: Duration,
) -> Duration {
    provided.or(activity).unwrap_or(default)
}

/// `_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's, else default.
///
/// `provided` is `None` only when the key was absent: the reference tests key presence, so an
/// explicit default still beats a disagreeing activity.
#[lk_agents_abi::host(export(name = "amd_resolve_max_endpointing_delay", pid = false))]
pub fn resolve_max_endpointing_delay_for_host(provided: Option<Duration>, #[host(required)] activity: Option<Duration>) -> Duration {
    resolve_max_endpointing_delay(provided, activity, MAX_ENDPOINTING_DELAY)
}

/// The detector's constants, as one dict so a host cannot pick up half of them.
///
/// These are the reference's, not tuning knobs: the default models are what the benchmark ran on.
#[lk_agents_abi::host(export(pid = false))]
pub fn amd_constants() -> lk_agents_abi::voice::amd::AmdConstants {
    use lk_agents_abi::voice::amd as a;
    // The classifier's own defaults, so `DetectionOptions`' merge has one source.
    let c = AmdConfig::default();
    lk_agents_abi::voice::amd::AmdConstants {
        default_llm_model: a::DEFAULT_LLM_MODEL.to_owned(),
        default_stt_model: a::DEFAULT_STT_MODEL.to_owned(),
        track_publication_timeout: a::TRACK_PUBLICATION_TIMEOUT.as_secs_f64(),
        sip_call_status_attr: a::SIP_CALL_STATUS_ATTR.to_owned(),
        sip_call_status_active: a::SIP_CALL_STATUS_ACTIVE.to_owned(),
        prompt: a::AMD_PROMPT.to_owned(),
        evaluated_llm_models: a::EVALUATED_LLM_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        evaluated_stt_models: a::EVALUATED_STT_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        human_speech_threshold: c.human_speech_threshold.as_secs_f64(),
        human_silence_threshold: c.human_silence_threshold.as_secs_f64(),
        machine_silence_threshold: c.machine_silence_threshold.as_secs_f64(),
        no_speech_threshold: c.no_speech_threshold.as_secs_f64(),
        detection_timeout: c.detection_timeout.as_secs_f64(),
        max_endpointing_delay: c.max_endpointing_delay.as_secs_f64(),
        max_extensions: MAX_EXTENSIONS,
        max_extension_secs: MAX_EXTENSION_SECS,
    }
}
