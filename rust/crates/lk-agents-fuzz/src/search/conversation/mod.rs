//! The search: a person on a phone, talking turn by turn to the agent, while the developer's
//! program calls the API and the models, outputs and room do what real ones do, failures
//! included.
//!
//! The uniform sampler reaches the second half of a chain by luck. A person on a phone does not:
//! they speak, a detector and a recogniser report what they heard in a fixed order with
//! provider-shaped delays, the agent answers, and they speak again -- sometimes over the agent.
//! This mode drives the microphone that way, so the sequences the turn machine was written for
//! (start of speech, a final, an end of speech, an endpointing delay) happen every turn rather
//! than once in ten runs.
//!
//! Everything an actor does follows from what it observed: a recogniser transcribes only audio
//! the driver forwarded to it, a speaker reports only audio it was handed, the person answers what
//! they heard. What varies is drawn once per run and holds for its length: who is on the line
//! (the [`Persona`]), the developer's program (the [`Host`]), how the models, outputs and room
//! behave and fail ([`Providers`]), and how random the run is ([`randomness`]): before
//! each move the conversation makes, a run may first make uniform moves -- any legal row, no
//! actor behind it -- at a rate from almost never to almost always, so the search spans a pure conversation, a
//! pure uniform sampler, and everything between. Uniform moves are labelled in the case
//! ([`Case::random`](crate::drive::Case::random)).
//!
//! There are no per-variant weights: every attempt at aiming the search at a variant was measured
//! on the survey's yardstick and lost.
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::{Range, RangeInclusive},
};

use fastrand::Rng;
use lk_agents_abi::voice::amd::AmdCategory;
use lk_agents_voice::{
    agent_session::IoKind,
    amd::classifier::{MAX_EXTENSION_SECS, POSTPONE_TERMINATION_TOOL, SAVE_PREDICTION_TOOL},
    events::SessionEvent,
    keyterm_detection::RECORD_KEYTERMS_TOOL,
    trace::{RealtimeStep, Step},
};

use super::uniform;

use crate::{
    drive::{AmdAnswer, Case, Outcome, Policy, Run, Setup},
    fakes::{MS_PER_WORD, Measured, Timing, delay, measured},
    input::{
        AGENT_TURNS, AgentText, AmdInput, Confidence, ConsoleAsk, Count, HookAnswer, Input, Make,
        Millis, NAMES, Ref, RoomIn, RtEvent, Sampler, Speaker, ToolAnswer, ToolArgs, TurnMode,
        USER_TURNS, UserText,
    },
    record::{Detail, Kind, Sink, Tag},
};

mod microphone;
mod providers;
mod speaker;

pub use providers::{STT_MODELS, SttModel};
use providers::{SttKnobs, recogniser, recogniser_noise};

use speaker::Playback;

// ==============================================================================================
// Who is on the line
// ==============================================================================================
//
// One caller per run, and the stretches of microphone they produce.

/// Who is on the line, drawn once per run. These rates are a person's, not the search's: an
/// impatient caller talks over most replies and a patient one over few, one detector is mostly
/// right and another often wrong, and a run is one such person for its whole length. Each rate is
/// drawn from a range wide enough that every kind of caller is some run's caller, and bounded so
/// that every caller is still a caller: a run whose every stretch is noise, or whose every reply
/// is talked over, reaches nothing the driver does with a conversation, and a run is the unit
/// the search pays for.
#[derive(Clone, Copy, Debug)]
struct Persona {
    /// How often the person pauses before speaking rather than starting straight away.
    pauses: u8,
    /// How often a stretch of speech is a pause for thought, more words following after a short
    /// silence rather than the agent's turn: what the end-of-turn predictor exists to tell apart.
    hesitates: u8,
    /// How often the person talks over the agent's reply instead of waiting it out.
    barges: u8,
    /// How often a stretch of speech is noise to the recogniser -- a cough, a door -- which the
    /// detector hears and the recogniser has no words for: the shape a false interruption has.
    noise: u8,
    /// How often a stretch comes from someone else in the room, whom the recogniser diarises
    /// apart.
    other_speaker: u8,
    /// How often the person drops off the call and comes back, per turn.
    leaves: u8,
    /// How fast they talk, in percent of a typical 120-210 words a minute.
    pace: u64,
    /// How long their silences are, in percent of a typical 300-1200 ms.
    silences: u64,
    /// How long they wait for the agent after speaking, in percent of a typical 300-1500 ms.
    patience: u64,
    /// How often, per beat while the agent speaks, they say "mm-hm" without wanting the floor.
    backchannels: u8,
    /// How long they sit in silence waiting for a reply before they speak again, in ms.
    gives_up_ms: u64,
}

impl Persona {
    fn draw(rng: &mut Rng) -> Self {
        Persona {
            pauses: rng.u8(0..=100),
            hesitates: rng.u8(0..=80),
            barges: rng.u8(0..=100),
            noise: rng.u8(0..=60),
            other_speaker: rng.u8(0..=60),
            leaves: if rng.u8(0..100) < 20 { rng.u8(5..=30) } else { 0 },
            pace: rng.u64(40..=250),
            silences: rng.u64(20..=500),
            patience: rng.u64(20..=500),
            backchannels: rng.u8(0..=15),
            gives_up_ms: rng.u64(300..=15_000),
        }
    }
}

/// One stretch of the person's microphone: speaking or silent, this long, at this pacing.
///
/// The one primitive. A pause is a silent segment, a interruption is a speech segment placed while
/// the agent's speech is live, a hesitation is a speech segment the person is not done after,
/// and the detector's and recogniser's verdicts are placed around a speech segment from the
/// models below. Not to be confused with [`Kind::Segment`], a flushed audio segment on the
/// agent's side.
struct Segment {
    speaking: bool,
    dur_ms: u64,
    pacing: Pacing,
    /// The words, when the person is speaking: the duration is theirs, and so is the transcript.
    text: Option<&'static str>,
    /// Whether this is the end of what the person meant to say -- what the end-of-turn predictor
    /// is asked to tell, and the ground truth its answer is sampled from.
    done: bool,
}

/// How a segment's frames reach the driver in virtual time. The frames always add up to the
/// segment; only their spacing differs.
enum Pacing {
    /// A beat's worth of frames every beat.
    Steady,
    /// Beats unevenly long around the nominal.
    Jitter { spread_ms: u64 },
    /// No frames for the stall, then one burst that catches up.
    Gap { stall_ms: u64 },
}

/// How fast the person talks, in words per second: from a slow 120 words a minute to a brisk
/// 210. The duration of what they say is their words at this rate, so "can you hear me" is
/// short and a rambling turn is long, as on a real line.
const WORDS_PER_SEC_X10: Range<u64> = 20..36;

/// The shortest stretch of speech, in milliseconds: one word takes at least this long to say.
const MIN_SPEECH_MS: u64 = 250;

/// How long a pause with nobody speaking lasts, inside a turn: from a breath to a pause for
/// thought long enough to cross a short endpointing delay.
const SILENCE_MS: Range<u64> = 300..1200;

/// Jitter around a beat: within one detector interval, up to most of a beat.
const JITTER_MS: Range<u64> = 10..60;

/// How long a gap stalls: from a network hiccup to a stall the driver's own timers notice.
const STALL_MS: Range<u64> = 100..500;

// ==============================================================================================
// What the providers say about a stretch
// ==============================================================================================
//
// The detector and the recogniser report what they heard around a stretch of speech, with
// provider-shaped delays; the predictor and the voice have failure modes of their own.

/// A provider's word about a speech segment, and when it is due, measured from the segment's
/// start. Made into an `Input` only when it is sent, because the utterance an STT verdict names
/// does not exist until `SttStart` has been applied.
struct Verdict {
    at_ms: u64,
    word: Word,
}

enum Word {
    VadStart,
    VadEnd,
    SttStart,
    SttInterim(String),
    SttPreflight(String),
    SttFinal(String),
    SttEnd,
    /// A usage report, which the corpus models send before every final.
    SttUsage,
    /// The realtime server's own VAD: `input_audio_buffer.speech_started` / `speech_stopped`.
    RtStart,
    RtStop,
    // The end-of-turn predictor has no word here: the driver asks it, so its answer is placed
    // when a `Kind::Prediction` is live, not on a schedule. See `Talk::predict`.
}

/// When the detector notices speech began, after the first frame of it.
const VAD_START_MS: Range<u64> = 30..150;

/// When the detector notices speech ended, after the last frame of it.
const VAD_END_MS: Range<u64> = 100..400;

const STT_LATE_FINAL_MS: Range<u64> = 1_500..3_000;

/// Detector confidence while speaking: straddles 50, so hysteresis and flicker are reached.
const SPEAKING_P: Range<u8> = 45..95;

/// Detector confidence while silent.
const SILENT_P: Range<u8> = 0..20;

/// The predictor's confidence that the turn is over when the person is done, and when they are
/// only pausing.
const EOT_DONE_P: Range<u8> = 70..101;

const EOT_MORE_P: Range<u8> = 0..31;

/// Speech over the agent shorter than this is judged a backchannel -- "mm-hm" -- not a interruption,
/// the distinction the adaptive detector exists to make. Matches the driver's default minimum
/// interruption duration, so by default the detector and the driver agree on what counts.
const BACKCHANNEL_MS: u64 = 500;

/// What a listener says over the agent without wanting the floor, and how long it takes.
const BACKCHANNELS: &[&str] = &["mm-hm", "yeah", "right", "okay", "uh-huh", "sure"];
const BACKCHANNEL_SAID_MS: RangeInclusive<u64> = 150..=450;

/// How long a call lasts, drawn per seed: log-normal around a five-minute median, so about four
/// calls in five last one to twenty minutes and about one in fifty over an hour, clamped to these
/// bounds. The run lasts as long as its call; the input budget is only a safety cap.
const CALL_MEDIAN_MS: f64 = 300_000.0;
const CALL_SPREAD: f64 = 1.2;
const CALL_LENGTH_MS: RangeInclusive<f64> = 20_000.0..=7_200_000.0;

// ==============================================================================================
// The developer's program
// ==============================================================================================

/// The developer's program, drawn once per run. Modelled by two rates so far: how often a call
/// lands unprompted before a stretch of speech, and how often one lands inside a window --
/// while the person speaks or barges in, while the agent thinks, runs a tool or speaks -- the
/// call at the wrong moment, which developers write. A reactive form, handlers firing on journal families and drawing from the host's rows,
/// was measured on 2026-09-23 and lost on variants and findings while failing most runs: a
/// handler that draws uniformly from the API is not a program, and a program needs a model of
/// what handlers do that this crate does not have yet.
#[derive(Clone, Debug)]
struct Host {
    /// The chance of an unprompted call before a stretch of speech.
    unprompted: u8,
    /// The chance, per beat, of a call inside whatever the conversation is doing.
    during: u8,
    /// The same while the agent's audio is playing, interruptions included: the window a developer
    /// most often acts in, and the one a run spends least time in.
    speaking: u8,
    /// How often the model answers with a tool call rather than words, when the agent has tools:
    /// the program's prompt and tools decide it.
    tool_calls: u8,
    /// How often it scopes a filler on a tool it is waiting on.
    fillers: u8,
    /// How often a tool it is waiting on runs a `VoiceTask` (collecting an email, say) before it
    /// returns: nested when the task's own tool does, cancelled when the body ends first.
    voice_tasks: u8,
}

impl Host {
    /// Most programs call the API now and then; three in ten are busy ones, updating the agent
    /// every turn or speaking on every event. The busy tier is where calling any method at any
    /// moment gets exercised: without it, measured on 2026-10-22 at 512 seeds, 136 of 520
    /// (call, phase) cells were never reached, against 24 with a uniform sampler alongside.
    fn draw(rng: &mut Rng) -> Self {
        let busy = rng.u8(0..100) < 30;
        Host {
            unprompted: if busy { rng.u8(20..=60) } else { rng.u8(0..=10) },
            during: if busy { rng.u8(10..=40) } else { rng.u8(0..=4) },
            speaking: if busy { rng.u8(20..=60) } else { rng.u8(0..=30) },
            tool_calls: rng.u8(0..=100),
            fillers: rng.u8(0..=100),
            voice_tasks: if rng.u8(0..100) < 50 { rng.u8(5..=40) } else { 0 },
        }
    }
}

// ==============================================================================================
// What goes wrong around the call
// ==============================================================================================

/// How the models, the outputs and the room behave this run: how fast, how flaky, how they fail.
/// Each failure kind has its own coin and its own stream off the seed, so some runs are clean and
/// others full of one kind of trouble, and adding a kind never moves another's draw (agent-store's
/// swarm hunt). The fakes stay dumb: this is what the case tells them, so the case is still the
/// whole of what happened.
#[derive(Clone, Copy, Debug)]
struct Providers {
    /// How slow every model answers, in percent of typical latencies.
    latency: u64,
    /// How often the recogniser's final comes 1.5-3 s late.
    late_finals: u8,
    /// How often an interim is mis-heard, then corrected.
    revisions: u8,
    /// How often the end-of-turn predictor loses a request or fails it.
    eot_lost: u8,
    /// How often the interruption detector agrees with the overlap's length.
    detector_right: u8,
    /// The model fails a request instead of answering it.
    llm_fails: u8,
    /// The recogniser fails partway through a stretch; its stream is spent and the driver does not
    /// reopen it, so the rest of the call is deaf: a rare, fatal failure.
    stt_fails: u8,
    /// The interruption detector fails its judgement of an overlap.
    overlap_fails: u8,
    /// The realtime server drops the socket during a reply, then comes back.
    socket_drops: u8,
    /// The realtime server errs during a reply: an `error` event, a failed response or
    /// transcription, a response of its own, a stray delta.
    realtime_errors: u8,
    /// The speaker throws its playback count away mid-playout, as a republished track does.
    sink_resets: u8,
    /// The voice is down as the call starts: every synthesis fails at the open until it comes
    /// back.
    voice_down: bool,
    /// A wait lands one millisecond before, on, or after a deadline the driver armed, where a
    /// driver's boundary races are and a drawn delay almost never lands.
    aims: u8,
}

impl Providers {
    fn draw(seed: u64) -> Self {
        // A stream per name, so adding a field never moves another's draw.
        let rng = |name: &str| {
            let key = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            });
            Rng::with_seed(seed ^ key)
        };
        // `on` in a hundred runs have the failure at all; those draw its rate from `rate`.
        let fails = |name: &str, on: u8, rate: RangeInclusive<u8>| {
            let mut rng = rng(name);
            if rng.u8(0..100) < on { rng.u8(rate) } else { 0 }
        };
        Providers {
            latency: rng("latency").u64(25..=400),
            late_finals: rng("late_finals").u8(0..=40),
            revisions: rng("revisions").u8(0..=60),
            eot_lost: rng("eot_lost").u8(0..=30),
            detector_right: rng("detector_right").u8(0..=100),
            llm_fails: fails("llm_fails", 30, 5..=50),
            stt_fails: fails("stt_fails", 30, 1..=10),
            overlap_fails: fails("overlap_fails", 30, 10..=80),
            socket_drops: fails("socket_drops", 30, 5..=30),
            realtime_errors: fails("realtime_errors", 30, 5..=30),
            sink_resets: fails("sink_resets", 20, 1..=10),
            voice_down: rng("voice_down").u8(0..100) < 12,
            aims: fails("aims", 50, 20..=80),
        }
    }
}

/// How random the run is: uniform moves per move the conversation makes, on average. Drawn flat
/// over orders of magnitude, from one in ten thousand (a conversation with nothing random in it)
/// to a thousand (uniform moves with a conversation step now and then), so every decade gets as
/// many seeds as every other. A flat draw of a probability instead spent nearly every seed in
/// runs a random close ended early: measured on 2026-10-22, interruptions went unreached.
pub fn randomness(seed: u64) -> f64 {
    10f64.powf(Rng::with_seed(seed ^ 0x636f_6865_7265_6e63).f64() * 7.0 - 4.0)
}

/// Beats the developer keeps calling the API after a close was asked for: calls made while the
/// session winds down are part of the contract, a person still talking into it is not.
const CLOSING_BEATS: usize = 20;

/// A long read-out (terms, a policy, a list of options): about fifty seconds of speech.
const READ_OUT: &str = "Before we continue, I need to read you our terms. This call may be recorded for \
quality and training purposes. Your information is used only to service your account and is \
never sold. You can ask for a copy of your data at any time, and you can ask us to delete it, \
except where the law requires us to keep it. Payments made today are processed within two \
business days, and refunds take five to seven business days to appear on your statement. If \
you cancel within thirty days you will receive a full refund, minus any shipping costs. After \
thirty days, cancellations are subject to a fee of ten percent of the remaining balance. By \
continuing, you confirm that you are the account holder or are authorized to act on their \
behalf, and that you agree to these terms. Do you agree to continue?";

/// How often the program makes its own calls in the instant the call ends, before the close and
/// with nothing settling between: teardown code (`session.input.audio = None`, a last update, a
/// goodbye) followed by `await session.aclose()`, or a call already on its way as the room goes.
const TEARDOWN: u8 = 30;

/// How often, per beat, a camera the case attached delivers frames.
const CAMERA: u8 = 30;

/// How often, per beat, a ringing SIP call AMD waits on is answered.
const SIP_ANSWERS: u8 = 20;

/// How often, per beat, a voice provider that is down comes back: outages last seconds.
const OUTAGE_ENDS: u8 = 2;

/// How often, per beat, the developer turns back on a leg it turned off: about two seconds on.
const RESTORES: u8 = 5;

/// How often, per beat, the developer leaves the `async with AMD(...)` block once `execute()`
/// returned.
const AMD_LEFT: u8 = 30;

// ==============================================================================================
// The beat, and the agent's turn
// ==============================================================================================
//
// A conversation advances a beat at a time: the frames, the detector's tick, the clock. Between
// the person's stretches the agent gets its turn: the model, the tools, the playout.

/// One beat of microphone time: a beat's frames in one `MicFrames`, one detector tick, then the
/// clock moves by the beat. A real detector reports every 32 ms, but three inputs per 32 ms beat
/// spend a 150-input budget on a second and a half of audio; 100 ms buys several turns.
const BEAT_MS: u64 = 100;

/// One microphone frame, as `MicFrames` defines a chunk.
const FRAME_MS: u64 = 10;

/// How long the person waits for the agent after speaking: from under the default endpointing
/// delay to well past it, so the turn commits on some waits and is still open on others.
const ENDPOINT_MS: Range<u64> = 300..1500;

/// A model's time to its first token.
const THINK_MS: Range<u64> = 100..600;

/// Rounds of answering what the agent's reply owes: the model, then synthesis, then playout.
const REPLY_ROUNDS: usize = 3;

// ==============================================================================================
// Running one conversation
// ==============================================================================================

/// Drive a conversation from a seed: speech segments with provider verdicts around them, agent
/// replies between them, until the budget of applied inputs runs out; then close the session down
/// exactly as [`explore`] does.
pub fn explore(setup: Setup, seed: u64, budget: usize) -> Outcome {
    explore_at(setup, seed, budget, Dials::of(seed))
}

/// [`explore`] with chosen dials instead of the seed's.
pub fn explore_at(setup: Setup, seed: u64, budget: usize, dials: Dials) -> Outcome {
    explore_with(setup, seed, budget, dials, Policy::of(seed))
}

/// [`explore_at`] under a chosen queue policy instead of the seed's.
pub fn explore_with(
    setup: Setup,
    seed: u64,
    budget: usize,
    dials: Dials,
    policy: Policy,
) -> Outcome {
    crate::drive::simulate_with(seed, converse_in(setup, seed, budget, dials, policy))
}

/// The two dials that shape a whole run rather than one actor.
#[derive(Clone, Copy, Debug)]
pub struct Dials {
    /// See [`randomness`].
    pub randomness: f64,
    /// How long the call lasts before someone ends it, in ms of virtual time.
    pub call_length_ms: f64,
}

impl Dials {
    /// The seed's own: randomness flat over orders of magnitude, a call length as calls are.
    pub fn of(seed: u64) -> Self {
        let mut rng = Rng::with_seed(seed ^ 0x6361_6c6c);
        // A standard normal from two uniforms (Box-Muller).
        let normal =
            (-2.0 * (1.0 - rng.f64()).ln()).sqrt() * (std::f64::consts::TAU * rng.f64()).cos();
        let length = (CALL_MEDIAN_MS.ln() + CALL_SPREAD * normal).exp();
        Dials {
            randomness: randomness(seed),
            call_length_ms: length.clamp(*CALL_LENGTH_MS.start(), *CALL_LENGTH_MS.end()),
        }
    }

    /// A pure conversation nobody hangs up: what the soak and the coherence tests want.
    pub fn conversation() -> Self {
        Dials { randomness: 0.0, call_length_ms: f64::INFINITY }
    }
}

async fn converse_in(
    mut setup: Setup,
    seed: u64,
    budget: usize,
    dials: Dials,
    policy: Policy,
) -> Outcome {
    // A provider delay strays from the measured ones as often as the run makes a uniform move.
    setup.stray = dials.randomness / (1.0 + dials.randomness);
    let mut talk = Talk {
        run: Run::start(&setup, seed, policy).await,
        outcome: Outcome {
            case: Case { setup, schedule: seed, policy, ..Case::default() },
            ..Outcome::default()
        },
        rng: Rng::with_seed(seed),
        budget,
        unanswered: BTreeSet::new(),
        playback: Default::default(),
        planned: BTreeSet::new(),
        streams: BTreeMap::new(),
        delivered: 0,
        ms_per_word: measured(&mut Rng::with_seed(seed ^ 0x0076_6f69_6365), &MS_PER_WORD),
        turn_words: 0,
        speaker: None,
        current_speaker: None,
        // Each per-run draw has its own stream off the seed -- the words "Persona" and "Host" as
        // bytes -- so a caller and a program never share a coin, and adding a rate to one never
        // changes the other.
        persona: Persona::draw(&mut Rng::with_seed(seed ^ 0x0050_6572_736f_6e61)),
        host: Host::draw(&mut Rng::with_seed(seed ^ 0x0000_0000_486f_7374)),
        providers: Providers::draw(seed),
        randomness: dials.randomness,
        call_length_ms: dials.call_length_ms,
        ending: false,
        tasks: Rng::with_seed(seed ^ 0x0000_0056_5461_736b),
    };
    talk.speaker = talk.draw::<Option<Speaker>>();
    match talk.converse().await {
        Err(Halt::Found(found)) => *found,
        Ok(()) | Err(Halt::Spent | Halt::Ended) => {
            crate::drive::finish(talk.run, talk.outcome).await
        }
    }
}

/// Why a conversation stops before its natural end: a contract failed, or the budget is spent.
enum Halt {
    Found(Box<Outcome>),
    Spent,
    /// The call is over: closed by the developer, the room or the session itself.
    Ended,
}

/// One conversation in progress: the run, what it has produced, and the dice.
struct Talk {
    run: Run,
    outcome: Outcome,
    rng: Rng,
    budget: usize,
    /// Predictions the predictor will never answer, so the driver's timeout runs on them.
    unanswered: BTreeSet<u32>,
    /// The speaker: the segments it reported started and how far, and whether the output reset.
    playback: Playback,
    /// The chat requests the model has planned its answer to: every one ever opened, once.
    planned: BTreeSet<u32>,
    /// The model's answers still streaming, by chat: each delivery and the instant it is due.
    streams: BTreeMap<u32, VecDeque<(u64, Input)>>,
    /// Model deliveries applied so far: whether a reply is still moving.
    delivered: usize,
    /// How fast the agent's voice speaks this run, in words per second times ten.
    ms_per_word: u64,
    /// Words the model has planned since the person last started speaking: how long the reply
    /// they are about to hear is.
    turn_words: u64,
    /// Who the recogniser says is speaking, when it diarises: one person, one id, all run long.
    speaker: Option<Speaker>,
    /// Whose voice the stretch being spoken is, as the recogniser will report it.
    current_speaker: Option<Speaker>,
    /// Who is on the line this run.
    persona: Persona,
    /// The developer's program this run.
    host: Host,
    /// How the models, outputs and room behave this run.
    providers: Providers,
    /// See [`randomness`].
    randomness: f64,
    /// How long this call lasts, in ms of virtual time; see [`CALL_LENGTH_MS`].
    call_length_ms: f64,
    /// The call is being ended: no second ending starts inside the first.
    ending: bool,
    /// The `VoiceTask` coins, a stream of their own: a run without tasks draws as it always did.
    tasks: Rng,
}

/// Plumbing: how an input reaches the driver, and the small draws.
impl Talk {
    /// Record, apply, check: the one way an input reaches the driver in this mode. Recorded before
    /// it is applied, so a case that ends in a panic still says what caused it.
    async fn send(&mut self, input: Input) -> Result<(), Halt> {
        while self.rng.f64() < self.randomness / (1.0 + self.randomness) {
            self.uniform().await?;
        }
        self.apply(input).await
    }

    /// One uniform move, then time passing half the time; labelled in the case. A uniform move
    /// that closes the session ends the call like any other close.
    async fn uniform(&mut self) -> Result<(), Halt> {
        let Some(row) = uniform::sample_next(&self.run.record, &mut self.rng) else {
            return Err(Halt::Ended);
        };
        self.outcome.case.random.push(self.outcome.case.inputs.len());
        self.apply(row).await?;
        if self.rng.u8(0..100) < uniform::TICK {
            let tick = uniform::tick(&self.run.record, &mut self.rng);
            self.outcome.case.random.push(self.outcome.case.inputs.len());
            self.apply(tick).await?;
        }
        Ok(())
    }

    /// One input to the driver, with no handlers run on what it did: what `send` and the
    /// handlers themselves go through. Before an input that lets time pass, the speaker finishes
    /// what it played to the end.
    async fn apply(&mut self, input: Input) -> Result<(), Halt> {
        if matches!(
            input,
            Input::Advance { .. }
                | Input::AwaitAnswer { .. }
                | Input::AwaitSpeech { .. }
                | Input::AwaitExit {}
        ) {
            self.finish_drained().await?;
        }
        self.apply_now(input).await
    }

    /// One input to the driver, as it is.
    async fn apply_now(&mut self, input: Input) -> Result<(), Halt> {
        if self.outcome.case.inputs.len() >= self.budget {
            return Err(Halt::Spent);
        }
        self.outcome.case.inputs.push(input.clone());
        self.run.apply(&input).await;
        match crate::drive::guard(&self.run, &mut self.outcome) {
            Some(found) => Err(Halt::Found(Box::new(found))),
            None => Ok(()),
        }
    }

    fn now_ms(&self) -> u64 {
        self.run.record.now().at_ms
    }

    /// One of the agent's lines.
    fn pick_rt<'a>(&mut self, of: &'a [RtEvent]) -> &'a RtEvent {
        &of[self.rng.usize(0..of.len())]
    }

    /// One of the agent's lines: a sentence or two, and now and then a long read-out that plays
    /// for most of a minute, during which the program and the person keep acting.
    fn line(&mut self) -> AgentText {
        match self.rng.usize(0..AGENT_TURNS.len() + 1) {
            i if i < AGENT_TURNS.len() => AgentText(AGENT_TURNS[i].to_owned()),
            _ => AgentText(READ_OUT.to_owned()),
        }
    }

    /// One value of a table type, drawn the way the table draws it.
    fn draw<T: Make>(&mut self) -> T {
        T::make(&mut Sampler::new(self.run.record.live(), &mut self.rng)).expect("a plain variant")
    }

    /// The first live resource of this kind: the microphone, detector or recogniser spoken into.
    fn first(&self, kind: Kind) -> Option<u32> {
        self.run.record.live().of(kind).first().copied()
    }

    /// Whether the realtime server is detecting turns now, and whether semantically: its socket
    /// is open, the setup has it detect, and the driver did not switch that off at the open.
    fn server_vad(&self) -> Option<bool> {
        let realtime = self.run.setup.realtime?;
        self.first(Kind::Realtime)?;
        let disabled = self
            .run
            .record
            .journal_where(|step| match step {
                Step::Realtime(RealtimeStep::Opened { turn_detection_disabled }) => {
                    Some(*turn_detection_disabled)
                }
                _ => None,
            })
            .last()
            .is_none_or(|(_, disabled)| disabled);
        (realtime.turn_detection && !disabled).then_some(realtime.semantic)
    }
}

/// The conversation: the person's stretches and the agent's replies, turn by turn.
impl Talk {
    /// One call the developer might make mid-call: a command, text or audio fed into a stream it
    /// opened, or a hook answered late -- never what a person, a model or an output does, which
    /// stay causal. Never a drain or a close: how the call ends is its own dial ([`Talk::hang_up`]).
    async fn stray(&mut self) -> Result<(), Halt> {
        let stray = {
            let mut sampler = Sampler::new(self.run.record.live(), &mut self.rng);
            let offered: Vec<Input> = Input::available(&mut sampler)
                .into_iter()
                .filter(|row| {
                    hosts(row) && !matches!(row, Input::Drain { .. } | Input::Close { .. })
                })
                .collect();
            (!offered.is_empty()).then(|| sampler.pick(&offered).clone())
        };
        match stray {
            Some(stray) => self.send(stray).await,
            None => Ok(()),
        }
    }

    /// A beat of whatever the conversation is doing: the developer may call the API, a camera
    /// delivers frames, a playing speaker may reset.
    async fn beat(&mut self) -> Result<(), Halt> {
        if !self.ending && self.run.record.now().at_ms as f64 >= self.call_length_ms {
            return self.hang_up().await;
        }
        self.act().await
    }

    /// What the developer, the camera and the speaker do in a beat.
    async fn act(&mut self) -> Result<(), Halt> {
        // The speaker plays on its own clock, whatever the person is doing: it finishes what has
        // played its length, starts what it was handed next, and reports what has played.
        self.finish_played().await?;
        self.hear().await?;
        self.report_progress().await?;
        let hearing = self.hearing();
        let rate = if hearing { self.host.speaking } else { self.host.during };
        if self.rng.u8(0..100) < rate {
            self.stray().await?;
        }
        if let Some(video) = self.first(Kind::Video)
            && self.rng.u8(0..100) < CAMERA
        {
            let frames = Count(self.rng.u32(1..=3));
            self.send(Input::VideoFrames { video: Ref::new(video), frames }).await?;
        }
        if hearing
            && self.rng.u8(0..100) < self.providers.sink_resets
            && let Some(output) = self.first(Kind::Output)
        {
            self.playback.reset = true;
            self.send(Input::SinkReset { output: Ref::new(output) }).await?;
        }
        // A voice provider that went down comes back, as outages end.
        if self.voice_down() && self.rng.u8(0..100) < OUTAGE_ENDS {
            self.send(Input::TtsOutage { down: false }).await?;
        }
        // A program that turns a leg off -- the agent's voice held, the microphone muted --
        // turns it back on a few seconds later.
        if self.rng.u8(0..100) < RESTORES {
            for restore in self.legs_off() {
                self.send(restore).await?;
            }
        }
        self.detect_machine().await
    }

    /// Whether the voice provider is down now: the last outage the case stated still stands.
    fn voice_down(&self) -> bool {
        self.run
            .record
            .applied()
            .iter()
            .rev()
            .find_map(|a| match a.input {
                Input::TtsOutage { down } if a.accepted => Some(down),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// The rows that turn back on whatever leg the session last had turned off.
    fn legs_off(&self) -> Vec<Input> {
        let mut off: BTreeMap<&'static str, Input> = BTreeMap::new();
        for applied in self.run.record.applied().iter().filter(|a| a.accepted) {
            match &applied.input {
                Input::SetAudioOutput { attach } => {
                    let restore = Input::SetAudioOutput { attach: true };
                    if *attach {
                        off.remove("voice")
                    } else {
                        off.insert("voice", restore)
                    };
                }
                Input::SetIoEnabled { leg, enabled } => {
                    let restore = Input::SetIoEnabled { leg: *leg, enabled: true };
                    let key = match leg {
                        IoKind::AudioInput => "mic",
                        IoKind::AudioOutput => "speaker",
                        IoKind::Transcription => "transcript",
                    };
                    if *enabled {
                        off.remove(key)
                    } else {
                        off.insert(key, restore)
                    };
                }
                _ => {}
            }
        }
        off.into_values().collect()
    }

    /// The developer's `async with AMD(...) as detector: result = await detector.execute()`: the
    /// SIP call is answered a few beats in, `execute()` returns once the verdict is out, and the
    /// block is left (`aclose()`) right after or later.
    async fn detect_machine(&mut self) -> Result<(), Halt> {
        let Some(amd) = self.run.setup.amd else { return Ok(()) };
        let sent = |run: &Run, step: AmdInput| {
            run.record.applied().iter().any(|a| a.accepted && a.input == Input::Amd { input: step })
        };
        if amd.answer == AmdAnswer::Sip
            && !sent(&self.run, AmdInput::Answer)
            && self.rng.u8(0..100) < SIP_ANSWERS
        {
            self.send(Input::Amd { input: AmdInput::Answer }).await?;
        }
        let verdict = self
            .run
            .record
            .host_view()
            .events()
            .iter()
            .any(|(_, ev)| matches!(ev, SessionEvent::AmdResult { .. }));
        if verdict && !sent(&self.run, AmdInput::Execute) {
            self.send(Input::Amd { input: AmdInput::Execute }).await?;
        }
        // `participant_missing` is the call that ended before anyone spoke: the developer hangs
        // up rather than leave the agent talking to an empty room, where its voice waits for a
        // listener that is not coming (the room's output plays once subscribed).
        let missing = self.run.record.host_view().events().iter().any(|(_, ev)| {
            matches!(ev, SessionEvent::AmdResult { verdict, .. }
                if verdict.reason == lk_agents_abi::voice::amd::AmdReason::ParticipantMissing)
        });
        if missing && !self.ending && !self.closing() {
            return Box::pin(self.hang_up()).await;
        }
        if sent(&self.run, AmdInput::Execute)
            && !sent(&self.run, AmdInput::Close)
            && self.rng.u8(0..100) < AMD_LEFT
        {
            self.send(Input::Amd { input: AmdInput::Close }).await?;
        }
        Ok(())
    }

    /// The session began closing: the host or the room asked, or the session gave up on its own.
    fn closing(&self) -> bool {
        self.run
            .record
            .applied()
            .iter()
            .any(|a| a.accepted && matches!(a.input, Input::Close { .. }))
            || self.run.record.internal_view().close_requested().is_some()
    }

    /// The call ends: the developer closes it, drains then closes, or the person hangs up or the
    /// room goes. The developer keeps calling the API while the session winds down.
    async fn hang_up(&mut self) -> Result<(), Halt> {
        self.ending = true;
        if self.rng.u8(0..100) < TEARDOWN {
            for _ in 0..self.rng.u8(1..=3) {
                self.stray().await?;
            }
        }
        let room = self.run.setup.room.is_some();
        match self.rng.u8(0..if room { 5 } else { 3 }) {
            0 => self.send(Input::Close { drain: false }).await?,
            1 => self.send(Input::Close { drain: true }).await?,
            2 => {
                let wait = self.rng.bool();
                self.send(Input::Drain { wait }).await?;
                let lingers = Millis(self.rng.u64(0..5_000));
                self.advance(lingers.0).await?;
                self.send(Input::Close { drain: false }).await?;
            }
            3 => self.send(Input::Room { event: RoomIn::Unlinked }).await?,
            _ => self.send(Input::Room { event: RoomIn::Closed }).await?,
        }
        // A room with nobody linked has nobody to leave it: the developer closes the session.
        let refused = self.run.record.applied().last().is_some_and(|a| !a.accepted);
        if refused {
            self.send(Input::Close { drain: false }).await?;
        }
        for _ in 0..CLOSING_BEATS {
            self.advance(BEAT_MS).await?;
            self.act().await?;
        }
        Err(Halt::Ended)
    }

    /// A wait of `ms`, or, at the run's `aims` rate, one landing on a deadline the driver armed.
    fn wait(&mut self, ms: Millis) -> Millis {
        if self.rng.u8(0..100) < self.providers.aims {
            let mut sampler = Sampler::new(self.run.record.live(), &mut self.rng);
            return uniform::aim_at_armed_timer(&self.run.record, &mut sampler).unwrap_or(ms);
        }
        ms
    }

    async fn converse(&mut self) -> Result<(), Halt> {
        // A run whose voice is down says so up front: the outage is standing, and every
        // synthesis the driver opens fails at the open until the provider comes back.
        if self.providers.voice_down {
            self.send(Input::TtsOutage { down: true }).await?;
        }
        let mut barged = false;
        loop {
            // The call is winding down: the person is gone, the developer is not.
            if self.closing() {
                for _ in 0..CLOSING_BEATS {
                    self.advance(BEAT_MS).await?;
                    self.act().await?;
                }
                return Err(Halt::Ended);
            }
            // The person drops off the call and comes back a few seconds later.
            if self.run.setup.room.is_some() && self.rng.u8(0..100) < self.persona.leaves {
                self.send(Input::Room { event: RoomIn::Unlinked }).await?;
                let away = Millis(self.rng.u64(500..5_000));
                self.advance(away.0).await?;
                self.send(Input::Room { event: RoomIn::Linked }).await?;
                continue;
            }
            // In a room, the person sometimes types instead of speaking, and their connection
            // sometimes drops and recovers: the same microphone, subscribed again.
            if self.run.setup.room_legs.is_some() && self.rng.u8(0..100) < self.persona.leaves {
                self.send(Input::Room { event: RoomIn::Chat }).await?;
            }
            if self.run.setup.room_legs.is_some() && self.rng.u8(0..100) < self.persona.leaves {
                self.send(Input::Room { event: RoomIn::Resubscribed }).await?;
            }
            // In a room, the agent's connection drops and resumes a few seconds later, and
            // someone else joins the call for a while.
            if self.run.setup.room.is_some() && self.rng.u8(0..100) < self.persona.leaves {
                self.send(Input::Room { event: RoomIn::StateChanged }).await?;
                let away = Millis(self.rng.u64(200..5_000));
                self.advance(away.0).await?;
                self.send(Input::Room { event: RoomIn::Reconnected }).await?;
            }
            if self.run.setup.room.is_some() && self.rng.u8(0..100) < self.persona.leaves {
                let guest = if self.rng.bool() { RoomIn::GuestJoined } else { RoomIn::GuestLeft };
                self.send(Input::Room { event: guest }).await?;
            }
            // On the console, the developer's TUI asks the session things as the call goes.
            if self.run.setup.console && self.rng.u8(0..100) < self.persona.leaves {
                let ask: ConsoleAsk = self.draw();
                self.send(Input::Console { ask }).await?;
            }
            // Nothing to speak into: the developer detached the microphone. The person keeps
            // talking into a dead line, which to the session is silence, so the line is silent for
            // a beat and the conversation carries on -- the developer may attach one again.
            let Some(mic) = self.first(Kind::Mic) else {
                self.advance(BEAT_MS).await?;
                self.beat().await?;
                continue;
            };
            if self.rng.u8(0..100) < self.host.unprompted {
                self.stray().await?;
            }
            // A person talking over the agent does not pause first.
            if !barged && self.rng.u8(0..100) < self.persona.pauses {
                let pause = self.segment(false, true);
                self.stretch(mic, &pause, Vec::new()).await?;
            }
            let done = self.rng.u8(0..100) >= self.persona.hesitates;
            self.turn_words = 0;
            let speech = self.segment(true, done);
            let verdicts = self.verdicts(&speech);
            self.stretch(mic, &speech, verdicts).await?;
            if done {
                // Under manual turn detection the host says when the turn is over; nothing else
                // will. The row is drawn from the table so its timeouts are the table's own.
                if self.run.setup.turn == TurnMode::Manual {
                    let commit = {
                        let mut sampler = Sampler::new(self.run.record.live(), &mut self.rng);
                        Input::available(&mut sampler)
                            .into_iter()
                            .find(|row| matches!(row, Input::CommitUserTurn { .. }))
                    };
                    if let Some(commit) = commit {
                        self.send(commit).await?;
                    }
                }
                barged = self.reply().await?;
            } else {
                // A pause for thought, then more words: the agent should not get a turn here,
                // and whether it takes one anyway is the endpointing decision under test.
                let pause = self.segment(false, false);
                self.stretch(mic, &pause, Vec::new()).await?;
                barged = false;
            }
        }
    }
}

/// The agent's turn.
impl Talk {
    /// Let the agent answer: wait out endpointing, then answer what its reply owes -- the tools,
    /// the hooks, the playout -- round by round, while the model streams on its own clock. Returns
    /// whether the person is about to talk over a playout still going.
    async fn reply(&mut self) -> Result<bool, Halt> {
        let wait = Millis(self.rng.u64(ENDPOINT_MS) * self.persona.patience / 100);
        let wait = self.wait(wait);
        self.advance(wait.0).await?;
        for _ in 0..REPLY_ROUNDS {
            // The person is done: the predictor, if asked, says so.
            self.predict(true).await?;
            self.plan_chats();
            let (tools, hooks, synthesising, playing, speaking, generations) = {
                let live = self.run.record.live();
                (
                    live.of(Kind::Tool),
                    live.of(Kind::Hook),
                    live.of(Kind::Tts),
                    live.of(Kind::Segment),
                    !live.of(Kind::Speech).is_empty(),
                    live.of(Kind::Generation),
                )
            };
            let mut rows = Vec::new();
            // The realtime server finishes its responses on its own clock. What it does wrong it
            // does at its own rate: drops the socket mid-reply and takes connections again, errs,
            // fails the response or the transcription, or opens a response of its own.
            if let Some(socket) = self.first(Kind::Realtime) {
                let socket_ref = Ref::new(socket);
                if !generations.is_empty() && self.rng.u8(0..100) < self.providers.realtime_errors {
                    let event = *self.pick_rt(&[
                        RtEvent::ServerError,
                        RtEvent::ResponseFailed,
                        RtEvent::TranscriptionFailed,
                        RtEvent::SecondGeneration,
                    ]);
                    rows.push(Input::RealtimeEvent { socket: socket_ref, event });
                }
                // A fallback adapter's model fails for good now and then: it swaps to the other.
                if self.run.setup.realtime.is_some_and(|r| r.fallback)
                    && self.rng.u8(0..100) < self.providers.socket_drops
                {
                    rows.push(Input::RealtimeEvent {
                        socket: socket_ref,
                        event: RtEvent::FatalError,
                    });
                }
                // The server drops the socket mid-reply and comes back.
                if self.rng.u8(0..100) < self.providers.socket_drops {
                    let socket = Ref::new(socket);
                    rows.push(Input::RealtimeEvent { socket, event: RtEvent::Disconnected });
                    rows.push(Input::Advance { ms: Millis(self.rng.u64(100..2_000)) });
                    rows.push(Input::RealtimeEvent { socket, event: RtEvent::Reconnected });
                }
            }
            // Tools and hooks answer with whatever the table can draw for them: a return, a
            // value, an error, a stop -- the agent's reply path under each is what is under test.
            for tool in tools {
                if self.run.tool_detached(tool) {
                    // A detached tool speaks again through the host, and ends on a final update.
                    let progress = self.draw::<Option<AgentText>>();
                    let end = self.draw::<ToolAnswer>();
                    rows.push(Input::ToolUpdate { tool: Ref::new(tool), progress, end });
                    continue;
                }
                if self.rng.u8(0..100) < self.host.fillers {
                    // The host scopes a filler on the tool it is waiting on, as `with_filler`
                    // does; the row is the table's, so its delays are the table's.
                    let filler = {
                        let mut sampler = Sampler::new(self.run.record.live(), &mut self.rng);
                        Input::available(&mut sampler)
                            .into_iter()
                            .find(|row| matches!(row, Input::FillerStart { .. }))
                    };
                    rows.extend(filler);
                }
                // A body awaiting its `VoiceTask` returns once the task answered, or now and then
                // first (the await cancelled); a body may start one instead of returning.
                if self.run.awaits_task(tool) && self.tasks.u8(0..100) >= 10 {
                    continue;
                }
                if !self.run.awaited_task(tool) && self.tasks.u8(0..100) < self.host.voice_tasks {
                    let agent = (*self.tasks.choice(&["collector", "speller"]).unwrap()).into();
                    let reply = self.tasks.bool();
                    rows.push(Input::VoiceTask { tool: Ref::new(tool), agent, reply });
                    continue;
                }
                let answer = self.draw::<ToolAnswer>();
                let text = self.line();
                rows.push(Input::ToolAnswered { tool: Ref::new(tool), answer, text });
            }
            if self.voice_down() {
                for tts in synthesising {
                    rows.push(Input::TtsFail { tts: Ref::new(tts) });
                }
            }
            for hook in hooks {
                let answer = self.draw::<HookAnswer>();
                let text = self.line();
                rows.push(Input::HookDone { hook: Ref::new(hook), answer, text });
            }
            if rows.is_empty() && self.streams.is_empty() && playing.is_empty() && !speaking {
                return Ok(false);
            }
            // The tools and the hooks answer whatever the person decides to do next: their
            // answers go out before the person hears the reply and makes up their mind, so a tool
            // that asks for idle has asked before any interruption lands on its step.
            self.beat().await?;
            for row in rows {
                self.send(row).await?;
            }
            if self.listen().await? {
                return Ok(true);
            }
            let think = Millis(self.rng.u64(THINK_MS) * self.providers.latency / 100);
            let think = self.wait(think);
            self.advance(think.0).await?;
            self.await_playout().await?;
        }
        Ok(false)
    }

    /// The reply plays out a beat at a time: the model streams on its own clock, the voice speaks
    /// what it has, the developer acts, and the person listens -- and may talk over it anywhere,
    /// from its first word to its last, or say "mm-hm" without wanting the floor. Returns whether
    /// the person barged in. A reply that stops moving -- nothing streams, nothing new plays -- is
    /// left after the person's patience, as a caller hearing silence speaks again.
    async fn listen(&mut self) -> Result<bool, Halt> {
        let mut barge_at: Option<u64> = None;
        let mut decided = false;
        let mut still_ms = 0;
        let mut last = self.progress_mark();
        loop {
            self.hear().await?;
            let hearing = self.hearing();
            // Whether to talk over this reply is decided as it begins, anywhere in what the
            // person can tell is coming: the words the model has planned so far, at the voice's
            // pace.
            if hearing && !decided {
                decided = true;
                if self.rng.u8(0..100) < self.persona.barges {
                    let reply_ms = self.turn_words * self.ms_per_word;
                    barge_at = Some(self.now_ms() + self.rng.u64(0..=reply_ms.max(BEAT_MS)));
                }
            }
            if hearing && barge_at.is_some_and(|at| self.now_ms() >= at) {
                self.report_progress().await?;
                return Ok(true);
            }
            let streaming = !self.streams.is_empty();
            let playing = self.first(Kind::Segment).is_some();
            if !streaming && !playing {
                return Ok(false);
            }
            let mark = self.progress_mark();
            if mark == last {
                still_ms += BEAT_MS;
                if still_ms >= self.persona.gives_up_ms {
                    return Ok(false);
                }
            } else {
                (last, still_ms) = (mark, 0);
            }
            let beat = match barge_at {
                Some(at) if hearing => at.saturating_sub(self.now_ms()).clamp(1, BEAT_MS),
                _ => BEAT_MS,
            };
            self.advance(beat).await?;
            self.beat().await?;
            if hearing
                && self.rng.u8(0..100) < self.persona.backchannels
                && let Some(mic) = self.first(Kind::Mic)
            {
                let said = self.backchannel();
                let verdicts = self.verdicts(&said);
                self.stretch(mic, &said, verdicts).await?;
            }
        }
    }

    /// What has moved in the reply: model deliveries applied, segments opened and closed.
    fn progress_mark(&self) -> (usize, u32, usize) {
        let live = self.run.record.live();
        (self.delivered, live.opened(Kind::Segment), live.of(Kind::Segment).len())
    }

    /// Wait, a beat at a time, for the agent's audio to reach the speaker -- the moment a listener
    /// hears it start -- so there is something to let finish or to cut into.
    ///
    /// The person cannot see the driver: they hear silence, and after their own patience they
    /// speak again, whatever the driver is doing. The wait also ends when the audio arrives, when
    /// the speech is gone, or when the driver is waiting on the case (a model, tool or hook still
    /// to answer: the next round's job). A silence the driver had no right to leave is for a rule
    /// to catch, not for the person to avoid.
    async fn await_playout(&mut self) -> Result<(), Halt> {
        let owed_by_case =
            |t: &Self| [Kind::Chat, Kind::Tool, Kind::Hook].iter().any(|k| t.first(*k).is_some());
        let mut waited = 0;
        while waited < self.persona.gives_up_ms
            && self.first(Kind::Segment).is_none()
            && self.first(Kind::Speech).is_some()
            && !owed_by_case(self)
        {
            self.advance(BEAT_MS).await?;
            self.beat().await?;
            self.predict(true).await?;
            waited += BEAT_MS;
        }
        Ok(())
    }
}

/// A row the host program sends: a command, text or audio fed into a stream it opened, or a hook
/// answered late. Everything else is a person, a model, an output, the room or the clock.
fn hosts(row: &Input) -> bool {
    row.is_command()
        || matches!(
            row,
            Input::FeedText { .. }
                | Input::FeedAudio { .. }
                | Input::FeedEnd { .. }
                | Input::HookDoneLate { .. }
        )
}
