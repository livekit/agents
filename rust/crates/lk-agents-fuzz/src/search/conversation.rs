//! Conversation mode: a person on a phone, talking turn by turn to the agent.
//!
//! The uniform sampler reaches the second half of a chain by luck. A person on a phone does not:
//! they speak, a detector and a recogniser report what they heard in a fixed order with
//! provider-shaped delays, the agent answers, and they speak again -- sometimes over the agent.
//! This mode drives the microphone that way, so the sequences the turn machine was written for
//! (start of speech, a final, an end of speech, an endpointing delay) happen every turn rather
//! than once in ten runs.
//!
//! Three things are drawn once per run and hold for its length: who is on the line (the
//! [`Persona`]), the developer's program (the [`Host`]), and whether the voice provider is down.
//! Within a run every distribution is one named range with the reason for it. There are no
//! per-variant weights: every attempt at aiming the mode at a variant was measured on the
//! survey's yardstick and lost.
use std::{collections::BTreeSet, ops::Range, time::Instant};

use fastrand::Rng;
use lk_agents_abi::voice::amd::AmdCategory;
use lk_agents_voice::{
    amd::{
        classifier::MAX_EXTENSION_SECS,
        detector::{POSTPONE_TERMINATION_TOOL, SAVE_PREDICTION_TOOL},
    },
    trace::{GenerationStep, Step},
};

use crate::{
    drive::{Case, Outcome, Run, Setup},
    input::{
        AGENT_TURNS, AgentText, Confidence, Count, HookAnswer, Input, Make, Millis, Ref, RoomIn,
        RtEvent, Sampler, Speaker, ToolAnswer, ToolArgs, TurnMode, USER_TURNS, UserText,
    },
    record::{Detail, Kind, Tag},
};

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
    /// apart: what a speaker filter is for.
    other_speaker: u8,
    /// How often the interruption detector's verdict agrees with the utterance's length -- a
    /// barge-in at or over the driver's minimum, a backchannel under it.
    detector_right: u8,
    /// How often the model answers with a tool call rather than words, when the agent has tools.
    tool_calls: u8,
    /// How often the host scopes a filler on a tool it is waiting on.
    fillers: u8,
}

impl Persona {
    fn draw(rng: &mut Rng) -> Self {
        Persona {
            pauses: rng.u8(0..=60),
            hesitates: rng.u8(0..=50),
            barges: rng.u8(0..=80),
            noise: rng.u8(0..=30),
            other_speaker: rng.u8(0..=30),
            detector_right: rng.u8(50..=100),
            tool_calls: rng.u8(0..=60),
            fillers: rng.u8(0..=60),
        }
    }
}

/// One stretch of the person's microphone: speaking or silent, this long, at this pacing.
///
/// The one primitive. A pause is a silent segment, a barge-in is a speech segment placed while
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

/// How long the person listens to a reply before cutting in: from at once, inside the driver's
/// backchannel boundary, to well past it.
const LISTEN_MS: Range<u64> = 0..2_000;

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
    // The end-of-turn predictor has no word here: the driver asks it, so its answer is placed
    // when a `Kind::Prediction` is live, not on a schedule. See `Talk::predict`.
}

/// When the detector notices speech began, after the first frame of it.
const VAD_START_MS: Range<u64> = 30..150;

/// When the detector notices speech ended, after the last frame of it.
const VAD_END_MS: Range<u64> = 100..400;

/// When the recogniser opens the utterance: slower than the detector, as it waits for words.
const STT_START_MS: Range<u64> = 50..300;

/// When the recogniser's final arrives, after speech ends: a provider's tail latency.
const STT_FINAL_MS: Range<u64> = 100..800;

/// One final in ten is slow: a provider under load answers after the endpointing delay has
/// already run out, so the driver's wait for a transcript, and its timeout, are exercised.
const STT_LATE_FINAL: u8 = 10;
const STT_LATE_FINAL_MS: Range<u64> = 1_500..3_000;

/// How often the recogniser streams an interim: roughly one every so many milliseconds of speech,
/// as a streaming provider does, so a long turn has many and a short one has one.
const STT_INTERIM_EVERY_MS: u64 = 400;

/// One turn in four has an interim the recogniser gets wrong and corrects in the next: a word
/// mis-heard, then revised, which is what a driver that acts on interims must survive.
const STT_REVISION: u8 = 25;

/// One turn in five that is long enough comes back as two finals, the provider having cut the
/// utterance at a breath: the turn is the two segments together, and the driver must not act on
/// the first as if it were the whole.
const STT_SPLIT_FINAL: u8 = 20;

/// Detector confidence while speaking: straddles 50, so hysteresis and flicker are reached.
const SPEAKING_P: Range<u8> = 45..95;

/// Detector confidence while silent.
const SILENT_P: Range<u8> = 0..20;

/// The predictor's confidence that the turn is over when the person is done, and when they are
/// only pausing.
const EOT_DONE_P: Range<u8> = 70..101;

const EOT_MORE_P: Range<u8> = 0..31;

/// How often the predictor lets the driver down: half of these fail the request, half never
/// answer it, so the driver's timeout path runs.
const EOT_LOST: u8 = 6;

/// One run in eight has a voice that is down for the whole run: every synthesis fails, so the
/// driver's retries run out and it loses its voice, rather than a failure here and there.
const VOICE_DOWN: u8 = 12;

/// Speech over the agent shorter than this is judged a backchannel -- "mm-hm" -- not a barge-in,
/// the distinction the adaptive detector exists to make. Matches the driver's default minimum
/// interruption duration, so by default the detector and the driver agree on what counts.
const BACKCHANNEL_MS: u64 = 500;

// ==============================================================================================
// The developer's program
// ==============================================================================================

/// The developer's program, drawn once per run. Modelled by one rate so far: how often a call
/// lands unprompted before a stretch of speech -- the call at the wrong moment, which developers
/// write. A reactive form, handlers firing on journal families and drawing from the host's rows,
/// was measured on 2026-09-23 and lost on variants and findings while failing most runs: a
/// handler that draws uniformly from the API is not a program, and a program needs a model of
/// what handlers do that this crate does not have yet.
#[derive(Clone, Debug)]
struct Host {
    /// The chance of an unprompted call before a stretch of speech.
    unprompted: u8,
}

impl Host {
    fn draw(rng: &mut Rng) -> Self {
        Host { unprompted: rng.u8(0..=10) }
    }
}

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

/// How long a flushed audio segment plays before the case reports it finished.
const PLAYOUT_MS: Range<u64> = 500..2000;

/// How fast the agent's voice speaks, in words per second: the playout of a reply is the length
/// of its words at this rate, not a draw, so a long sentence takes long to say.
const AGENT_WORDS_PER_SEC_X10: Range<u64> = 23..30;

/// Rounds of answering what the agent's reply owes: the model, then synthesis, then playout.
const REPLY_ROUNDS: usize = 3;

// ==============================================================================================
// Running one conversation
// ==============================================================================================

/// Drive a conversation from a seed: speech segments with provider verdicts around them, agent
/// replies between them, until the budget of applied inputs runs out; then close the session down
/// exactly as [`explore`] does.
pub fn explore_conversation(setup: Setup, seed: u64, budget: usize) -> Outcome {
    crate::drive::simulate_with(seed, converse_in(setup, seed, budget))
}

async fn converse_in(setup: Setup, seed: u64, budget: usize) -> Outcome {
    let started = Instant::now();
    let mut talk = Talk {
        run: Run::start(&setup).await,
        outcome: Outcome {
            case: Case { setup, inputs: Vec::new(), schedule: seed },
            ..Outcome::default()
        },
        rng: Rng::with_seed(seed),
        started,
        budget,
        unanswered: BTreeSet::new(),
        heard: BTreeSet::new(),
        speaker: None,
        voice_down: false,
        current_speaker: None,
        // Each per-run draw has its own stream off the seed -- the words "Persona" and "Host" as
        // bytes -- so a caller and a program never share a coin, and adding a rate to one never
        // changes the other.
        persona: Persona::draw(&mut Rng::with_seed(seed ^ 0x0050_6572_736f_6e61)),
        host: Host::draw(&mut Rng::with_seed(seed ^ 0x0000_0000_486f_7374)),
    };
    talk.speaker = talk.draw::<Option<Speaker>>();
    talk.voice_down = talk.rng.u8(0..100) < VOICE_DOWN;
    match talk.converse().await {
        Err(Halt::Found(found)) => *found,
        Ok(()) | Err(Halt::Spent) => crate::drive::finish(talk.run, talk.outcome, started).await,
    }
}

/// Why a conversation stops before its natural end: a contract failed, or the budget is spent.
enum Halt {
    Found(Box<Outcome>),
    Spent,
}

/// One conversation in progress: the run, what it has produced, and the dice.
struct Talk {
    run: Run,
    outcome: Outcome,
    rng: Rng,
    started: Instant,
    budget: usize,
    /// Predictions the predictor will never answer, so the driver's timeout runs on them.
    unanswered: BTreeSet<u32>,
    /// Segments whose audio the person has heard begin: the speaker reported them started.
    heard: BTreeSet<u32>,
    /// Who the recogniser says is speaking, when it diarises: one person, one id, all run long.
    speaker: Option<Speaker>,
    /// The voice is down for the whole run: every synthesis the driver asks for fails.
    voice_down: bool,
    /// Whose voice the stretch being spoken is, as the recogniser will report it.
    current_speaker: Option<Speaker>,
    /// Who is on the line this run.
    persona: Persona,
    /// The developer's program this run.
    host: Host,
}

/// Plumbing: how an input reaches the driver, and the small draws.
impl Talk {
    /// Record, apply, check: the one way an input reaches the driver in this mode. Recorded before
    /// it is applied, so a case that ends in a panic still says what caused it.
    async fn send(&mut self, input: Input) -> Result<(), Halt> {
        self.apply(input).await
    }

    /// One input to the driver, with no handlers run on what it did: what `send` and the
    /// handlers themselves go through.
    async fn apply(&mut self, input: Input) -> Result<(), Halt> {
        if self.outcome.case.inputs.len() >= self.budget {
            return Err(Halt::Spent);
        }
        self.outcome.case.inputs.push(input.clone());
        self.run.apply(&input).await;
        match crate::drive::guard(&self.run, &mut self.outcome, self.started) {
            Some(found) => Err(Halt::Found(Box::new(found))),
            None => Ok(()),
        }
    }

    /// One of the agent's lines.
    fn line(&mut self) -> AgentText {
        AgentText(AGENT_TURNS[self.rng.usize(0..AGENT_TURNS.len())].to_owned())
    }

    /// One value of a table type, drawn the way the table draws it.
    fn draw<T: Make>(&mut self) -> T {
        T::make(&mut Sampler::new(self.run.record.live(), &mut self.rng)).expect("a plain variant")
    }

    /// The first live resource of this kind: the microphone, detector or recogniser spoken into.
    fn first(&self, kind: Kind) -> Option<u32> {
        self.run.record.live().of(kind).first().copied()
    }
}

/// The conversation: the person's stretches and the agent's replies, turn by turn.
impl Talk {
    async fn converse(&mut self) -> Result<(), Halt> {
        // A run whose voice is down says so up front: the outage is standing, and every
        // synthesis the driver opens for the rest of the run fails at the open.
        if self.voice_down {
            self.send(Input::TtsOutage { down: true }).await?;
        }
        let mut barged = false;
        loop {
            // Nothing to speak into: the developer detached the microphone. The person keeps
            // talking into a dead line, which to the session is silence, so the line is silent for
            // a beat and the conversation carries on -- the developer may attach one again.
            let Some(mic) = self.first(Kind::Mic) else {
                self.send(Input::Advance { ms: Millis(BEAT_MS) }).await?;
                continue;
            };
            if self.rng.u8(0..100) < self.host.unprompted {
                let stray = {
                    let mut sampler = Sampler::new(self.run.record.live(), &mut self.rng);
                    // Anything the developer might do mid-call, except end the call: a drain or a
                    // close winds the session down, the room closing or the person leaving hangs
                    // it up, and after any of them every reply is refused as draining while the
                    // person here keeps talking into the microphone. Those rows are the runner's,
                    // at the end of the run; chaos mode sends them anywhere.
                    let ends_the_call = |row: &Input| {
                        matches!(
                            row,
                            Input::Drain { .. }
                                | Input::Close { .. }
                                | Input::Room { event: RoomIn::Closed | RoomIn::Unlinked }
                        )
                    };
                    let offered: Vec<Input> = Input::available(&mut sampler)
                        .into_iter()
                        .filter(|row| !ends_the_call(row))
                        .collect();
                    (!offered.is_empty()).then(|| sampler.pick(&offered).clone())
                };
                if let Some(stray) = stray {
                    self.send(stray).await?;
                }
            }
            // A person talking over the agent does not pause first.
            if !barged && self.rng.u8(0..100) < self.persona.pauses {
                let pause = self.segment(false, true);
                self.stretch(mic, &pause, Vec::new()).await?;
            }
            let done = self.rng.u8(0..100) >= self.persona.hesitates;
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

/// The person's side: a stretch of microphone, and what the providers say about it.
impl Talk {
    fn segment(&mut self, speaking: bool, done: bool) -> Segment {
        let rng = &mut self.rng;
        // What the person says decides how long they speak: their words at their pace.
        let text = speaking.then(|| USER_TURNS[rng.usize(0..USER_TURNS.len())]);
        let dur_ms = match text {
            Some(text) => {
                let words = text.split_whitespace().count() as u64;
                (words * 10_000 / rng.u64(WORDS_PER_SEC_X10)).max(MIN_SPEECH_MS)
            }
            None => rng.u64(SILENCE_MS),
        };
        let pacing = match rng.u8(0..3) {
            0 => Pacing::Steady,
            1 => Pacing::Jitter { spread_ms: rng.u64(JITTER_MS) },
            _ => Pacing::Gap { stall_ms: rng.u64(STALL_MS) },
        };
        Segment { speaking, dur_ms, pacing, text, done }
    }

    /// What the detector and the recogniser will say about this speech, in due order.
    ///
    /// The recogniser's end of speech is placed before or after the detector's with equal
    /// probability. That race is the point: the turn machine has two owners for the same edge
    /// and the journal says which one it believed.
    fn verdicts(&mut self, speech: &Segment) -> Vec<Verdict> {
        let live = self.run.record.live();
        let vad = !live.of(Kind::Vad).is_empty();
        let noise = self.persona.noise;
        let rng = &mut self.rng;
        // Noise: the detector hears speech, the recogniser has nothing to transcribe.
        let stt = !live.of(Kind::Stt).is_empty() && rng.u8(0..100) >= noise;
        let dur = speech.dur_ms;
        let mut out = Vec::new();
        if vad {
            out.push(Verdict { at_ms: rng.u64(VAD_START_MS), word: Word::VadStart });
        }
        let mut vad_end_at = dur + rng.u64(VAD_END_MS);
        if let (true, Some(text)) = (stt, speech.text) {
            let words: Vec<&str> = text.split(' ').collect();
            let start = rng.u64(STT_START_MS);
            out.push(Verdict { at_ms: start, word: Word::SttStart });
            // A provider under load answers late now and then; usually its tail latency is short.
            let final_at = dur
                + if rng.u8(0..100) < STT_LATE_FINAL {
                    rng.u64(STT_LATE_FINAL_MS)
                } else {
                    rng.u64(STT_FINAL_MS)
                };
            // A long enough turn may come back as two finals, cut at a breath: the first covers
            // the words up to the cut and arrives while the person is still talking.
            let cut = (words.len() >= 4 && rng.u8(0..100) < STT_SPLIT_FINAL)
                .then(|| rng.usize(2..words.len() - 1));
            // Interims stream as a provider's do: one every so often, each a longer prefix of
            // the words so far, and one of them now and then mis-heard, then corrected.
            let interims =
                (dur / STT_INTERIM_EVERY_MS).clamp(1, words.len().saturating_sub(1).max(1) as u64);
            let revised = (rng.u8(0..100) < STT_REVISION).then(|| rng.u64(1..=interims));
            for i in 1..=interims {
                let at_ms = start + dur.saturating_sub(start) * i / (interims + 1);
                let upto = words.len() * i as usize / (interims as usize + 1);
                let mut prefix: Vec<&str> = words[..upto.max(1)].to_vec();
                if revised == Some(i) {
                    // The last word so far, mis-heard as a word from some other turn.
                    let other = USER_TURNS[rng.usize(0..USER_TURNS.len())];
                    let stray = other
                        .split(' ')
                        .nth(rng.usize(0..other.split(' ').count()))
                        .unwrap_or("uh");
                    *prefix.last_mut().unwrap() = stray;
                }
                match cut {
                    // Past the cut, an interim is the second final's own prefix.
                    Some(cut) if upto > cut => {
                        out.push(Verdict {
                            at_ms,
                            word: Word::SttInterim(prefix[cut..].join(" ")),
                        });
                    }
                    _ => out.push(Verdict { at_ms, word: Word::SttInterim(prefix.join(" ")) }),
                }
            }
            match cut {
                Some(cut) => {
                    let first_at =
                        start + dur.saturating_sub(start) * cut as u64 / words.len() as u64;
                    out.push(Verdict {
                        at_ms: first_at,
                        word: Word::SttFinal(words[..cut].join(" ")),
                    });
                    out.push(Verdict {
                        at_ms: final_at,
                        word: Word::SttFinal(words[cut..].join(" ")),
                    });
                }
                None => {
                    if rng.bool() {
                        let at_ms = dur + (final_at - dur) / 2;
                        out.push(Verdict { at_ms, word: Word::SttPreflight(text.to_owned()) });
                    }
                    out.push(Verdict { at_ms: final_at, word: Word::SttFinal(text.to_owned()) });
                }
            }
            // The utterance closes at the first of final and end, so the end never precedes the
            // final; the coin decides which side of the detector's end it lands on.
            let stt_end_first = rng.bool();
            let end_at = if stt_end_first { final_at } else { final_at.max(vad_end_at) };
            if stt_end_first {
                vad_end_at = vad_end_at.max(end_at);
                out.push(Verdict { at_ms: end_at, word: Word::SttEnd });
                if vad {
                    out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
                }
            } else {
                if vad {
                    out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
                }
                out.push(Verdict { at_ms: end_at, word: Word::SttEnd });
            }
        } else if vad {
            out.push(Verdict { at_ms: vad_end_at, word: Word::VadEnd });
        }
        // Stable, so two verdicts due at the same instant keep the order chosen above.
        out.sort_by_key(|v| v.at_ms);
        out
    }

    /// Play one segment into the microphone beat by beat -- frames, a detector tick, the clock --
    /// saying each verdict once it falls due, and keep the frames flowing through the verdicts'
    /// tail: a real microphone never stops.
    ///
    /// Verdicts come after the beat's advance, because the driver only runs while the clock
    /// moves: the frames a verdict is about must have reached the providers before it is said.
    async fn stretch(
        &mut self,
        mic: u32,
        seg: &Segment,
        verdicts: Vec<Verdict>,
    ) -> Result<(), Halt> {
        let stall_at = match seg.pacing {
            Pacing::Gap { .. } => self.rng.u64(0..seg.dur_ms.max(1)),
            _ => u64::MAX,
        };
        let mut verdicts = verdicts.into_iter().peekable();
        let mut elapsed = 0;
        let mut owed_ms = 0;
        let mut stalled = false;
        // Where the forwarding log stood when this segment began, and when its `SttStart` was
        // said. A provider knows only what the driver handed it: a recogniser does not start on,
        // or transcribe, audio it never received -- and the driver withholds the microphone from
        // it while an uninterruptible reply plays -- and a detector hears no speech in frames it
        // was not given. So every verdict waits until microphone audio has reached its stream,
        // and one still waiting when the speech is over is never said: that provider heard
        // silence, and silence has nothing to report.
        let segment_start = self.run.record.forwarded().len();
        let mut stt_start: Option<usize> = None;
        // An overlap: this speech starts while the agent's audio is coming out of the speaker.
        // Audio only handed to the output is not being spoken yet -- the driver marks the agent
        // speaking on the speaker's own started report -- so a segment merely live is not enough.
        // The detector is asked for its verdict one beat after `VadStart`, once the driver has
        // opened the overlap.
        let overlapping = seg.speaking && self.hearing();
        let mut vad_started_at: Option<u64> = None;
        let mut vad_ended = false;
        let mut judged = false;
        // Whose voice this stretch is: the person's, or now and then someone else in the room.
        self.current_speaker = if self.rng.u8(0..100) < self.persona.other_speaker {
            self.draw::<Option<Speaker>>()
        } else {
            self.speaker
        };
        while elapsed < seg.dur_ms || verdicts.peek().is_some() {
            let (advance, defer) = match seg.pacing {
                Pacing::Steady => (BEAT_MS, false),
                Pacing::Jitter { spread_ms } => {
                    (BEAT_MS - spread_ms + self.rng.u64(0..=2 * spread_ms), false)
                }
                Pacing::Gap { stall_ms } if elapsed >= stall_at && !stalled => {
                    stalled = true;
                    (stall_ms, true)
                }
                Pacing::Gap { .. } => (BEAT_MS, false),
            };
            // Land exactly on the segment's end, so the detector's verdict flips there.
            let advance =
                if elapsed < seg.dur_ms { advance.min(seg.dur_ms - elapsed) } else { advance };
            let advance = advance.max(1);
            owed_ms += advance;
            let chunks = if defer { 0 } else { owed_ms / FRAME_MS };
            owed_ms -= chunks * FRAME_MS;
            if chunks > 0 {
                let mic = Ref::new(mic);
                self.send(Input::MicFrames { mic, chunks: Count(chunks as u32) }).await?;
            }
            let was_speaking = seg.speaking && elapsed < seg.dur_ms;
            let beat_start = self.run.record.forwarded().len();
            self.send(Input::Advance { ms: Millis(advance) }).await?;
            elapsed += advance;
            // The detector's tick on this beat's frames, once the driver has handed them over:
            // speech only in frames that carried the microphone, silence in anything else.
            if chunks > 0
                && let Some(vad) = self.first(Kind::Vad)
            {
                let speaking = was_speaking && self.reached(Kind::Vad, vad, beat_start);
                let p = self.rng.u8(if speaking { SPEAKING_P } else { SILENT_P });
                // What the detector reports: the speech's length while speaking, the silence's
                // since the speech ended otherwise.
                let duration = Millis(if speaking {
                    elapsed.min(seg.dur_ms)
                } else {
                    elapsed.saturating_sub(if seg.speaking { seg.dur_ms } else { 0 })
                });
                let vad = Ref::new(vad);
                self.send(Input::VadInference {
                    vad,
                    probability: Confidence(p),
                    speaking,
                    duration,
                })
                .await?;
            }
            // Due verdicts whose stream has heard the microphone are said; one still unheard once
            // the speech is over is dropped, along with everything that would have followed it.
            let over = elapsed >= seg.dur_ms;
            while let Some(verdict) = verdicts.next_if(|v| {
                v.at_ms <= elapsed && (over || self.sendable(&v.word, segment_start, stt_start))
            }) {
                if !self.sendable(&verdict.word, segment_start, stt_start) {
                    continue;
                }
                if matches!(verdict.word, Word::SttStart) {
                    stt_start = Some(self.run.record.forwarded().len());
                }
                if matches!(verdict.word, Word::VadStart) {
                    vad_started_at = Some(elapsed);
                }
                if matches!(verdict.word, Word::VadEnd) {
                    vad_ended = true;
                }
                self.say(seg, verdict).await?;
            }
            // A prediction the driver asked for shows up here too, after the advance.
            self.predict(seg.done).await?;
            // The detector answers when it has heard enough: a stretch long enough to be a
            // barge-in is judged as soon as it is one, a shorter one once it has ended.
            let evidence = seg.dur_ms >= BACKCHANNEL_MS || vad_ended;
            if overlapping && !judged && evidence && vad_started_at.is_some_and(|at| elapsed > at) {
                let barge = seg.dur_ms >= BACKCHANNEL_MS;
                let right = self.rng.u8(0..100) < self.persona.detector_right;
                self.judge(barge == right).await?;
                judged = true;
            }
        }
        Ok(())
    }

    /// Send one verdict against what is live now; dropped when what it names is gone, because a
    /// recogniser whose utterance already closed has nobody to report to.
    async fn say(&mut self, seg: &Segment, verdict: Verdict) -> Result<(), Halt> {
        // The newest utterance: the one this segment's `SttStart` opened.
        let utterance = self.run.record.live().of(Kind::Utterance).into_iter().max().map(Ref::new);
        let vad = self.first(Kind::Vad).map(Ref::new);
        let stt = self.first(Kind::Stt).map(Ref::new);
        let input = match verdict.word {
            Word::VadStart => {
                vad.map(|vad| Input::VadStart { vad, speech_ms: Millis(verdict.at_ms) })
            }
            Word::VadEnd => vad.map(|vad| Input::VadEnd {
                vad,
                silence_ms: Millis(verdict.at_ms.saturating_sub(seg.dur_ms)),
            }),
            Word::SttStart => stt.map(|stt| Input::SttStart { stt }),
            Word::SttInterim(text) => utterance.map(|utterance| Input::SttInterim {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttPreflight(text) => utterance.map(|utterance| Input::SttPreflight {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttFinal(text) => utterance.map(|utterance| Input::SttFinal {
                utterance,
                text: UserText(text),
                speaker: self.current_speaker,
            }),
            Word::SttEnd => utterance.map(|utterance| Input::SttEnd { utterance }),
        };
        match input {
            Some(input) => self.send(input).await,
            None => Ok(()),
        }
    }

    /// Whether a verdict's stream has heard what the verdict is about: the detector's, the
    /// microphone since the segment began; the recogniser's start, the same; the recogniser's
    /// words, the microphone since the start that opened the utterance.
    fn sendable(&self, word: &Word, segment_start: usize, stt_start: Option<usize>) -> bool {
        match word {
            Word::VadStart | Word::VadEnd => {
                self.first(Kind::Vad).is_some_and(|vad| self.reached(Kind::Vad, vad, segment_start))
            }
            Word::SttStart => {
                self.first(Kind::Stt).is_some_and(|stt| self.reached(Kind::Stt, stt, segment_start))
            }
            Word::SttInterim(_) | Word::SttPreflight(_) | Word::SttFinal(_) | Word::SttEnd => {
                match (self.first(Kind::Stt), stt_start) {
                    (Some(stt), Some(since)) => self.reached(Kind::Stt, stt, since),
                    _ => false,
                }
            }
        }
    }

    /// Whether microphone audio has reached this provider stream since a point in the forwarding
    /// log. The driver's own record of what it handed over, not the case's record of what it sent.
    fn reached(&self, kind: Kind, nth: u32, since: usize) -> bool {
        self.run.record.forwarded()[since..].iter().any(|(_, k, n, tags)| {
            *k == kind && *n == nth && tags.iter().any(|tag| matches!(tag, Tag::Mic { .. }))
        })
    }
}

/// The listener's side: hearing the agent, judging an overlap, answering the predictor.
impl Talk {
    /// Whether the person is hearing the agent right now: a segment the speaker reported started
    /// and has not finished.
    fn hearing(&self) -> bool {
        self.run.record.live().of(Kind::Segment).iter().any(|s| self.heard.contains(s))
    }

    /// The speaker starts emitting whatever segments it was handed and has not started yet: the
    /// moment a listener hears the reply begin, and the moment the driver marks the agent speaking.
    async fn hear(&mut self) -> Result<(), Halt> {
        let unheard: Vec<u32> = self
            .run
            .record
            .live()
            .of(Kind::Segment)
            .into_iter()
            .filter(|s| !self.heard.contains(s))
            .collect();
        for segment in unheard {
            self.heard.insert(segment);
            self.send(Input::PlayoutStarted { segment: Ref::new(segment) }).await?;
        }
        Ok(())
    }

    /// The speaker reports this much of every segment it is playing as played.
    async fn report_progress(&mut self, ms: u64) -> Result<(), Halt> {
        for segment in self.run.record.live().of(Kind::Segment) {
            if self.heard.contains(&segment) {
                self.send(Input::PlayoutProgressed { segment: Ref::new(segment), ms: Millis(ms) })
                    .await?;
            }
        }
        Ok(())
    }

    /// The interruption detector's verdict on the overlap in progress, if there is a detector.
    async fn judge(&mut self, is_interruption: bool) -> Result<(), Halt> {
        let Some(port) = self.first(Kind::Interruption) else { return Ok(()) };
        let port = Ref::new(port);
        // Whether the agent had already stopped by the time the detector answered is the
        // detector's to report, and either answer is legal against any driver state.
        let agent_ended = self.rng.bool();
        self.send(Input::OverlapVerdict { port, is_interruption, agent_ended }).await
    }

    /// Answer the end-of-turn predictor's newest pending request, if the driver made one:
    /// confident when the person is done, doubtful when they are only pausing. That is the
    /// predictor's job, and the driver's endpointing is what its answer feeds.
    async fn predict(&mut self, done: bool) -> Result<(), Halt> {
        let pending = self.run.record.live().of(Kind::Prediction).into_iter().max();
        let Some(nth) = pending.filter(|nth| !self.unanswered.contains(nth)) else {
            return Ok(());
        };
        let prediction = Ref::new(nth);
        if self.rng.u8(0..100) < EOT_LOST {
            if self.rng.bool() {
                self.unanswered.insert(nth);
                return Ok(());
            }
            return self.send(Input::EotFail { prediction }).await;
        }
        let probability = Confidence(self.rng.u8(if done { EOT_DONE_P } else { EOT_MORE_P }));
        let backchannel = self.draw::<Option<Confidence>>();
        self.send(Input::EotAnswer { prediction, probability, backchannel }).await
    }
}

/// The agent's turn.
impl Talk {
    /// Let the agent answer: wait out endpointing, then answer what its reply owes -- the model,
    /// the tools, the hooks, the playout -- round by round. Returns whether the person is about to
    /// talk over a playout still going.
    async fn reply(&mut self) -> Result<bool, Halt> {
        let wait = Millis(self.rng.u64(ENDPOINT_MS));
        self.send(Input::Advance { ms: wait }).await?;
        for _ in 0..REPLY_ROUNDS {
            // The person is done: the predictor, if asked, says so.
            self.predict(true).await?;
            let (chats, tools, hooks, synthesising, playing, speaking, generations) = {
                let live = self.run.record.live();
                (
                    live.of(Kind::Chat),
                    live.of(Kind::Tool),
                    live.of(Kind::Hook),
                    live.of(Kind::Tts),
                    live.of(Kind::Segment),
                    !live.of(Kind::Speech).is_empty(),
                    live.of(Kind::Generation),
                )
            };
            let mut rows = Vec::new();
            // The realtime server finishes the turns it is speaking, oldest first.
            if let Some(socket) = self.first(Kind::Realtime) {
                for _ in generations {
                    rows.push(Input::RealtimeEvent {
                        socket: Ref::new(socket),
                        event: RtEvent::GenerationDone,
                    });
                }
            }
            for chat in chats {
                // A request that offered the answering-machine classifier's tool is the
                // classifier's, and is answered as a model would answer it: with that tool. The
                // person on the line is a person, and saying so is the classifier's whole job.
                let classifying =
                    match self.run.record.resource(Kind::Chat, chat).map(|r| r.detail()) {
                        Some(Detail::Chat { tools, .. }) => {
                            tools.iter().any(|t| t == SAVE_PREDICTION_TOOL)
                        }
                        _ => false,
                    };
                if classifying {
                    // Any answer the model could give, drawn uniformly: each label the tool
                    // accepts, a postponement (including one past the clamp), and an answer
                    // without the label, which the classifier must ignore.
                    let choices = AmdCategory::ALL.len() + 2;
                    let (name, args) = match self.rng.usize(0..choices) {
                        i if i < AmdCategory::ALL.len() => (
                            SAVE_PREDICTION_TOOL,
                            format!(r#"{{"label":"{}"}}"#, AmdCategory::ALL[i].as_str()),
                        ),
                        i if i == AmdCategory::ALL.len() => (
                            POSTPONE_TERMINATION_TOOL,
                            format!(
                                r#"{{"seconds":{}}}"#,
                                self.rng.u64(0..=MAX_EXTENSION_SECS as u64 + 2)
                            ),
                        ),
                        _ => (SAVE_PREDICTION_TOOL, "{}".to_owned()),
                    };
                    rows.push(Input::LlmToolCall {
                        chat: Ref::new(chat),
                        name: name.to_owned(),
                        args: ToolArgs(args),
                    });
                } else if !self.run.setup.tools.is_empty()
                    && self.rng.u8(0..100) < self.persona.tool_calls
                {
                    // The model reaches for one of the agent's tools instead of answering.
                    let tools = &self.run.setup.tools;
                    let name = tools[self.rng.usize(0..tools.len())].clone();
                    let args = self.draw::<ToolArgs>();
                    rows.push(Input::LlmToolCall { chat: Ref::new(chat), name, args });
                } else {
                    let text = self.line();
                    rows.push(Input::LlmText { chat: Ref::new(chat), text });
                }
                rows.push(Input::LlmEnd { chat: Ref::new(chat) });
            }
            // Tools and hooks answer with whatever the table can draw for them: a return, a
            // value, an error, a stop -- the agent's reply path under each is what is under test.
            for tool in tools {
                if self.run.tool_detached(tool) {
                    // A detached tool speaks again through the host, and ends on a final update.
                    let progress = self.draw::<Option<AgentText>>();
                    rows.push(Input::ToolUpdate { tool: Ref::new(tool), progress });
                    continue;
                }
                if self.rng.u8(0..100) < self.persona.fillers {
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
                let answer = self.draw::<ToolAnswer>();
                let text = self.line();
                rows.push(Input::ToolAnswered { tool: Ref::new(tool), answer, text });
            }
            if self.voice_down {
                for tts in synthesising {
                    rows.push(Input::TtsFail { tts: Ref::new(tts) });
                }
            }
            for hook in hooks {
                let answer = self.draw::<HookAnswer>();
                let text = self.line();
                rows.push(Input::HookDone { hook: Ref::new(hook), answer, text });
            }
            if rows.is_empty() && playing.is_empty() && !speaking {
                return Ok(false);
            }
            // The model, the tools and the hooks answer whatever the person decides to do next:
            // their answers go out before the person hears the reply and makes up their mind,
            // so a tool that asks for idle has asked before any barge-in lands on its step.
            for row in std::mem::take(&mut rows) {
                self.send(row).await?;
            }
            if !playing.is_empty() {
                // The person hears the reply begin before deciding whether to let it finish or
                // to talk over it: an overlap is over audio being spoken, not audio in a buffer.
                self.hear().await?;
                if self.rng.u8(0..100) < self.persona.barges {
                    // The person listens for a moment first; how long decides which side of the
                    // driver's backchannel boundary the overlap falls on.
                    let listened = self.rng.u64(LISTEN_MS);
                    self.send(Input::Advance { ms: Millis(listened) }).await?;
                    self.report_progress(listened).await?;
                    return Ok(true);
                }
                // What is playing takes as long as its words take to say; a reply with no words
                // the run knows of -- audio the developer fed -- plays for a drawn while.
                let played = match self.words_playing() {
                    Some(words) => {
                        (words * 10_000 / self.rng.u64(AGENT_WORDS_PER_SEC_X10)).max(BEAT_MS)
                            / playing.len().max(1) as u64
                    }
                    None => self.rng.u64(PLAYOUT_MS),
                };
                rows.push(Input::Advance { ms: Millis(played) });
                // The speaker reports what played before it reports the end, as a real output
                // does at the discontinuity.
                rows.extend(playing.iter().map(|s| Input::PlayoutProgressed {
                    segment: Ref::new(*s),
                    ms: Millis(played),
                }));
                rows.extend(
                    playing.into_iter().map(|s| Input::PlayoutFinished { segment: Ref::new(s) }),
                );
            }
            for row in rows {
                self.send(row).await?;
            }
            let think = Millis(self.rng.u64(THINK_MS));
            self.send(Input::Advance { ms: think }).await?;
            self.await_playout().await?;
        }
        Ok(false)
    }

    /// How many words the driver asked the voice to say for the speeches now live: the length of
    /// what the speaker is playing. `None` when no live speech asked for any.
    fn words_playing(&self) -> Option<u64> {
        let record = &self.run.record;
        let mut words = 0u64;
        let mut any = false;
        for nth in record.live().of(Kind::Speech) {
            let Some(crate::record::Id::Speech(id)) =
                record.resource(Kind::Speech, nth).map(|r| r.id())
            else {
                continue;
            };
            for (_, step) in record.internal_view().speech(id.0).steps() {
                if let Step::Generation(GenerationStep::TtsRequested { text, .. }) = step {
                    any = true;
                    words += text.split_whitespace().count() as u64;
                }
            }
        }
        any.then_some(words)
    }

    /// Wait, a beat at a time, for the agent's audio to reach the speaker -- the moment a listener
    /// hears it start -- so there is something to let finish or to cut into.
    ///
    /// Bounded by the driver, not by a number. The wait goes on while the driver is at work on
    /// the reply -- every beat moved its journal -- and ends when the audio arrives, when the
    /// speech is gone, when the driver is waiting on the case (a model, tool or hook still to
    /// answer: the next round's job, not a wait), or at the first beat the driver did nothing in.
    /// A reply the driver is holding back -- authorization paused, a floor held -- is silence to
    /// the person, who does what people do with silence: speaks again. The harness's release cap
    /// is the backstop, never the reason the wait ends.
    async fn await_playout(&mut self) -> Result<(), Halt> {
        let owed_by_case = |t: &Self| {
            [Kind::Chat, Kind::Tool, Kind::Hook, Kind::Generation]
                .iter()
                .any(|k| t.first(*k).is_some())
        };
        let mut waited = 0;
        while waited < crate::drive::RELEASE_CAP_MS
            && self.first(Kind::Segment).is_none()
            && self.first(Kind::Speech).is_some()
            && !owed_by_case(self)
        {
            let before = self.run.record.journal().len();
            self.send(Input::Advance { ms: Millis(BEAT_MS) }).await?;
            self.predict(true).await?;
            waited += BEAT_MS;
            // A beat the driver did nothing in is silence on the line: the person stops waiting.
            if self.run.record.journal().len() == before {
                break;
            }
        }
        Ok(())
    }
}
