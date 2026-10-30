//! `use_tts_aligned_transcript`: the transcript is the voice's own stamped words exactly when the
//! option (the agent's, else the session's) is on, and a synchroniser shows each at its stamp.
//!
//! Off, nothing forwarded carries a stamp: the transcript is the model's text. On, every word
//! forwarded with a stamp is one the voice stamped, with that stamp (a voice that does not stream
//! is stamped by the session's stream adapter, sentence by sentence); and a synthesised speech
//! that completed through untouched outputs, its voice stamping every word and never failing,
//! forwarded the voice's words, not the model's. A voice that advertises alignment and stamps
//! nothing falls back to the model's text (judged by `text_conservation`).
//!
//! With a synchroniser, each word shown starts where the voice stamped it: the first token of a
//! stamped word at the stamp, the rest of it (punctuation the tokeniser splits off) no earlier.
//! Words are matched to stamps by position in the text, whitespace ignored.
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    trace::{GenerationStep, PlayoutStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording, Shown, TranscriptSetup},
};

/// What float arithmetic may leave between a stamp and the pacer's start, in seconds.
const EPSILON: f64 = 1e-6;

pub struct AlignedTranscriptFollowsTts;

impl Rule for AlignedTranscriptFollowsTts {
    const CONTRACT: Contract = Contract {
        id: "aligned_transcript_follows_tts",
        guarantee: "The transcript is the voice's stamped words exactly when use_tts_aligned_transcript is on, each shown at its stamp.",
        examples: &[aligned_and_shown_at_stamps, off_forwards_the_model_text],
        counterexamples: &[
            stamped_while_off,
            model_text_while_on,
            a_stamp_the_voice_never_gave,
            shown_off_its_stamp,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let setup = r.transcript();
        let (inputs, host, internal) = (r.inputs_view(), r.host_view(), r.internal_view());
        let stamped = r.stamped();
        for speech in host.speeches() {
            let trace = internal.speech(speech.id);
            let forwarded: Vec<(&str, Option<f64>)> = trace
                .steps()
                .filter_map(|(_, step)| match step {
                    Step::Playout(PlayoutStep::TranscriptForwarded {
                        text, start_time, ..
                    }) => Some((text.as_str(), *start_time)),
                    _ => None,
                })
                .collect();
            for (text, start) in &forwarded {
                check.expect(
                    speech,
                    setup.aligned || start.is_none(),
                    format!("{text:?} was forwarded with the voice's stamp, the option being off"),
                );
                // An adapted voice's stamps are the adapter's, one per sentence.
                let Some(start) = start.filter(|_| !setup.adapted) else { continue };
                check.expect(
                    speech,
                    stamped.iter().any(|(_, w, t)| w == text && (t - start).abs() <= EPSILON),
                    format!(
                        "{text:?} was forwarded stamped {start:.4}s, which the voice never sent"
                    ),
                );
            }

            let until = speech.done_at().unwrap_or(r.now());
            if setup.aligned
                && setup.stamps
                && speech.completed_normally()
                && trace.synthesis_requested_at().is_some()
                && trace.audio_played()
                && !inputs.voice_failed()
                && !inputs.outputs_touched_by(until)
            {
                let unstamped: Vec<&str> =
                    forwarded.iter().filter(|(_, s)| s.is_none()).map(|(t, _)| *t).collect();
                check.expect(
                    speech,
                    unstamped.is_empty(),
                    format!("the option on, forwarded the model's {unstamped:?}"),
                );
            }

            if setup.synchronized && forwarded.iter().all(|(_, s)| s.is_some()) {
                // Segment by segment: each synthesis is stamped from its own start.
                type Segment<'a> = (Vec<(&'a str, Option<f64>)>, Vec<Shown<'a>>);
                let mut segments: Vec<Segment<'_>> = Vec::new();
                for (at, step) in trace.steps() {
                    match step {
                        Step::Generation(GenerationStep::TtsRequested { .. }) => {
                            segments.push(Default::default())
                        }
                        Step::Playout(PlayoutStep::TranscriptForwarded {
                            text,
                            start_time,
                            ..
                        }) => {
                            if let Some((words, _)) = segments.last_mut() {
                                words.push((text, *start_time));
                            }
                        }
                        Step::Playout(PlayoutStep::TranscriptShown {
                            text,
                            start,
                            end,
                            played,
                            ..
                        }) => {
                            if let Some((_, shown)) = segments.last_mut() {
                                shown.push(Shown {
                                    at,
                                    text,
                                    start: *start,
                                    end: *end,
                                    played: *played,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                for (words, shown) in &segments {
                    shown_at_stamps(&speech, words, shown, check);
                }
            }
        }
    }
}

/// Walk the shown tokens along the stamped words, by their position in the joined text.
fn shown_at_stamps(
    speech: &crate::record::SpeechView<'_>,
    forwarded: &[(&str, Option<f64>)],
    shown: &[Shown<'_>],
    check: &mut Check,
) {
    // Each stamped word's first and past-last character, whitespace ignored, and its stamp.
    let mut words: Vec<(usize, usize, f64)> = Vec::new();
    let mut at = 0;
    for (text, start) in forwarded {
        let len = text.chars().filter(|c| !c.is_whitespace()).count();
        words.push((at, at + len, start.unwrap_or_default()));
        at += len;
    }
    let mut at = 0;
    for token in shown {
        let len = token.text.chars().filter(|c| !c.is_whitespace()).count();
        if len == 0 {
            continue;
        }
        let Some(&(first, _, stamp)) =
            words.iter().find(|(first, last, _)| *first <= at && at < *last)
        else {
            // Shown past what was forwarded: `shown_transcript_is_conserved`'s to judge.
            return;
        };
        if at == first {
            check.expect(
                speech,
                (token.start - stamp).abs() <= EPSILON,
                format!(
                    "{:?} starts at {:.4}s, the voice stamped {stamp:.4}s",
                    token.text, token.start
                ),
            );
        } else {
            check.expect(
                speech,
                token.start >= stamp - EPSILON,
                format!(
                    "{:?} starts at {:.4}s, before its word's stamp {stamp:.4}s",
                    token.text, token.start
                ),
            );
        }
        at += len;
    }
}

fn forwarded(text: &str, start_time: Option<f64>) -> Step {
    Step::Playout(PlayoutStep::TranscriptForwarded {
        speech: SpeechId(1),
        text: text.into(),
        start_time,
    })
}

fn shown(text: &str, start: f64) -> Step {
    Step::Playout(PlayoutStep::TranscriptShown {
        speech: SpeechId(1),
        text: text.into(),
        start,
        end: start + 0.005,
        played: std::time::Duration::from_secs_f64(start),
    })
}

fn requested() -> Step {
    Step::Generation(GenerationStep::TtsRequested {
        speech: SpeechId(1),
        segment: 0,
        text: "Sure, I".into(),
    })
}

fn setup(aligned: bool) -> TranscriptSetup {
    TranscriptSetup { synchronized: true, aligned, stamps: true, ..TranscriptSetup::default() }
}

/// "Sure, I" stamped at 0 and 5 ms, forwarded as stamped, shown at the stamps ("," after "Sure").
fn aligned_and_shown_at_stamps() -> Recording {
    Authored::new()
        .transcript(setup(true))
        .stamped(0, 0, "Sure, I", &[("Sure, ", 0.0), ("I ", 0.005)])
        .journal(requested())
        .journal(forwarded("Sure, ", Some(0.0)))
        .journal(forwarded("I ", Some(0.005)))
        .journal(shown("Sure", 0.0))
        .journal(shown(",", 0.002))
        .journal(shown(" ", 0.004))
        .journal(shown("I ", 0.005))
        .speech(1, Handle::finished())
        .done()
}

fn off_forwards_the_model_text() -> Recording {
    Authored::new()
        .transcript(setup(false))
        .stamped(0, 0, "Sure, I", &[("Sure, ", 0.0), ("I ", 0.005)])
        .journal(forwarded("Sure, I", None))
        .speech(1, Handle::finished())
        .done()
}

fn stamped_while_off() -> Recording {
    Authored::new()
        .transcript(setup(false))
        .stamped(0, 0, "Sure", &[("Sure ", 0.0)])
        .journal(forwarded("Sure ", Some(0.0)))
        .speech(1, Handle::finished())
        .done()
}

/// The option on, a voice stamping every word, and the model's text forwarded.
fn model_text_while_on() -> Recording {
    Authored::new()
        .transcript(setup(true))
        .journal(requested())
        .stamped(0, 0, "Sure", &[("Sure ", 0.0)])
        .journal(forwarded("Sure", None))
        .journal(Step::Playout(PlayoutStep::AudioPlayed { speech: SpeechId(1), frames: 1 }))
        .speech(1, Handle::finished())
        .done()
}

fn a_stamp_the_voice_never_gave() -> Recording {
    Authored::new()
        .transcript(setup(true))
        .stamped(0, 0, "Sure", &[("Sure ", 0.0)])
        .journal(forwarded("Sure ", Some(0.25)))
        .speech(1, Handle::finished())
        .done()
}

fn shown_off_its_stamp() -> Recording {
    Authored::new()
        .transcript(setup(true))
        .stamped(0, 0, "Sure, I", &[("Sure, ", 0.0), ("I ", 0.005)])
        .journal(requested())
        .journal(forwarded("Sure, ", Some(0.0)))
        .journal(forwarded("I ", Some(0.005)))
        .journal(shown("Sure", 0.0))
        .journal(shown(",", 0.002))
        .journal(shown(" I ", 0.004))
        .speech(1, Handle::finished())
        .done()
}
