//! A synchronised transcript never shows a word before its audio starts playing.
//!
//! The synchroniser releases each word against the segment's played position: a paced word at
//! the middle of its estimated audio, an aligned word at the voice's stamp. Each release is
//! journalled with where the word's audio starts and how much of the segment had played, so a
//! word shown with its start still ahead of the playhead is a caption running ahead of the voice.
//!
//! A finish is a playhead too: a cut segment shows the words whose audio started before where the
//! cut says it stopped, with no progress reported in between. And a segment the sink reported
//! finished whole shows the words still queued at once, their audio being over. Those come in one
//! go, right before the segment's uninterrupted finish, with nothing played in between.
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    trace::{PlayoutStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording},
};

/// What two clocks' rounding may leave between a stamp and the playhead, in seconds.
const EPSILON: f64 = 1e-6;

pub struct TranscriptFollowsAudio;

impl Rule for TranscriptFollowsAudio {
    const CONTRACT: Contract = Contract {
        id: "transcript_follows_audio",
        guarantee: "A synchronised transcript shows no word before the segment has played up to where the word's audio starts, except the rest of a segment the sink reported finished whole.",
        examples: &[shown_as_played, rest_shown_at_the_end],
        counterexamples: &[shown_ahead_of_the_voice, shown_ahead_then_cut, shown_whole_with_no_finish],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let internal = r.internal_view();
        for speech in host.speeches() {
            // Words shown ahead of the playhead, owed an excuse by the segment's whole finish.
            let mut ahead: Vec<(String, f64, f64)> = Vec::new();
            // `reached`: how far a finish says the segment played, if one came next.
            let judge = |check: &mut Check, ahead: &mut Vec<(String, f64, f64)>, reached: f64| {
                for (text, start, played) in ahead.drain(..) {
                    check.expect(
                        &speech,
                        start <= reached + EPSILON,
                        format!(
                            "{text:?} was shown with {played:.3}s of its segment played, its audio \
                             starting at {start:.3}s"
                        ),
                    );
                }
            };
            for (_, step) in internal.speech(speech.id).steps() {
                let Step::Playout(step) = step else { continue };
                match step {
                    PlayoutStep::TranscriptShown { text, start, played, .. } => {
                        let played = played.as_secs_f64();
                        // Its stamps excuse nothing: a word stamped as ending where it starts
                        // (a whole finish's clamp) is still owed that finish next (seed 19354).
                        if *start > played + EPSILON {
                            ahead.push((text.clone(), *start, played));
                        } else {
                            check.expect(&speech, true, "");
                        }
                    }
                    // The segment played on: what was shown ahead was not its end. (A start is
                    // journalled at a finish the sink reported without one.)
                    PlayoutStep::PlaybackProgressed { .. } => {
                        judge(check, &mut ahead, f64::NEG_INFINITY)
                    }
                    PlayoutStep::PlaybackFinished { interrupted, position, .. } => {
                        let reached =
                            if *interrupted { position.as_secs_f64() } else { f64::INFINITY };
                        judge(check, &mut ahead, reached)
                    }
                    _ => {}
                }
            }
            judge(check, &mut ahead, f64::NEG_INFINITY);
        }
    }
}

fn shown(text: &str, start: f64, played_ms: u64) -> Step {
    Step::Playout(PlayoutStep::TranscriptShown {
        speech: SpeechId(1),
        text: text.into(),
        start,
        end: start + 0.1,
        played: std::time::Duration::from_millis(played_ms),
    })
}

fn progressed(to_ms: u64) -> Step {
    Step::Playout(PlayoutStep::PlaybackProgressed {
        speech: SpeechId(1),
        offset: std::time::Duration::ZERO,
        duration: std::time::Duration::from_millis(to_ms),
    })
}

fn finished(at_ms: u64, interrupted: bool) -> Step {
    Step::Playout(PlayoutStep::PlaybackFinished {
        speech: SpeechId(1),
        position: std::time::Duration::from_millis(at_ms),
        interrupted,
    })
}

fn speech() -> Authored {
    Authored::new().speech(1, Handle::finished())
}

/// Each word shown once the segment had played past its start.
fn shown_as_played() -> Recording {
    speech()
        .journal(progressed(60))
        .journal(shown("one", 0.0, 60))
        .journal(progressed(160))
        .journal(shown(" two", 0.1, 160))
        .done()
}

/// The segment finished whole: the last word, its start past the reported playhead, comes at once.
fn rest_shown_at_the_end() -> Recording {
    speech()
        .journal(progressed(60))
        .journal(shown("one", 0.0, 60))
        .journal(shown(" two", 0.1, 60))
        .journal(finished(200, false))
        .done()
}

/// A word shown with its audio still ahead, and the segment playing on.
fn shown_ahead_of_the_voice() -> Recording {
    speech().journal(progressed(60)).journal(shown(" two", 0.1, 60)).journal(progressed(160)).done()
}

/// Words stamped as a whole finish stamps them, shown before any audio, and no finish followed.
fn shown_whole_with_no_finish() -> Recording {
    let mut word = shown(" two", 0.1, 0);
    if let Step::Playout(PlayoutStep::TranscriptShown { end, .. }) = &mut word {
        *end = 0.1;
    }
    speech().journal(word).journal(progressed(160)).done()
}

/// A word shown with its audio still ahead, then the segment cut: it was never heard.
fn shown_ahead_then_cut() -> Recording {
    speech()
        .journal(progressed(60))
        .journal(shown(" two", 0.1, 60))
        .journal(finished(60, true))
        .done()
}
