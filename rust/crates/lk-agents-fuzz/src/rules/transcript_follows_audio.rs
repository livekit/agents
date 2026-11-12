use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    trace::{PlayoutStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording},
};

/// What two clocks' rounding may leave between a stamp and the playhead, in seconds.
const EPSILON: f64 = 1e-6;

/// The journal's stamps are whole milliseconds: what one may hide of the time since a report.
const STAMP_SECS: f64 = 1e-3;

// A paced word is released at the middle of its estimated audio, an aligned word at its stamp. A
// finish is a playhead too: a cut shows words started before the cut point, and a whole finish
// releases the remaining words at once. Between the sink's reports the playhead moves with
// elapsed time: a speaker that played to `p` by its last report (its start, or a stretch played)
// has played no further than `p` plus the time since, whether or not it reports progress.

pub struct TranscriptFollowsAudio;

impl Rule for TranscriptFollowsAudio {
    const CONTRACT: Contract = Contract {
        id: "transcript_follows_audio",
        guarantee: "A synchronised transcript shows no word before the segment could have played up to where the word's audio starts (the sink's last reported playhead plus the time since that report), except the rest of a segment the sink reported finished whole.",
        examples: &[shown_as_played, rest_shown_at_the_end, shown_as_time_passes],
        counterexamples: &[
            shown_ahead_of_the_voice,
            shown_ahead_then_cut,
            shown_whole_with_no_finish,
            shown_ahead_of_the_time_since_the_start,
        ],
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
                        speech,
                        start <= reached + EPSILON,
                        format!(
                            "{text:?} was shown with {played:.3}s of its segment played, its audio \
                             starting at {start:.3}s"
                        ),
                    );
                }
            };
            // When the sink last reported where it was: its start or a stretch it played.
            let mut reported: Option<u64> = None;
            for (at, step) in internal.speech(speech.id).steps() {
                let Step::Playout(step) = step else { continue };
                match step {
                    PlayoutStep::PlaybackStarted { .. } | PlayoutStep::AudioPlayed { .. } => {
                        reported = Some(at.at_ms);
                    }
                    PlayoutStep::TranscriptShown { text, start, played, .. } => {
                        let since = reported.map_or(0.0, |r| {
                            at.at_ms.saturating_sub(r) as f64 / 1000.0 + STAMP_SECS
                        });
                        let played = played.as_secs_f64() + since;
                        // Its stamps excuse nothing: a word stamped as ending where it starts
                        // (a whole finish's clamp) is still owed that finish next (seed 19354).
                        if *start > played + EPSILON {
                            ahead.push((text.clone(), *start, played));
                        } else {
                            check.expect(speech, true, "");
                        }
                    }
                    // The segment played on: what was shown ahead was not its end. (A start is
                    // journalled at a finish the sink reported without one.)
                    PlayoutStep::PlaybackProgressed { .. } => {
                        reported = Some(at.at_ms);
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

fn started() -> Step {
    Step::Playout(PlayoutStep::PlaybackStarted { speech: SpeechId(1) })
}

/// A sink that reports no progress: each word shown once the time since the start reached it.
fn shown_as_time_passes() -> Recording {
    speech()
        .journal(started())
        .later(250)
        .journal(shown("one", 0.0, 0))
        .later(300)
        .journal(shown(" two", 0.4, 0))
        .journal(finished(800, false))
        .done()
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

/// A sink that reports no progress: a word shown 100 ms after the start with its audio 2.5 s in,
/// and the segment playing on.
fn shown_ahead_of_the_time_since_the_start() -> Recording {
    speech()
        .journal(started())
        .later(100)
        .journal(shown(" twelve", 2.5, 0))
        .journal(progressed(3_000))
        .done()
}
