use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Ref, TurnMode, UserText},
    record::{Kind, Recognised, Recording, SttReport},
};

// A provider that detects speech may report the end while it still owes the speech's words; the
// stream keeps the speech open for them. What it waits on must end it when it comes, or the
// person stays "speaking" past their own words.

pub struct RecognisedEndIsReleased;

impl Rule for RecognisedEndIsReleased {
    const CONTRACT: Contract = Contract {
        id: "recognised_end_is_released",
        guarantee: "An end of speech a recognition stream holds for owed words is released by \
                    the next final, start of speech or connection end on that stream.",
        examples: &[released_by_the_final, released_by_the_next_start],
        counterexamples: &[held_past_the_final, held_past_the_drop],
        probe: Some(an_end_before_the_final),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Per stream: in a speech, and whether its end is held.
        let mut state: BTreeMap<u32, (bool, Option<u64>)> = BTreeMap::new();
        for rec in r.recognition() {
            let (speaking, held) = state.entry(rec.stt).or_default();
            let releases = matches!(rec.report, SttReport::Final(_) | SttReport::Started)
                || rec.report.is_boundary();
            if let Some(since) = *held
                && releases
            {
                check.expect(
                    Subject::Resource(Kind::Stt, rec.stt),
                    rec.emitted.contains(&Recognised::End),
                    format!(
                        "stt#{} held the end it was told at {since} ms past {:?} at {} ms",
                        rec.stt, rec.report, rec.at.at_ms
                    ),
                );
                *held = None;
            }
            let was_speaking = *speaking;
            for event in &rec.emitted {
                match event {
                    Recognised::Start => *speaking = true,
                    Recognised::End => {
                        *speaking = false;
                        *held = None;
                    }
                    _ => {}
                }
            }
            if rec.report == SttReport::Ended && was_speaking && *speaking {
                *held = Some(rec.at.at_ms);
            }
        }
    }
}

fn released_by_the_final() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Interim("hi".into()), vec![Recognised::Start, interim("hi")])
        .recognised(0, SttReport::Ended, vec![])
        .recognised(0, SttReport::Final("hi".into()), vec![final_("hi"), Recognised::End])
        .done()
}

fn released_by_the_next_start() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Ended, vec![])
        .recognised(0, SttReport::Started, vec![Recognised::End, Recognised::Start])
        .done()
}

/// The final the end waited for came, and the speech went on.
fn held_past_the_final() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Interim("hi".into()), vec![Recognised::Start, interim("hi")])
        .recognised(0, SttReport::Ended, vec![])
        .recognised(0, SttReport::Final("hi".into()), vec![final_("hi")])
        .done()
}

/// The connection dropped and the held end stayed held.
fn held_past_the_drop() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Ended, vec![])
        .recognised(0, SttReport::Dropped, vec![])
        .done()
}

fn interim(text: &str) -> Recognised {
    Recognised::Interim(text.into())
}

fn final_(text: &str) -> Recognised {
    Recognised::Final(text.into())
}

/// The provider's detector ends the speech while its words are an interim; the final follows.
fn an_end_before_the_final() -> Case {
    Case {
        setup: Setup { stt: true, turn: TurnMode::Stt, ..Setup::default() },
        inputs: vec![
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim { utterance: Ref::new(0), text: UserText("I need".into()), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            Input::SttFinal { utterance: Ref::new(0), text: UserText("I need".into()), speaker: None },
            Input::Advance { ms: Millis(500) },
        ],
        ..Case::default()
    }
}
