use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Ref, TurnMode, UserText},
    record::{Kind, Recognised, Recording, SttReport},
};

// A connection that ends mid-speech takes the provider's unfinished words with it: the interim it
// had sent is all the person's turn will ever have of them, so the stream makes it final before
// the speech ends.

pub struct RecognisedBoundaryKeepsWords;

impl Rule for RecognisedBoundaryKeepsWords {
    const CONTRACT: Contract = Contract {
        id: "recognised_boundary_keeps_words",
        guarantee: "When a recogniser's connection ends mid-speech, the words of its last interim \
                    no final replaced reach the reader as a final before the speech ends.",
        examples: &[promoted_at_the_drop],
        counterexamples: &[lost_at_the_drop, lost_at_the_failure],
        probe: Some(a_drop_mid_word),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // The words each stream's open speech still owes: its last interim no final replaced.
        let mut owed: BTreeMap<u32, Option<String>> = BTreeMap::new();
        for rec in r.recognition() {
            let words = owed.entry(rec.stt).or_default();
            if rec.report.is_boundary()
                && let Some(text) = words.clone()
            {
                let promoted = rec.emitted.iter().position(|e| *e == Recognised::Final(text.clone()));
                let ended = rec.emitted.iter().position(|e| *e == Recognised::End);
                check.expect(
                    Subject::Resource(Kind::Stt, rec.stt),
                    promoted.is_some_and(|p| ended.is_none_or(|e| p < e)),
                    format!(
                        "stt#{} {:?} at {} ms owing {text:?}, and handed the reader {:?}",
                        rec.stt, rec.report, rec.at.at_ms, rec.emitted
                    ),
                );
            }
            for event in &rec.emitted {
                *words = match event {
                    Recognised::Interim(t) | Recognised::Preflight(t) if !t.is_empty() => {
                        Some(t.clone())
                    }
                    // an empty interim is the words retracted
                    Recognised::Interim(_) | Recognised::Preflight(_) => None,
                    Recognised::Start | Recognised::Final(_) | Recognised::End => None,
                };
            }
        }
    }
}

fn promoted_at_the_drop() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Interim("I need".into()), vec![Recognised::Interim("I need".into())])
        .recognised(0, SttReport::Dropped, vec![Recognised::Final("I need".into()), Recognised::End])
        .done()
}

/// The connection dropped and the stream ended the speech without the words it held.
fn lost_at_the_drop() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Interim("I need".into()), vec![Recognised::Interim("I need".into())])
        .recognised(0, SttReport::Dropped, vec![Recognised::End])
        .done()
}

/// The recogniser gave up and its stream failed with the words of a preflight never final.
fn lost_at_the_failure() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Preflight("a table".into()), vec![Recognised::Preflight(
            "a table".into(),
        )])
        .recognised(0, SttReport::Failed, vec![Recognised::End])
        .done()
}

/// The connection drops while the person's words are still an interim.
fn a_drop_mid_word() -> Case {
    Case {
        setup: Setup { stt: true, turn: TurnMode::Stt, ..Setup::default() },
        inputs: vec![
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim { utterance: Ref::new(0), text: UserText("I need".into()), speaker: None },
            Input::SttDrop { stt: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ],
        ..Case::default()
    }
}
