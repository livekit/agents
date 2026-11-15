use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Ref, TurnMode, UserText},
    record::{Kind, Recognised, Recording, SttReport},
};

// A speech a recognition stream left open keeps the session hearing a person nobody reports on:
// the turn never ends and every reply waits for them. The stream's own end and a connection's are
// the last things that can end it; once the case answered everything, nothing else will.

pub struct RecognisedSpeechEnds;

impl Rule for RecognisedSpeechEnds {
    const CONTRACT: Contract = Contract {
        id: "recognised_speech_ends",
        guarantee: "Every speech a recognition stream starts ends on that stream: never past a \
                    connection's end or the stream's, and by the time the case has answered \
                    everything, unless the session closed the stream.",
        examples: &[ended_by_the_final, ended_by_the_drop],
        counterexamples: &[open_past_a_failure, open_once_answered],
        probe: Some(a_held_end),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let answered = r.inputs_view().everything_answered_at().filter(|_| !r.hung());
        let mut open: BTreeMap<u32, bool> = BTreeMap::new();
        let judge_open = |open: &BTreeMap<u32, bool>, check: &mut Check| {
            let Some(answered) = answered else { return };
            for (stt, speaking) in open {
                let closed = r
                    .closed()
                    .iter()
                    .any(|(at, kind, nth)| *kind == Kind::Stt && nth == stt && *at <= answered);
                check.expect(
                    Subject::Resource(Kind::Stt, *stt),
                    !speaking || closed,
                    format!(
                        "stt#{stt} was still in a speech once everything was answered at {} ms",
                        answered.at_ms
                    ),
                );
            }
        };
        let mut judged = false;
        for rec in r.recognition() {
            if !judged && answered.is_some_and(|answered| rec.at > answered) {
                judge_open(&open, check);
                judged = true;
            }
            let speaking = open.entry(rec.stt).or_default();
            for event in &rec.emitted {
                match event {
                    Recognised::Start => *speaking = true,
                    Recognised::End => *speaking = false,
                    _ => {}
                }
            }
            if rec.report.is_boundary() {
                check.expect(
                    Subject::Resource(Kind::Stt, rec.stt),
                    !*speaking,
                    format!(
                        "stt#{} {:?} at {} ms and its speech went on",
                        rec.stt, rec.report, rec.at.at_ms
                    ),
                );
            }
        }
        if !judged {
            judge_open(&open, check);
        }
    }
}

fn ended_by_the_final() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Interim("hi".into()), vec![Recognised::Start, interim("hi")])
        .recognised(0, SttReport::Final("hi".into()), vec![Recognised::Final("hi".into())])
        .recognised(0, SttReport::Ended, vec![Recognised::End])
        .everything_answered()
        .done()
}

fn ended_by_the_drop() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Interim("hi".into()), vec![Recognised::Start, interim("hi")])
        .recognised(0, SttReport::Dropped, vec![Recognised::Final("hi".into()), Recognised::End])
        .everything_answered()
        .done()
}

/// The recogniser gave up mid-speech and its stream failed with the speech still open.
fn open_past_a_failure() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Started, vec![Recognised::Start])
        .recognised(0, SttReport::Failed, vec![])
        .done()
}

/// The provider's end was held for a final that never came, and nothing released it.
fn open_once_answered() -> Recording {
    Authored::new()
        .recognised(0, SttReport::Interim("hi".into()), vec![Recognised::Start, interim("hi")])
        .recognised(0, SttReport::Ended, vec![])
        .everything_answered()
        .done()
}

fn interim(text: &str) -> Recognised {
    Recognised::Interim(text.into())
}

/// The provider ends a speech whose words are still an interim and never sends the final.
fn a_held_end() -> Case {
    Case {
        setup: Setup { stt: true, turn: TurnMode::Stt, ..Setup::default() },
        inputs: vec![
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim { utterance: Ref::new(0), text: UserText("I need".into()), speaker: None },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(500) },
        ],
        ..Case::default()
    }
}
