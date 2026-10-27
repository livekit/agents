//! A latency contract: once a speech may play, its first audio reaches the speaker within a budget
//! of the product having it.
//!
//! The product has a speech's first audio when its source hands over that source's first chunk:
//! the voice (`tts`, after the voice's calibrated time to first audio) or the realtime server (its
//! first audio delta, after the server's calibrated delays). The speech may play once the
//! scheduler authorized it. From the later of the two, the first audio is on the speaker within
//! [`BUDGET_MS`] of virtual time, unless the speech was paused, cut or the session closing, or the
//! output replaced, in between. What is left is audio the product holds for no reason: a reply's
//! voice buffered until the server finished generating it (what c8a832860 fixed in the Python
//! glue), or until the synthesis ended.
use std::collections::BTreeMap;

use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    agent_session::effects::{GateCause, GateOp},
    trace::{CloseStep, InterruptStep, PlayoutStep, SinkCall, SpeechStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Kind, Observed, Recording, Sink, Tag},
};

/// What the product may add between having a speech's first audio and the speaker playing it.
pub const BUDGET_MS: u64 = 100;

pub struct FirstAudioIsPrompt;

impl Rule for FirstAudioIsPrompt {
    const CONTRACT: Contract = Contract {
        id: "first_audio_is_prompt",
        guarantee: "Once a speech may play, its first audio reaches the speaker within a budget of the product having it, unless the speech was paused, cut, its output replaced or the session closing meanwhile.",
        examples: &[pushed_as_it_arrived, held_while_paused],
        counterexamples: &[held_until_the_generation_ended],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let delivered = r.delivered();
        // Each speech's first push: the first audio pushed before the speech's first push record
        // (the playout records a stretch of pushes at its end, one speech playing at a time).
        let mut first: BTreeMap<u64, (Observed, Tag)> = BTreeMap::new();
        let pushes: Vec<(Observed, Tag)> = r
            .sink()
            .iter()
            .filter_map(|(at, sink)| match sink {
                Sink::Audio { tags, .. } => tags.first().map(|t| (*at, *t)),
                _ => None,
            })
            .collect();
        let mut next_push = 0;
        let mut seen_since_record: Option<(Observed, Tag)> = None;
        for (at, step) in journal {
            while next_push < pushes.len() && pushes[next_push].0.seq < at.seq {
                seen_since_record.get_or_insert(pushes[next_push]);
                next_push += 1;
            }
            if let Step::Playout(PlayoutStep::SinkCalled { speech: Some(speech), call }) = step {
                if let Some(push) = seen_since_record.take()
                    && matches!(call, SinkCall::Push { .. })
                {
                    first.entry(speech.0).or_insert(push);
                }
            }
        }
        for (speech, (pushed, tag)) in first {
            // Only a source's first chunk is the speech's first audio as the source made it.
            let source = match tag {
                Tag::Synthesized { tts, chunk: 0 } => (Kind::Tts, tts),
                Tag::Generated { generation, chunk: 0 } => (Kind::Generation, generation),
                _ => continue,
            };
            // The last authorization before it played: one a hold (AMD's) took back is renewed.
            let may = journal.iter().take_while(|(a, _)| a.seq < pushed.seq).filter_map(|(a, s)| {
                matches!(s, Step::Speech(SpeechStep::Authorized { speech: sp }) if sp.0 == speech)
                    .then_some(*a)
            });
            let (Some(had), Some(may)) = (delivered.get(&source), may.last()) else {
                continue;
            };
            let from = (*had).max(may);
            if pushed.at_ms <= from.at_ms + BUDGET_MS {
                check.expect_with(Subject::Speech(speech), true, String::new);
                continue;
            }
            let excused = journal
                .iter()
                .filter(|(a, _)| a.seq >= may.seq && a.seq <= pushed.seq)
                .any(|(_, s)| excuses(s, speech));
            if excused {
                check.expect_with(Subject::Speech(speech), true, String::new);
                continue;
            }
            check.expect_with(Subject::Speech(speech), false, || {
                format!(
                    "first audio of speech {speech} ({source:?}) reached the speaker at {} ms; the \
                     product had it at {} ms and could play it from {} ms (budget {BUDGET_MS} ms)",
                    pushed.at_ms, had.at_ms, from.at_ms
                )
            });
        }
    }
}

/// What lets a speech's audio wait: a pause on it (the person's voice, the launch hold), its cut,
/// its output replaced, the session closing.
fn excuses(step: &Step, speech: u64) -> bool {
    match step {
        Step::Interrupt(InterruptStep::PlayoutGate { speech: s, op: GateOp::Pause, .. }) => {
            s.0 == speech
        }
        Step::Interrupt(InterruptStep::Interrupted { speech: s, .. }) => s.0 == speech,
        Step::Playout(PlayoutStep::OutputReplaced { .. }) => true,
        Step::Close(CloseStep::Requested { .. }) => true,
        _ => false,
    }
}

fn authorized(speech: u64) -> Step {
    Step::Speech(SpeechStep::Authorized { speech: SpeechId(speech) })
}

fn push_record(speech: u64) -> Step {
    Step::Playout(PlayoutStep::SinkCalled {
        speech: Some(SpeechId(speech)),
        call: SinkCall::Push { frames: 1 },
    })
}

fn first_chunk() -> Sink {
    Sink::Audio { output: 0, tags: vec![Tag::Generated { generation: 0, chunk: 0 }] }
}

/// Authorized, the first delta in, on the speaker at once.
fn pushed_as_it_arrived() -> Recording {
    Authored::new()
        .journal(authorized(4))
        .later(300)
        .delivered(Kind::Generation, 0)
        .sink(first_chunk())
        .journal(push_record(4))
        .done()
}

/// Paused for the person while the delta came in: it waits for the resume.
fn held_while_paused() -> Recording {
    Authored::new()
        .journal(authorized(4))
        .delivered(Kind::Generation, 0)
        .journal(Step::Interrupt(InterruptStep::PlayoutGate {
            speech: SpeechId(4),
            op: GateOp::Pause,
            cause: GateCause::ByAudioActivity,
        }))
        .later(2_000)
        .sink(first_chunk())
        .journal(push_record(4))
        .done()
}

/// The first delta in, and the speaker silent until the generation ended a second later.
fn held_until_the_generation_ended() -> Recording {
    Authored::new()
        .journal(authorized(4))
        .delivered(Kind::Generation, 0)
        .later(1_000)
        .sink(first_chunk())
        .journal(push_record(4))
        .done()
}
