use std::collections::BTreeSet;

use lk_agents_voice::trace::{SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::amd::SILENCE,
    input::{AmdVerdict, Input, Millis, Ref},
    record::{Recording, amd::Answer},
};

// Spec §14 invariant 7: a machine-stage reply never plays before
// the participant has been silent for `machine_silence_threshold`. Silence is read from the fake
// VAD's ground truth -- the boundaries the case reported, backdated by what it said it measured --
// not from the session's or AMD's view. A detach releases the hold (spec §10.3). On a realtime
// model the reply starts generating only after the silence (spec §10.3, "A realtime reply starts
// generating only after the silence"): its response request is judged as the release.

pub struct MachineReplyWaitsForSilence;

impl Rule for MachineReplyWaitsForSilence {
    const CONTRACT: Contract = Contract {
        id: "machine_reply_waits_for_silence",
        guarantee: "A reply AMD allowed after silence is first heard only once the participant \
                    has been silent for the threshold, unless AMD completed first.",
        examples: &[heard_after_the_silence, released_by_detach, realtime_asked_after_the_silence],
        counterexamples: &[
            heard_inside_the_silence,
            heard_while_speaking,
            let_go_before_the_detach,
            realtime_asked_inside_the_silence,
        ],
        probe: Some(a_voicemail_greeting),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let threshold = SILENCE.as_millis() as u64;
        for turn in amd.gated_turns() {
            let Some(Answer::Decide { verdict, at: decided, .. }) = amd.deciding(&turn) else {
                continue;
            };
            if !verdict.waits_for_silence() {
                continue;
            }
            let released = amd
                .answers()
                .into_iter()
                .find(|a| matches!(a, Answer::Detach { run, .. } if *run == turn.run) && a.at() > decided)
                .map(|a| a.at());
            // The first response request: a tool follow-up comes after the release, under
            // ordinary interruption. The server takes one request at a time, so the request may
            // leave after the release; it must not leave before the participant kept the
            // threshold's silence once since the turn's speech.
            if let Some(response) = amd.rt_responses_answering(&turn).first() {
                let asked = response.at;
                if released.is_some_and(|d| d < asked) {
                    check.expect(Subject::Session, asked > decided, "asked before its decision");
                } else {
                    let silences = silences(&amd.vad_boundaries());
                    let spoke_until = (amd.vad_boundaries().into_iter())
                        .filter(|(speaking, _, seen)| !speaking && *seen <= turn.committed)
                        .map(|(_, boundary, _)| boundary)
                        .next_back()
                        .unwrap_or(0);
                    let kept = silences.iter().any(|(from, to)| {
                        *from >= spoke_until
                            && from + threshold <= asked.at_ms
                            && from + threshold <= *to
                    });
                    check.expect_with(Subject::Session, kept, || {
                        format!(
                            "turn {}'s realtime reply was asked at {} ms before the participant \
                             kept {threshold} ms of silence since {spoke_until} ms ({silences:?})",
                            turn.turn, asked.at_ms
                        )
                    });
                }
            }
            let speeches: BTreeSet<u64> =
                amd.requests_answering(&turn).iter().filter_map(|q| q.speech).collect();
            for speech in speeches {
                let Some(heard) = amd.first_audio(speech) else { continue };
                // Judged when the gate let the reply go: speech after that release goes through
                // ordinary interruption, and a false interruption's resume may play it sooner, as
                // in Python (`detector.py` `_update_idle` resumes authorization once, then leaves
                // the reply to the session).
                let let_go = r
                    .journal()
                    .iter()
                    .find(|(at, step)| {
                        *at > decided
                            && matches!(step, Step::Speech(SpeechStep::Authorized { speech: s })
                                if s.0 == speech)
                    })
                    .map_or(heard, |(at, _)| (*at).min(heard));
                // Completion frees the held reply at once (spec §10.3): a reply let go after it
                // owes no silence.
                if released.is_some_and(|d| d < let_go) {
                    check.expect(Subject::Speech(speech), heard > decided, "heard before its decision");
                    continue;
                }
                let since = amd.silent_since(let_go);
                check.expect_with(
                    Subject::Speech(speech),
                    since.is_some_and(|boundary| let_go.at_ms >= boundary + threshold),
                    || {
                        format!(
                            "turn {}'s reply was let go at {} ms; the participant was silent \
                             since {since:?} ms, short of {threshold} ms",
                            turn.turn, let_go.at_ms
                        )
                    },
                );
            }
        }
    }
}

/// The fake VAD's silences, as `(from, to)` ms, `to` the next speech's backdated start.
fn silences(boundaries: &[(bool, u64, crate::record::Observed)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut from = Some(0);
    for (speaking, boundary, _) in boundaries {
        match (speaking, from) {
            (true, Some(start)) => {
                out.push((start, *boundary));
                from = None;
            }
            (false, None) => from = Some(*boundary),
            _ => {}
        }
    }
    if let Some(start) = from {
        out.push((start, u64::MAX));
    }
    out
}

fn held(silence_ms: u64) -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) })
        .later(1_000)
        .sent(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(silence_ms) })
        .user_turn(Some(1), 1, "you have reached the screener")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Silence })
        .request(10, 1, "you have reached the screener", &[crate::fakes::amd::STAGE], &[])
}

fn heard_after_the_silence() -> Recording {
    held(0).later(1_600).pushed(10).done()
}

fn released_by_detach() -> Recording {
    held(0)
        .sent(Input::AmdDetach { category: AmdVerdict::Silence, forbid: false })
        .pushed(10)
        .done()
}

fn realtime_held(silence_ms: u64) -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) })
        .later(1_000)
        .sent(Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(silence_ms) })
        .user_turn(Some(1), 1, "you have reached the screener")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Silence })
}

fn realtime_asked_after_the_silence() -> Recording {
    realtime_held(0).later(1_600).rt_response("rt_1", Some(crate::fakes::amd::STAGE)).done()
}

/// The response is asked at the decision; only its playout waits.
fn realtime_asked_inside_the_silence() -> Recording {
    realtime_held(0).later(100).rt_response("rt_1", Some(crate::fakes::amd::STAGE)).done()
}

fn heard_inside_the_silence() -> Recording {
    held(0).later(500).pushed(10).done()
}

/// Let go inside the silence and heard only after a later detach: the detach did not release
/// it.
fn let_go_before_the_detach() -> Recording {
    held(0)
        .later(500)
        .journal(Step::Speech(SpeechStep::Authorized {
            speech: lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId(10),
        }))
        .sent(Input::AmdDetach { category: AmdVerdict::Silence, forbid: false })
        .pushed(10)
        .done()
}

fn heard_while_speaking() -> Recording {
    held(0)
        .later(1_600)
        .sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) })
        .pushed(10)
        .done()
}

/// A voicemail greeting's reply waits for the silence.
fn a_voicemail_greeting() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("please leave a message", AmdVerdict::Silence)
}
