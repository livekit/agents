use std::collections::BTreeSet;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    fakes::amd::SILENCE,
    input::{AmdVerdict, Input, Millis, Ref},
    record::{Recording, amd::Answer},
};

// Spec §14 invariant 7, the AMD proposal "Silence": a machine-stage reply never plays before
// the participant has been silent for `machine_silence_threshold`. Silence is read from the fake
// VAD's ground truth -- the boundaries the case reported, backdated by what it said it measured --
// not from the session's or AMD's view. A detach releases the hold (spec §10.3).

pub struct MachineReplyWaitsForSilence;

impl Rule for MachineReplyWaitsForSilence {
    const CONTRACT: Contract = Contract {
        id: "machine_reply_waits_for_silence",
        guarantee: "A reply AMD allowed after silence is first heard only once the participant \
                    has been silent for the threshold, unless AMD completed first.",
        examples: &[heard_after_the_silence, released_by_detach],
        counterexamples: &[heard_inside_the_silence, heard_while_speaking],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let threshold = SILENCE.as_millis() as u64;
        for turn in amd.gated_turns() {
            let Some(Answer::Decide {
                verdict: AmdVerdict::Silence | AmdVerdict::Ivr, at: decided, ..
            }) = amd.allowed_at(&turn)
            else {
                continue;
            };
            let released = amd
                .answers()
                .into_iter()
                .find(|a| matches!(a, Answer::Detach { run, .. } if *run == turn.run) && a.at() > decided)
                .map(|a| a.at());
            let speeches: BTreeSet<u64> =
                amd.requests_answering(&turn).iter().filter_map(|q| q.speech).collect();
            for speech in speeches {
                let Some(heard) = amd.first_audio(speech) else { continue };
                // Completion frees the held reply at once (spec §10.3): heard after it.
                if released.is_some_and(|d| d < heard) {
                    check.expect(Subject::Speech(speech), heard > decided, "heard before its decision");
                    continue;
                }
                let since = amd.silent_since(heard);
                check.expect_with(
                    Subject::Speech(speech),
                    since.is_some_and(|boundary| heard.at_ms >= boundary + threshold),
                    || {
                        format!(
                            "turn {}'s reply was heard at {} ms; the participant was silent \
                             since {since:?} ms, short of {threshold} ms",
                            turn.turn, heard.at_ms
                        )
                    },
                );
            }
        }
    }
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

fn heard_inside_the_silence() -> Recording {
    held(0).later(500).pushed(10).done()
}

fn heard_while_speaking() -> Recording {
    held(0)
        .later(1_600)
        .sent(Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) })
        .pushed(10)
        .done()
}
