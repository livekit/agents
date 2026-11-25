use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording, amd::Answer},
};

// the AMD proposal, "DST" and "Detach" step 3: no turn whose AMD category at completion is
// `wait` gets a reply from `Detach`. `pending` replaces any decision the session has not applied
// yet, including one stored for a turn still in its customer hook, so an earlier allow stored
// for that turn must not answer it.

pub struct CompletionRespectsWait;

impl Rule for CompletionRespectsWait {
    const CONTRACT: Contract = Contract {
        id: "completion_respects_wait",
        guarantee: "A turn not yet answered when AMD completes in the wait category is never \
                    answered afterwards.",
        examples: &[waiting_turn_stays_silent],
        counterexamples: &[stored_allow_answers_after_wait],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for answer in amd.answers() {
            let Answer::Detach { at, run, category: AmdVerdict::Wait, .. } = answer else {
                continue;
            };
            for turn in amd.gated_turns().into_iter().filter(|t| t.run == run && t.committed < at)
            {
                if turn.superseded.is_some_and(|s| s < at) {
                    continue;
                }
                let requests = amd.requests_answering(&turn);
                if requests.iter().any(|q| !amd.preemptive(q) && q.opened < at) {
                    continue;
                }
                let after = requests.iter().find(|q| {
                    (!amd.preemptive(q) && q.opened > at)
                        || q.speech.and_then(|s| amd.first_audio(s)).is_some_and(|heard| heard > at)
                });
                check.expect_with(
                    Subject::Resource(Kind::AmdTurn, turn.turn as u32),
                    after.is_none(),
                    || {
                        format!(
                            "turn {} was unanswered when run {run} completed in `wait` at {} ms, \
                             and request {:?} answered it after",
                            turn.turn,
                            at.at_ms,
                            after.map(|q| q.chat)
                        )
                    },
                );
            }
        }
    }
}

fn waiting() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "please hold")
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .sent(Input::AmdDetach { category: AmdVerdict::Wait, forbid: false })
}

fn waiting_turn_stays_silent() -> Recording {
    waiting().later(2_000).done()
}

fn stored_allow_answers_after_wait() -> Recording {
    waiting().request(10, 1, "please hold", &[], &[]).pushed(10).done()
}
