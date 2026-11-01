//! A reply the realtime server started on its own is cancelled by the client only for a reason.
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_voice::trace::{CloseStep, SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    record::{Breach, Detail, Kind, Recording},
};

pub struct RealtimeReplyCancelledWithCause;

impl Rule for RealtimeReplyCancelledWithCause {
    const CONTRACT: Contract = Contract {
        id: "realtime_reply_cancelled_with_cause",
        guarantee: "A reply the realtime server started on its own (its VAD answering the person's turn) is never cancelled by the client before it heard of it (unless the host interrupted in that moment), nor in the very moment it heard of it with nothing else happening (no input, no time passing, the driver not declining the reply's speech, the session not closing): such a cancel was meant for another reply, and leaves the person unanswered.",
        examples: &[
            cancelled_after_an_input,
            cancelled_while_closing,
            cancelled_unheard_of_by_the_host,
        ],
        counterexamples: &[cancelled_on_announcement, cancelled_unheard_of],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inference = r.inference_view();
        let generations = inference.streams(Kind::Generation);
        let inputs = r.inputs_view();
        let steps = r.internal_view().steps();
        for (at, socket, breach) in inference.realtime_breaches() {
            let Breach::CancelledUnprompted { response, generation } = breach else { continue };
            let subject = Subject::Resource(Kind::Realtime, *socket);
            // The host's own `interrupt()` in the same moment cancels whatever the server has
            // (`_rt_session.interrupt()` is unconditional): the reply its VAD opened as the
            // cancel crossed it on the wire, announced or still on its way to the client (seeds
            // 82311, 19868).
            let host_interrupt = |ms: u64| {
                inputs.accepted().any(|a| {
                    matches!(a.input, Input::Interrupt { .. }) && a.at.at_ms == ms && a.at < *at
                })
            };
            let Some(generation) = generation else {
                check.expect(subject, host_interrupt(at.at_ms), format!("a client cancel at {} ms stopped {response}, which the client had not heard of", at.at_ms));
                continue;
            };
            let announced =
                generations.iter().find(|g| g.nth() == *generation).map(|g| g.opened_at());
            // Time passed (the driver's own timers decide too), an input came, the host
            // interrupted in that moment, or the driver declined the reply's speech (draining or
            // closing: `ScheduleRejected`). A reply the server opens while the session closes is
            // cancelled: nothing plays it, and left generating it holds the server (seed 623).
            let closing = steps.iter().any(|(when, step)| {
                when < at && matches!(step, Step::Close(CloseStep::Requested { .. }))
            });
            let caused = closing
                || announced.is_some_and(|from| {
                    at.at_ms > from.at_ms
                        || host_interrupt(from.at_ms)
                        || inputs.accepted_between(from, *at).next().is_some()
                        || steps.iter().any(|(when, step)| {
                            from < *when
                                && when < at
                                && matches!(step, Step::Speech(SpeechStep::ScheduleRejected { .. }))
                        })
                });
            check.expect(
                subject,
                caused,
                format!("a client cancel at {} ms stopped {response} (generation {generation}) in the moment the client heard of it", at.at_ms),
            );
        }
    }
}

fn cancel(generation: Option<u32>) -> Breach {
    Breach::CancelledUnprompted { response: "resp_1".into(), generation }
}

/// An input came after the server's reply was announced, and the client cancelled it then.
fn cancelled_after_an_input() -> Recording {
    Authored::new()
        .opens(Kind::Generation, Detail::None)
        .sent(Input::Advance { ms: crate::input::Millis(20) })
        .breach_at_once(0, cancel(Some(0)))
        .done()
}

/// The host interrupted, and the cancel stopped the reply the server opened as it crossed it.
fn cancelled_unheard_of_by_the_host() -> Recording {
    Authored::new()
        .sent(Input::Interrupt { force: false, wait: false })
        .breach_at_once(0, cancel(None))
        .done()
}

/// The session is closing when the server opens a reply, and the client cancels it at once.
fn cancelled_while_closing() -> Recording {
    Authored::new()
        .journal(Step::Close(CloseStep::Requested {
            reason: CloseReason::JobShutdown,
            drain: false,
        }))
        .opens(Kind::Generation, Detail::None)
        .breach_at_once(0, cancel(Some(0)))
        .done()
}

/// The client cancelled the server's reply as it heard of it, nothing having happened.
fn cancelled_on_announcement() -> Recording {
    Authored::new().opens(Kind::Generation, Detail::None).breach_at_once(0, cancel(Some(0))).done()
}

/// The client's cancel stopped a reply it had not heard of.
fn cancelled_unheard_of() -> Recording {
    Authored::new().breach(0, cancel(None)).done()
}
