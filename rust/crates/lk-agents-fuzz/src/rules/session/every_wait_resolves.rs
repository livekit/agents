use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup, WAIT_BOUND_MS},
    input::{AgentText, Input, Millis, Ref, SayInput, SpeechWait},
    record::{Fault, Observed, Phase, Recording, Site, Wait},
};

// A host that awaits a speech, a play or a close must get its answer once the session closes,
// whatever a provider or host call did: panicked, or never answered. The bound runs from the
// close's start, or from the arming when that is later: the close watchdog, then the host
// teardown bound. A driver that died of an injected panic owes every wait an end, however late.

pub struct EveryWaitResolves;

impl Rule for EveryWaitResolves {
    const CONTRACT: Contract = Contract {
        id: "every_wait_resolves",
        guarantee: "Every host wait armed on the session resolves within the close watchdog and \
                    the host teardown bound past the close's start (or its arming, if later), \
                    injected panics and hangs included.",
        examples: &[resolved_in_time, resolved_after_an_injected_hang],
        counterexamples: &[never_resolved, resolved_past_the_bound],
        probe: Some(a_probed_say),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Judged once the host awaited the close out (`AwaitExit`), as a host does.
        if !r.driver_finished() || r.phase() < Phase::Closed || r.livelock().is_some() {
            return;
        }
        let close = r.close_begun();
        if close.is_none() && r.driver_panic().is_none() {
            return;
        }
        for probe in r.probes() {
            let bound = close.map(|close| probe.armed.at_ms.max(close.at_ms) + WAIT_BOUND_MS);
            let held = probe.resolved.is_some_and(|at| bound.is_none_or(|bound| at.at_ms <= bound));
            check.expect_with(Subject::Session, held, || match probe.resolved {
                None => format!(
                    "{:?}, armed at {} ms, never resolved ({} faults injected)",
                    probe.wait,
                    probe.armed.at_ms,
                    r.faults().len()
                ),
                Some(at) => format!(
                    "{:?}, armed at {} ms, resolved at {} ms, past its bound at {} ms",
                    probe.wait,
                    probe.armed.at_ms,
                    at.at_ms,
                    bound.unwrap_or_default()
                ),
            });
        }
    }
}

/// A close, then a wait armed with it, resolved at `resolved_ms`.
fn waited(resolved_ms: Option<u64>) -> Recording {
    let mut r = Authored::new().sent(Input::Close { drain: false }).driver_finished().done();
    let probe = r.record_probe_armed(Observed::at(1, 1), Wait::Playout(1));
    if let Some(ms) = resolved_ms {
        r.record_probe_resolved(probe, Observed::at(10, ms));
    }
    r
}

fn resolved_in_time() -> Recording {
    waited(Some(1 + WAIT_BOUND_MS))
}

/// A provider never answered; the close still ended the wait.
fn resolved_after_an_injected_hang() -> Recording {
    let mut r = waited(Some(1_000));
    r.record_fault(Observed::at(1, 1), Site::ChatNext, Fault::Hang);
    r
}

fn never_resolved() -> Recording {
    waited(None)
}

fn resolved_past_the_bound() -> Recording {
    waited(Some(2 + WAIT_BOUND_MS))
}

/// A say, its playout awaited before the speech exists.
pub(super) fn a_probed_say() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![
            Input::Say {
                text: AgentText("Hello there.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Probe { speech: Ref::new(0), wait: SpeechWait::Playout },
            Input::Probe { speech: Ref::new(0), wait: SpeechWait::Reported },
            Input::Advance { ms: Millis(1_000) },
        ],
        ..Case::default()
    }
}
