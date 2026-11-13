use lk_agents_voice::trace::{GenerationStep, ProviderKind, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    record::{Kind, NodeUse, Recording},
};

// One layer owns a request's retries (`retry::owner`): each retry the session arms is one
// recoverable `error` event on the provider whose attempt failed. A nested retry layer reports its
// own attempts on top; a silent one reports none. That a request reports one final failure is
// held by inference's and the host's tests: the fakes fail outside any one attempt count. A node override's own code can fail with no
// provider behind it, and the answering-machine detector's classifier is no speech's request, so a
// leg either served is not judged.

pub struct ProviderReportsEachFailureOnce;

impl Rule for ProviderReportsEachFailureOnce {
    const CONTRACT: Contract = Contract {
        id: "provider_reports_each_failure_once",
        guarantee: "Each retried model or voice attempt is one recoverable error event on its provider: no retry is silent, none is reported by a second layer.",
        examples: &[retried_then_given_up],
        counterexamples: &[nested_retries, silent_retry],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let legs = [
            (Kind::Chat, ProviderKind::Llm, NodeUse::Llm),
            (Kind::Tts, ProviderKind::Tts, NodeUse::Tts),
        ];
        let amd = r.inputs_view().sent().iter().any(|sent| matches!(sent.input, Input::Amd { .. }));
        for (kind, leg, node) in legs {
            if r.inference_view().nodes().any(|(_, _, used)| used == node)
                || (leg == ProviderKind::Llm && amd)
            {
                continue;
            }
            let events = r.inference_view().provider_failed();
            let heard = |recoverable: bool| {
                events.iter().filter(|(_, k, r)| *k == kind && *r == recoverable).count()
            };
            let armed = r
                .internal_view()
                .steps()
                .iter()
                .filter(|(_, step)| {
                    matches!(step, Step::Generation(GenerationStep::RetryArmed { leg: armed, .. })
                        if *armed == leg)
                })
                .count();
            check.expect(
                Subject::Session,
                heard(true) == armed,
                format!(
                    "the {leg:?} provider heard {} recoverable failures for {armed} retries",
                    heard(true)
                ),
            );
        }
    }
}

fn armed(attempt: u32) -> Step {
    use lk_agents_abi::Ts;
    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    Step::Generation(GenerationStep::RetryArmed {
        speech: SpeechId(4),
        leg: ProviderKind::Llm,
        attempt,
        at: Ts::ZERO,
    })
}

/// One retried attempt, then the request gives up: one recoverable and one final event.
fn retried_then_given_up() -> Recording {
    Authored::new()
        .chat(4)
        .provider_failed(Kind::Chat, true)
        .journal(armed(1))
        .chat(4)
        .provider_failed(Kind::Chat, false)
        .done()
}

/// A nested retry layer: the provider hears each attempt twice.
fn nested_retries() -> Recording {
    Authored::new()
        .chat(4)
        .provider_failed(Kind::Chat, true)
        .provider_failed(Kind::Chat, true)
        .journal(armed(1))
        .done()
}

/// A retry the provider never heard of.
fn silent_retry() -> Recording {
    Authored::new().chat(4).journal(armed(1)).chat(4).done()
}
