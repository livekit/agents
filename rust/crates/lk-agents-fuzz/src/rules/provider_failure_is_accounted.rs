//! A provider failure the case injected is accounted for: retried, surfaced, or closed over.
//!
//! The fake fails only when the case tells it to, so every accepted failure is a known input. The
//! driver may answer it in more than one legitimate way -- arm a retry for that leg, fail the
//! speech and say so on the handle, or begin closing once its error tolerance is spent -- but it
//! may not swallow it. The host's own close, which ends every run, is not the driver closing over
//! anything. This is the ownership half of provider failure; whether the retry count
//! and tolerance are right is a separate contract that needs the configured values.
use lk_agents_voice::trace::{GenerationStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{Input, Ref},
    record::{Handle, Recording},
};

pub struct ProviderFailureIsAccounted;

impl Rule for ProviderFailureIsAccounted {
    const CONTRACT: Contract = Contract {
        id: "provider_failure_is_accounted",
        guarantee: "Every injected model failure is retried, surfaced on its speech, or followed by the session closing.",
        examples: &[failure_then_retry, failure_then_failed_speech],
        counterexamples: &[failure_swallowed],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let (host, internal, inference) = (r.host_view(), r.internal_view(), r.inference_view());
        for sent in inputs.sent() {
            let Input::LlmFail { chat } = &sent.input else { continue };
            if !sent.accepted {
                continue;
            }
            // The driver gets a turn to react before this is a failure to react.
            if !inputs.driver_ran_after(sent.at) {
                continue;
            }
            // Which speech the driver planned this request for; a request attributed to no speech
            // belongs to nobody we can name, so there is nothing to check against.
            let Some(id) = inference.request_for(chat.nth) else { continue };
            let speech = host.speech(id);

            let retried = internal.speech(id).retry_armed_at().is_some_and(|armed| armed > sent.at);
            let surfaced = speech.error().is_some();
            // Only the driver's own error close counts as closing over the failure: the host closing
            // the session at the end of every run says nothing about what the driver did with it.
            let closing = internal.error_close_requested_after(sent.at);

            check.expect(
                &speech,
                retried || surfaced || closing,
                format!(
                    "a model failure on {chat:?} for speech {id} was neither retried, surfaced nor \
                     closed over"
                ),
            );
        }
    }
}

fn failed_request(speech: u64) -> Authored {
    Authored::new().chat(speech).sent(Input::LlmFail { chat: Ref::new(0) }).then_act()
}

/// The driver armed a retry for the speech after the failure.
fn failure_then_retry() -> Recording {
    use lk_agents_abi::Ts;
    use lk_agents_voice::{agent_activity::scheduling::SpeechId, trace::ProviderKind};
    failed_request(4)
        .journal(Step::Generation(GenerationStep::RetryArmed {
            speech: SpeechId(4),
            leg: ProviderKind::Llm,
            attempt: 1,
            at: Ts::ZERO,
        }))
        .done()
}

/// The speech carries the error, so the caller was told.
fn failure_then_failed_speech() -> Recording {
    failed_request(4).speech(4, Handle::finished().failed("injected provider failure")).done()
}

/// The failure was delivered, the driver ran, and nothing came of it anywhere.
fn failure_swallowed() -> Recording {
    failed_request(4).speech(4, Handle::finished()).done()
}
