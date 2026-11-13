use lk_agents_abi::InputState;
use lk_agents_abi::errors::ProviderError;
use lk_agents_voice::events::{ErrorEvent, ErrorSource, SessionEvent};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Input, Ref},
    record::{Handle, Host, Recording},
};

// An input called after its provider failed answers with that failure (`InputState`): a push the
// session makes into a failed synthesis or recognition fails the speech or reports the error with
// the provider's own failure, never a generic refusal that hides it.

pub struct InputRefusalIsTheFailure;

impl Rule for InputRefusalIsTheFailure {
    const CONTRACT: Contract = Contract {
        id: "input_refusal_is_the_failure",
        guarantee: "No speech fails, and the session reports no error, with a closed input's refusal: an input called after its provider failed returns that provider's failure.",
        examples: &[failed_with_the_providers_error],
        counterexamples: &[failed_with_a_closed_input, reported_a_closed_input],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let closed = closed();
        let said = closed.to_string();
        for speech in r.host_view().speeches() {
            let Some(error) = speech.error() else { continue };
            check.expect(
                speech,
                !error.contains(&said),
                format!("speech failed with a closed input's refusal ({error}), not its failure"),
            );
        }
        for (at, event) in r.host() {
            let Host::Session(SessionEvent::Error(error)) = event else { continue };
            check.expect(
                Subject::Session,
                error.error != closed,
                format!(
                    "the {:?} error at seq {} is a closed input's refusal",
                    error.source, at.seq
                ),
            );
        }
    }
}

/// What an input that ended normally answers.
fn closed() -> ProviderError {
    InputState::Ended.admit().expect_err("an ended input refuses")
}

/// The voice failed, and the speech failed with what it said.
fn failed_with_the_providers_error() -> Recording {
    Authored::new()
        .tts()
        .sent(Input::TtsFail { tts: Ref::new(0) })
        .speech(4, Handle::finished().failed("injected synthesis failure"))
        .done()
}

/// The voice failed, and the speech's next push was refused as closed.
fn failed_with_a_closed_input() -> Recording {
    Authored::new()
        .tts()
        .sent(Input::TtsFail { tts: Ref::new(0) })
        .speech(4, Handle::finished().failed(&closed().to_string()))
        .done()
}

/// The session reported a refused push in place of the failure behind it.
fn reported_a_closed_input() -> Recording {
    Authored::new()
        .session(SessionEvent::Error(Box::new(ErrorEvent::provider(
            ErrorSource::Tts,
            false,
            closed(),
            None,
        ))))
        .done()
}
