//! A speech fails only from a failure the case injected into one of its own requests, or from
//! closing before it was scheduled, from a synthesis the case failed while it was live, or from a
//! refusal the driver journals itself.
use lk_agents_voice::trace::{SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{Input, Ref},
    record::{Applied, Handle, Recording},
};

pub struct SpeechErrorIsInjected;

impl Rule for SpeechErrorIsInjected {
    const CONTRACT: Contract = Contract {
        id: "speech_error_is_injected",
        guarantee: "A failed speech had a provider failure delivered to one of its own requests, was refused by Close before scheduling, or was a say the driver refused on the journal.",
        examples: &[failed_by_injection],
        counterexamples: &[failed_without_injection],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let (host, internal, inference) = (
            r.host_view(),
            r.internal_view(),
            r.inference_view(),
        );
        for speech in host.speeches() {
            let Some(error) = speech.error() else { continue };

            // The only provider that can fail is our fake, and it fails only when told to. So a
            // failed speech must have had a failure sent to one of the chat requests the run
            // attributes to it, unless Close refused it before it was ever scheduled.
            let requests = inference.requests_for(speech.id);
            // A refused failure never reached a provider, so it cannot be what broke the speech.
            let injected = inputs.accepted().any(
                |sent| matches!(&sent.input, Input::LlmFail { chat } if requests.contains(&chat.nth)),
            );
            // A synthesis failure is attributed by time rather than by request: the fake voice fails
            // only when told to, and a speech that ended after one was told to has its cause.
            let before_done = |sent: &&Applied| speech.done_at().is_none_or(|done| sent.at < done);
            let voice_failed = inputs
                .accepted()
                .filter(before_done)
                .any(|sent| matches!(sent.input, Input::TtsFail { .. }));
            // The voice was down when the speech ended: the outage as the case last set it before.
            let voice_down = inputs
                .accepted()
                .filter(before_done)
                .filter_map(|sent| match sent.input {
                    Input::TtsOutage { down } => Some(down),
                    _ => None,
                })
                .last()
                .is_some_and(|down| down);
            let refused_by_close = inputs.close_asked_at().is_some() && !speech.is_scheduled();
            // A `say` the driver refused on its own terms -- words with no voice to speak them --
            // fails without ever making a request; the refusal is on the journal.
            let refused_by_driver = requests.is_empty()
                && internal
                    .steps()
                    .iter()
                    .any(|(_, step)| matches!(step, Step::Speech(SpeechStep::SayRefused { .. })));
            let accounted =
                injected || voice_failed || voice_down || refused_by_close || refused_by_driver;
            check.expect(
                &speech,
                accounted,
                format!(
                    "speech failed with {error} but no failure was sent to its requests {requests:?}"
                ),
            );
        }
    }
}

/// The case failed the speech's own chat request, and the speech failed.
fn failed_by_injection() -> Recording {
    Authored::new()
        .chat(4)
        .sent(Input::LlmFail { chat: Ref::new(0) })
        .speech(4, Handle::finished().failed("injected provider failure"))
        .done()
}

/// A speech failed; nothing was ever injected and nobody closed.
fn failed_without_injection() -> Recording {
    Authored::new().speech(4, Handle::finished().failed("boom")).done()
}
