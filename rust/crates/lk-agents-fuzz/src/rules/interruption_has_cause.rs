//! Every interruption the driver journals has an input that permits its stated cause.
//!
//! The journal says why (`InterruptCause`); this rule checks the why against what the test did:
//! a host interrupt, a Close, or user speech and turn input. Causes the runtime owns alone
//! (superseded, watchdog, schedule rejected, handoff, detector) need no input.
//!
//! `Host` is broader than the two interrupt calls. Swapping the audio output while a speech is
//! playing also interrupts it as the host's own doing, because a segment already playing into
//! the old sink cannot move; the driver takes that path without a `HostInterrupt` step, so the
//! sink swap counts as a permitted input here.
use lk_agents_voice::trace::{InterruptCause, InterruptStep::Interrupted, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::Input,
    record::{InputsView, Interruption, Observed, Recording},
};

pub struct InterruptionHasCause;

impl Rule for InterruptionHasCause {
    const CONTRACT: Contract = Contract {
        id: "interruption_has_cause",
        guarantee: "A journalled interruption's cause matches an input that permits it: for Host, an interrupt the case issued or a sink swap it forced.",
        examples: &[host_interrupt_then_journal, sink_swap_then_journal],
        counterexamples: &[host_cause_without_host_call],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let internal = r.internal_view();
        for Interruption { at, speech, cause } in internal.interruptions() {
            // The journalled cause names who asked; that party must actually have acted before it.
            let permitted = match cause {
                InterruptCause::Host => host_interrupted_before(&inputs, at),
                InterruptCause::Closing => inputs.close_asked_at().is_some_and(|c| c < at),
                InterruptCause::ByAudioActivity | InterruptCause::UserTurn => {
                    user_input_before(&inputs, at)
                }
                InterruptCause::Detector
                | InterruptCause::Superseded
                | InterruptCause::ScheduleRejected
                | InterruptCause::Watchdog
                | InterruptCause::Handoff => true,
            };
            check.expect(
                Subject::Speech(speech),
                permitted,
                format!("interrupted for {cause:?} with no matching input before seq {}", at.seq),
            );
        }
    }
}

/// Something the case did before `at` that the driver treats as the host interrupting: an
/// interrupt call, or an audio-output swap, which cuts whatever is playing into the old sink.
fn host_interrupted_before(inputs: &InputsView, at: Observed) -> bool {
    inputs.sent().iter().any(|sent| {
        sent.at < at
            && matches!(
                sent.input,
                Input::Interrupt { .. }
                    | Input::InterruptSpeech { .. }
                    | Input::SetAudioOutput { .. }
            )
    })
}

/// Any recognition event delivered, or a user-turn claim or commit issued, before `at`.
fn user_input_before(inputs: &InputsView, at: Observed) -> bool {
    // A recognition event the stream refused never reached the turn, so it permits nothing. A
    // turn command is judged by its issue: whether the driver acted on it is what is in question.
    let recognized = inputs.sent().iter().any(|sent| {
        sent.at < at
            && sent.accepted
            && matches!(
                sent.input,
                Input::SttStart { .. }
                    | Input::SttInterim { .. }
                    | Input::SttPreflight { .. }
                    | Input::SttFinal { .. }
                    | Input::SttEnd { .. }
            )
    });
    let claimed = inputs.sent().iter().any(|sent| {
        sent.at < at
            && matches!(
                sent.input,
                Input::ClaimUserTurn {} | Input::CommitUserTurn { .. } | Input::CommitTurn { .. }
            )
    });
    recognized || claimed
}

/// The host interrupted, and the journal blames the host.
fn host_interrupt_then_journal() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Authored::new()
        .sent(Input::Interrupt { force: false, wait: false })
        .journal(Step::Interrupt(Interrupted {
            speech: SpeechId(3),
            cause: InterruptCause::Host,
            force: false,
            position: None,
        }))
        .done()
}

/// The case swapped the audio output, which cuts whatever was playing into the old one.
fn sink_swap_then_journal() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Authored::new()
        .sent(Input::SetAudioOutput { attach: false })
        .journal(Step::Interrupt(Interrupted {
            speech: SpeechId(3),
            cause: InterruptCause::Host,
            force: true,
            position: None,
        }))
        .done()
}

/// The journal blames the host, but the host never interrupted anything.
fn host_cause_without_host_call() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Authored::new()
        .journal(Step::Interrupt(Interrupted {
            speech: SpeechId(3),
            cause: InterruptCause::Host,
            force: false,
            position: None,
        }))
        .done()
}
