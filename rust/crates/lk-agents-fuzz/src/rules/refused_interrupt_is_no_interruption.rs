//! A host `interrupt()` the session refused, because the speech disallowed interruptions by the
//! time it landed, interrupts nothing: the speech is not cut for it, and not reported interrupted
//! when it ends. The handle kept the request after the refusal and turned it into an interruption
//! when the speech played out (the reference raises in `interrupt()` and nothing changes).
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct RefusedInterruptIsNoInterruption;

impl Rule for RefusedInterruptIsNoInterruption {
    const CONTRACT: Contract = Contract {
        id: "refused_interrupt_is_no_interruption",
        guarantee: "A host interrupt the session refused is never journalled as the speech's interruption.",
        examples: &[refused_then_asked_again],
        counterexamples: &[refused_then_interrupted],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // The speeches whose last host interrupt was refused.
        let mut refused: Vec<SpeechId> = Vec::new();
        for (at, step) in r.journal() {
            match step {
                Step::Interrupt(InterruptStep::HostInterruptRefused { speech }) => {
                    refused.push(*speech);
                }
                Step::Interrupt(InterruptStep::HostInterrupt { speech: Some(speech), .. }) => {
                    refused.retain(|s| s != speech);
                }
                Step::Interrupt(InterruptStep::HostInterrupt { speech: None, .. }) => {
                    refused.clear();
                }
                Step::Interrupt(InterruptStep::Interrupted {
                    speech,
                    cause: InterruptCause::Host,
                    ..
                }) if refused.contains(speech) => {
                    check.expect(
                        Subject::Session,
                        false,
                        format!("{speech:?} journalled interrupted by a refused host interrupt (seq {})", at.seq),
                    );
                }
                Step::Interrupt(InterruptStep::Interrupted { speech, .. }) => {
                    check.expect(Subject::Session, true, String::new());
                    refused.retain(|s| s != speech);
                }
                _ => {}
            }
        }
    }
}

fn refused(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::HostInterruptRefused { speech: SpeechId(speech) })
}

fn asked(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::HostInterrupt { speech: Some(SpeechId(speech)), force: true })
}

fn interrupted(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::Interrupted {
        speech: SpeechId(speech),
        cause: InterruptCause::Host,
        force: true,
    })
}

/// Refused, then asked again with `force`: that one interrupts.
fn refused_then_asked_again() -> Recording {
    Authored::new().journal(refused(2)).journal(asked(2)).journal(interrupted(2)).done()
}

/// Refused, then reported interrupted as it ended.
fn refused_then_interrupted() -> Recording {
    Authored::new().journal(refused(2)).journal(interrupted(2)).done()
}
