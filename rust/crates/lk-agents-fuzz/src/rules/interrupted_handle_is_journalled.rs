//! A handle that reports itself interrupted has an `Interrupted` step in the journal.
use lk_agents_voice::trace::{InterruptStep::Interrupted, Step::Interrupt};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording},
};

pub struct InterruptedHandleIsJournalled;

impl Rule for InterruptedHandleIsJournalled {
    const CONTRACT: Contract = Contract {
        id: "interrupted_handle_is_journalled",
        guarantee: "Every interrupted handle has a journalled Interrupted step naming it.",
        examples: &[interrupted_and_journalled],
        counterexamples: &[interrupted_without_journal],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let internal = r.internal_view();
        for speech in host.speeches() {
            if !(speech.is_known() && speech.is_interrupted()) {
                continue;
            }
            // The handle's flag and the driver's own record must agree: an interruption the journal
            // does not know about was set by a path that bypassed the one writer.
            let journalled = internal.speech(speech.id).interrupted_at().is_some();
            check.expect(
                &speech,
                journalled,
                "handle interrupted, journal has no Interrupted step for it",
            );
        }
    }
}

/// The handle says interrupted and the journal agrees.
fn interrupted_and_journalled() -> Recording {
    use lk_agents_voice::{agent_activity::scheduling::SpeechId, trace::InterruptCause};
    Authored::new()
        .speech(3, Handle::finished().interrupted())
        .journal(Interrupt(Interrupted {
            speech: SpeechId(3),
            cause: InterruptCause::Host,
            force: false,
            position: None,
        }))
        .done()
}

/// The handle says interrupted; the journal never said so.
fn interrupted_without_journal() -> Recording {
    Authored::new().speech(3, Handle::finished().interrupted()).done()
}
