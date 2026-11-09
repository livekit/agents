use lk_agents_abi::resources::ResourceId;
use lk_agents_voice::{
    agent_session::effects::{ProviderCall, StreamKind},
    trace::{InputStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Models},
    record::Recording,
};

// An effect naming a released id is a use after free: the table answers it with nothing.

pub struct ReleasedProviderUnused;

impl Rule for ReleasedProviderUnused {
    const CONTRACT: Contract = Contract {
        id: "released_provider_unused",
        guarantee: "No effect names a provider after the core released it.",
        examples: &[opened_then_released],
        counterexamples: &[opened_after_release, released_twice],
        probe: Some(models_swapped_twice),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let mut released = std::collections::BTreeSet::new();
        for (at, step) in r.internal_view().steps() {
            let Step::Input(InputStep::ProviderCalled(call)) = step else { continue };
            let id = match *call {
                ProviderCall::OpenLeg { resource, .. } => resource,
                ProviderCall::Prewarm(id)
                | ProviderCall::Release(id)
                | ProviderCall::CloseRealtime(id) => id,
                ProviderCall::FlushStt | ProviderCall::FlushVad => continue,
            };
            check.expect(
                Subject::Session,
                !released.contains(&id),
                format!("{call:?} at seq {} names provider {} after its release", at.seq, id.0),
            );
            if matches!(call, ProviderCall::Release(_)) {
                released.insert(id);
            }
        }
    }
}

fn called(call: ProviderCall) -> Step {
    Step::Input(InputStep::ProviderCalled(call))
}

fn open(id: u64) -> Step {
    called(ProviderCall::OpenLeg { leg: StreamKind::Stt, open: 1, resource: ResourceId(id) })
}

/// The STT opened, then released once nothing held it.
fn opened_then_released() -> Recording {
    Authored::new().journal(open(1)).journal(called(ProviderCall::Release(ResourceId(1)))).done()
}

fn opened_after_release() -> Recording {
    Authored::new().journal(called(ProviderCall::Release(ResourceId(1)))).journal(open(1)).done()
}

fn released_twice() -> Recording {
    let release = || called(ProviderCall::Release(ResourceId(1)));
    Authored::new().journal(release()).journal(release()).done()
}

/// Prerequisite: every model replaced twice, so the opening providers and then the first
/// replacements are released while the STT reopens on each.
fn models_swapped_twice() -> Case {
    Case {
        setup: Setup { stt: true, vad: true, ..Setup::default() },
        inputs: vec![
            Input::UpdateModels { models: Models::All },
            Input::Advance { ms: Millis(50) },
            Input::UpdateModels { models: Models::All },
            Input::Advance { ms: Millis(50) },
        ],
        ..Case::default()
    }
}
