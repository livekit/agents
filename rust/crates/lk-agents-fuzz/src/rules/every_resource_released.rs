//! Once the session has ended, its resource table is empty.
//!
//! The host registers every provider, output and microphone it hands over, and the data plane
//! registers every stream sink it feeds; the table keeps each alive until released. Whatever owns
//! an id releases it: the decision core what it held or was handed in a command it never took, the
//! runtime the microphone and the sinks. An id still in the table once the driver has exited and
//! the session's tasks are joined has nobody left to release it: the provider behind it is kept
//! alive for as long as the host keeps the session object.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{Input, Millis, Models},
    record::{Phase, Recording},
};

pub struct EveryResourceReleased;

impl Rule for EveryResourceReleased {
    const CONTRACT: Contract = Contract {
        id: "every_resource_released",
        guarantee: "Once the driver has exited, the session's resource table holds nothing: every \
                    provider, output, microphone and stream sink registered was released.",
        examples: &[released_before_exit],
        counterexamples: &[left_at_exit],
        probe: Some(models_swapped_then_closed),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if r.phase() != Phase::Closed || !r.driver_finished() {
            return;
        }
        check.expect(
            Subject::Session,
            r.resources_left().is_empty(),
            format!("still in the table after the driver exited: {:?}", r.resources_left()),
        );
    }
}

fn released_before_exit() -> Recording {
    Authored::new().sent(Input::Close { drain: false }).driver_finished().done()
}

fn left_at_exit() -> Recording {
    Authored::new()
        .sent(Input::Close { drain: false })
        .driver_finished()
        .resource_left(1, "audio sink")
        .done()
}

/// Prerequisite: providers handed over mid-call, streams opened on them, then the close.
fn models_swapped_then_closed() -> Case {
    Case {
        setup: Setup { stt: true, vad: true, ..Setup::default() },
        inputs: vec![
            Input::UpdateModels { models: Models::All },
            Input::Advance { ms: Millis(50) },
        ],
        ..Case::default()
    }
}
