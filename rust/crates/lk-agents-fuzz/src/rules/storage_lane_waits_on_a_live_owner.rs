use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup, StorageSetup},
    input::{Input, Millis},
    record::{Recording, StorageOperation, StorageStep},
};
use lk_agents_runtime::storage::thread::LeaseTiming;

pub struct StorageLaneWaitsOnALiveOwner;

/// How long past its renewal interval a live owner's next request may come: its renewals are
/// sent start to start, whatever the answers do.
const SLACK_MS: u64 = 1_000;

impl Rule for StorageLaneWaitsOnALiveOwner {
    const CONTRACT: Contract = Contract {
        id: "storage_lane_waits_on_a_live_owner",
        guarantee: "A turn's acquire of the voice lane waits only on a live owner: every lease that refuses it was renewed, committed or released within one renewal interval.",
        examples: &[taken_at_once, waited_on_a_renewing_owner],
        counterexamples: &[waited_on_an_orphan],
        probe: Some(a_stored_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let limit = LeaseTiming::default().renew_interval.as_millis() as u64 + SLACK_MS;
        let storage = r.host_view().storage();
        for (i, (requested, step)) in storage.iter().enumerate() {
            let StorageStep::Requested { speech, operation: StorageOperation::Acquire } = step
            else {
                continue;
            };
            let idle = storage[i + 1..]
                .iter()
                .take_while(|(_, step)| {
                    !matches!(step, StorageStep::Answered { speech: id, operation: StorageOperation::Acquire, .. } if id == speech)
                })
                .filter_map(|(_, step)| match step {
                    StorageStep::LaneHeld { idle_ms } => Some(*idle_ms),
                    _ => None,
                })
                .max();
            check.expect_with(Subject::Speech(*speech), idle.is_none_or(|ms| ms <= limit), || {
                format!(
                    "the acquire at {} ms waited on a lease whose owner was silent for {} ms",
                    requested.at_ms,
                    idle.unwrap_or_default()
                )
            });
        }
    }
}

fn acquire() -> StorageStep {
    StorageStep::Requested { speech: 1, operation: StorageOperation::Acquire }
}

fn acquired() -> StorageStep {
    StorageStep::Answered {
        speech: 1,
        operation: StorageOperation::Acquire,
        result: Ok(()),
        delivered: true,
    }
}

fn taken_at_once() -> Recording {
    Authored::new().storage(acquire()).later(5).storage(acquired()).done()
}

fn waited_on_a_renewing_owner() -> Recording {
    Authored::new()
        .storage(acquire())
        .storage(StorageStep::LaneHeld { idle_ms: 2_500 })
        .later(1_000)
        .storage(acquired())
        .done()
}

/// A lease nobody renews held the lane until its TTL ran out.
fn waited_on_an_orphan() -> Recording {
    Authored::new()
        .storage(acquire())
        .later(6_000)
        .storage(StorageStep::LaneHeld { idle_ms: 6_000 })
        .later(5_000)
        .storage(acquired())
        .done()
}

pub(crate) fn a_stored_turn() -> Case {
    Case {
        setup: Setup {
            tts: false,
            storage: Some(StorageSetup::default()),
            ..Setup::default()
        },
        inputs: vec![
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(100) },
        ],
        ..Case::default()
    }
}
