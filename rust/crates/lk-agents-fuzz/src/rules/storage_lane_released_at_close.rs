use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageStep},
};

pub struct StorageLaneReleasedAtClose;

impl Rule for StorageLaneReleasedAtClose {
    const CONTRACT: Contract = Contract {
        id: "storage_lane_released_at_close",
        guarantee: "Once the call's store closed, every run it started on the voice lane has ended, cancelled ones included, and no lease is left held, unless the database was down or lost a write.",
        examples: &[all_ended],
        counterexamples: &[left_held, run_never_ended],
        probe: Some(super::storage_lane_waits_on_a_live_owner::a_stored_turn),
    };

    /// A store that met an outage or a lost write may leave its lease to lapse: not judged.
    fn on_complete(r: &Recording, check: &mut Check) {
        for (_, step) in r.host_view().storage() {
            let StorageStep::LaneSettled { unended, held, faulted: false } = *step else {
                continue;
            };
            check.expect_with(Subject::Session, unended == 0 && !held, || {
                format!("the closed store left {unended} runs unended, the lease held: {held}")
            });
        }
    }
}

fn settled(unended: u32, held: bool, faulted: bool) -> Recording {
    Authored::new().storage(StorageStep::LaneSettled { unended, held, faulted }).done()
}

fn all_ended() -> Recording {
    settled(0, false, false)
}

fn left_held() -> Recording {
    settled(0, true, false)
}

fn run_never_ended() -> Recording {
    settled(1, false, false)
}
