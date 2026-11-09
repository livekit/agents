use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageFault, StorageStep},
};

// Another worker that took the voice lane's lease over (`StorageFault::Stolen`) owns the lane
// until it lets go: every write the call sends under its old token must be refused by the
// lane's fence, not applied over the new owner's.

pub struct StolenLaneFencesItsHolder;

impl Rule for StolenLaneFencesItsHolder {
    const CONTRACT: Contract = Contract {
        id: "stolen_lane_fences_its_holder",
        guarantee: "A write under a voice lane lease another worker took over changes nothing \
                    while that worker holds the lane.",
        examples: &[fenced],
        counterexamples: &[applied],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (at, step) in r.host_view().storage() {
            let StorageStep::StaleWrite { applied } = step else { continue };
            check.expect_with(Subject::Session, !applied, || {
                format!("a write under a stolen lease changed the lane at {} ms", at.at_ms)
            });
        }
    }
}

fn stale(applied: bool) -> Recording {
    Authored::new()
        .storage(StorageStep::Fault { fault: StorageFault::Stolen })
        .storage(StorageStep::StaleWrite { applied })
        .done()
}

fn fenced() -> Recording {
    stale(false)
}

fn applied() -> Recording {
    stale(true)
}
