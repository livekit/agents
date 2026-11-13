use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageFault, StorageStep},
};

// Another worker that took the voice partition's lease over (`StorageFault::Stolen`) owns the partition
// until it lets go: every write the call sends under its old token must be refused by the
// partition's fence, not applied over the new owner's.

pub struct StolenPartitionFencesItsHolder;

impl Rule for StolenPartitionFencesItsHolder {
    const CONTRACT: Contract = Contract {
        id: "stolen_partition_fences_its_holder",
        guarantee: "A write under a voice partition_key lease another worker took over changes nothing \
                    while that worker holds the partition_key.",
        examples: &[fenced],
        counterexamples: &[applied],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (at, step) in r.host_view().storage() {
            let StorageStep::StaleWrite { applied } = step else { continue };
            check.expect_with(Subject::Session, !applied, || {
                format!("a write under a stolen lease changed the partition_key at {} ms", at.at_ms)
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
