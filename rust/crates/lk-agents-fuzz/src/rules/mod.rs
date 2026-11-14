use crate::check::{Registered, Rule};

// `build.rs` generates the `mod` declarations and `RULES` from `src/rules/<domain>/*.rs`: a file
// holding a unit struct `<Id>` that implements `Rule` is registered by existing.
include!(concat!(env!("OUT_DIR"), "/rules.rs"));

#[cfg(test)]
mod registry {
    use super::*;

    /// Every contract accepts each of its examples while making at least one expectation on it,
    /// and rejects each of its counterexamples. A rule that asserts nothing fails here, as does
    /// a duplicate id or a contract missing either list. This validates the checkers, not the
    /// runtime.
    #[test]
    fn every_contract_rejects_its_counterexamples() {
        let mut ids = std::collections::BTreeSet::new();
        for rule in RULES {
            let contract = &rule.contract;
            assert!(ids.insert(contract.id), "duplicate contract id {}", contract.id);
            assert!(!contract.guarantee.is_empty(), "{} has no guarantee", contract.id);
            assert!(!contract.examples.is_empty(), "{} has no accepted example", contract.id);
            assert!(!contract.counterexamples.is_empty(), "{} has no counterexample", contract.id);
            for (index, example) in contract.examples.iter().enumerate() {
                let check = rule.run(&example());
                assert!(
                    check.failures.is_empty(),
                    "{} rejected its example {index}: {:?}",
                    contract.id,
                    check.failures
                );
                assert!(
                    check.expectations > 0,
                    "{} made no expectation on its example {index}: the rule would be a no-op",
                    contract.id
                );
            }
            for (index, counterexample) in contract.counterexamples.iter().enumerate() {
                let recording = counterexample();
                let check = rule.run(&recording);
                assert!(
                    check.failures.iter().any(|f| f.contract == contract.id),
                    "{} accepted counterexample {index}",
                    contract.id
                );
            }
            let empty = crate::record::Recording::default();
            let check = rule.run(&empty);
            assert!(check.failures.is_empty(), "{} fails on an empty recording", contract.id);
        }
    }
}
