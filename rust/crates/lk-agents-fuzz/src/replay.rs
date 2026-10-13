//! A failing run, saved as a file and replayed forever.
//!
//! What is saved is the case, never the seed: a seed only reproduces while the sampler is
//! unchanged, which is exactly the case that needs no help.
use std::path::Path;

use crate::{
    check::Subject,
    drive::{Case, Outcome},
};

/// One corpus file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Saved {
    pub case: Case,
    pub expect: Expect,
}

/// The verdict the case is pinned to. Compared on identity only -- `Failure::detail` is written
/// for a person and is free to change.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expect {
    Passes,
    Fails { contract: String, subject: Subject },
}

impl Expect {
    /// The first failure names the case. A run reports every finding it has, so on replay the
    /// finding is looked for among all of them, not only at the front.
    pub fn of(outcome: &Outcome) -> Expect {
        match outcome.failures.first() {
            None => Expect::Passes,
            Some(failure) => Expect::Fails {
                contract: failure.contract.clone(),
                subject: failure.subject.clone(),
            },
        }
    }
}

pub fn save(path: impl AsRef<Path>, case: &Case, outcome: &Outcome) -> std::io::Result<()> {
    let saved = Saved { case: case.clone(), expect: Expect::of(outcome) };
    let json = serde_json::to_string_pretty(&saved).expect("a case serialises");
    std::fs::write(path, json + "\n")
}

pub fn load(path: impl AsRef<Path>) -> Saved {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Replay a saved case and compare the verdict to the one it was saved with.
///
/// A file that now passes is as much a finding as one that now fails differently: the fix is real
/// and the file needs updating, which cannot happen if replay goes quietly green.
pub fn verify(saved: &Saved) -> Result<(), String> {
    let outcome = crate::drive::run(saved.case.clone());
    let reproduced = match &saved.expect {
        Expect::Passes => outcome.failures.is_empty(),
        Expect::Fails { contract, subject } => {
            outcome.failures.iter().any(|f| f.contract == *contract && f.subject == *subject)
        }
    };
    if reproduced {
        return Ok(());
    }
    let got = Expect::of(&outcome);
    Err(match (&saved.expect, &got) {
        (Expect::Fails { contract, subject }, Expect::Passes) => format!(
            "expected {contract} to fail on {subject:?}, but the case now passes: the file is \
             stale and should be re-saved or deleted"
        ),
        (Expect::Passes, Expect::Fails { contract, subject }) => {
            format!("expected the case to pass, but {contract} failed on {subject:?}")
        }
        (expected, actual) => format!("expected {expected:?}, got {actual:?}"),
    })
}
