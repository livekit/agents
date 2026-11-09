use std::collections::BTreeMap;

use crate::{
    drive::Case,
    record::{Kind, Recording},
    rules,
};

// A finding's identity is its contract plus subject, never its message: shrinking compares
// identities, and the wording changes as the case changes.

/// What a failed expectation is about.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Subject {
    /// The run as a whole.
    Session,
    /// One speech, by the id the driver minted for it.
    Speech(u64),
    /// The n-th resource of a kind, named the way a case names one.
    Resource(Kind, u32),
}

impl Subject {
    /// What the subject is, without which one: two speeches failing one contract are one finding.
    pub fn kind(&self) -> String {
        match self {
            Subject::Session => "Session".into(),
            Subject::Speech(_) => "Speech".into(),
            Subject::Resource(kind, _) => format!("{kind:?}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub contract: String,
    pub subject: Subject,
    /// Diagnostic text. Read by a person, never compared.
    pub detail: String,
}

/// Accumulates one contract's expectations over one recording.
pub struct Check {
    contract: &'static str,
    /// How many expectations this contract actually made. Zero means it never applied to this
    /// run, which a report must distinguish from passing.
    pub expectations: u32,
    pub failures: Vec<Failure>,
}

impl Check {
    pub fn new(contract: &'static str) -> Self {
        Check { contract, expectations: 0, failures: Vec::new() }
    }

    /// Record one expectation about a subject. Reaching this call is what marks the rule as
    /// applying to the run: a rule that returns before its first `expect` did not apply, and the
    /// report says so rather than counting it as a pass.
    pub fn expect(&mut self, subject: impl Into<Subject>, holds: bool, detail: impl Into<String>) {
        self.expectations += 1;
        if !holds {
            self.failures.push(Failure {
                contract: self.contract.to_string(),
                subject: subject.into(),
                detail: detail.into(),
            });
        }
    }

    /// [`Check::expect`] with a lazily built detail, for per-frame loops where formatting would
    /// dominate the rule's time.
    pub fn expect_with(
        &mut self,
        subject: impl Into<Subject>,
        holds: bool,
        detail: impl FnOnce() -> String,
    ) {
        self.expectations += 1;
        if !holds {
            self.failures.push(Failure {
                contract: self.contract.to_string(),
                subject: subject.into(),
                detail: detail(),
            });
        }
    }
}

/// An open finding the launch gate tolerates: a contract failing on a kind of subject.
pub struct Known {
    pub contract: &'static str,
    /// [`Subject::kind`]: which speech or resource never splits a key.
    pub subject: &'static str,
    /// The regression test (`src/<name>_tests.rs`) whose fix closes it, once one exists:
    /// a fixed finding the full window no longer reaches fails the gate until it is removed.
    pub fixed_by: Option<&'static str>,
    pub why: &'static str,
}

/// The open findings from `known_findings.tsv`. Any other (contract, subject kind) fails the
/// gate, so a new kind of defect cannot hide among known ones; the file only shrinks.
/// `session_settles` is the release phase's own verdict, not a rule.
pub static KNOWN_FINDINGS: std::sync::LazyLock<Vec<Known>> = std::sync::LazyLock::new(|| {
    include_str!("known_findings.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            let [contract, subject, fixed_by, why] = fields[..] else {
                panic!("known_findings.tsv: not four tab-separated fields: {line:?}")
            };
            let fixed_by = (fixed_by != "-").then_some(fixed_by);
            Known { contract, subject, fixed_by, why }
        })
        .collect()
});

impl Failure {
    /// Whether this failure is one of the open findings.
    pub fn known(&self) -> bool {
        let kind = self.subject.kind();
        KNOWN_FINDINGS.iter().any(|k| k.contract == self.contract && k.subject == kind)
    }
}

/// One guarantee the driver must keep.
///
/// A rule runs once over the complete recording and states a property of the whole timeline,
/// never of the moment it is read. A rule that only applies once the session has closed reads
/// [`Recording::phase`] and returns early otherwise.
pub trait Rule {
    /// What the rule promises, and the recordings that prove the promise can be checked.
    const CONTRACT: Contract;

    /// The run is complete: judge it.
    fn on_complete(r: &Recording, check: &mut Check);
}

/// What a rule promises, and how it proves it can tell.
///
/// Each `example` must be accepted while making at least one expectation, and each
/// `counterexample` must be rejected, so a rule that never applies or cannot fail is caught. A
/// `probe` reaches the rule's prerequisite deterministically; it is a whole [`Case`] because some
/// prerequisites are fixed when the session is built.
pub struct Contract {
    /// The rule's name in reports: its file name.
    pub id: &'static str,
    /// The guarantee in one sentence, as a person would say it.
    pub guarantee: &'static str,
    pub examples: &'static [fn() -> Recording],
    pub counterexamples: &'static [fn() -> Recording],
    pub probe: Option<fn() -> Case>,
}

/// One rule as the registry holds it, generated by `build.rs`.
pub struct Registered {
    pub contract: Contract,
    pub on_complete: fn(&Recording, &mut Check),
}

impl Registered {
    pub fn id(&self) -> &'static str {
        self.contract.id
    }

    pub fn run(&self, recording: &Recording) -> Check {
        let mut check = Check::new(self.contract.id);
        (self.on_complete)(recording, &mut check);
        check
    }
}

/// Runs every contract over a finished recording; returns the failures and each contract's
/// expectation count, so a contract with zero is reported as unexercised, not passed.
pub fn check_all(recording: &Recording) -> (Vec<Failure>, BTreeMap<&'static str, u32>) {
    let mut failures = Vec::new();
    let mut expectations = BTreeMap::new();
    for rule in rules::RULES {
        let check = rule.run(recording);
        *expectations.entry(rule.id()).or_default() += check.expectations;
        failures.extend(check.failures);
    }
    (failures, expectations)
}
