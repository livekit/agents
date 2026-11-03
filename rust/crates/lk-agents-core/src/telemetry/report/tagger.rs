//! The session's tags, judgments and outcome, and the "recording disabled" response every Cloud
//! upload recognizes.

use lk_agents_abi::telemetry::report::{
    EvaluationRecord, Outcome, ReportOutcome, ReportTag, TAG_FAIL, TAG_SUCCESS,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::time::SystemTime;

/// The 401 that means "the project owner turned recording off", matched case-insensitively.
/// Other 401s must not disable uploads.
pub const DISABLED_MARKERS: [&str; 2] = ["data recording is disabled", "disabled by owner"];

/// Whether a Cloud response is the project's "recording disabled" refusal.
pub fn is_disabled_response(status_code: u16, body: &[u8]) -> bool {
    if status_code != 401 {
        return false;
    }
    let text = String::from_utf8_lossy(body).to_lowercase();
    DISABLED_MARKERS.iter().any(|m| text.contains(m))
}

/// The host log exporter's per-session upload gate: one transition, one warning.
#[derive(Debug, Default)]
pub struct UploadGate {
    disabled: bool,
    warned: bool,
}

impl lk_agents_abi::telemetry::report::UploadGate for UploadGate {
    fn new() -> Self {
        UploadGate::default()
    }

    fn reset(&mut self) {
        self.disabled = false;
        self.warned = false;
    }

    fn disabled(&self) -> bool {
        self.disabled
    }

    fn disable(&mut self) -> bool {
        if self.disabled {
            return false;
        }
        self.disabled = true;
        !std::mem::replace(&mut self.warned, true)
    }

    fn is_disabled_response(status_code: u16, body: Vec<u8>) -> bool {
        is_disabled_response(status_code, &body)
    }
}

/// One tag: its metadata and when it was added, from the session clock.
#[derive(Debug, Clone, PartialEq)]
struct TagEntry {
    metadata: Option<Value>,
    timestamp: SystemTime,
}

/// The session's tags, judgments and outcome. Times are supplied by the caller.
#[derive(Debug, Default)]
pub struct Tagger {
    tags: Vec<(String, TagEntry)>,
    evaluations: Vec<EvaluationRecord>,
    outcome_reason: Option<String>,
}

impl Tagger {
    /// Clears any existing `lk.fail` first.
    pub fn success(&mut self, reason: Option<String>, at: SystemTime) {
        self.remove(TAG_FAIL);
        self.add(TAG_SUCCESS.to_owned(), None, at);
        self.outcome_reason = reason;
    }

    pub fn fail(&mut self, reason: Option<String>, at: SystemTime) {
        self.remove(TAG_SUCCESS);
        self.add(TAG_FAIL.to_owned(), None, at);
        self.outcome_reason = reason;
    }

    /// Replaces a tag in place (keeping its position), else appends it.
    pub fn add(&mut self, tag: String, metadata: Option<Value>, at: SystemTime) {
        let entry = TagEntry { metadata, timestamp: at };
        match self.tags.iter_mut().find(|(k, _)| *k == tag) {
            Some(slot) => slot.1 = entry,
            None => self.tags.push((tag, entry)),
        }
    }

    pub fn remove(&mut self, tag: &str) {
        self.tags.retain(|(k, _)| k != tag);
    }

    pub fn tags(&self) -> BTreeSet<String> {
        self.tags.iter().map(|(k, _)| k.clone()).collect()
    }

    /// The tags in insertion order.
    pub fn entries(&self) -> Vec<ReportTag> {
        self.tags
            .iter()
            .map(|(name, e)| ReportTag {
                name: name.clone(),
                metadata: e.metadata.clone(),
                timestamp: lk_agents_abi::unix_secs(e.timestamp),
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    pub fn evaluations(&self) -> &[EvaluationRecord] {
        &self.evaluations
    }

    /// From tag presence; `lk.success` wins if both are set.
    pub fn outcome(&self) -> Option<Outcome> {
        if self.tags.iter().any(|(k, _)| k == TAG_SUCCESS) {
            Some(Outcome::Success)
        } else if self.tags.iter().any(|(k, _)| k == TAG_FAIL) {
            Some(Outcome::Fail)
        } else {
            None
        }
    }

    /// Only another `success`/`fail` overwrites it.
    pub fn outcome_reason(&self) -> Option<&str> {
        self.outcome_reason.as_deref()
    }

    /// The outcome and its reason, as the session report carries them.
    pub fn report_outcome(&self) -> Option<ReportOutcome> {
        Some(ReportOutcome { outcome: self.outcome()?, reason: self.outcome_reason.clone() })
    }

    pub fn record_judgment(
        &mut self,
        name: String,
        verdict: String,
        reasoning: String,
        instructions: String,
        at: SystemTime,
    ) {
        let tag = format!("lk.judge.{name}:{verdict}");
        self.add(tag.clone(), None, at);
        self.evaluations.push(EvaluationRecord { name, tag, verdict, reasoning, instructions });
    }

    pub fn has_evals(&self) -> bool {
        !self.evaluations.is_empty() || self.outcome().is_some()
    }
}
