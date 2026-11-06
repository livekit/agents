//! The `Tagger` the host constructs: the session's tags, judgments and outcome, stamped by the
//! process clock.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::SystemTime;

use lk_agents_abi::Clock;
use lk_agents_abi::evals::JudgeVerdict;
use lk_agents_abi::telemetry::report::{
    EvaluationRecord, Outcome, ReportOutcome, ReportTag, SessionReport, TAG_FAIL, TAG_SUCCESS,
};
use serde_json::Value;

use lk_agents_net::env::ProcessEnv;

/// One tag: its metadata and when it was added, from the session clock.
#[derive(Debug, Clone, PartialEq)]
struct TagEntry {
    metadata: Option<Value>,
    timestamp: SystemTime,
}

pub struct Tagger {
    clock: Arc<dyn Clock>,
    tags: Vec<(String, TagEntry)>,
    evaluations: Vec<EvaluationRecord>,
    outcome_reason: Option<String>,
}

impl Tagger {
    fn with_clock(clock: Arc<dyn Clock>) -> Self {
        Tagger { clock, tags: Vec::new(), evaluations: Vec::new(), outcome_reason: None }
    }

    fn has(&self, tag: &str) -> bool {
        self.tags.iter().any(|(k, _)| k == tag)
    }
}

impl lk_agents_abi::telemetry::report::Tagger for Tagger {
    type Env = ProcessEnv;

    /// Timestamps are strictly increasing, so same-instant tags keep insertion order.
    fn new(env: &ProcessEnv) -> Self {
        Self::with_clock(env.clock.clone())
    }

    /// Clears any existing `lk.fail` first.
    fn success(&mut self, reason: Option<String>) {
        self.remove(TAG_FAIL.to_owned());
        self.add(TAG_SUCCESS.to_owned(), None);
        self.outcome_reason = reason;
    }

    fn fail(&mut self, reason: Option<String>) {
        self.remove(TAG_SUCCESS.to_owned());
        self.add(TAG_FAIL.to_owned(), None);
        self.outcome_reason = reason;
    }

    /// Replaces a tag in place (keeping its position), else appends it.
    fn add(&mut self, tag: String, metadata: Option<Value>) {
        let entry = TagEntry { metadata, timestamp: self.clock.wall() };
        match self.tags.iter_mut().find(|(k, _)| *k == tag) {
            Some(slot) => slot.1 = entry,
            None => self.tags.push((tag, entry)),
        }
    }

    fn remove(&mut self, tag: String) {
        self.tags.retain(|(k, _)| *k != tag);
    }

    fn tags(&self) -> BTreeSet<String> {
        self.tags.iter().map(|(k, _)| k.clone()).collect()
    }

    /// The tags in insertion order.
    fn entries(&self) -> Vec<ReportTag> {
        self.tags
            .iter()
            .map(|(name, e)| ReportTag {
                name: name.clone(),
                metadata: e.metadata.clone(),
                timestamp: lk_agents_abi::unix_secs(e.timestamp),
            })
            .collect()
    }

    fn evaluations(&self) -> Vec<EvaluationRecord> {
        self.evaluations.clone()
    }

    /// From tag presence; `lk.success` wins if both are set.
    fn outcome(&self) -> Option<Outcome> {
        if self.has(TAG_SUCCESS) {
            Some(Outcome::Success)
        } else if self.has(TAG_FAIL) {
            Some(Outcome::Fail)
        } else {
            None
        }
    }

    /// Only another `success`/`fail` overwrites it.
    fn outcome_reason(&self) -> Option<String> {
        self.outcome_reason.clone()
    }

    fn _core_record_judgment(
        &mut self,
        name: String,
        verdict: JudgeVerdict,
        reasoning: String,
        instructions: String,
    ) {
        let tag = format!("lk.judge.{name}:{verdict}");
        self.add(tag.clone(), None);
        self.evaluations.push(EvaluationRecord { name, tag, verdict, reasoning, instructions });
    }

    fn _core_stamp(&self, report: SessionReport) -> SessionReport {
        let reason = self.outcome_reason();
        SessionReport {
            tags: self.entries(),
            evaluations: self.evaluations(),
            outcome: self.outcome().map(|outcome| ReportOutcome { outcome, reason }),
            ..report
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lk_agents_abi::VirtualClock;
    use lk_agents_abi::telemetry::report::Tagger as _;
    use serde_json::json;

    use super::*;

    // Catches: tag timestamps from the wall clock rather than the session clock.
    #[test]
    fn tag_timestamps_come_from_the_session_clock() {
        let clock = Arc::new(VirtualClock::new());
        let mut t = Tagger::with_clock(clock.clone());
        t.add("a:1".into(), None);
        clock.advance(Duration::from_secs(10));
        t.add("b:1".into(), None);
        let ts: Vec<SystemTime> = t
            .entries()
            .into_iter()
            .map(|t| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(t.timestamp))
            .collect();
        // Not exactly 10s: `Clock::wall` nudges by 1 µs per read.
        let delta = ts[1].duration_since(ts[0]).unwrap();
        assert!(delta > Duration::from_millis(9_990) && delta < Duration::from_secs(11), "{ts:?}");
    }

    // Catches: tags losing insertion order, or an outcome surviving its tag's removal.
    #[test]
    fn the_tagger_reports_its_state() {
        let mut tagger = Tagger::with_clock(Arc::new(VirtualClock::new()));
        tagger.add("b:1".into(), Some(json!({"k": 1})));
        tagger.success(Some("done".into()));
        tagger.add("b:1".into(), None);
        let names: Vec<String> = tagger.entries().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["b:1", "lk.success"]);
        assert_eq!(tagger.outcome(), Some(Outcome::Success));
        assert_eq!(tagger.outcome_reason().as_deref(), Some("done"));
        tagger.fail(None);
        assert_eq!(tagger.outcome(), Some(Outcome::Fail));
        tagger.remove("lk.fail".into());
        assert_eq!(tagger.outcome(), None);
    }

    // Catches: a stamped report keeping stale tags, or an outcome without its reason.
    #[test]
    fn a_stamped_report_carries_the_tags_evaluations_and_outcome_now() {
        let mut tagger = Tagger::with_clock(Arc::new(VirtualClock::new()));
        tagger.add("k:v".into(), Some(json!({"a": 1})));
        tagger._core_record_judgment(
            "tone".into(),
            JudgeVerdict::Pass,
            "ok".into(),
            "be nice".into(),
        );
        let report = SessionReport { job_id: "AJ_1".into(), ..SessionReport::default() };
        let stamped = tagger._core_stamp(report.clone());
        assert_eq!(stamped.job_id, "AJ_1");
        let names: Vec<String> = stamped.tags.into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["k:v", "lk.judge.tone:pass"]);
        assert_eq!(stamped.evaluations.len(), 1);
        assert!(stamped.outcome.is_none());
        tagger.fail(Some("hung up".into()));
        let outcome = tagger._core_stamp(report).outcome.unwrap();
        assert_eq!((outcome.outcome, outcome.reason.as_deref()), (Outcome::Fail, Some("hung up")));
    }
}
