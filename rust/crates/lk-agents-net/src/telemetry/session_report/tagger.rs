//! The `Tagger` the host constructs: the core's tag state, stamped by the process clock.

use std::collections::BTreeSet;
use std::sync::Arc;

use lk_agents_abi::Clock;
use lk_agents_abi::telemetry::report::{EvaluationRecord, Outcome, ReportTag};
use lk_agents_core::telemetry::report as records;
use serde_json::Value;

use crate::env::ProcessEnv;

pub struct Tagger {
    clock: Arc<dyn Clock>,
    state: records::Tagger,
}

impl lk_agents_abi::telemetry::report::Tagger for Tagger {
    type Env = ProcessEnv;

    /// Timestamps are strictly increasing, so same-instant tags keep insertion order.
    fn new(env: &ProcessEnv) -> Self {
        Tagger { clock: env.clock.clone(), state: records::Tagger::default() }
    }

    fn success(&mut self, reason: Option<String>) {
        self.state.success(reason, self.clock.wall());
    }

    fn fail(&mut self, reason: Option<String>) {
        self.state.fail(reason, self.clock.wall());
    }

    fn add(&mut self, tag: String, metadata: Option<Value>) {
        self.state.add(tag, metadata, self.clock.wall());
    }

    fn remove(&mut self, tag: String) {
        self.state.remove(&tag);
    }

    fn tags(&self) -> BTreeSet<String> {
        self.state.tags()
    }

    fn sorted_tags(&self) -> Vec<String> {
        self.state.tags().into_iter().collect()
    }

    fn entries(&self) -> Vec<ReportTag> {
        self.state.entries()
    }

    fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    fn evaluations(&self) -> Vec<EvaluationRecord> {
        self.state.evaluations().to_vec()
    }

    fn outcome(&self) -> Option<Outcome> {
        self.state.outcome()
    }

    fn outcome_reason(&self) -> Option<String> {
        self.state.outcome_reason().map(str::to_owned)
    }

    fn _core_record_judgment(
        &mut self,
        name: String,
        verdict: String,
        reasoning: String,
        instructions: String,
    ) {
        self.state.record_judgment(name, verdict, reasoning, instructions, self.clock.wall());
    }

    fn has_evals(&self) -> bool {
        self.state.has_evals()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use lk_agents_abi::VirtualClock;
    use lk_agents_abi::telemetry::report::Tagger as _;

    use super::*;

    // Catches: tag timestamps from the wall clock rather than the session clock.
    #[test]
    fn tag_timestamps_come_from_the_session_clock() {
        let clock = Arc::new(VirtualClock::new());
        let mut t = Tagger { clock: clock.clone(), state: records::Tagger::default() };
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
}
