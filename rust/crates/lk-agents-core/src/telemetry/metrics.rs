//! Admission for framework metric series. Subject-owned folds still decide measurements;
//! this destination boundary refuses unbounded labels before an SDK creates a new series.

use std::collections::{BTreeMap, BTreeSet};

const MAX_SERIES: usize = 1024;
const MAX_LABELS: usize = 8;
const MAX_LABEL_BYTES: usize = 256;

type Series = (String, BTreeMap<String, String>);

pub struct MetricSeries {
    series: BTreeSet<Series>,
}

impl lk_agents_abi::telemetry::policy::MetricSeries for MetricSeries {
    fn new() -> Self {
        Self { series: BTreeSet::new() }
    }

    fn admit(&mut self, name: String, attributes: Vec<(String, String)>) -> bool {
        let attributes: BTreeMap<_, _> = attributes.into_iter().collect();
        let instrument = lk_agents_abi::metrics::otel::INSTRUMENTS.iter().any(|(n, _)| *n == name)
            || matches!(
                name.as_str(),
                "gen_ai.invoke_agent.duration"
                    | "gen_ai.execute_tool.duration"
                    | "lk.agents.event_loop.blocked_duration"
            );
        if !instrument
            || attributes.len() > MAX_LABELS
            || attributes.iter().any(|(key, value)| {
                !matches!(
                    key.as_str(),
                    "model_name"
                        | "model_provider"
                        | "connection_reused"
                        | "gen_ai.operation.name"
                        | "gen_ai.agent.name"
                        | "gen_ai.tool.name"
                        | "severity"
                ) || value.len() > MAX_LABEL_BYTES
            })
        {
            return false;
        }
        let key = (name, attributes);
        if self.series.contains(&key) {
            return true;
        }
        if self.series.len() == MAX_SERIES {
            return false;
        }
        self.series.insert(key);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::telemetry::policy::MetricSeries as _;

    #[test]
    fn configured_names_survive_but_identity_and_unbounded_series_are_refused() {
        let mut metrics = MetricSeries::new();
        let tool = || "gen_ai.execute_tool.duration".to_owned();
        let attributes = vec![
            ("gen_ai.operation.name".into(), "execute_tool".into()),
            ("gen_ai.tool.name".into(), "lookup".into()),
            ("gen_ai.agent.name".into(), "frontdesk".into()),
        ];
        assert!(metrics.admit(tool(), attributes.clone()));
        assert!(metrics.admit(tool(), attributes));
        for invalid in
            ["job_id", "room_id", "thread_id", "customer", "exception.message", "lk.pii.input"]
        {
            assert!(!metrics.admit(tool(), vec![(invalid.into(), "private".into())]));
        }
        for index in 1..MAX_SERIES {
            assert!(
                metrics.admit(
                    tool(),
                    vec![("gen_ai.tool.name".into(), format!("configured-{index}"))]
                )
            );
        }
        assert!(!metrics.admit(tool(), vec![("gen_ai.tool.name".into(), "one more".into())]));
        assert!(!metrics.admit("arbitrary metric".into(), Vec::new()));
    }
}
