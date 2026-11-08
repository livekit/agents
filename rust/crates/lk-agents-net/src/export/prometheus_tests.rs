use super::*;
use lk_agents_abi::telemetry::instruments::{
    WorkerIdleProcesses, WorkerIdleProcessesTarget, WorkerJobs, WorkerLoad,
    WorkerProcessReadyDuration,
};

/// The sample lines of `exposition`, without comments: `(series, value)`.
fn samples(exposition: &str) -> Vec<(&str, &str)> {
    exposition
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.rsplit_once(' ').expect("a sample is `series value`"))
        .collect()
}

/// A metric name the format accepts: `[a-zA-Z_:][a-zA-Z0-9_:]*`.
fn is_metric_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

// Catches: a documented metric renamed (a scraper's dashboards and alerts read these names), a
// name the format refuses, or a series a worker serves without its `# TYPE`.
#[test]
fn the_worker_instruments_serve_under_their_documented_names() {
    let instruments = Instruments::new();
    instruments.set::<WorkerLoad>(0.5);
    instruments.set::<WorkerJobs>(3.0);
    instruments.set::<WorkerIdleProcesses>(2.0);
    instruments.set::<WorkerIdleProcessesTarget>(4.0);
    instruments.observe::<WorkerProcessReadyDuration>(0.3);
    let out = exposition(&instruments);
    let types: Vec<_> = out.lines().filter_map(|line| line.strip_prefix("# TYPE ")).collect();
    assert_eq!(
        types,
        [
            "lk_agents_worker_load_ratio gauge",
            "lk_agents_worker_jobs gauge",
            "lk_agents_worker_idle_processes gauge",
            "lk_agents_worker_idle_processes_target gauge",
            "lk_agents_worker_process_ready_duration_seconds histogram",
        ]
    );
    for (series, _) in samples(&out) {
        let name = series.split('{').next().unwrap();
        assert!(is_metric_name(name), "{name}");
        let family = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .unwrap_or(name);
        assert!(types.iter().any(|t| t.starts_with(&format!("{family} "))), "{name} untyped");
    }
}

// Catches: a gauge that does not serve the value set, or that a later set does not replace.
#[test]
fn a_gauge_serves_its_last_value() {
    let instruments = Instruments::new();
    instruments.set::<WorkerJobs>(3.0);
    instruments.set::<WorkerJobs>(1.0);
    instruments.set::<WorkerLoad>(0.25);
    assert_eq!(
        samples(&exposition(&instruments)),
        [("lk_agents_worker_jobs", "1"), ("lk_agents_worker_load_ratio", "0.25")]
    );
}

// Catches: buckets that are not cumulative, an observation on a bound in the next bucket, a
// missing `+Inf` bucket, or a count or sum that disagrees with the observations.
#[test]
fn a_histogram_serves_cumulative_buckets_its_count_and_sum() {
    let instruments = Instruments::new();
    for seconds in [0.0625, 0.5, 0.75, 2.0, 2.0, 12.0] {
        instruments.observe::<WorkerProcessReadyDuration>(seconds);
    }
    let name = "lk_agents_worker_process_ready_duration_seconds";
    let bucket = |le: &str| format!("{name}_bucket{{le=\"{le}\"}}");
    let expected = [
        (bucket("0.1"), "1"),
        (bucket("0.5"), "2"),
        (bucket("1"), "3"),
        (bucket("2"), "5"),
        (bucket("5"), "5"),
        (bucket("10"), "5"),
        (bucket("+Inf"), "6"),
        (format!("{name}_sum"), "17.3125"),
        (format!("{name}_count"), "6"),
    ];
    let out = exposition(&instruments);
    let actual: Vec<_> = samples(&out).into_iter().map(|(s, v)| (s.to_owned(), v)).collect();
    assert_eq!(actual, expected);
}

// Catches: Rust's `inf` reaching a scraper, which refuses the whole scrape for one sample.
#[test]
fn special_values_use_the_format_spelling() {
    let instruments = Instruments::new();
    instruments.set::<WorkerLoad>(f64::NAN);
    instruments.set::<WorkerJobs>(f64::INFINITY);
    instruments.set::<WorkerIdleProcesses>(f64::NEG_INFINITY);
    assert_eq!(
        samples(&exposition(&instruments)),
        [
            ("lk_agents_worker_load_ratio", "NaN"),
            ("lk_agents_worker_jobs", "+Inf"),
            ("lk_agents_worker_idle_processes", "-Inf"),
        ]
    );
}

// Catches: a series served before the worker recorded it (a scraper would read a false zero).
#[test]
fn nothing_recorded_serves_an_empty_exposition() {
    assert_eq!(exposition(&Instruments::new()), "");
}
