use crate::load::*;
use crate::config::{DEFAULT_LOAD_THRESHOLD_DEV, DEFAULT_LOAD_THRESHOLD_PROD, EnvOption};

fn inputs() -> LoadInputs {
    LoadInputs {
        worker_load: 0.0,
        active_jobs: 0,
        reserved_slots: 0,
        load_threshold: DEFAULT_LOAD_THRESHOLD_PROD,
        num_idle_processes: 3,
        draining: false,
        simulation: false,
    }
}

// Catches: skipping the 5-sample average, so one CPU spike marks a healthy worker as full.
#[test]
fn default_load_calc_averages_five_samples() {
    let mut c = DefaultLoadCalc::new();
    assert_eq!(c.get(), 0.0, "empty window must read 0, or a cold worker is never available");
    for s in [1.0, 0.0, 0.0, 0.0, 0.0] {
        c.add_sample(s);
    }
    assert!((c.get() - 0.2).abs() < 1e-6, "{}", c.get());
    // Window slides: the spike falls out after five more samples.
    for _ in 0..5 {
        c.add_sample(0.0);
    }
    assert_eq!(c.get(), 0.0);
}

// Catches, per row: dropping dev mode's infinite-threshold branch (inf/NaN); charging reserved
// slots nothing while idle (two early requests both accepted); dividing by `num_idle_processes = 0`;
// still guessing per-job cost once jobs run instead of measuring it; WS_AVAILABLE while draining;
// a simulated worker reported full; `>` for `>=` at the threshold.
#[test]
fn effective_load_availability_and_status() {
    use WorkerStatus::{Available, Full};
    let prod = DEFAULT_LOAD_THRESHOLD_PROD;
    let rows = [
        (LoadInputs { load_threshold: DEFAULT_LOAD_THRESHOLD_DEV, reserved_slots: 2, ..inputs() }, Some(0.0), true, None),
        // 0.7 / max(3, 1) = 0.2333 per reserved slot.
        (LoadInputs { reserved_slots: 2, ..inputs() }, Some(0.466_666_7), true, None),
        (LoadInputs { reserved_slots: 3, ..inputs() }, None, false, None),
        (LoadInputs { reserved_slots: 1, num_idle_processes: 0, ..inputs() }, Some(prod), false, None),
        (LoadInputs { worker_load: 0.4, active_jobs: 2, reserved_slots: 1, ..inputs() }, Some(0.6), true, None),
        (LoadInputs { draining: true, ..inputs() }, None, false, Some(Full)),
        (LoadInputs { simulation: true, worker_load: 9.0, ..inputs() }, None, true, Some(Available)),
        (LoadInputs { worker_load: prod, ..inputs() }, None, false, Some(Full)),
    ];
    for (i, want_load, available, status) in rows {
        if let Some(want) = want_load {
            assert!((i.effective_load() - want).abs() < 1e-5, "{} vs {want}", i.effective_load());
        }
        assert_eq!(i.is_available(), available, "{i:?}");
        if let Some(status) = status {
            assert_eq!(i.worker_status(), status, "{i:?}");
        }
    }
}

// Catches: setting the idle target to 0 when job_load is 0, instead of leaving it alone.
#[test]
fn zero_job_load_leaves_the_idle_target_alone() {
    let i = LoadInputs { worker_load: 0.0, active_jobs: 2, ..inputs() };
    assert_eq!(i.target_idle_processes(), None);
    // ..whereas with no jobs at all it resets to the configured default.
    let i = LoadInputs { active_jobs: 0, ..i };
    assert_eq!(i.target_idle_processes(), Some(3));
    // ..and dev mode never touches the pool.
    let i = LoadInputs { load_threshold: f32::INFINITY, ..i };
    assert_eq!(i.target_idle_processes(), None);
}

// Catches: dropping the `num_idle_processes` cap on the idle target.
#[test]
fn idle_target_is_capped_at_the_configured_default() {
    let i = LoadInputs { worker_load: 0.01, active_jobs: 1, num_idle_processes: 3, ..inputs() };
    // available 0.69 / job_load 0.01 = 69 -> capped at 3.
    assert_eq!(i.target_idle_processes(), Some(3));
}

// Catches: resolving a ServerEnvOption against the wrong mode.
#[test]
fn env_option_resolves_per_mode() {
    let o = EnvOption::Split { dev: f32::INFINITY, prod: 0.7 };
    assert!(o.get(true).is_infinite());
    assert_eq!(o.get(false), 0.7);
    assert_eq!(EnvOption::Fixed(0.5).get(true), 0.5);
}
