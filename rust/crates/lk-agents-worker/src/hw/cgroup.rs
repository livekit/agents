//! Linux cgroup detection, CPU quotas and usage counters; parsing is split from I/O so it tests
//! on any platform.

use std::time::Duration;

/// Source of CPU capacity and usage measurements: `"cgroup_v2"`, `"cgroup_v1"` or `"default"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuMonitorKind {
    CgroupV2,
    CgroupV1,
    Default,
}

/// `"max 100000"` or `"50000 100000"`. Malformed input means unrestricted; default period 100000.
pub fn parse_cpu_max(contents: &str) -> (Option<u64>, u64) {
    let mut it = contents.split_whitespace();
    let quota = it.next();
    let period = it.next().and_then(|p| p.parse::<u64>().ok()).unwrap_or(100_000);
    match quota {
        None | Some("max") => (None, period),
        Some(q) => match q.parse::<u64>() {
            Ok(v) => (Some(v), period),
            Err(_) => (None, 100_000),
        },
    }
}

/// Keeps fractional quotas. The env override wins; unrestricted uses the system CPU count.
pub fn cpu_count_v2(env_cpus: Option<f32>, cpu_max: &str, system: f32) -> f32 {
    if let Some(n) = env_cpus {
        return n;
    }
    match parse_cpu_max(cpu_max) {
        (None, _) => system,
        (Some(quota), period) => quota as f32 / period.max(1) as f32,
    }
}

/// Preserve fractional quotas; unrestricted groups use the effective system CPU count.
pub fn cpu_count_v1(
    env_cpus: Option<f32>,
    quota: Option<i64>,
    period: Option<i64>,
    system: f32,
) -> f32 {
    if let Some(n) = env_cpus {
        return n;
    }
    match (quota, period) {
        (Some(q), Some(p)) if q > 0 && p > 0 => (q as f32 / p as f32).min(system),
        _ => system,
    }
}

/// `usage_delta` is in the file's unit (µs v2, ns v1). Clamped to `0.0..=1.0`: a negative delta
/// is a counter reset, not spare capacity.
pub fn busy_fraction(usage_delta: i128, unit_per_sec: f64, interval: Duration, cpus: f32) -> f32 {
    let seconds = usage_delta as f64 / unit_per_sec;
    let denom = interval.as_secs_f64() * cpus.max(f32::MIN_POSITIVE) as f64;
    if denom <= 0.0 {
        return 0.0;
    }
    ((seconds / denom) as f32).clamp(0.0, 1.0)
}

/// cgroup v2, then v1, then the system monitor.
pub fn detect() -> CpuMonitorKind {
    super::paths::CpuCgroup::discover().map_or(CpuMonitorKind::Default, |g| g.kind)
}

/// `usage_usec` out of `/sys/fs/cgroup/cpu.stat`.
pub fn parse_usage_usec(contents: &str) -> Option<i128> {
    contents
        .lines()
        .find_map(|l| l.strip_prefix("usage_usec"))
        .and_then(|rest| rest.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both controller versions must preserve sub-CPU limits.
    #[test]
    fn both_cgroup_versions_preserve_fractional_capacity() {
        // v2: fractional limits survive, an unset quota is the node count
        assert_eq!(cpu_count_v2(None, "50000 100000", 64.0), 0.5);
        assert_eq!(cpu_count_v2(None, "max 100000", 64.0), 64.0);
        // v1: unlimited uses effective system capacity; fractional limits survive.
        assert_eq!(cpu_count_v1(None, None, None, 64.0), 64.0);
        assert_eq!(cpu_count_v1(None, Some(-1), Some(100_000), 64.0), 64.0);
        assert_eq!(cpu_count_v1(None, Some(50_000), Some(100_000), 64.0), 0.5);
        assert_eq!(cpu_count_v1(None, Some(400_000), Some(100_000), 64.0), 4.0);
    }

    // Catches: a corrupt `cpu.max` reading as zero CPUs (`unwrap_or(0)`) instead of unrestricted.
    #[test]
    fn a_malformed_cpu_max_means_unrestricted_not_zero() {
        assert_eq!(parse_cpu_max("max"), (None, 100_000));
        assert_eq!(parse_cpu_max(""), (None, 100_000));
        assert_eq!(parse_cpu_max("garbage 100000"), (None, 100_000));
        assert_eq!(parse_cpu_max("50000"), (Some(50_000), 100_000));
        assert_eq!(cpu_count_v2(None, "", 8.0), 8.0);
    }

    // Catches: an unclamped fraction; a counter reset or a throttled burst would skew the average.
    #[test]
    fn the_busy_fraction_is_clamped_at_both_ends() {
        let half = Duration::from_millis(500);
        // 0.25 CPU-seconds over 0.5 s on 1 CPU = 50 %
        assert!((busy_fraction(250_000, 1e6, half, 1.0) - 0.5).abs() < 1e-6);
        // the same work spread over 4 CPUs
        assert!((busy_fraction(250_000, 1e6, half, 4.0) - 0.125).abs() < 1e-6);
        assert_eq!(busy_fraction(-5_000, 1e6, half, 1.0), 0.0);
        assert_eq!(busy_fraction(10_000_000, 1e6, half, 1.0), 1.0);
        assert_eq!(busy_fraction(1_000, 1e6, Duration::ZERO, 1.0), 0.0);
    }

    // Catches: a loose match on `usage_usec` picking up `user_usec` or `system_usec` instead.
    #[test]
    fn usage_usec_is_read_off_cpu_stat_and_not_its_neighbours() {
        let s = "usage_usec 123456\nuser_usec 100000\nsystem_usec 23456\nnr_periods 0\n";
        assert_eq!(parse_usage_usec(s), Some(123_456));
        assert_eq!(parse_usage_usec("user_usec 1\nsystem_usec 2\n"), None);
    }
}
