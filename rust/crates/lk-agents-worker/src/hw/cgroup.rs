//! Linux cgroup detection, CPU quotas and usage counters; parsing is split from I/O so it tests
//! on any platform.

use std::time::Duration;

/// The v1 files under both mount spellings: `cpu/` + `cpuacct/`, and the combined `cpu,cpuacct/`.
pub const V1_QUOTA: &[&str] =
    &["/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us"];
pub const V1_PERIOD: &[&str] =
    &["/sys/fs/cgroup/cpu/cpu.cfs_period_us", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us"];
pub const V1_USAGE: &[&str] =
    &["/sys/fs/cgroup/cpuacct/cpuacct.usage", "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"];

/// Source of CPU capacity and usage measurements: `"cgroup_v2"`, `"cgroup_v1"` or `"default"`.
#[lk_agents_abi::host(wire_enum)]
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

/// The env override wins. Unset quotas are 2.0 CPUs; explicit ones floor at 1.0.
pub fn cpu_count_v1(env_cpus: Option<f32>, quota: Option<i64>, period: Option<i64>) -> f32 {
    if let Some(n) = env_cpus {
        return n;
    }
    match (quota, period) {
        (Some(q), Some(p)) if q >= 0 && p > 0 => (q as f32 / p as f32).max(1.0),
        _ => 2.0,
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


pub fn first_existing(paths: &[&'static str]) -> Option<&'static str> {
    paths.iter().copied().find(|p| std::path::Path::new(p).exists())
}

/// cgroup v2, then v1, then the system monitor.
pub fn detect() -> CpuMonitorKind {
    if std::path::Path::new("/sys/fs/cgroup/cpu.stat").exists() {
        return CpuMonitorKind::CgroupV2;
    }
    if [V1_QUOTA, V1_PERIOD, V1_USAGE].iter().any(|ps| first_existing(ps).is_some()) {
        return CpuMonitorKind::CgroupV1;
    }
    CpuMonitorKind::Default
}

pub(crate) fn read_first_int(path: &str) -> Option<i64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
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

    // Catches: one shared rule for v1 and v2; v1 answers 2.0 for an unset quota and floors at 1.0.
    #[test]
    fn the_two_cgroup_versions_disagree_on_purpose() {
        // v2: fractional limits survive, an unset quota is the node count
        assert_eq!(cpu_count_v2(None, "50000 100000", 64.0), 0.5);
        assert_eq!(cpu_count_v2(None, "max 100000", 64.0), 64.0);
        // v1: unset -> 2.0 (NOT 64), and a fractional quota is floored at 1.0
        assert_eq!(cpu_count_v1(None, None, None), 2.0);
        assert_eq!(cpu_count_v1(None, Some(-1), Some(100_000)), 2.0);
        assert_eq!(cpu_count_v1(None, Some(50_000), Some(100_000)), 1.0);
        assert_eq!(cpu_count_v1(None, Some(400_000), Some(100_000)), 4.0);
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
