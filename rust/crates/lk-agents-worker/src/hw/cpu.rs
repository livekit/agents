//! CPU capacity and usage: the default load calculator's input and the basis of
//! `num_idle_processes`' prod default. cgroup monitors live in [`crate::hw::cgroup`]; the bare-host
//! one uses `sysinfo`, so it works on macOS and Windows.

use crate::hw::cgroup::read_first_int;
pub use crate::hw::cgroup::{
    CpuMonitorKind, busy_fraction, cpu_count_v1, cpu_count_v2, detect, parse_cpu_max,
    parse_usage_usec,
};
use crate::hw::cgroup::{V1_PERIOD, V1_QUOTA, V1_USAGE, first_existing};

/// Overrides every other answer in all three monitors.
pub const NUM_CPUS_ENV: &str = "NUM_CPUS";

/// `min(ceil(cpu_count), 4)`; a positive fractional count still allows one idle process.
pub(crate) fn idle_process_target(cpu_count: f32) -> u32 {
    (cpu_count.max(0.0).ceil() as u32).min(4)
}

/// Invalid values return `None`.
pub fn cpu_count_from_env(raw: Option<&str>) -> Option<f32> {
    raw.and_then(|v| v.trim().parse::<f32>().ok()).filter(|n| n.is_finite() && *n > 0.0)
}

/// Falls back to one.
pub fn system_cpu_count() -> f32 {
    std::thread::available_parallelism().map(|n| n.get() as f32).unwrap_or(1.0)
}

/// `utils/hw/cpu.py`'s three monitors as one. Starts no threads, so it is safe before core init.
///
/// Samples CPU busy-ness across calls; one per worker. Uses `Instant`, not `Ts`, because the
/// sampling interval is the caller's, not the machine's.
#[lk_agents_abi::class(name = "CpuMonitor", mutable)]
pub struct CpuMonitor {
    kind: CpuMonitorKind,
    cpus: f32,
    source: Source,
    primed: bool,
}

/// A cgroup usage counter differenced against the previous call, or `sysinfo`'s per-CPU counters.
enum Source {
    Cgroup { path: &'static str, unit_per_sec: f64, last: Option<(i128, std::time::Instant)> },
    Host { system: sysinfo::System, primed: bool },
}

/// `utils/hw/cpu.py`'s three monitors as one. Starts no threads, so it is safe before core init.
#[lk_agents_abi::host(object(pid = false))]
impl CpuMonitor {
    #[host(new)]
    pub fn new() -> Self {
        let kind = detect();
        let env = cpu_count_from_env(std::env::var(NUM_CPUS_ENV).ok().as_deref());
        let (cpus, source) = match kind {
            CpuMonitorKind::CgroupV2 => (
                cpu_count_v2(
                    env,
                    &std::fs::read_to_string("/sys/fs/cgroup/cpu.max").unwrap_or_default(),
                    system_cpu_count(),
                ),
                Source::Cgroup { path: "/sys/fs/cgroup/cpu.stat", unit_per_sec: 1e6, last: None },
            ),
            CpuMonitorKind::CgroupV1 => (
                cpu_count_v1(
                    env,
                    first_existing(V1_QUOTA).and_then(read_first_int),
                    first_existing(V1_PERIOD).and_then(read_first_int),
                ),
                Source::Cgroup {
                    path: first_existing(V1_USAGE).unwrap_or(V1_USAGE[0]),
                    unit_per_sec: 1e9,
                    last: None,
                },
            ),
            // `System::new()` reads nothing and starts no threads: safe at import and across `fork()`.
            CpuMonitorKind::Default => (
                env.unwrap_or_else(system_cpu_count),
                Source::Host { system: sysinfo::System::new(), primed: false },
            ),
        };
        CpuMonitor { kind, cpus, source, primed: false }
    }

    /// Which monitor was picked.
    #[host(getter)]
    pub fn kind(&self) -> CpuMonitorKind {
        self.kind
    }

    /// `cpu_count()`: logical CPUs, fractional under a cgroup quota.
    #[host(method)]
    pub fn cpu_count(&self) -> f32 {
        self.cpus
    }

    /// Busy fraction since the previous call, `0.0..=1.0`, non-blocking. `None` on the first
    /// sample or an unreadable cgroup counter: keep the previous estimate.
    pub fn cpu_percent(&mut self) -> Option<f32> {
        let cpus = self.cpus;
        match &mut self.source {
            Source::Cgroup { path, unit_per_sec, last } => {
                let usage = match *unit_per_sec {
                    // `cpu.stat` is a key/value file; `cpuacct.usage` is one integer.
                    u if u == 1e6 => parse_usage_usec(&std::fs::read_to_string(*path).ok()?)?,
                    _ => read_first_int(path)? as i128,
                };
                let now = std::time::Instant::now();
                let prev = last.replace((usage, now));
                prev.map(|(p, at)| {
                    busy_fraction(usage - p, *unit_per_sec, now.duration_since(at), cpus)
                })
            }
            Source::Host { system, primed } => {
                // `global_cpu_usage` is the delta between the last two refreshes, in percent.
                // sysinfo wants at least 200 ms between them; the worker's 500 ms tick clears it.
                system.refresh_cpu_usage();
                if !*primed {
                    *primed = true;
                    return None;
                }
                Some((system.global_cpu_usage() / 100.0).clamp(0.0, 1.0))
            }
        }
    }

    /// `cpu_percent(interval)`: the busy fraction over the next `interval` seconds, sleeping the
    /// calling thread (the binding releases the GIL); the first call primes the counter. `None`
    /// means a cgroup file became unreadable: keep the last value rather than reading it as `0.0`.
    #[host(method(name = "cpu_percent", detach, defaults(interval = 0.5)))]
    pub fn cpu_percent_over(&mut self, interval: f64) -> Option<f32> {
        if !self.primed {
            self.cpu_percent();
            self.primed = true;
        }
        std::thread::sleep(std::time::Duration::from_secs_f64(interval.max(0.0)));
        self.cpu_percent()
    }
}

impl Default for CpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: NUM_CPUS being consulted in only one of the three monitors.
    #[test]
    fn num_cpus_overrides_every_monitor() {
        assert_eq!(cpu_count_v2(Some(3.0), "50000 100000", 64.0), 3.0);
        assert_eq!(cpu_count_v1(Some(3.0), Some(400_000), Some(100_000)), 3.0);
        assert_eq!(cpu_count_from_env(Some("2.5")), Some(2.5));
        // junk, zero and negatives fall back rather than propagating
        assert_eq!(cpu_count_from_env(Some("many")), None);
        assert_eq!(cpu_count_from_env(Some("0")), None);
        assert_eq!(cpu_count_from_env(Some("-4")), None);
    }

    // Catches: dropping the `min(.., 4)` (right on every box up to four cores), or a floor at 1 in
    // `cpu_count_v2` hiding a fractional limit. The prod default is this rule at an unbounded count.
    #[test]
    fn the_idle_process_target_is_capped_at_four_and_floored_at_one() {
        for (cpus, want) in [(32.0, 4), (f32::INFINITY, 4), (4.0, 4), (3.0, 3), (0.1, 1), (0.0, 0)] {
            assert_eq!(idle_process_target(cpus), want, "{cpus}");
        }
        assert_eq!(idle_process_target(cpu_count_v2(None, "50000 100000", 8.0)), 1);
    }

    // Catches: a monitor that never answers, leaving the load at 0.0 forever on macOS/Windows.
    #[test]
    fn the_second_sample_answers_on_every_platform() {
        let mut m = CpuMonitor::new();
        assert_eq!(m.cpu_percent(), None, "the first sample has nothing to difference against");
        let second = m.cpu_percent();
        match m.kind() {
            CpuMonitorKind::Default => {
                let v = second.expect("the host monitor answers from its second refresh on");
                assert!((0.0..=1.0).contains(&v), "{v}");
            }
            // A cgroup read can still fail between two calls (a container being torn down).
            _ => {
                if let Some(v) = second {
                    assert!((0.0..=1.0).contains(&v), "{v}");
                }
            }
        }
        assert!(m.cpu_count() > 0.0 || matches!(m.kind(), CpuMonitorKind::CgroupV2));
    }
}
