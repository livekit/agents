//! CPU capacity and usage: the default load calculator's input and the basis of
//! `num_idle_processes`' prod default. cgroup monitors live in [`crate::hw::cgroup`]; the bare-host
//! one uses `sysinfo`, so it works on macOS and Windows.

pub use crate::hw::cgroup::{
    CpuMonitorKind, busy_fraction, cpu_count_v1, cpu_count_v2, detect, parse_cpu_max,
    parse_usage_usec,
};
use crate::hw::paths::CpuCgroup;
use std::time::{Duration, Instant};

/// Overrides every other answer in all three monitors.
pub const NUM_CPUS_ENV: &str = "NUM_CPUS";

/// Invalid values return `None`.
pub fn cpu_count_from_env(raw: Option<&str>) -> Option<f32> {
    raw.and_then(|v| v.trim().parse::<f32>().ok()).filter(|n| n.is_finite() && *n > 0.0)
}

/// CPUs allowed by process affinity, before cgroup quotas are applied.
pub fn system_cpu_count() -> f32 {
    #[cfg(target_os = "linux")]
    {
        // available_parallelism also rounds cgroup quotas to integers. Using it as the
        // ceiling would incorrectly turn a 1.5-CPU quota into 1 CPU. Read affinity instead.
        // This mask covers 65,536 CPUs on 64-bit Linux, including large kernel CPU sets.
        let mut mask = [0_usize; 1024];
        // SAFETY: the writable mask is aligned and its byte size is passed to the kernel.
        let result = unsafe {
            libc::sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr().cast())
        };
        if result == 0 {
            let count: u32 = mask.iter().map(|word| word.count_ones()).sum();
            if count > 0 {
                return count as f32;
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get() as f32).unwrap_or(1.0)
}

/// `utils/hw/cpu.py`'s three monitors as one. Starts no threads, so it is safe before core init.
///
/// Samples CPU busy-ness across calls; one per worker. Uses `Instant`, not `Ts`, because the
/// sampling interval is the caller's, not the machine's.
pub struct CpuMonitor {
    kind: CpuMonitorKind,
    cpus: f32,
    source: Source,
}

/// A cgroup usage counter differenced against the previous call, or `sysinfo`'s per-CPU counters.
enum Source {
    Cgroup { path: std::path::PathBuf, unit_per_sec: f64, last: Option<(i128, Instant)> },
    Host { system: sysinfo::System, last: Option<Instant> },
}

/// `utils/hw/cpu.py`'s three monitors as one. Starts no threads, so it is safe before core init.
impl CpuMonitor {
    pub fn new() -> Self {
        let group = CpuCgroup::discover();
        let kind = group.as_ref().map_or(CpuMonitorKind::Default, |g| g.kind);
        let env = cpu_count_from_env(std::env::var(NUM_CPUS_ENV).ok().as_deref());
        let cpus = env.unwrap_or_else(|| {
            group.as_ref().map_or_else(system_cpu_count, |g| g.cpu_count(system_cpu_count()))
        });
        let source = match group {
            Some(group) => Source::Cgroup {
                path: group.usage,
                unit_per_sec: if kind == CpuMonitorKind::CgroupV2 { 1e6 } else { 1e9 },
                last: None,
            },
            None => Source::Host { system: sysinfo::System::new(), last: None },
        };
        CpuMonitor { kind, cpus, source }
    }

    /// Which monitor was picked.
    pub fn kind(&self) -> CpuMonitorKind {
        self.kind
    }

    /// `cpu_count()`: logical CPUs, fractional under a cgroup quota.
    pub fn cpu_count(&self) -> f32 {
        self.cpus
    }

    /// Busy fraction since the previous reading. Requests closer than 500 ms do not
    /// refresh counters, so bursts cannot distort the utilization window.
    pub fn cpu_percent(&mut self) -> Option<f32> {
        self.sample().map(|(value, _)| value)
    }

    /// The utilization and the actual interval it describes. Missing or reset counters
    /// do not manufacture idle samples; freshness is enforced by the worker.
    pub(crate) fn sample(&mut self) -> Option<(f32, Duration)> {
        let now = Instant::now();
        match &mut self.source {
            Source::Cgroup { path, unit_per_sec, last } => {
                if last.is_some_and(|(_, at)| now.duration_since(at) < crate::UPDATE_LOAD_INTERVAL)
                {
                    return None;
                }
                let raw = std::fs::read_to_string(path).ok()?;
                let usage = if *unit_per_sec == 1e6 {
                    parse_usage_usec(&raw)?
                } else {
                    raw.trim().parse::<i128>().ok()?
                };
                let (previous, at) = last.replace((usage, now))?;
                if usage < previous {
                    return None;
                }
                let interval = now.duration_since(at);
                Some((
                    busy_fraction(usage - previous, *unit_per_sec, interval, self.cpus),
                    interval,
                ))
            }
            Source::Host { system, last } => {
                if last.is_some_and(|at| now.duration_since(at) < crate::UPDATE_LOAD_INTERVAL) {
                    return None;
                }
                system.refresh_cpu_usage();
                let at = last.replace(now)?;
                Some(((system.global_cpu_usage() / 100.0).clamp(0.0, 1.0), now.duration_since(at)))
            }
        }
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
        assert_eq!(cpu_count_v1(Some(3.0), Some(400_000), Some(100_000), 64.0), 3.0);
        assert_eq!(cpu_count_from_env(Some("2.5")), Some(2.5));
        // junk, zero and negatives fall back rather than propagating
        assert_eq!(cpu_count_from_env(Some("many")), None);
        assert_eq!(cpu_count_from_env(Some("0")), None);
        assert_eq!(cpu_count_from_env(Some("-4")), None);
    }

    #[test]
    fn rapid_requests_do_not_refresh_cpu_counters() {
        let mut monitor = CpuMonitor::new();
        assert_eq!(monitor.sample(), None);
        assert_eq!(monitor.sample(), None);
        assert!(monitor.cpu_count() > 0.0);
    }

    #[test]
    fn host_samples_report_the_interval_they_measure() {
        let mut monitor = CpuMonitor {
            kind: CpuMonitorKind::Default,
            cpus: 1.0,
            source: Source::Host {
                system: sysinfo::System::new(),
                last: Some(Instant::now() - Duration::from_secs(1)),
            },
        };
        let (value, interval) = monitor.sample().unwrap();
        assert!((0.0..=1.0).contains(&value));
        assert!(interval >= Duration::from_secs(1));
        assert_eq!(monitor.sample(), None);
    }
}
