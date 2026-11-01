//! Worker CPU smoothing, capacity, status, and idle-process calculations.

use lk_agents_abi::Ts;
use std::collections::VecDeque;
use std::time::Duration;

pub(crate) const STALE_AFTER: Duration = Duration::from_secs(5);
pub(crate) const STARTUP_WINDOW: Duration = Duration::from_secs(5);

#[derive(Default)]
pub(crate) struct LoadHealth {
    pub last_valid: Option<Ts>,
    full: bool,
}

impl LoadHealth {
    pub fn fresh(&self, now: Ts) -> bool {
        self.last_valid.is_some_and(|at| now.saturating_duration_since(at) < STALE_AFTER)
    }

    pub fn update(&mut self, now: Ts, load: f32, threshold: f32) -> bool {
        if threshold.is_infinite() {
            return true;
        }
        if !self.fresh(now) {
            return false;
        }
        // Reopen only after 10% headroom has returned; use the same latch for status and offers.
        if self.full {
            self.full = load >= threshold * 0.9;
        } else {
            self.full = load >= threshold;
        }
        !self.full
    }
}

/// Time-weighted CPU utilization over the most recent 2.5 seconds.
#[derive(Debug, Default)]
pub struct DefaultLoadCalc {
    samples: VecDeque<(f64, std::time::Duration)>,
    duration: std::time::Duration,
}

impl DefaultLoadCalc {
    pub const WINDOW: std::time::Duration = std::time::Duration::from_millis(2500);

    pub fn new() -> Self {
        Self::default()
    }

    /// A sample describes the entire interval since the preceding CPU reading.
    pub fn add_sample(&mut self, cpu_fraction: f32, interval: std::time::Duration) {
        if !cpu_fraction.is_finite() || interval.is_zero() {
            return;
        }
        let interval = interval.min(Self::WINDOW);
        self.samples.push_back((f64::from(cpu_fraction.clamp(0.0, 1.0)), interval));
        self.duration += interval;
        while self.duration > Self::WINDOW {
            let excess = self.duration - Self::WINDOW;
            let (_, oldest) = self.samples.front_mut().expect("nonempty window");
            let removed = excess.min(*oldest);
            *oldest -= removed;
            self.duration -= removed;
            if oldest.is_zero() {
                self.samples.pop_front();
            }
        }
    }

    /// The caller separately tracks whether a valid measurement exists and is fresh.
    pub fn get(&self) -> f32 {
        if self.duration.is_zero() {
            0.0
        } else {
            (self.samples.iter().map(|(v, d)| v * d.as_secs_f64()).sum::<f64>()
                / self.duration.as_secs_f64()) as f32
        }
    }
}

/// Gathered so the load rules can be tested without a worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadInputs {
    pub worker_load: f32,
    pub active_jobs: usize,
    pub starting_jobs: usize,
    /// Minimum estimated normalized CPU cost of an additional job.
    pub job_load: f32,
    /// Stale measurements or the hysteresis latch prohibit admission.
    pub unavailable: bool,
    pub reserved_slots: usize,
    /// Resolved for the current mode.
    pub load_threshold: f32,
    /// Resolved for the current mode.
    pub num_idle_processes: u32,
    pub draining: bool,
}

impl LoadInputs {
    /// Keep a minimum reservation while jobs start and while new jobs await assignment.
    /// Prewarming configuration does not determine admission capacity.
    pub fn effective_load(&self) -> f32 {
        if self.load_threshold.is_infinite() {
            return self.worker_load;
        }
        let per_job = if self.active_jobs > 0 {
            (self.worker_load / self.active_jobs as f32).max(self.job_load)
        } else {
            self.job_load
        };
        // CPU from existing jobs must not erase reservations for jobs that just launched.
        self.worker_load
            + self.starting_jobs as f32 * self.job_load
            + self.reserved_slots as f32 * per_job
    }

    /// Draining takes precedence over disabled capacity limits.
    pub fn is_available(&self) -> bool {
        if self.draining {
            return false;
        }
        if self.load_threshold.is_infinite() {
            return true;
        }
        !self.unavailable && self.effective_load() < self.load_threshold
    }

    /// Status and admission use the same rules.
    pub fn worker_status(&self) -> WorkerStatus {
        if self.is_available() { WorkerStatus::Available } else { WorkerStatus::Full }
    }

    /// `None` leaves the pool unchanged: infinite threshold, or no measurable per-job load.
    pub fn target_idle_processes(&self) -> Option<u32> {
        if self.load_threshold.is_infinite() {
            return None;
        }
        if self.active_jobs == 0 {
            return Some(self.num_idle_processes);
        }
        let job_load = self.worker_load / self.active_jobs as f32;
        if job_load <= 0.0 {
            return None;
        }
        let available_load = (self.load_threshold - self.worker_load).max(0.0);
        let ceil = (available_load / job_load).ceil() as i64;
        Some((ceil.max(0) as u32).min(self.num_idle_processes))
    }
}

/// `agent.WorkerStatus`, mirrored so the calculator does not depend on prost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    Available,
    Full,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_LOAD_THRESHOLD_DEV, DEFAULT_LOAD_THRESHOLD_PROD};

    #[test]
    fn hysteresis_and_freshness_share_one_admission_latch() {
        let mut health = LoadHealth::default();
        assert!(!health.update(Ts::ZERO, 0.0, 0.7));
        health.last_valid = Some(Ts::ZERO);
        assert!(health.update(Ts::ZERO, 0.69, 0.7));
        assert!(!health.update(Ts::ZERO, 0.7, 0.7));
        assert!(!health.update(Ts::ZERO, 0.68, 0.7));
        assert!(health.update(Ts::ZERO, 0.62, 0.7));
        assert!(!health.update(Ts::ZERO.saturating_add(STALE_AFTER), 0.0, 0.7));
        assert!(health.update(Ts::ZERO.saturating_add(STALE_AFTER), 0.0, f32::INFINITY));
    }

    fn inputs() -> LoadInputs {
        LoadInputs {
            worker_load: 0.0,
            active_jobs: 0,
            starting_jobs: 0,
            job_load: DEFAULT_LOAD_THRESHOLD_PROD / 3.0,
            unavailable: false,
            reserved_slots: 0,
            load_threshold: DEFAULT_LOAD_THRESHOLD_PROD,
            num_idle_processes: 3,
            draining: false,
        }
    }

    // Disabled limits, reservations with no prewarming, measured job costs and drain precedence.
    #[test]
    fn effective_load_availability_and_status() {
        use WorkerStatus::{Available, Full};
        let prod = DEFAULT_LOAD_THRESHOLD_PROD;
        let rows = [
            (
                LoadInputs {
                    load_threshold: DEFAULT_LOAD_THRESHOLD_DEV,
                    reserved_slots: 2,
                    ..inputs()
                },
                Some(0.0),
                true,
                None,
            ),
            // Explicit per-job cost, independent of the idle pool.
            (LoadInputs { reserved_slots: 2, ..inputs() }, Some(0.466_666_7), true, None),
            (LoadInputs { reserved_slots: 3, ..inputs() }, None, false, None),
            (
                LoadInputs { reserved_slots: 1, num_idle_processes: 0, ..inputs() },
                Some(prod / 3.0),
                true,
                None,
            ),
            (
                LoadInputs { worker_load: 0.4, active_jobs: 2, reserved_slots: 1, ..inputs() },
                Some(0.633_333_3),
                true,
                None,
            ),
            (LoadInputs { draining: true, ..inputs() }, None, false, Some(Full)),
            (
                LoadInputs { load_threshold: f32::INFINITY, worker_load: 9.0, ..inputs() },
                None,
                true,
                Some(Available),
            ),
            (LoadInputs { worker_load: prod, ..inputs() }, None, false, Some(Full)),
        ];
        for (i, want_load, available, status) in rows {
            if let Some(want) = want_load {
                assert!(
                    (i.effective_load() - want).abs() < 1e-5,
                    "{} vs {want}",
                    i.effective_load()
                );
            }
            assert_eq!(i.is_available(), available, "{i:?}");
            if let Some(status) = status {
                assert_eq!(i.worker_status(), status, "{i:?}");
            }
        }
    }

    #[test]
    fn existing_cpu_usage_does_not_absorb_startup_reservations() {
        let i = LoadInputs {
            worker_load: 0.4,
            active_jobs: 2,
            starting_jobs: 2,
            job_load: 0.175,
            ..inputs()
        };
        assert!((i.effective_load() - 0.75).abs() < 1e-6);
        assert!(!i.is_available());
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
}
