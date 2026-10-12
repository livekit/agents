//! Worker capacity, status, and idle-process calculations.

use lk_agents_abi::MovingAverage;

/// Sampled independently by each worker.
#[derive(Debug)]
pub struct DefaultLoadCalc {
    avg: MovingAverage,
}

impl Default for DefaultLoadCalc {
    fn default() -> Self {
        Self::new()
    }
}

#[lk_agents_abi::host(export)]
impl DefaultLoadCalc {
    /// Five samples at the load-update interval cover 2.5 seconds.
    #[host(export(name = "default_load_window", pid = false))]
    pub const WINDOW: usize = 5;

    pub fn new() -> Self {
        DefaultLoadCalc { avg: MovingAverage::new(Self::WINDOW) }
    }

    /// Normalised to `0.0..=1.0`.
    pub fn add_sample(&mut self, cpu_fraction: f32) {
        self.avg.add_sample(cpu_fraction as f64);
    }

    /// Zero before the first sample, so a fresh worker is available on cold start.
    pub fn get(&self) -> f32 {
        self.avg.get_avg() as f32
    }
}

/// Gathered so the load rules can be tested without a worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadInputs {
    pub worker_load: f32,
    pub active_jobs: usize,
    pub reserved_slots: usize,
    /// Resolved for the current mode.
    pub load_threshold: f32,
    /// Resolved for the current mode.
    pub num_idle_processes: u32,
    pub draining: bool,
    pub simulation: bool,
}

impl LoadInputs {
    /// Current load plus an estimated cost per reserved slot: measured per-job cost when jobs run,
    /// zero under an infinite threshold, else estimated (works with zero idle processes).
    pub fn effective_load(&self) -> f32 {
        let per_job = if self.active_jobs > 0 {
            self.worker_load / self.active_jobs as f32
        } else if self.load_threshold.is_infinite() {
            0.0
        } else {
            self.load_threshold / self.num_idle_processes.max(1) as f32
        };
        self.worker_load + self.reserved_slots as f32 * per_job
    }

    /// Draining takes precedence over simulation.
    pub fn is_available(&self) -> bool {
        if self.draining {
            return false;
        }
        if self.simulation {
            return true;
        }
        if self.load_threshold.is_infinite() {
            return true;
        }
        self.effective_load() < self.load_threshold
    }

    /// Draining reports full; otherwise the threshold decides unless simulating.
    pub fn worker_status(&self) -> WorkerStatus {
        if self.draining {
            return WorkerStatus::Full;
        }
        let is_full = !self.simulation && self.effective_load() >= self.load_threshold;
        if is_full { WorkerStatus::Full } else { WorkerStatus::Available }
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

