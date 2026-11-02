//! Load sampling I/O and freshness. Capacity decisions stay in `load`.

use super::{Completion, Outbox, Worker};
use crate::CpuMonitor;
use crate::load::{STALE_AFTER, STARTUP_WINDOW};
use futures_util::FutureExt;
use std::time::Duration;

pub(super) type CpuSampleResult =
    Result<(Box<CpuMonitor>, Option<(f32, Duration)>), lk_agents_net::blocking_io::JobPanicked>;

impl Worker {
    /// Samples the user's function, else CPU. A hung or invalid custom function cannot report idle.
    pub(super) fn sample_load(&mut self) {
        if self.load_pending {
            return;
        }
        self.load_pending = true;
        self.load_requested = self.clock.now();
        let host = self.host.clone();
        self.operations.push(
            async move {
                let load = tokio::time::timeout(STALE_AFTER, host.user_load())
                    .await
                    .unwrap_or(Some(f32::NAN));
                Completion::Load(load)
            }
            .boxed(),
        );
    }

    pub(super) fn complete_user_load(&mut self, load: Option<f32>, outbox: &mut Outbox) {
        if let Some(load) = load {
            self.load_pending = false;
            if load.is_finite() && (0.0..=1.0).contains(&load) {
                self.worker_load = load;
                self.load_health.last_valid = Some(self.load_requested);
            } else {
                self.host
                    .logger()
                    .emit(log::Level::Warn, "invalid or timed out worker load sample".into());
            }
            self.finish_load(outbox);
            return;
        }
        let blocking_io = self.blocking_io.clone();
        let permit = match blocking_io.try_admit(()) {
            Ok(permit) => permit,
            Err(error) => {
                self.load_pending = false;
                self.host
                    .logger()
                    .emit(log::Level::Debug, format!("worker CPU sample deferred error={error}"));
                self.finish_load(outbox);
                return;
            }
        };
        let cpu = self.cpu.take();
        let sample = blocking_io.spawn_admitted(permit, move || {
            let mut cpu = cpu.unwrap_or_default();
            let sample = cpu.sample();
            (cpu, sample)
        });
        self.operations.push(async move { Completion::CpuSample(sample.await) }.boxed());
    }

    pub(super) fn complete_cpu_sample(&mut self, result: CpuSampleResult, outbox: &mut Outbox) {
        self.load_pending = false;
        match result {
            Ok((cpu, sample)) => {
                self.cpu = Some(cpu);
                if let Some((value, interval)) = sample {
                    self.default_load.add_sample(value, interval);
                    self.worker_load = self.default_load.get();
                    self.load_health.last_valid = Some(self.load_requested);
                }
            }
            Err(error) => self
                .host
                .logger()
                .emit(log::Level::Error, format!("worker CPU sampler failed error={error}")),
        }
        self.finish_load(outbox);
    }

    pub(super) fn refresh_capacity(&mut self) {
        self.starting_jobs
            .retain(|_, at| self.clock.now().saturating_duration_since(*at) < STARTUP_WINDOW);
        let inputs = self.load_inputs();
        self.load_available = self.load_health.update(
            self.clock.now(),
            inputs.effective_load(),
            inputs.load_threshold,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultLoadCalc;

    #[test]
    fn short_samples_have_proportional_weight_and_old_samples_expire() {
        let mut calc = DefaultLoadCalc::new();
        calc.add_sample(1.0, Duration::from_secs(2));
        for _ in 0..50 {
            calc.add_sample(0.0, Duration::from_millis(10));
        }
        assert!((calc.get() - 0.8).abs() < 1e-6);
        calc.add_sample(0.0, Duration::from_secs(2));
        assert_eq!(calc.get(), 0.0);
    }
}
