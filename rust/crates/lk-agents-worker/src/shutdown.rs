//! Drain options shared by launchers and worker lifecycle configuration.

use std::time::Duration;

/// Time allowed for draining active jobs and completing process shutdown.
#[derive(Debug, Clone, Copy)]
pub struct DrainOptions {
    total: Duration,
    drain: Duration,
}

impl DrainOptions {
    pub fn new(total: f64, drain: f64) -> Result<Self, String> {
        let duration = |name: &str, seconds: f64| {
            Duration::try_from_secs_f64(seconds)
                .ok()
                .filter(|value| value.as_secs_f64() <= 3_155_760_000.0)
                .ok_or_else(|| format!("{name} must be finite and nonnegative"))
        };
        let total = duration("shutdown_timeout", total)?;
        let drain = duration("drain_timeout", drain)?;
        if total.is_zero() || drain >= total {
            return Err("shutdown_timeout must be positive and greater than drain_timeout".into());
        }
        Ok(Self { total, drain })
    }

    pub fn total(self) -> Duration {
        self.total
    }

    pub fn drain_remaining(self, remaining: Duration) -> Duration {
        let reserve = self.total - self.drain;
        self.drain.min(remaining.saturating_sub(reserve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_keeps_the_cleanup_reserve_even_after_delays() {
        let options = DrainOptions::new(55.0, 45.0).unwrap();
        assert_eq!(options.drain_remaining(Duration::from_secs(55)), Duration::from_secs(45));
        assert_eq!(options.drain_remaining(Duration::from_secs(40)), Duration::from_secs(30));
        assert_eq!(options.drain_remaining(Duration::from_secs(5)), Duration::ZERO);
        assert!(DrainOptions::new(45.0, 45.0).is_err());
        assert!(DrainOptions::new(f64::INFINITY, 0.0).is_err());
    }
}
