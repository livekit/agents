//! Recording upload attempts use supplied time, response policy and jitter.

pub const MAX_ATTEMPTS: u8 = 4;
const MAX_RETRY_DELAY_MS: u64 = 8_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadOutcome {
    Success,
    Disabled,
    Retryable(Option<u64>),
    Permanent,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Ready(u64),
    InFlight,
    Finished,
}

pub struct UploadPolicy {
    phase: Phase,
    attempts: u8,
}

impl UploadPolicy {
    pub fn new(disabled: bool) -> Self {
        Self { phase: if disabled { Phase::Finished } else { Phase::Ready(0) }, attempts: 0 }
    }

    pub fn next_attempt(&mut self, now_ms: u64) -> Option<u8> {
        if matches!(self.phase, Phase::Ready(at) if now_ms >= at) {
            self.attempts += 1;
            self.phase = Phase::InFlight;
            Some(self.attempts)
        } else {
            None
        }
    }

    /// `jitter` is supplied entropy in [0, 1]. Server delays are bounded by the upload budget.
    pub fn observed(&mut self, now_ms: u64, outcome: UploadOutcome, jitter: f64) {
        if !matches!(self.phase, Phase::InFlight) {
            return;
        }
        self.phase = match outcome {
            UploadOutcome::Retryable(delay) if self.attempts < MAX_ATTEMPTS => {
                let backoff = (1_000u64 << (self.attempts - 1)).min(MAX_RETRY_DELAY_MS);
                let jitter = if jitter.is_finite() { jitter.clamp(0.0, 1.0) } else { 0.0 };
                let delay = delay.unwrap_or((backoff as f64 * jitter) as u64);
                Phase::Ready(now_ms.saturating_add(delay.min(900_000)))
            }
            _ => Phase::Finished,
        };
    }

    pub fn retry_at(&self) -> Option<u64> {
        match self.phase {
            Phase::Ready(at) => Some(at),
            _ => None,
        }
    }

    pub fn finished(&self) -> bool {
        matches!(self.phase, Phase::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_policy_supplied_time() {
        let mut disabled = UploadPolicy::new(true);
        assert_eq!(disabled.next_attempt(0), None);
        let mut upload = UploadPolicy::new(false);
        assert_eq!(upload.next_attempt(10_000), Some(1));
        upload.observed(10_100, UploadOutcome::Retryable(None), 0.5);
        assert_eq!(upload.next_attempt(10_599), None);
        assert_eq!(upload.next_attempt(10_600), Some(2));
        upload.observed(11_000, UploadOutcome::Retryable(Some(2_000)), 0.0);
        assert_eq!(upload.next_attempt(12_999), None);
        assert_eq!(upload.next_attempt(13_000), Some(3));
        upload.observed(13_001, UploadOutcome::Retryable(None), 1.0);
        assert_eq!(upload.next_attempt(17_000), None);
        assert_eq!(upload.next_attempt(17_001), Some(4));
        upload.observed(17_002, UploadOutcome::Retryable(Some(1)), 0.0);
        assert!(upload.finished());
        assert_eq!(upload.next_attempt(100_000), None);
        for outcome in [UploadOutcome::Success, UploadOutcome::Disabled, UploadOutcome::Permanent] {
            let mut upload = UploadPolicy::new(false);
            assert_eq!(upload.next_attempt(0), Some(1));
            upload.observed(1, outcome, 0.5);
            assert!(upload.finished());
            assert_eq!(upload.next_attempt(100_000), None);
        }
    }
}
