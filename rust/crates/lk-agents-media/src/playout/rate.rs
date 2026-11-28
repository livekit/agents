use std::time::Duration;

/// Validated, immutable source/output rate. Time windows retain their physical duration.
#[derive(Debug, Clone, Copy)]
pub(super) struct Rate(u32);

impl Default for Rate {
    fn default() -> Self {
        Self(super::DEFAULT_SAMPLE_RATE)
    }
}

impl Rate {
    pub(super) fn new(hz: u32) -> Option<Self> {
        matches!(hz, 16_000 | 24_000 | 48_000).then_some(Self(hz))
    }

    pub(super) fn hz(self) -> u32 {
        self.0
    }

    pub(super) fn frame(self) -> usize {
        self.0 as usize / 50
    }

    pub(super) fn overlap(self) -> usize {
        self.0 as usize / 200
    }

    pub(super) fn period_min(self) -> usize {
        self.0 as usize / 400
    }

    pub(super) fn period_max(self) -> usize {
        self.0 as usize * 15 / 1000
    }

    pub(super) fn history(self) -> usize {
        self.0 as usize / 25
    }

    // Reference tuning uses 24 kHz samples; round fractional windows outward.

    pub(super) fn scale(self, reference: usize) -> usize {
        (reference * self.0 as usize).div_ceil(24_000)
    }

    pub(super) fn samples(self, duration: Duration) -> usize {
        (duration.as_nanos().saturating_mul(u128::from(self.0)) / 1_000_000_000)
            .min(usize::MAX as u128) as usize
    }

    pub(super) fn clock_samples(self, duration: Duration) -> i64 {
        (duration.as_nanos().saturating_mul(u128::from(self.0)) / 1_000_000_000)
            .min((i64::MAX / 4) as u128) as i64
    }
}
