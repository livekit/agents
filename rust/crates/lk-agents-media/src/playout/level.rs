// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// Buffer-level filter adapted under the upstream notices in the playout README.md.

//! Smooth timing decisions on a fixed 10 ms source-independent rendering cadence. Account
//! immediately for a successful time-scale operation so it cannot feed back into the filter.

use super::Rate;

#[derive(Default)]
pub(super) struct Level {
    rate: Rate,
    value: Option<i64>,
    tick: u64,
}

impl Level {
    pub(super) fn new(rate: Rate) -> Self {
        Self { rate, value: None, tick: 0 }
    }
    pub(super) fn update(&mut self, samples: usize, target_ms: usize, rendered: u64) -> usize {
        // A minute is already far beyond the useful decision range. Bounding the filter also
        // makes arbitrarily large supplied-clock gaps safe for its fixed-point arithmetic.
        let input = samples.min(self.rate.hz() as usize * 60) as i64 * 256;
        let tick = rendered / (u64::from(self.rate.hz()) / 100);
        let Some(mut value) = self.value else {
            self.value = Some(input);
            self.tick = tick;
            return samples.min(self.rate.hz() as usize * 60);
        };
        let factor = match target_ms {
            0..=20 => 251,
            21..=60 => 252,
            61..=140 => 253,
            _ => 254,
        };
        for _ in self.tick..tick {
            value = (factor * value + (256 - factor) * input) / 256;
        }
        self.tick = tick;
        self.value = Some(value);
        (value / 256) as usize
    }

    /// Positive means source samples removed; negative means output samples inserted.
    pub(super) fn correct(&mut self, change: isize) {
        if let Some(value) = &mut self.value {
            *value = (*value - change as i64 * 256).max(0);
        }
    }
}

#[cfg(test)]
#[path = "level_tests.rs"]
mod tests;
