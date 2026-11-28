// Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
// Adapted under the upstream notices in the playout README.md.

//! Bounded Q30 delay distribution. Values beyond the final bucket saturate rather than
//! disappearing from the estimate. Early observations adapt quickly; older data decays.

pub(super) const BIN_MS: usize = 20;
const PROBABILITY: i64 = 1 << 30;
const QUANTILE: i64 = PROBABILITY * 95 / 100;
const FORGET: i64 = 32_210;

pub(super) struct Histogram {
    histogram: [i64; 100],
    forget: i64,
    updates: u64,
    target_ms: usize,
}

impl Default for Histogram {
    fn default() -> Self {
        let mut histogram = [0; 100];
        let mut probability = 0x4002;
        for bucket in &mut histogram {
            probability >>= 1;
            *bucket = probability << 16;
        }
        Self { histogram, forget: 0, updates: 0, target_ms: 80 }
    }
}

impl Histogram {
    pub(super) fn observe(&mut self, delay_ms: usize) {
        let value = (delay_ms / BIN_MS).min(self.histogram.len() - 1);
        let mut sum = 0;
        for bucket in &mut self.histogram {
            *bucket = (*bucket * self.forget) >> 15;
            sum += *bucket;
        }
        let addition = (32768 - self.forget) << 15;
        self.histogram[value] += addition;
        sum += addition - PROBABILITY;
        if sum != 0 {
            let sign = if sum > 0 { -1 } else { 1 };
            for bucket in &mut self.histogram {
                let correction = sign * sum.abs().min(*bucket >> 4);
                *bucket += correction;
                sum += correction;
                if sum == 0 {
                    break;
                }
            }
        }
        self.updates += 1;
        self.forget = FORGET.min(32768 * (self.updates as i64 - 1) / (self.updates as i64 + 1));
        let mut sum = PROBABILITY;
        let mut quantile = self.histogram.len() - 1;
        for (index, probability) in self.histogram.iter().enumerate() {
            sum -= probability;
            if sum <= PROBABILITY - QUANTILE {
                quantile = index;
                break;
            }
        }
        self.target_ms = (quantile + 1) * 20;
    }

    pub(super) fn observations(&self) -> u64 {
        self.updates
    }

    pub(super) fn quantile_ms(&self) -> usize {
        self.target_ms
    }
}

#[cfg(test)]
#[path = "histogram_tests.rs"]
mod tests;
