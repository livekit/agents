// Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
// Adapted under the BSD license and patent grant in fixtures/playout/LICENSE.webrtc
// and fixtures/playout/PATENTS.webrtc.

//! Arrival-envelope tracking and an underrun-delay histogram following NetEq's delay manager.
//! Ported histogram arithmetic is covered by the WebRTC license in `fixtures/playout/`.
//! Ordered PCM needs no RTP reordering model. Source time, not chunk count, bounds the window;
//! transport batching is represented as simultaneous <=20 ms source units.

use std::time::Duration;

use super::SAMPLE_RATE;

const UNIT: u64 = SAMPLE_RATE as u64 / 50;
const WINDOW: u64 = 100;
const BINS: usize = 128;
const PROBABILITY: i64 = 1 << 30;
const QUANTILE: i64 = (PROBABILITY * 95) / 100;
const FORGET: i64 = 32_210; // 0.983 in Q15.

#[derive(Clone, Copy)]
struct Arrival {
    bucket: u64,
    minimum: i64,
    maximum: i64,
}

const EMPTY: Arrival = Arrival { bucket: u64::MAX, minimum: 0, maximum: 0 };

pub(super) struct Delay {
    arrivals: [Arrival; BINS],
    newest: u64,
    minimum: i64,
    maximum: i64,
    seen: bool,
    histogram: [i64; 100],
    forget: i64,
    updates: u64,
    interval_start: Option<Duration>,
    interval_peak_ms: usize,
    target_ms: usize,
}

impl Default for Delay {
    fn default() -> Self {
        let mut histogram = [0; 100];
        let mut probability = 0x4002;
        for bucket in &mut histogram {
            probability >>= 1;
            *bucket = probability << 16;
        }
        Self {
            arrivals: [EMPTY; BINS],
            newest: 0,
            minimum: 0,
            maximum: 0,
            seen: false,
            histogram,
            forget: 0,
            updates: 0,
            interval_start: None,
            interval_peak_ms: 0,
            target_ms: 80,
        }
    }
}

impl Delay {
    pub(super) fn update(&mut self, now: Duration, source_start: u64, length: usize) {
        let arrival = clock_samples(now);
        let end = source_start.saturating_add(length as u64);
        let mut position = source_start;
        let mut last_offset = self.minimum;
        while position < end {
            let bucket = position / UNIT;
            let slot = &mut self.arrivals[bucket as usize % BINS];
            let source = position.min((i64::MAX / 4) as u64) as i64;
            let offset = arrival - source;
            if slot.bucket != bucket {
                *slot = Arrival { bucket, minimum: offset, maximum: offset };
            } else {
                slot.minimum = slot.minimum.min(offset);
                slot.maximum = slot.maximum.max(offset);
            }
            self.newest = bucket;
            self.seen = true;
            last_offset = offset;
            position = end.min((bucket + 1) * UNIT);
        }
        self.minimum = i64::MAX;
        self.maximum = i64::MIN;
        for entry in &self.arrivals {
            if entry.bucket != u64::MAX && entry.bucket + WINDOW >= self.newest {
                self.minimum = self.minimum.min(entry.minimum);
                self.maximum = self.maximum.max(entry.maximum);
            }
        }
        // Sample the maximum observed delay per 500ms interval, so a large transport chunk
        // cannot overweight the distribution merely because it was split into DSP blocks.
        let relative =
            ((last_offset - self.minimum).max(0) as usize).div_ceil(SAMPLE_RATE as usize / 1000);
        if let Some(start) = self.interval_start {
            if now.saturating_sub(start) > Duration::from_millis(500) {
                self.add(self.interval_peak_ms / 20);
                self.interval_start = Some(now);
                self.interval_peak_ms = 0;
            }
        } else {
            self.interval_start = Some(now);
        }
        // Include the full arrival envelope, not only the last simultaneous unit.
        self.interval_peak_ms = self.interval_peak_ms.max(relative).max(self.max_delay_ms());
    }

    fn add(&mut self, value: usize) {
        if value >= self.histogram.len() {
            return;
        }
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

    pub(super) fn target(&self, minimum: Duration, maximum: Duration) -> usize {
        let minimum = minimum.as_millis() as usize;
        let maximum = maximum.as_millis() as usize;
        self.target_ms.clamp(minimum, maximum) * SAMPLE_RATE as usize / 1000
    }

    pub(super) fn max_delay_ms(&self) -> usize {
        if self.seen {
            ((self.maximum - self.minimum).max(0) as usize).div_ceil(SAMPLE_RATE as usize / 1000)
        } else {
            0
        }
    }

    /// Source-to-render delay relative to the fastest recent delivery path. This includes
    /// packetization, unlike queue occupancy; draining a normal large chunk is not jitter.
    pub(super) fn playout_delay(&self, now: Duration, source_position: u64) -> usize {
        if !self.seen {
            return 0;
        }
        let source = source_position.min((i64::MAX / 4) as u64) as i64;
        (clock_samples(now) - source - self.minimum).max(0) as usize
    }
}

fn clock_samples(now: Duration) -> i64 {
    (now.as_nanos().saturating_mul(u128::from(SAMPLE_RATE)) / 1_000_000_000)
        .min((i64::MAX / 4) as u128) as i64
}

#[cfg(test)]
#[path = "delay_tests.rs"]
mod tests;
