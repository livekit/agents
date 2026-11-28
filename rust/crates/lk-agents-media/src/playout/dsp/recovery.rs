// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// NetEq processing adapted under LICENSE.webrtc and PATENTS.webrtc.

use std::collections::VecDeque;

use super::conceal::Concealer;
use super::{FRAME, OVERLAP, PERIOD_MAX, correlation, rms, sample};

const MATCH: usize = 240;
const MAX_MERGE: usize = 480;

/// A bounded continuation prefix aligns returning speech without skipping source samples.
/// Two-window phase validation precedes an energy-normalized merge and, after a faded
/// outage, a bounded gain ramp back to full level.
pub(in crate::playout) struct Recovery {
    continuation: [i16; MAX_MERGE],
    prefix: [i16; PERIOD_MAX],
    prefix_length: usize,
    prefix_position: usize,
    length: usize,
    position: usize,
    gain: f64,
    ramp: usize,
    ramp_position: usize,
    similarity: f64,
}

impl Default for Recovery {
    fn default() -> Self {
        Self {
            continuation: [0; MAX_MERGE],
            prefix: [0; PERIOD_MAX],
            prefix_length: 0,
            prefix_position: 0,
            length: 0,
            position: 0,
            gain: 1.0,
            ramp: 0,
            ramp_position: 0,
            similarity: 0.0,
        }
    }
}

impl Recovery {
    pub(in crate::playout) fn active(&self) -> bool {
        self.prefix_position < self.prefix_length
            || self.position < self.length
            || self.ramp_position < self.ramp
    }

    pub(in crate::playout) fn start(
        &mut self,
        conceal: &Concealer,
        queue: &VecDeque<i16>,
        allow_prefix: bool,
    ) {
        self.position = 0;
        self.ramp_position = 0;
        self.prefix_length = 0;
        self.prefix_position = 0;
        let available = queue.len().min(MATCH);
        if available == 0 {
            self.length = 0;
            self.ramp = 0;
            return;
        }
        let mut model = conceal.clone();
        let mut prediction = [0; PERIOD_MAX + MAX_MERGE];
        for out in &mut prediction {
            *out = model.next();
        }
        let new_rms = rms(|i| queue[i], available);
        let old_rms = rms(|i| prediction[i], available);
        let mut offset = 0;
        // Both halves must match: an isolated consonant edge must not establish a false
        // pitch alignment. Short arrivals merge promptly without speculative phase search.
        let score = |lag| {
            let half = available / 2;
            if half == 0 {
                return 0.0;
            }
            correlation(|i| prediction[lag + i], |i| queue[i], half).min(correlation(
                |i| prediction[lag + half + i],
                |i| queue[half + i],
                available - half,
            ))
        };
        let original = score(0);
        let mut similarity = original;
        if conceal.voiced() > 0.4 && conceal.gain() > 0.25 && available == MATCH {
            for lag in 1..conceal.period() {
                let candidate = score(lag);
                if candidate > similarity {
                    similarity = candidate;
                    offset = lag;
                }
            }
        }
        // A new unvoiced onset must emerge promptly; a periodic, well-matched return permits
        // a longer blend. Length is in samples, independent of render-call/chunk boundaries.
        // Align by continuing the predicted waveform before the returning source, as NetEq
        // Merge does. Never rotate the source or abruptly warp the generated waveform phase.
        // A weak/ambiguous match uses the short merge instead of adding speculative delay.
        if allow_prefix && similarity > 0.7 && similarity > original + 0.1 {
            self.prefix_length = offset;
            self.prefix[..offset].copy_from_slice(&prediction[..offset]);
            if offset > 0 {
                self.prefix[0] = conceal.last_generated();
            }
        } else {
            offset = 0;
            similarity = original;
        }
        self.length = if similarity > 0.7 && conceal.voiced() > 0.4 {
            (conceal.period() * 2).clamp(OVERLAP, MAX_MERGE).min(queue.len())
        } else {
            48
        };
        self.similarity = similarity.max(0.0);
        // Keep a short-gap return at original gain. Only a substantially faded interruption
        // warrants gradual unmuting; never boost an incoming quiet consonant to match a vowel.
        self.gain = if conceal.duration() > 480 && old_rms < new_rms * 0.7 {
            (old_rms / new_rms.max(1.0)).clamp(0.0, 1.0).max(conceal.gain())
        } else {
            1.0
        };
        self.ramp = if self.gain < 1.0 { FRAME } else { 0 };
        self.continuation[..self.length].copy_from_slice(&prediction[offset..offset + self.length]);
        if self.prefix_length == 0 {
            self.continuation[0] = conceal.last_generated();
        }
    }

    pub(in crate::playout) fn next_prefix(&mut self) -> Option<i16> {
        if self.prefix_position == self.prefix_length {
            return None;
        }
        let sample = self.prefix[self.prefix_position];
        self.prefix_position += 1;
        Some(sample)
    }

    pub(in crate::playout) fn apply(&mut self, input: i16) -> i16 {
        let gain = if self.ramp_position < self.ramp {
            let t = self.ramp_position as f64 / self.ramp as f64;
            self.ramp_position += 1;
            self.gain + (1.0 - self.gain) * smooth(t)
        } else {
            1.0
        };
        let input = f64::from(input) * gain;
        if self.position >= self.length {
            return sample(input);
        }
        let at = self.position;
        self.position += 1;
        let b = smooth(at as f64 / self.length as f64);
        let a = 1.0 - b;
        // Normalize decorrelated signals without the correlated +3 dB bump of an equal-power
        // crossfade. At a strong phase match, weights reduce to an ordinary complementary fade.
        let power = (a * a + b * b + 2.0 * self.similarity * a * b).sqrt().max(0.707);
        sample((f64::from(self.continuation[at]) * a + input * b) / power)
    }
}

pub(in crate::playout) fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}
