use std::collections::VecDeque;

use super::conceal::Concealer;
use super::{FRAME, OVERLAP, PERIOD_MAX, correlation, rms, sample};

const MATCH: usize = 120;
const MAX_MERGE: usize = 240;

/// Aligns the synthetic continuation to real audio, never the other way around: all source
/// samples retain their order. The generated phase is eased over 2 ms, followed by an energy-
/// normalized crossfade and, after a faded outage, a bounded gain ramp back to full level.
pub(in crate::playout) struct Recovery {
    continuation: [i16; MAX_MERGE],
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
        self.position < self.length || self.ramp_position < self.ramp
    }

    pub(in crate::playout) fn start(&mut self, conceal: &Concealer, queue: &VecDeque<i16>) {
        self.position = 0;
        self.ramp_position = 0;
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
        let mut similarity = correlation(|i| prediction[i], |i| queue[i], available);
        if conceal.voiced() > 0.4 && available >= 48 {
            for lag in 1..conceal.period() {
                let score = correlation(|i| prediction[lag + i], |i| queue[i], available);
                if score > similarity {
                    similarity = score;
                    offset = lag;
                }
            }
        }
        // A new unvoiced onset must emerge promptly; a periodic, well-matched return permits
        // a longer blend. Length is in samples, independent of render-call/chunk boundaries.
        self.length = if similarity > 0.7 && conceal.voiced() > 0.4 {
            conceal.period().clamp(OVERLAP, MAX_MERGE)
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
        for i in 0..self.length {
            let phase_weight = smooth((i as f64 / 48.0).min(1.0));
            self.continuation[i] = sample(
                f64::from(prediction[i]) * (1.0 - phase_weight)
                    + f64::from(prediction[i + offset]) * phase_weight,
            );
        }
        // Maintain the last already-rendered sample exactly at the boundary, including cases
        // where the model's amplitude envelope reached zero between calls.
        self.continuation[0] = conceal.last_generated();
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
