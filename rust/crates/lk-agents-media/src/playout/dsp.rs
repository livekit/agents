use std::collections::VecDeque;

pub(super) const FRAME: usize = 480;
pub(super) const OVERLAP: usize = 120;
pub(super) const PERIOD_MIN: usize = 60;
pub(super) const PERIOD_MAX: usize = 360;
pub(super) const HISTORY: usize = 960;
const FADE: usize = 120;

pub(super) fn mix(a: i16, b: i16, position: usize, length: usize) -> i16 {
    let position = position.min(length) as i64;
    ((i64::from(a) * (length as i64 - position) + i64::from(b) * position) / length as i64) as i16
}

pub(super) fn correlation(
    a: impl Fn(usize) -> i16,
    b: impl Fn(usize) -> i16,
    length: usize,
) -> f64 {
    let mut cross = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..length {
        let a = f64::from(a(i));
        let b = f64::from(b(i));
        sum_a += a;
        sum_b += b;
        cross += a * b;
        aa += a * a;
        bb += b * b;
    }
    let length = length as f64;
    aa = (aa - sum_a * sum_a / length).max(0.0);
    bb = (bb - sum_b * sum_b / length).max(0.0);
    cross -= sum_a * sum_b / length;
    if aa < 1.0 || bb < 1.0 { 0.0 } else { cross / (aa * bb).sqrt() }
}

/// Conservative periodic continuation, with a bounded fade. Unvoiced or insufficient history
/// falls back to a short ramp from the last sample; it never loops arbitrary speech indefinitely.
pub(super) struct Concealer {
    cycle: [i16; PERIOD_MAX],
    period: usize,
    position: usize,
    limit: usize,
    last: i16,
}

impl Default for Concealer {
    fn default() -> Self {
        Self { cycle: [0; PERIOD_MAX], period: 0, position: 0, limit: 0, last: 0 }
    }
}

impl Concealer {
    pub(super) fn start(&mut self, history: &VecDeque<i16>, maximum: usize, last: i16) {
        self.position = 0;
        self.last = last;
        self.period = 0;
        self.limit = FADE;
        if maximum == 0 || history.len() < PERIOD_MAX + FRAME {
            return;
        }
        let end = history.len();
        let mut best = 0.65;
        for lag in (PERIOD_MIN..=PERIOD_MAX).step_by(2) {
            let similarity = correlation(
                |i| history[end - FRAME + i],
                |i| history[end - FRAME - lag + i],
                FRAME,
            );
            if similarity > best {
                best = similarity;
                self.period = lag;
            }
        }
        if self.period > 0 {
            for i in 0..self.period {
                self.cycle[i] = history[end - self.period + i];
            }
            self.limit = maximum;
        }
    }

    pub(super) fn active(&self) -> bool {
        self.position <= self.limit
    }

    pub(super) fn next(&mut self) -> i16 {
        let at = self.position;
        self.position = self.position.saturating_add(1);
        if at >= self.limit {
            return 0;
        }
        let sample = if self.period == 0 {
            self.last
        } else {
            let predicted = self.cycle[at % self.period];
            if at < FADE { mix(self.last, predicted, at, FADE) } else { predicted }
        };
        // Fade over the whole bounded continuation. This is deliberately conservative about
        // sustained vowels, and guarantees silence by the configured limit.
        mix(sample, 0, at, self.limit)
    }
}
