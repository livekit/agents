use std::collections::VecDeque;

use super::noise::{Background, Noise};
use super::{FRAME, HISTORY, OVERLAP, PERIOD_MAX, PERIOD_MIN, correlation, rms, sample};

const JOIN: usize = 48; // Two milliseconds, independent of render block size.
const HOLD: usize = 240; // Preserve the first 10 ms before fading a continued phoneme.
const BACKGROUND_FADE: usize = 48_000;

/// Mixed voiced/unvoiced continuation. Pitch uses two real periods; unvoiced energy is
/// synthesized through a stable LPC model. The estimate is frozen for an outage, never
/// retrained on generated sound. Speech fades into independently learned room tone.
#[derive(Clone)]
pub(in crate::playout) struct Concealer {
    cycle: [i16; PERIOD_MAX],
    previous_cycle: [i16; PERIOD_MAX],
    period: usize,
    phase: usize,
    position: usize,
    limit: usize,
    last: i16,
    slope: f64,
    transition_bound: f64,
    last_generated: i16,
    voice: f64,
    unvoiced: Noise,
    background: Noise,
    background_model: Background,
    analysis: [i16; FRAME],
    observed: usize,
}

impl Default for Concealer {
    fn default() -> Self {
        Self {
            cycle: [0; PERIOD_MAX],
            previous_cycle: [0; PERIOD_MAX],
            period: 0,
            phase: 0,
            position: 0,
            limit: 0,
            last: 0,
            slope: 0.0,
            transition_bound: 0.0,
            last_generated: 0,
            voice: 0.0,
            unvoiced: Noise::default(),
            background: Noise::default(),
            background_model: Background::default(),
            analysis: [0; FRAME],
            observed: 0,
        }
    }
}

impl Concealer {
    /// Train room tone from unmodified source samples, at a fixed cadence unrelated to callers.
    pub(in crate::playout) fn observe(&mut self, source: i16) {
        self.analysis[self.observed] = source;
        self.observed = (self.observed + 1) % FRAME;
        if self.observed == 0 {
            self.background_model.observe(&self.analysis);
        }
    }

    pub(in crate::playout) fn start(&mut self, history: &VecDeque<i16>, maximum: usize, last: i16) {
        self.position = 0;
        self.phase = 0;
        self.last = last;
        self.last_generated = last;
        self.transition_bound =
            history.iter().map(|&x| f64::from(x).abs()).fold(f64::from(last).abs(), f64::max);
        self.period = 0;
        self.voice = 0.0;
        self.limit = maximum;
        self.background = self.background_model.model();
        self.unvoiced = Noise::default();
        self.slope = if history.len() > 1 {
            f64::from(last) - f64::from(history[history.len() - 2])
        } else {
            0.0
        };
        let available = history.len().min(FRAME);
        let mut analysis = [0; FRAME];
        for i in 0..available {
            analysis[FRAME - available + i] = history[history.len() - available + i];
        }
        if maximum == 0 || history.len() < FRAME + PERIOD_MAX {
            return;
        }
        self.unvoiced.analyze(&analysis);
        let end = history.len();
        // Search a full analysis window rather than just one short waveform fragment.
        // Prefer the first strong peak; selecting a multiple causes an artificial low pitch.
        let mut best = 0.0;
        let mut scores = [0.0; PERIOD_MAX + 1];
        for (lag, score) in scores.iter_mut().enumerate().take(PERIOD_MAX + 1).skip(PERIOD_MIN) {
            *score = correlation(
                |i| history[end - FRAME + i],
                |i| history[end - FRAME - lag + i],
                FRAME,
            );
            best = f64::max(best, *score);
        }
        if best > 0.45 {
            let threshold = (best * 0.93).max(0.45);
            for lag in PERIOD_MIN + 1..PERIOD_MAX {
                if scores[lag] >= threshold
                    && scores[lag] >= scores[lag - 1]
                    && scores[lag] > scores[lag + 1]
                {
                    self.period = lag;
                    break;
                }
            }
            if self.period == 0 {
                self.period = (PERIOD_MIN..=PERIOD_MAX)
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                    .expect("pitch interval is nonempty");
            }
            self.voice = ((scores[self.period] - 0.45) / 0.5).clamp(0.0, 1.0);
            for i in 0..self.period {
                self.cycle[i] = history[end - self.period + i];
                self.previous_cycle[i] = history[end - 2 * self.period + i];
            }
            // Match the repeated period's RMS to recent input without increasing level.
            let cycle_rms = rms(|i| self.cycle[i], self.period).max(1.0);
            let gain = (rms(|i| analysis[i], FRAME) / cycle_rms).min(1.0);
            for i in 0..self.period {
                self.cycle[i] = sample(f64::from(self.cycle[i]) * gain);
                self.previous_cycle[i] = sample(f64::from(self.previous_cycle[i]) * gain);
            }
        }
    }

    pub(in crate::playout) fn period(&self) -> usize {
        self.period
    }
    pub(in crate::playout) fn voiced(&self) -> f64 {
        self.voice
    }
    pub(in crate::playout) fn duration(&self) -> usize {
        self.position
    }
    pub(in crate::playout) fn last_generated(&self) -> i16 {
        self.last_generated
    }

    pub(in crate::playout) fn active(&self) -> bool {
        self.position <= self.limit.max(JOIN)
            || !self.background.quiet() && self.position < self.limit + BACKGROUND_FADE
    }

    pub(in crate::playout) fn next(&mut self) -> i16 {
        let at = self.position;
        self.position = self.position.saturating_add(1);
        let background_fade =
            1.0 - (at.saturating_sub(self.limit) as f64 / BACKGROUND_FADE as f64).min(1.0);
        let background = f64::from(self.background.next()) * background_fade;
        let hold = HOLD.min(self.limit / 2);
        let gain = if at < hold {
            1.0
        } else {
            1.0 - (at - hold) as f64 / self.limit.saturating_sub(hold).max(1) as f64
        }
        .clamp(0.0, 1.0);
        let mut generated = background;
        if self.limit > 0 && at < self.limit {
            let periodic = if self.period > 0 {
                let index = (at + self.phase) % self.period;
                // Slightly mix the preceding period without jittering pitch or replaying words.
                0.85 * f64::from(self.cycle[index]) + 0.15 * f64::from(self.previous_cycle[index])
            } else {
                0.0
            };
            let voiced_gain = self.voice.sqrt();
            let unvoiced_gain = (1.0 - self.voice).sqrt();
            let speech = periodic * voiced_gain + f64::from(self.unvoiced.next()) * unvoiced_gain;
            generated += speech * gain;
        }
        if at < JOIN {
            // Hermite extrapolation starts at the last value and derivative, then blends into
            // the model. A nonperiodic transient never turns into a held DC step.
            let t = at as f64 / JOIN as f64;
            let extrapolated = (f64::from(self.last) + self.slope * at as f64 * (1.0 - t).powi(2))
                .clamp(-self.transition_bound, self.transition_bound);
            let weight = t * t * (3.0 - 2.0 * t);
            generated = extrapolated * (1.0 - weight) + generated * weight;
        }
        self.last_generated = sample(generated);
        self.last_generated
    }
}

const _: () = assert!(HISTORY >= FRAME + PERIOD_MAX && HISTORY >= 2 * PERIOD_MAX);
const _: () = assert!(OVERLAP >= JOIN);
