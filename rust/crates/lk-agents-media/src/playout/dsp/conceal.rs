// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// Expand analysis adapted under the BSD license and patent grant in
// fixtures/playout/LICENSE.webrtc and fixtures/playout/PATENTS.webrtc.

use std::collections::VecDeque;

use super::noise::{Background, Noise};
use super::pitch;
use super::{FRAME, HISTORY, OVERLAP, PERIOD_MAX, sample};

const JOIN: usize = 48; // Two milliseconds, independent of render block size.
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
    lags: [usize; 3],
    lag_index: usize,
    segment_position: usize,
    segment_gain: f64,
    mute_slope: f64,
    segments: usize,
    onset: bool,
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
            lags: [0; 3],
            lag_index: 0,
            segment_position: 0,
            segment_gain: 1.0,
            mute_slope: 0.0,
            segments: 0,
            onset: false,
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
        self.lags = [0; 3];
        self.lag_index = 0;
        self.segment_position = 0;
        self.segment_gain = 1.0;
        self.segments = 0;
        self.mute_slope = 0.0;
        self.onset = false;
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
        self.unvoiced.analyze_speech(&analysis);
        let pitch = pitch::analyze(history);
        self.lags = pitch.lags;
        self.period = pitch.lags[0];
        self.voice = if pitch.correlation > 0.48 {
            let x = pitch.correlation;
            ((-5179.0 + 19931.0 * x - 16422.0 * x * x + 5776.0 * x * x * x) / 4096.0)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let ratio = pitch.amplitude_ratio;
        self.onset = ratio > 1.5;
        self.mute_slope = if self.onset {
            (1.0 - 1.0 / ratio) / self.period as f64 / if ratio > 1.8 { 2.0 } else { 8.0 }
        } else if self.voice <= 0.8 {
            ((1.0 - ratio) / self.period as f64).max(0.005 / 3.0)
        } else if ratio > 0.98 {
            0.0
        } else {
            ((1.0 - ratio) / self.period as f64).max(0.0)
        };
        let end = history.len();
        let max_lag = *self.lags.iter().max().expect("three pitch lags");
        for i in 0..max_lag {
            self.cycle[i] = history[end - max_lag + i];
            let previous = f64::from(history[end - max_lag - self.period + i]);
            self.previous_cycle[i] = sample(previous * ratio);
        }
    }

    pub(in crate::playout) fn passive_level(&self) -> f64 {
        self.background_model.passive_level()
    }

    pub(in crate::playout) fn gain(&self) -> f64 {
        let gain =
            (self.segment_gain * (1.0 - self.segment_position as f64 * self.mute_slope)).max(0.0);
        if self.position >= self.limit { 0.0 } else { gain }
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
        let mut generated = background;
        if self.limit > 0 && at < self.limit {
            let current_lag = self.lags[self.lag_index].max(1);
            let max_lag = (*self.lags.iter().max().expect("three pitch lags")).max(1);
            let index = (max_lag - current_lag + self.segment_position).min(PERIOD_MAX - 1);
            let weight = match self.lag_index {
                0 => 0.0,
                1 => 0.25,
                _ => 0.5,
            };
            let periodic = (1.0 - weight) * f64::from(self.cycle[index])
                + weight * f64::from(self.previous_cycle[index]);
            // Begin with a short, continuous waveform extrapolation and move smoothly toward
            // the measured voiced/unvoiced mix, rather than injecting full-strength noise.
            let transition = if max_lag <= 93 {
                192.0
            } else if max_lag <= 189 {
                384.0
            } else {
                768.0
            };
            let voice = 1.0 - (1.0 - self.voice) * (at as f64 / transition).min(1.0);
            let mut gain = self.gain();
            // A configured hard ceiling must itself end smoothly.
            gain *= (self.limit.saturating_sub(at) as f64 / 120.0).min(1.0);
            let speech = periodic * voice + f64::from(self.unvoiced.next()) * (1.0 - voice);
            generated += speech * gain;
            self.segment_position += 1;
            if self.segment_position >= current_lag {
                if self.segments > 0 || !self.onset {
                    self.segment_gain *= (1.0 - current_lag as f64 * self.mute_slope).max(0.0);
                }
                self.segment_position = 0;
                self.segments += 1;
                self.lag_index = match self.segments % 4 {
                    0 => 0,
                    1 | 3 => 1,
                    _ => 2,
                };
                if self.segments >= 3 {
                    self.mute_slope = self.mute_slope.max(0.001 / 3.0);
                }
                if self.segments >= 7 {
                    self.mute_slope = self.mute_slope.max(0.002 / 3.0);
                }
            }
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
