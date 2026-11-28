//! Bounded linear-prediction analysis and deterministic spectrally shaped excitation.

use super::{FRAME, sample};

const ORDER: usize = 8;

#[derive(Clone)]
pub(super) struct Noise {
    coefficients: [f64; ORDER],
    state: [f64; ORDER],
    gain: f64,
    mean: f64,
    bound: f64,
    seed: u32,
}

impl Default for Noise {
    fn default() -> Self {
        Self {
            coefficients: [0.0; ORDER],
            state: [0.0; ORDER],
            gain: 0.0,
            mean: 0.0,
            bound: 0.0,
            seed: 0x4e455451,
        }
    }
}

impl Noise {
    /// Stabilized Levinson recursion; returns the prediction-error/input-energy ratio.
    /// No unstable model is installed. Lag damping and reflection limits deliberately trade
    /// spectral resolution for bounded synthesis on transients and nearly periodic input.
    pub(super) fn analyze(&mut self, input: &[i16]) -> f64 {
        if input.len() < FRAME {
            self.gain = 0.0;
            return 0.0;
        }
        self.mean = input.iter().map(|&x| f64::from(x)).sum::<f64>() / input.len() as f64;
        let mut r = [0.0; ORDER + 1];
        for lag in 0..=ORDER {
            for i in lag..input.len() {
                r[lag] +=
                    (f64::from(input[i]) - self.mean) * (f64::from(input[i - lag]) - self.mean);
            }
            r[lag] *= (-(lag as f64 * lag as f64) / 800.0).exp();
        }
        if r[0] < input.len() as f64 {
            self.coefficients.fill(0.0);
            self.state.fill(0.0);
            self.gain = 0.0;
            self.bound = 0.0;
            return 0.0;
        }
        let energy = r[0] * 1.0001;
        let mut error = energy;
        let mut a = [0.0; ORDER];
        for order in 0..ORDER {
            let mut residual = r[order + 1];
            for j in 0..order {
                residual += a[j] * r[order - j];
            }
            let reflection = (-residual / error.max(1.0)).clamp(-0.95, 0.95);
            let previous = a;
            for j in 0..order {
                a[j] = previous[j] + reflection * previous[order - j - 1];
            }
            a[order] = reflection;
            error *= 1.0 - reflection * reflection;
        }
        self.coefficients = a;
        self.gain = (error / input.len() as f64).max(0.0).sqrt();
        let rms = (energy / input.len() as f64).sqrt();
        self.bound = (rms * 6.0).min(32_767.0);
        for i in 0..ORDER {
            self.state[i] = f64::from(input[input.len() - i - 1]) - self.mean;
        }
        error / energy
    }

    pub(super) fn quiet(&self) -> bool {
        self.gain < 0.5
    }

    pub(super) fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    pub(super) fn next(&mut self) -> i16 {
        if self.quiet() {
            return 0;
        }
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        // Uniform excitation has variance 1/3; scale to unit variance.
        let excitation =
            (f64::from(self.seed) / f64::from(u32::MAX) * 2.0 - 1.0) * 3.0_f64.sqrt() * self.gain;
        let mut value = excitation;
        for i in 0..ORDER {
            value -= self.coefficients[i] * self.state[i];
        }
        // Clamp the internal state as well as output; a bad model cannot accumulate infinity.
        value = value.clamp(-self.bound, self.bound);
        self.state.copy_within(..ORDER - 1, 1);
        self.state[0] = value;
        // Do not synthesize source DC into a long gap.
        sample(value)
    }
}

/// Learns only low-energy non-periodic real audio. Synthesized audio never trains this model.
#[derive(Clone)]
pub(super) struct Background {
    model: Noise,
    energies: [f64; 128],
    count: usize,
    index: usize,
    initialized: bool,
}

impl Default for Background {
    fn default() -> Self {
        Self {
            model: Noise::default(),
            energies: [0.0; 128],
            count: 0,
            index: 0,
            initialized: false,
        }
    }
}

impl Background {
    pub(super) fn observe(&mut self, input: &[i16]) {
        let mean = input.iter().map(|&x| f64::from(x)).sum::<f64>() / input.len() as f64;
        let energy =
            input.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / input.len() as f64;
        self.energies[self.index] = energy;
        self.index = (self.index + 1) % self.energies.len();
        self.count = (self.count + 1).min(self.energies.len());
        if self.count < 10 {
            return;
        }
        let mut sorted = self.energies;
        sorted[..self.count].sort_unstable_by(f64::total_cmp);
        let floor = sorted[(self.count - 1) / 5].max(1.0);
        if energy > floor * 1.5 {
            return;
        }
        let mut candidate = self.model.clone();
        let flatness = candidate.analyze(input);
        if energy < 1.0 || flatness > 0.15 {
            // A speech-only stream must not learn loud unvoiced consonants as room tone.
            // Without a measured quiet interval, the safe background model is silence.
            if energy.sqrt() <= 512.0 {
                candidate.reset_state();
                self.model = candidate;
                self.initialized = true;
            }
        }
    }

    pub(super) fn model(&self) -> Noise {
        if self.initialized { self.model.clone() } else { Noise::default() }
    }
}
