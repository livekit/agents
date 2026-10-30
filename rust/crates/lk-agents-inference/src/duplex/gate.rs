//! `AudioGate` (`llm/duplex_adapter.py`): which frames of a duplex model's continuous output carry
//! speech worth playing. Durations count audio, not wall clock, so one set of defaults ports across
//! providers, frame sizes and networks.

use std::collections::VecDeque;

use lk_agents_abi::AudioFrame;

/// A floor this low is digital silence; it keeps the gate's ratios finite when a model emits zeros.
pub const SILENCE_FLOOR: f64 = 1e-4;

/// `update` is true while the frame belongs to an open burst of output; `deactivate` ends the open
/// burst, keeping whatever has been learned about the model. `Err` is a gate that failed (a host
/// gate raised): the output stream is then unusable.
pub trait AudioGate: Send {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String>;
    fn deactivate(&mut self);
}

/// The frame's RMS over 16-bit PCM, in `[0, 1]`.
pub fn rms(frame: &AudioFrame) -> f64 {
    let n = frame.data.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = frame
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            let s = f64::from(i16::from_le_bytes([b[0], b[1]]));
            s * s
        })
        .sum();
    (sum / n as f64).sqrt() / 32768.0
}

/// The hysteresis both gates share: open above `floor * activation`, close after
/// `min_silence_duration` of audio below `floor * deactivation`.
#[derive(Debug, Clone)]
struct Hysteresis {
    activation_ratio: f64,
    deactivation_ratio: f64,
    min_silence_duration: f64,
    open: bool,
    quiet: f64,
}

impl Hysteresis {
    fn step(&mut self, level: f64, floor: f64, duration: f64) -> bool {
        if !self.open {
            if level > floor * self.activation_ratio {
                self.open = true;
                self.quiet = 0.0;
            }
        } else if level < floor * self.deactivation_ratio {
            self.quiet += duration;
            if self.quiet >= self.min_silence_duration {
                self.open = false;
            }
        } else {
            self.quiet = 0.0;
        }
        self.open
    }

    fn deactivate(&mut self) {
        self.open = false;
        self.quiet = 0.0;
    }
}

/// `FixedGate`: opens on output that stands out from a silence the plugin already knows.
#[derive(Debug, Clone)]
pub struct FixedGate {
    floor: f64,
    h: Hysteresis,
}

impl FixedGate {
    pub fn new(
        silence: f64,
        activation_ratio: f64,
        deactivation_ratio: f64,
        min_silence_duration: f64,
    ) -> Self {
        FixedGate {
            floor: silence.max(SILENCE_FLOOR),
            h: Hysteresis {
                activation_ratio,
                deactivation_ratio,
                min_silence_duration,
                open: false,
                quiet: 0.0,
            },
        }
    }
}

impl AudioGate for FixedGate {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String> {
        Ok(self.h.step(rms(frame), self.floor, frame.duration()))
    }

    fn deactivate(&mut self) {
        self.h.deactivate();
    }
}

/// `AdaptiveNoiseGate`: the floor is the quietest `min_silence_duration` stretch the model produced
/// while not speaking, within `window`; speech never raises it, and no single frame defines it.
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseGate {
    window: f64,
    history: VecDeque<(f64, f64)>,
    history_duration: f64,
    stretch_sum: f64,
    stretch_duration: f64,
    h: Hysteresis,
}

impl AdaptiveNoiseGate {
    pub fn new(
        activation_ratio: f64,
        deactivation_ratio: f64,
        min_silence_duration: f64,
        window: f64,
    ) -> Self {
        AdaptiveNoiseGate {
            window,
            history: VecDeque::new(),
            history_duration: 0.0,
            stretch_sum: 0.0,
            stretch_duration: 0.0,
            h: Hysteresis {
                activation_ratio,
                deactivation_ratio,
                min_silence_duration,
                open: false,
                quiet: 0.0,
            },
        }
    }
}

impl Default for AdaptiveNoiseGate {
    fn default() -> Self {
        AdaptiveNoiseGate::new(3.0, 1.8, 0.5, 10.0)
    }
}

impl AudioGate for AdaptiveNoiseGate {
    fn update(&mut self, frame: &AudioFrame) -> Result<bool, String> {
        let level = rms(frame);
        let duration = frame.duration();
        if !self.h.open {
            // only silence teaches the floor, and only a whole stretch of it
            self.stretch_sum += level * duration;
            self.stretch_duration += duration;
            if self.stretch_duration >= self.h.min_silence_duration {
                let mean = self.stretch_sum / self.stretch_duration;
                self.history.push_back((mean, self.stretch_duration));
                self.history_duration += self.stretch_duration;
                self.stretch_sum = 0.0;
                self.stretch_duration = 0.0;
                while self.history_duration > self.window && self.history.len() > 1 {
                    let (_, d) = self.history.pop_front().expect("len > 1");
                    self.history_duration -= d;
                }
            }
        }
        let floor = if !self.history.is_empty() {
            self.history.iter().map(|(l, _)| *l).fold(f64::INFINITY, f64::min)
        } else if self.stretch_duration > 0.0 {
            self.stretch_sum / self.stretch_duration
        } else {
            level
        };
        Ok(self.h.step(level, floor.max(SILENCE_FLOOR), duration))
    }

    fn deactivate(&mut self) {
        self.h.deactivate();
    }
}
