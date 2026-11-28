//! PCM recovery DSP: source-only noise analysis, voiced/unvoiced extrapolation, phase-aligned
//! recovery and pitch-preserving overlap/add. All state and scratch storage are bounded.

mod conceal;
#[cfg(test)]
mod dsp_tests;
mod noise;
mod pitch;
mod recovery;
mod stretch;

use super::Rate;

pub(super) use conceal::Concealer;
pub(super) use recovery::Recovery;
pub(super) use stretch::{Stretch, preemptive, stretch};

#[cfg(test)]
pub(super) const FRAME: usize = 480;
#[cfg(test)]
pub(super) const PERIOD_MIN: usize = 60;
#[cfg(test)]
pub(super) const PERIOD_MAX: usize = 360;
#[cfg(test)]
pub(super) const HISTORY: usize = 960;

pub(super) const MAX_FRAME: usize = 960;
pub(super) const MAX_PERIOD: usize = 720;

pub(super) fn sample(value: f64) -> i16 {
    value.round().clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

pub(super) fn mix(a: i16, b: i16, position: usize, length: usize) -> i16 {
    let position = position.min(length) as i64;
    ((i64::from(a) * (length as i64 - position) + i64::from(b) * position) / length as i64) as i16
}

pub(super) fn rms(signal: impl Fn(usize) -> i16, length: usize) -> f64 {
    if length == 0 {
        return 0.0;
    }
    ((0..length).map(|i| f64::from(signal(i)).powi(2)).sum::<f64>() / length as f64).sqrt()
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
    if aa < 1.0 || bb < 1.0 { 0.0 } else { (cross / (aa * bb).sqrt()).clamp(-1.0, 1.0) }
}
