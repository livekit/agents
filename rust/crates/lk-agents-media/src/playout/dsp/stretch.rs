// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// NetEq processing adapted under fixtures/playout/LICENSE.webrtc and PATENTS.webrtc.

use std::collections::VecDeque;

use super::recovery::smooth;
use super::{FRAME, OVERLAP, PERIOD_MAX, PERIOD_MIN, correlation, rms, sample};

const CORRELATION: f64 = 0.9;
const FAST_CORRELATION: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::playout) enum Stretch {
    Expand,
    Accelerate,
    FastAccelerate,
}

/// Finds a periodic splice over two windows, rather than matching an isolated transient.
/// Low-energy input may use a short splice without reliable pitch; noisy speech is left
/// untouched when no safe splice exists. Output is exactly FRAME samples; returns consumed
/// source samples and the signed timing change. The caller owns the rate budget.
pub(in crate::playout) fn stretch(
    queue: &VecDeque<i16>,
    operation: Stretch,
    budget: usize,
    passive_level: f64,
    output: &mut [i16; FRAME],
) -> Option<(usize, usize)> {
    if queue.len() < FRAME + PERIOD_MAX || budget < PERIOD_MIN {
        return None;
    }
    let low_energy = rms(|i| queue[i], FRAME + PERIOD_MAX) < passive_level;
    let mut best = -1.0;
    let mut period = 0;
    let mut scores = [-1.0; PERIOD_MAX + 1];
    let length = FRAME - OVERLAP; // all candidate comparisons use exactly the same length
    for lag in PERIOD_MIN..=PERIOD_MAX.min(budget) {
        let a = correlation(|i| queue[i], |i| queue[lag + i], length);
        let b = correlation(|i| queue[OVERLAP + i], |i| queue[OVERLAP + lag + i], length);
        let score = a.min(b);
        scores[lag] = score;
        if score > best {
            best = score;
            period = lag;
        }
    }
    let threshold =
        if operation == Stretch::FastAccelerate { FAST_CORRELATION } else { CORRELATION };
    if best < threshold && !low_energy {
        return None;
    }
    if low_energy && best < threshold {
        period = PERIOD_MIN;
    }
    // Choose the first sufficiently strong local period rather than a long multiple.
    if !low_energy {
        for lag in PERIOD_MIN + 1..period {
            if scores[lag] >= best * 0.95
                && scores[lag] >= scores[lag - 1]
                && scores[lag] > scores[lag + 1]
            {
                period = lag;
                break;
            }
        }
    }
    if operation == Stretch::FastAccelerate && !low_energy {
        // NetEq's fast mode removes a multiple of the pitch lag. Verify the actual long
        // splice, rather than assuming a harmonic remains valid across changing speech.
        for multiple in (2..=PERIOD_MAX.min(budget) / period).rev() {
            let lag = period * multiple;
            let a = correlation(|i| queue[i], |i| queue[lag + i], length);
            let b = correlation(|i| queue[OVERLAP + i], |i| queue[OVERLAP + lag + i], length);
            if a.min(b) >= threshold {
                period = lag;
                break;
            }
        }
    }
    let expand = operation == Stretch::Expand;
    let splice = if expand { period } else { 0 };
    for (i, out) in output.iter_mut().enumerate() {
        *out = if i < splice {
            queue[i]
        } else if i < splice + OVERLAP {
            let at = i - splice;
            let weight = smooth(at as f64 / OVERLAP as f64);
            let other = if expand { i - period } else { i + period };
            sample(f64::from(queue[i]) * (1.0 - weight) + f64::from(queue[other]) * weight)
        } else {
            queue[if expand { i - period } else { i + period }]
        };
    }
    Some((if expand { FRAME - period } else { FRAME + period }, period))
}

/// Builds reserve even with 20 ms deliveries, borrowing already rendered history for
/// analysis and one repeated period. A strong match across both history and new source
/// protects consonants and onsets; no future 35 ms chunk is required.
pub(in crate::playout) fn preemptive(
    history: &VecDeque<i16>,
    queue: &VecDeque<i16>,
    budget: usize,
    passive_level: f64,
    output: &mut [i16; FRAME],
) -> Option<(usize, usize)> {
    if history.len() < FRAME + PERIOD_MAX || budget < PERIOD_MIN {
        return None;
    }
    let end = history.len();
    let quiet = rms(|i| history[end - FRAME + i], FRAME) < passive_level
        && rms(|i| queue[i], queue.len().min(FRAME)) < passive_level;
    let mut best = CORRELATION;
    let mut period = 0;
    for lag in PERIOD_MIN..=PERIOD_MAX.min(budget) {
        if queue.len() < (FRAME - lag).max(lag) {
            continue;
        }
        let previous = rms(|i| history[end - lag + i], lag);
        let next = rms(|i| queue[i], lag);
        if !quiet && (next < previous * 0.8 || next > previous * 1.25) {
            continue;
        }
        let past =
            correlation(|i| history[end - FRAME + i], |i| history[end - FRAME - lag + i], FRAME);
        let upcoming = correlation(|i| history[end - lag + i], |i| queue[i], lag);
        let score = past.min(upcoming);
        if quiet || score > best + 1e-9 {
            best = score;
            period = lag;
            if quiet {
                break;
            }
        }
    }
    if period == 0 {
        return None;
    }
    for (i, out) in output.iter_mut().enumerate() {
        *out = if i < period {
            // Start on the real next sample; finish on the repeated history period before
            // joining source[0]. The complete-period match protects both boundaries.
            let b = smooth(i as f64 / period as f64);
            sample(f64::from(queue[i]) * (1.0 - b) + f64::from(history[end - period + i]) * b)
        } else {
            queue[i - period]
        };
    }
    Some((FRAME - period, period))
}
