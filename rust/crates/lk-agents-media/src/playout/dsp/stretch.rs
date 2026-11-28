// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// NetEq processing adapted under the upstream notices in the playout README.md.

use std::collections::VecDeque;

use super::recovery::smooth;
use super::{Rate, correlation, rms, sample};

const CORRELATION: f64 = 0.9;
const FAST_CORRELATION: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::playout) enum Stretch {
    Expand,
    Accelerate,
    FastAccelerate,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::playout) struct Scale {
    pub consumed: usize,
    pub produced: usize,
    pub change: usize,
    pub passive: bool,
}

#[derive(Clone, Copy)]
struct Match {
    period: usize,
    splice: usize,
    overlap: usize,
    score: f64,
    passive: bool,
}

impl Match {
    // Ranking only breaks ties within the same class; voiced audio never replaces quiet.
    fn prefer(self, candidate: Self, better: impl FnOnce(Self, Self) -> bool) -> Self {
        let replace = match (self.passive, candidate.passive) {
            (false, true) => true,
            (true, false) => false,
            _ => better(candidate, self),
        };
        if replace { candidate } else { self }
    }
}

fn matching_seams(
    rate: Rate,
    left: impl Fn(usize) -> i16,
    right: impl Fn(usize) -> i16,
    period: usize,
    threshold: f64,
    passive_level: f64,
) -> bool {
    // Check local matches independently of the crossfade length: full-period correlation
    // can mask a brief mismatched transient inside a long vowel.
    let window = rate.overlap().min(period);
    for start in (0..period).step_by(window) {
        let start = start.min(period - window);
        let a = |i| left(start + i);
        let b = |i| right(start + i);
        if rms(a, window) < passive_level && rms(b, window) < passive_level {
            continue;
        }
        if correlation(a, b, window) < threshold {
            return false;
        }
    }
    true
}

/// Match adjacent periods in NetEq's 30ms window and crossfade over the whole period.
/// Prefer a low-energy splice and preserve the onset before it. Return the actual produced
/// length: expansion adds one period, acceleration removes it.
pub(in crate::playout) fn stretch(
    rate: Rate,
    queue: &VecDeque<i16>,
    operation: Stretch,
    maximum_change: usize,
    passive_level: f64,
    output: &mut [i16],
) -> Option<Scale> {
    if queue.len() < rate.stretch_window() || maximum_change < rate.period_min() {
        return None;
    }
    let expand = operation == Stretch::Expand;
    let threshold =
        if operation == Stretch::FastAccelerate { FAST_CORRELATION } else { CORRELATION };
    let evaluate = |period: usize| -> Option<Match> {
        // NetEq splices at 15ms with a full pitch-period crossfade, over 30ms of input.
        let anchor = rate.period_max();
        let splice = if expand { anchor } else { anchor - period };
        let overlap = period;
        let before = rms(|i| queue[anchor - period + i], period);
        let after = rms(|i| queue[anchor + i], period);
        let passive = before < passive_level
            && after < passive_level
            && rms(|i| queue[splice + i], overlap) < passive_level;
        if !passive && (after < before * 0.8 || after > before * 1.25) {
            return None;
        }
        let score = correlation(|i| queue[anchor - period + i], |i| queue[anchor + i], period);
        if !passive
            && (score < threshold
                || !matching_seams(
                    rate,
                    |i| queue[anchor - period + i],
                    |i| queue[anchor + i],
                    period,
                    threshold,
                    passive_level,
                ))
        {
            return None;
        }
        Some(Match { period, splice, overlap, score, passive })
    };
    let mut best: Option<Match> = None;
    for period in rate.period_min()..=rate.period_max().min(maximum_change) {
        let Some(candidate) = evaluate(period) else {
            continue;
        };
        best = Some(best.map_or(candidate, |old| {
            old.prefer(candidate, |candidate, old| candidate.score > old.score + 1e-9)
        }));
        if candidate.passive {
            break;
        }
    }
    let mut best = best?;
    if operation == Stretch::FastAccelerate {
        let period = best.period;
        for multiple in 2..=rate.period_max().min(maximum_change) / period {
            if let Some(candidate) = evaluate(period * multiple) {
                best = best.prefer(candidate, |candidate, old| candidate.period > old.period);
            }
        }
    }
    let produced = if expand {
        rate.stretch_window() + best.period
    } else {
        rate.stretch_window() - best.period
    };
    for (i, out) in output[..produced].iter_mut().enumerate() {
        *out = if i < best.splice {
            queue[i]
        } else if i < best.splice + best.overlap {
            let weight = smooth((i - best.splice) as f64 / best.overlap as f64);
            let other = if expand { i - best.period } else { i + best.period };
            sample(f64::from(queue[i]) * (1.0 - weight) + f64::from(queue[other]) * weight)
        } else {
            queue[if expand { i - best.period } else { i + best.period }]
        };
    }
    Some(Scale {
        consumed: rate.stretch_window(),
        produced,
        change: best.period,
        passive: best.passive,
    })
}

/// Builds reserve even with 20 ms deliveries, borrowing already rendered history for
/// analysis and one repeated period. A strong match across both history and new source
/// protects consonants and onsets; no future 35 ms chunk is required.
pub(in crate::playout) fn preemptive(
    rate: Rate,
    history: &VecDeque<i16>,
    queue: &VecDeque<i16>,
    maximum_change: usize,
    passive_level: f64,
    output: &mut [i16],
) -> Option<Scale> {
    if history.len() < 2 * rate.period_min() || maximum_change < rate.period_min() {
        return None;
    }
    let end = history.len();
    let mut quiet = false;
    let mut best = CORRELATION;
    let mut period = 0;
    for lag in rate.period_min()..=rate.period_max().min(maximum_change).min(end / 2) {
        if queue.len() < (rate.frame() - lag).max(lag) {
            continue;
        }
        let previous = rms(|i| history[end - lag + i], lag);
        let next = rms(|i| queue[i], lag);
        let passive = previous < passive_level && next < passive_level;
        if !passive && (next < previous * 0.8 || next > previous * 1.25) {
            continue;
        }
        let past = correlation(|i| history[end - lag + i], |i| history[end - 2 * lag + i], lag);
        let upcoming = correlation(|i| history[end - lag + i], |i| queue[i], lag);
        let score = past.min(upcoming);
        if !passive && score <= best + 1e-9 {
            continue;
        }
        if !passive
            && (!matching_seams(
                rate,
                |i| history[end - lag + i],
                |i| history[end - 2 * lag + i],
                lag,
                CORRELATION,
                passive_level,
            ) || !matching_seams(
                rate,
                |i| history[end - lag + i],
                |i| queue[i],
                lag,
                CORRELATION,
                passive_level,
            ))
        {
            continue;
        }
        if passive || score > best + 1e-9 {
            best = score;
            period = lag;
            quiet = passive;
            if passive {
                break;
            }
        }
    }
    if period == 0 {
        return None;
    }
    for (i, out) in output[..rate.frame()].iter_mut().enumerate() {
        *out = if i < period {
            // Start on the real next sample; finish on the repeated history period before
            // joining source[0]. The complete-period match protects both boundaries.
            let b = smooth(i as f64 / period as f64);
            sample(f64::from(queue[i]) * (1.0 - b) + f64::from(history[end - period + i]) * b)
        } else {
            queue[i - period]
        };
    }
    Some(Scale {
        consumed: rate.frame() - period,
        produced: rate.frame(),
        change: period,
        passive: quiet,
    })
}
