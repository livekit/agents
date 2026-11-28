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

/// Validate adjacent pitch periods and the actual overlap, rather than requiring an entire
/// 20ms of stationary speech. Preserve the onset before the splice. A genuine low-energy
/// interval is preferred to changing voiced speech, even if the latter correlates perfectly.
pub(in crate::playout) fn stretch(
    rate: Rate,
    queue: &VecDeque<i16>,
    operation: Stretch,
    budget: usize,
    passive_level: f64,
    output: &mut [i16],
) -> Option<Scale> {
    if queue.len() < rate.frame() + rate.period_max() || budget < rate.period_min() {
        return None;
    }
    let expand = operation == Stretch::Expand;
    let threshold =
        if operation == Stretch::FastAccelerate { FAST_CORRELATION } else { CORRELATION };
    let evaluate = |period: usize| -> Option<Match> {
        let anchor = if expand {
            rate.frame() - rate.overlap()
        } else {
            (rate.frame() - rate.overlap()).max(period + rate.overlap())
        };
        let splice = if expand { anchor } else { anchor - period };
        let overlap = rate.overlap().min(rate.frame() - splice);
        let before = rms(|i| queue[anchor - period + i], period);
        let after = rms(|i| queue[anchor + i], period);
        let passive = before < passive_level
            && after < passive_level
            && rms(|i| queue[splice + i], overlap) < passive_level;
        if !passive && (after < before * 0.8 || after > before * 1.25) {
            return None;
        }
        let other = if expand { splice - period } else { splice + period };
        let score = correlation(|i| queue[anchor - period + i], |i| queue[anchor + i], period)
            .min(correlation(|i| queue[splice + i], |i| queue[other + i], overlap));
        if !passive && score < threshold {
            return None;
        }
        Some(Match { period, splice, overlap, score, passive })
    };
    let mut best: Option<Match> = None;
    for period in rate.period_min()..=rate.period_max().min(budget) {
        let Some(candidate) = evaluate(period) else {
            continue;
        };
        if best.is_none_or(|old| {
            candidate.passive && !old.passive
                || candidate.passive == old.passive && candidate.score > old.score + 1e-9
        }) {
            best = Some(candidate);
        }
        if candidate.passive {
            break;
        }
    }
    let mut best = best?;
    if operation == Stretch::FastAccelerate {
        for multiple in (2..=rate.period_max().min(budget) / best.period).rev() {
            if let Some(candidate) = evaluate(best.period * multiple) {
                best = candidate;
                break;
            }
        }
    }
    for (i, out) in output.iter_mut().enumerate() {
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
        consumed: if expand { rate.frame() - best.period } else { rate.frame() + best.period },
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
    budget: usize,
    passive_level: f64,
    output: &mut [i16],
) -> Option<Scale> {
    if history.len() < 2 * rate.period_min() || budget < rate.period_min() {
        return None;
    }
    let end = history.len();
    let mut quiet = false;
    let mut best = CORRELATION;
    let mut period = 0;
    for lag in rate.period_min()..=rate.period_max().min(budget).min(end / 2) {
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
    Some(Scale { consumed: rate.frame() - period, change: period, passive: quiet })
}
