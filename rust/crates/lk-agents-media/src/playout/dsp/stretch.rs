use std::collections::VecDeque;

use super::recovery::smooth;
use super::{FRAME, OVERLAP, PERIOD_MAX, PERIOD_MIN, correlation, rms, sample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::playout) enum Stretch {
    Expand,
    Accelerate,
}

/// Finds a periodic splice over two windows, rather than matching an isolated transient.
/// Low-energy input may use a short splice without reliable pitch; noisy speech is left
/// untouched when no safe splice exists. Output is exactly FRAME samples; returns consumed
/// source samples and the signed timing change. The caller owns the rate budget.
pub(in crate::playout) fn stretch(
    queue: &VecDeque<i16>,
    operation: Stretch,
    budget: usize,
    output: &mut [i16; FRAME],
) -> Option<(usize, usize)> {
    if queue.len() < FRAME + PERIOD_MAX || budget < PERIOD_MIN {
        return None;
    }
    let low_energy = rms(|i| queue[i], FRAME + PERIOD_MAX) < 40.0;
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
    if best < 0.8 && !low_energy {
        return None;
    }
    if low_energy && best < 0.8 {
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
