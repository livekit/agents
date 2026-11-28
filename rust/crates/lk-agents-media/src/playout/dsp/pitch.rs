// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// Expand analysis adapted under the BSD license and patent grant in
// the upstream notices in the playout README.md.

//! Multi-candidate correlation/distortion search following NetEq's expand analysis. This is
//! floating-point PCM analysis at the configured rate, not its fixed-point, downsampled implementation.

use super::{MAX_PERIOD, Rate, correlation, rms};
use std::collections::VecDeque;

pub(super) struct Pitch {
    pub lags: [usize; 3],
    pub correlation: f64,
    pub amplitude_ratio: f64,
}

pub(super) fn analyze(rate: Rate, history: &VecDeque<i16>) -> Pitch {
    let end = history.len();
    let mut scores = [0.0; MAX_PERIOD + 1];
    for (lag, score) in
        scores.iter_mut().enumerate().take(rate.period_max() + 1).skip(rate.period_min())
    {
        *score = correlation(
            |i| history[end - rate.frame() + i],
            |i| history[end - rate.frame() - lag + i],
            rate.frame(),
        );
    }
    let mut selected = [rate.period_min(); 3];
    let mut scratch = scores;
    for candidate in &mut selected {
        *candidate = (rate.period_min()..=rate.period_max())
            .max_by(|&a, &b| scratch[a].total_cmp(&scratch[b]).then_with(|| b.cmp(&a)))
            .expect("pitch search interval is nonempty");
        let from = candidate.saturating_sub(rate.scale(12)).max(rate.period_min());
        let to = (*candidate + rate.scale(12)).min(rate.period_max());
        scratch[from..=to].fill(f64::NEG_INFINITY);
    }
    let mut winner = (f64::NEG_INFINITY, rate.period_min(), rate.period_min());
    for correlation_lag in selected {
        let mut distortion_lag = correlation_lag;
        let mut distortion = f64::INFINITY;
        for lag in correlation_lag.saturating_sub(rate.scale(12)).max(rate.period_min())
            ..=(correlation_lag + rate.scale(12)).min(rate.period_max())
        {
            let difference = (0..rate.scale(60))
                .map(|i| {
                    (f64::from(history[end - rate.scale(60) + i])
                        - f64::from(history[end - rate.scale(60) - lag + i]))
                    .abs()
                })
                .sum::<f64>();
            if difference < distortion {
                distortion = difference;
                distortion_lag = lag;
            }
        }
        let merit = scores[correlation_lag].max(0.0) / (distortion + 1.0);
        if merit > winner.0 {
            winner = (merit, distortion_lag, correlation_lag);
        }
    }
    let (_, distortion, correlation_lag) = winner;
    let length = (distortion + rate.scale(10)).clamp(rate.scale(180), rate.period_max());
    let mut best = 0.0_f64;
    for lag in distortion.min(correlation_lag)..=distortion.max(correlation_lag) {
        best = best.max(correlation(
            |i| history[end - length + i],
            |i| history[end - length - lag + i],
            length,
        ));
    }
    let recent = rms(|i| history[end - distortion + i], distortion);
    let previous = rms(|i| history[end - 2 * distortion + i], distortion);
    let ratio = if previous > 0.0 { recent / previous } else { 2.0 };
    Pitch {
        lags: [
            distortion,
            (distortion + correlation_lag) / 2,
            if distortion > correlation_lag {
                (distortion + correlation_lag - 1) / 2
            } else {
                (distortion + correlation_lag).div_ceil(2)
            },
        ],
        correlation: best,
        amplitude_ratio: ratio.clamp(0.5, 2.0),
    }
}
