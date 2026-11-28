// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
// Expand analysis adapted under the BSD license and patent grant in
// fixtures/playout/LICENSE.webrtc and fixtures/playout/PATENTS.webrtc.

//! Multi-candidate correlation/distortion search following NetEq's expand analysis. This is
//! floating-point PCM analysis at 24 kHz, not its fixed-point, downsampled implementation.

use super::{FRAME, PERIOD_MAX, PERIOD_MIN, correlation, rms};
use std::collections::VecDeque;

pub(super) struct Pitch {
    pub lags: [usize; 3],
    pub correlation: f64,
    pub amplitude_ratio: f64,
}

pub(super) fn analyze(history: &VecDeque<i16>) -> Pitch {
    let end = history.len();
    let mut scores = [0.0; PERIOD_MAX + 1];
    for (lag, score) in scores.iter_mut().enumerate().skip(PERIOD_MIN) {
        *score =
            correlation(|i| history[end - FRAME + i], |i| history[end - FRAME - lag + i], FRAME);
    }
    let mut selected = [PERIOD_MIN; 3];
    let mut scratch = scores;
    for candidate in &mut selected {
        *candidate = (PERIOD_MIN..=PERIOD_MAX)
            .max_by(|&a, &b| scratch[a].total_cmp(&scratch[b]).then_with(|| b.cmp(&a)))
            .expect("pitch search interval is nonempty");
        let from = candidate.saturating_sub(12).max(PERIOD_MIN);
        let to = (*candidate + 12).min(PERIOD_MAX);
        scratch[from..=to].fill(f64::NEG_INFINITY);
    }
    let mut winner = (f64::NEG_INFINITY, PERIOD_MIN, PERIOD_MIN);
    for correlation_lag in selected {
        let mut distortion_lag = correlation_lag;
        let mut distortion = f64::INFINITY;
        for lag in correlation_lag.saturating_sub(12).max(PERIOD_MIN)
            ..=(correlation_lag + 12).min(PERIOD_MAX)
        {
            let difference = (0..60)
                .map(|i| {
                    (f64::from(history[end - 60 + i]) - f64::from(history[end - 60 - lag + i]))
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
    let length = (distortion + 10).clamp(180, PERIOD_MAX);
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
