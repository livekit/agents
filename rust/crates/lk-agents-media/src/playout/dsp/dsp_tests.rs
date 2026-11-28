use std::collections::VecDeque;

use super::*;

fn sine(length: usize, period: usize, phase: usize, gain: f64) -> Vec<i16> {
    (0..length)
        .map(|i| sample(((i + phase) as f64 * std::f64::consts::TAU / period as f64).sin() * gain))
        .collect()
}

fn noise(length: usize, gain: f64) -> Vec<i16> {
    let mut seed = 734_u32;
    (0..length)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            sample((f64::from(seed) / f64::from(u32::MAX) * 2.0 - 1.0) * gain)
        })
        .collect()
}

fn begin(input: &[i16], limit: usize) -> Concealer {
    let mut c = Concealer::default();
    for &s in input {
        c.observe(s);
    }
    let history = input[input.len().saturating_sub(HISTORY)..].iter().copied().collect();
    c.start(&history, limit, *input.last().unwrap());
    c
}

#[test]
fn voiced_short_gap_preserves_pitch_and_energy_instead_of_a_dc_hold() {
    for period in [65, 120, 240, 350] {
        let source = sine(HISTORY, period, 21, 10_000.0);
        let mut c = begin(&source, 2400);
        assert!((c.period() as isize - period as isize).abs() <= 2);
        let generated: Vec<_> = (0..480).map(|_| c.next()).collect();
        assert!(rms(|i| generated[i + 60], 240) > 4000.0);
        let score = correlation(|i| generated[60 + i], |i| generated[60 + period + i], 60);
        assert!(score > 0.95, "period={period} score={score}");
        assert_eq!(generated[0], *source.last().unwrap());
    }
}

#[test]
fn unvoiced_speech_retains_noise_energy_and_does_not_become_a_tone() {
    let input = noise(24_000, 9000.0);
    let mut c = begin(&input, 2400);
    assert!(c.voiced() < 0.2);
    let out: Vec<_> = (0..480).map(|_| c.next()).collect();
    assert!(rms(|i| out[60 + i], 240) > 2000.0);
    for lag in [60, 120, 240] {
        assert!(correlation(|i| out[i], |i| out[i + lag], 120).abs() < 0.4);
    }
}

#[test]
fn room_tone_continues_after_speech_fades_then_eventually_mutes() {
    let input = noise(24_000, 200.0);
    let mut c = begin(&input, 960);
    let out: Vec<_> = (0..3000).map(|_| c.next()).collect();
    let level = rms(|i| out[1200 + i], 1000);
    assert!(level > 30.0 && level < 250.0, "room tone RMS={level}");
    for _ in 0..60_000 {
        c.next();
    }
    assert_eq!(c.next(), 0);
    assert!(!c.active());
}

#[test]
fn speech_only_does_not_train_loud_consonants_as_background() {
    let input = noise(24_000, 8000.0);
    let mut c = begin(&input, 960);
    for _ in 0..1000 {
        c.next();
    }
    assert_eq!(c.next(), 0);
}

#[test]
fn phase_aligned_recovery_keeps_the_boundary_and_preserves_incoming_energy() {
    for phase in [0, 30, 60, 90] {
        let mut c = begin(&sine(HISTORY, 120, 7, 12_000.0), 2400);
        for _ in 0..480 {
            c.next();
        }
        let last = c.last_generated();
        let input = sine(720, 120, phase, 12_000.0);
        let queue = input.iter().copied().collect();
        let mut recovery = Recovery::default();
        recovery.start(&c, &queue, true);
        let mut out = Vec::new();
        while let Some(sample) = recovery.next_prefix() {
            out.push(sample);
        }
        let prefix = out.len();
        assert!(prefix < c.period());
        out.extend(input.iter().map(|&s| recovery.apply(s)));
        assert_eq!(out[0], last);
        assert!(rms(|i| out[240 + i], 240) > 7500.0);
        assert_eq!(&out[prefix + 480..], &input[480..]);
        assert!(!recovery.active());
        let jump = out.windows(2).map(|p| (i32::from(p[1]) - i32::from(p[0])).abs()).max().unwrap();
        assert!(jump < 3000, "phase={phase} jump={jump}");
    }
}

#[test]
fn long_outage_returns_without_an_abrupt_full_gain_onset() {
    let mut c = begin(&sine(HISTORY, 120, 0, 10_000.0), 960);
    for _ in 0..8000 {
        c.next();
    }
    let input = [12_000; 960];
    let mut recovery = Recovery::default();
    recovery.start(&c, &input.iter().copied().collect(), true);
    let out: Vec<_> = input.into_iter().map(|s| recovery.apply(s)).collect();
    assert_eq!(out[0], 0);
    assert!(out[24] < 1000);
    assert_eq!(out[480], 12_000);
    assert!(out.windows(2).all(|pair| (i32::from(pair[1]) - i32::from(pair[0])).abs() < 300));
}

#[test]
fn lpc_handles_dc_full_scale_impulses_and_silence_without_unbounded_states() {
    for input in [
        vec![0; HISTORY],
        vec![i16::MAX; HISTORY],
        vec![i16::MIN; HISTORY],
        (0..HISTORY).map(|i| if i.is_multiple_of(2) { i16::MAX } else { i16::MIN }).collect(),
        (0..HISTORY).map(|i| if i == HISTORY - 1 { i16::MAX } else { 0 }).collect(),
    ] {
        let mut a = begin(&input, 960);
        let mut b = begin(&input, 960);
        for _ in 0..60_000 {
            assert_eq!(a.next(), b.next());
        }
        assert_eq!(a.next(), 0);
    }
}

#[test]
fn time_stretch_preserves_tone_pitch_and_handles_quiet_input() {
    for operation in [Stretch::Expand, Stretch::Accelerate] {
        let input: VecDeque<_> = sine(FRAME + PERIOD_MAX, 120, 23, 8000.0).into();
        let mut output = [0; FRAME];
        let scale = stretch(&input, operation, PERIOD_MAX, 40.0, &mut output).unwrap();
        let (consumed, period) = (scale.consumed, scale.change);
        assert!((period as isize - 120).abs() < 5);
        assert_eq!(
            consumed,
            if operation == Stretch::Expand { FRAME - period } else { FRAME + period }
        );
        assert!(correlation(|i| output[i], |i| output[i + 120], FRAME - 120) > 0.98);
        let quiet = VecDeque::from(vec![12; FRAME + PERIOD_MAX]);
        stretch(&quiet, operation, PERIOD_MAX, 40.0, &mut output).unwrap();
        assert_eq!(output, [12; FRAME]);
    }
}

#[test]
fn unvoiced_transients_are_not_spliced_without_a_safe_match() {
    let input = noise(FRAME + PERIOD_MAX, 10_000.0).into();
    let mut output = [0; FRAME];
    assert!(stretch(&input, Stretch::Accelerate, PERIOD_MAX, 40.0, &mut output).is_none());
}

#[test]
fn pitch_search_rejects_a_strong_half_period_harmonic() {
    let input: Vec<_> = (0..HISTORY)
        .map(|i| {
            let phase = i as f64 * std::f64::consts::TAU / 120.0;
            sample(1000.0 * phase.sin() + 9000.0 * (2.0 * phase).sin())
        })
        .collect();
    let c = begin(&input, 4800);
    assert!((c.period() as isize - 120).abs() <= 2, "selected period={}", c.period());
}

#[test]
fn stable_voiced_continuation_survives_a_thirty_ms_gap_without_premature_muting() {
    let mut c = begin(&sine(HISTORY, 120, 23, 10_000.0), 4800);
    let out: Vec<_> = (0..960).map(|_| c.next()).collect();
    let level = rms(|i| out[600 + i], 240);
    assert!(level > 4500.0, "voiced RMS during 25–35ms gap={level}");
    assert!(correlation(|i| out[600 + i], |i| out[720 + i], 120) > 0.98);
}

#[test]
fn a_quiet_phoneme_does_not_force_the_returning_onset_to_near_silence() {
    let mut c = begin(&sine(HISTORY, 120, 23, 1500.0), 4800);
    for _ in 0..720 {
        c.next();
    }
    let input = sine(720, 120, 17, 12000.0);
    let mut recovery = Recovery::default();
    recovery.start(&c, &input.iter().copied().collect(), true);
    let out: Vec<_> = input.iter().map(|&s| recovery.apply(s)).collect();
    assert!(rms(|i| out[240 + i], 120) > 6000.0);
    assert_eq!(&out[480..], &input[480..]);
}

#[test]
fn preemptive_expansion_uses_history_with_only_twenty_ms_of_new_audio() {
    let history: VecDeque<_> = sine(HISTORY, 120, 23, 8000.0).into();
    let input: VecDeque<_> = sine(FRAME, 120, HISTORY + 23, 8000.0).into();
    let mut output = [0; FRAME];
    let scale = preemptive(&history, &input, PERIOD_MAX, 40.0, &mut output).unwrap();
    let (consumed, inserted) = (scale.consumed, scale.change);
    assert_eq!(inserted, 120);
    assert_eq!(consumed + inserted, FRAME);
    assert!(correlation(|i| output[i], |i| output[i + 120], FRAME - 120) > 0.99);
    assert_eq!(output[0], input[0]);
    assert!(output.windows(2).all(|p| (i32::from(p[1]) - i32::from(p[0])).abs() < 600));
    // A voiced history is not permission to duplicate a returning consonant or new onset.
    let transient: VecDeque<_> = noise(FRAME, 12000.0).into();
    assert!(preemptive(&history, &transient, PERIOD_MAX, 40.0, &mut output).is_none());
    assert!(preemptive(&history, &VecDeque::new(), PERIOD_MAX, 40.0, &mut output).is_none());
}

#[test]
fn fast_acceleration_removes_multiple_periods_without_changing_tone_pitch() {
    let input: VecDeque<_> = sine(FRAME + PERIOD_MAX, 120, 23, 8000.0).into();
    let mut normal = [0; FRAME];
    let mut fast = [0; FRAME];
    let regular =
        stretch(&input, Stretch::Accelerate, PERIOD_MAX, 40.0, &mut normal).unwrap().change;
    let scale = stretch(&input, Stretch::FastAccelerate, PERIOD_MAX, 40.0, &mut fast).unwrap();
    let (consumed, removed) = (scale.consumed, scale.change);
    assert_eq!(regular, 120);
    assert_eq!(removed, 360);
    assert_eq!(consumed, FRAME + removed);
    assert!(correlation(|i| fast[i], |i| fast[i + 120], FRAME - 120) > 0.99);
    let transient = noise(FRAME + PERIOD_MAX, 10000.0).into();
    assert!(stretch(&transient, Stretch::FastAccelerate, PERIOD_MAX, 40.0, &mut fast).is_none());
}

#[test]
fn recovery_never_adds_alignment_delay_for_an_unvoiced_onset_or_eof() {
    let mut c = begin(&sine(HISTORY, 120, 7, 12000.0), 2400);
    for _ in 0..480 {
        c.next();
    }
    for input in [noise(720, 12000.0), vec![0; 720]] {
        let mut recovery = Recovery::default();
        recovery.start(&c, &input.iter().copied().collect(), true);
        assert!(recovery.next_prefix().is_none());
    }
    let voiced = sine(720, 120, 60, 12000.0).into();
    let mut recovery = Recovery::default();
    recovery.start(&c, &voiced, false);
    assert!(recovery.next_prefix().is_none());
    recovery.start(&c, &voiced, true);
    assert!(recovery.next_prefix().is_some());
    let remaining = std::iter::from_fn(|| recovery.next_prefix()).count();
    assert!(remaining < c.period());
}

#[test]
fn learned_room_tone_can_be_time_scaled_without_treating_it_as_voiced_speech() {
    let background = noise(24000, 200.0);
    let c = begin(&background, 2400);
    assert!(c.passive_level() > 200.0);
    let input: VecDeque<_> = noise(FRAME + PERIOD_MAX, 200.0).into();
    let mut output = [0; FRAME];
    assert!(stretch(&input, Stretch::Accelerate, PERIOD_MAX, 40.0, &mut output).is_none());
    assert!(
        stretch(&input, Stretch::Accelerate, PERIOD_MAX, c.passive_level(), &mut output).is_some()
    );
    assert!(output.iter().all(|s| i32::from(*s).abs() <= 200));
    let speech_only = begin(&noise(24000, 12000.0), 2400);
    assert_eq!(speech_only.passive_level(), 40.0);
}

#[test]
fn scaling_prefers_a_pause_and_preserves_the_word_attack_before_it() {
    let mut input = sine(FRAME + PERIOD_MAX, 120, 23, 12000.0);
    input[300..600].fill(0);
    let queue = input.iter().copied().collect();
    let mut out = [0; FRAME];
    let scale = stretch(&queue, Stretch::Expand, PERIOD_MAX, 40.0, &mut out).unwrap();
    assert!(scale.passive);
    assert_eq!(&out[..300], &input[..300]);
    assert_eq!(out[FRAME - 1], 0);
    assert!(scale.change >= PERIOD_MIN);
}

#[test]
fn time_scaling_defers_a_sharp_consonant_onset() {
    let mut input = vec![0; FRAME + PERIOD_MAX];
    let onset = noise(FRAME + PERIOD_MAX - 300, 12000.0);
    input[300..].copy_from_slice(&onset);
    let queue = input.into();
    let mut out = [0; FRAME];
    assert!(stretch(&queue, Stretch::Expand, PERIOD_MAX, 40.0, &mut out).is_none());
}
