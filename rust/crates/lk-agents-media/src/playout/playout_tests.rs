use super::*;

fn push<Mode>(p: &mut Playout<Mode>, now: Duration, pcm: &[i16]) {
    let (result, bytes) = crate::test_alloc::allocated(|| p.push(now, pcm));
    assert_eq!(bytes, 0, "push allocated on the audio path");
    result.unwrap();
}

fn render<Mode>(p: &mut Playout<Mode>, now: Duration, output: &mut [i16]) -> Render {
    let (result, bytes) = crate::test_alloc::allocated(|| p.render(now, output));
    assert_eq!(bytes, 0, "render allocated on the audio path");
    result.unwrap()
}

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}

fn plain() -> Playout {
    Playout::continuous(Config {
        minimum_delay: Duration::ZERO,
        startup_delay: Duration::ZERO,
        maximum_delay: Duration::ZERO,
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap()
}

fn voice(length: usize, offset: usize) -> Vec<i16> {
    (offset..offset + length)
        .map(|i| ((i as f64 * std::f64::consts::TAU / 120.0).sin() * 12_000.0) as i16)
        .collect()
}

fn account<Mode>(p: &Playout<Mode>, render: Render) {
    let s = p.statistics();
    assert_eq!(s.source_received, s.source_consumed + render.queued_source as u64);
    assert_eq!(
        s.rendered as i128,
        s.source_consumed as i128 + s.stretched as i128 - s.accelerated as i128
            + s.concealed as i128
            + s.silent as i128
            - render.pending_output as i128
    );
}

#[test]
fn arbitrary_chunk_boundaries_preserve_clean_samples() {
    let source = voice(24_000, 0);
    let mut p = plain();
    let mut offset = 0;
    for length in [1, 17, 2400, 7, 4800, 100, 16_675] {
        push(&mut p, ms(0), &source[offset..offset + length]);
        offset += length;
    }
    assert_eq!(offset, source.len());
    let mut rendered = Vec::new();
    let mut block = [0; 137];
    for tick in 0..source.len() / block.len() {
        let r = render(&mut p, ms(tick as u64 * 6), &mut block);
        rendered.extend_from_slice(&block);
        account(&p, r);
    }
    assert_eq!(rendered, source[..rendered.len()]);
    assert_eq!(p.statistics().smallest_chunk, 1);
    assert_eq!(p.statistics().largest_chunk, 16_675);
}

#[test]
fn chunk_size_can_change_during_a_stream() {
    let mut p = plain();
    let mut output = [0; 240];
    let mut source = 0;
    let mut consumed = 0;
    for (now, length) in [(0, 2400), (100, 480), (120, 1920), (200, 2400)] {
        push(&mut p, ms(now), &voice(length, source));
        source += length;
        for at in (now..now + length as u64 / 24).step_by(10) {
            let r = render(&mut p, ms(at), &mut output);
            assert_eq!(output.as_slice(), voice(240, consumed));
            consumed += 240;
            account(&p, r);
        }
    }
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn twenty_ms_lateness_does_not_conceal_a_hundred_ms_chunk() {
    let mut p = plain();
    push(&mut p, ms(0), &voice(2400, 0));
    let mut output = [0; 240];
    for at in (0..120).step_by(10) {
        let r = render(&mut p, ms(at), &mut output);
        account(&p, r);
    }
    assert_eq!(p.statistics().concealed, 480);
    push(&mut p, ms(120), &voice(2400, 2400));
    let r = render(&mut p, ms(120), &mut output);
    account(&p, r);
    assert_eq!(p.statistics().underruns, 1);
    assert_eq!(p.statistics().source_received, 4800);
    assert_eq!(p.statistics().source_consumed, 2640);
}

#[test]
fn concealment_and_recovery_bound_discontinuities_and_fade_out() {
    let mut p = plain();
    push(&mut p, ms(0), &voice(2400, 23));
    let mut output = [0; 240];
    let mut last = 0_i16;
    for at in (0..200).step_by(10) {
        if at == 120 {
            push(&mut p, ms(at), &voice(2400, 61));
        }
        render(&mut p, ms(at), &mut output);
        if at >= 10 {
            for &s in &output {
                assert!((i32::from(s) - i32::from(last)).abs() < 2500);
                last = s;
            }
        } else {
            last = output[239];
        }
    }
    for at in (200..500).step_by(10) {
        render(&mut p, ms(at), &mut output);
    }
    assert_eq!(output, [0; 240]);
}

#[test]
fn unvoiced_outage_and_disabled_plc_ramp_to_silence() {
    let mut p = Playout::continuous(Config {
        minimum_delay: Duration::ZERO,
        startup_delay: Duration::ZERO,
        maximum_concealment: Duration::ZERO,
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap();
    push(&mut p, ms(0), &[12_000; 240]);
    let mut output = [0; 240];
    render(&mut p, ms(0), &mut output);
    render(&mut p, ms(10), &mut output);
    assert_eq!(output[0], 12_000);
    assert!(output[..120].windows(2).all(|pair| pair[1] <= pair[0]));
    assert_eq!(output[119], 0);
    assert_eq!(output[120..], [0; 120]);
}

#[test]
fn a_reserve_absorbs_short_delivery_stalls() {
    let mut p = Playout::continuous(Config {
        startup_delay: ms(20),
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap();
    push(&mut p, ms(0), &voice(2400, 0));
    let mut output = [0; 240];
    for at in (0..220).step_by(10) {
        if at == 120 {
            push(&mut p, ms(at), &voice(2400, 2400));
        }
        render(&mut p, ms(at), &mut output);
    }
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn queue_overflow_and_reversed_time_are_atomic() {
    let mut p = plain();
    push(&mut p, ms(10), &[4; 240]);
    let before = p.statistics();
    assert_eq!(p.push(ms(9), &[5]), Err(PlayoutError::TimeReversed));
    assert!(matches!(p.push(ms(10), &[0; 48_000]), Err(PlayoutError::Full { .. })));
    let mut rejected = [123; 240];
    assert_eq!(p.render(ms(9), &mut rejected), Err(PlayoutError::TimeReversed));
    assert_eq!(rejected, [123; 240]);
    assert_eq!(p.statistics(), before);
    push(&mut p, ms(10), &[5]);
    let mut output = [0; 240];
    render(&mut p, ms(10), &mut output);
    assert_eq!(output, [4; 240]);
}

#[test]
fn finish_drains_without_inventing_speech_and_reset_clears_epoch() {
    let mut p = plain();
    push(&mut p, ms(0), &[4000; 333]);
    p.finish();
    assert_eq!(p.push(ms(0), &[0]), Err(PlayoutError::Finished));
    let mut output = [0; 240];
    assert!(!render(&mut p, ms(0), &mut output).drained);
    let r = render(&mut p, ms(10), &mut output);
    assert!(r.drained);
    assert_eq!(&output[93..], &[0; 147]);
    assert_eq!(output[92], 0);
    account(&p, r);
    assert_eq!(p.statistics().concealed, 0);
    p.reset();
    push(&mut p, ms(0), &[900; 240]);
    render(&mut p, ms(0), &mut output);
    assert_eq!(output, [900; 240]);
    assert_eq!(p.statistics().source_received, 240);
}

#[test]
fn voiced_time_stretching_accounts_for_every_source_sample() {
    let mut p = Playout::continuous(Config {
        minimum_delay: Duration::ZERO,
        startup_delay: Duration::ZERO,
        maximum_stretch: 0.05,
        ..Config::default()
    })
    .unwrap();
    let mut output = [0; 240];
    // Forty-millisecond deliveries leave enough lookahead to build the initial target.
    for at in (0..500).step_by(10) {
        if at % 40 == 0 {
            push(&mut p, ms(at), &voice(960, at as usize * 24));
        }
        let report = render(&mut p, ms(at), &mut output);
        account(&p, report);
    }
    assert!(p.statistics().stretched > 0);
    p.reset();
    push(&mut p, ms(0), &voice(12_000, 0));
    // Delay pulling an already queued signal: this is real playout delay to remove.
    for at in (100..310).step_by(10) {
        let report = render(&mut p, ms(at), &mut output);
        account(&p, report);
    }
    assert!(p.statistics().accelerated > 0);
}

#[test]
fn stalls_bursts_and_clock_drift_remain_bounded() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut output = [0; 240];
    let mut source = 0;
    let mut next = 0;
    let mut seed = 734_u32;
    for at in (0..30_000).step_by(10) {
        if at >= next {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let count = [480, 960, 2400, 1200][seed as usize % 4];
            push(&mut p, ms(at), &voice(count, source));
            source += count;
            next += count as u64 / 24 + u64::from(seed.is_multiple_of(19)) * 30;
        }
        let r = render(&mut p, ms(at), &mut output);
        account(&p, r);
        assert!(r.queued_source <= 48_000);
        assert!(r.pending_output <= FRAME);
        assert!(r.target_delay <= ms(100));
        let s = p.statistics();
        assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
    }
    p.finish();
    let mut drained = false;
    for at in (30_000..32_000).step_by(10) {
        let r = render(&mut p, ms(at), &mut output);
        account(&p, r);
        drained = r.drained;
    }
    assert!(drained);
    assert_eq!(p.statistics().source_consumed, source as u64);
}

#[test]
fn invalid_configuration_and_render_leave_state_unchanged() {
    for config in [
        Config { maximum_stretch: f32::NAN, ..Config::default() },
        Config { minimum_delay: ms(101), ..Config::default() },
        Config { capacity: ms(1), ..Config::default() },
    ] {
        assert!(matches!(Playout::continuous(config), Err(PlayoutError::Configuration)));
    }
    let mut p = plain();
    assert_eq!(p.render(ms(100), &mut []), Err(PlayoutError::OutputSize));
    let mut oversized = [123; FRAME + 1];
    assert_eq!(p.render(ms(100), &mut oversized), Err(PlayoutError::OutputSize));
    assert_eq!(oversized, [123; FRAME + 1]);
    assert_eq!(p.statistics(), Statistics::default());
    push(&mut p, ms(0), &[0]);
}

#[test]
fn a_permanent_delivery_shift_is_not_counted_twice_as_buffer_delay() {
    let mut p = Playout::continuous(Config { maximum_delay: ms(60), ..Config::default() }).unwrap();
    let mut output = [0; 240];
    let mut source = 0;
    for at in (0..5000_u64).step_by(10) {
        if at < 1000 && at.is_multiple_of(100) || at >= 1040 && (at - 40).is_multiple_of(100) {
            push(&mut p, ms(at), &voice(2400, source));
            source += 2400;
        }
        let r = render(&mut p, ms(at), &mut output);
        account(&p, r);
    }
    let s = p.statistics();
    // The network's permanent 40ms shift is separate from the learned reserve. The
    // reserve may grow to its configured 60ms target, plus at most one pitch splice.
    assert!(s.rendered - s.source_consumed <= 24 * (40 + 60) + PERIOD_MAX as u64, "{s:?}");
    assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
}

#[test]
fn periodic_pcm_short_stall_has_no_hard_silence_edges() {
    let source = voice(192_000, 23);
    let mut p = plain();
    let mut baseline = VecDeque::new();
    let mut untreated = Vec::new();
    let mut candidate = Vec::new();
    let mut block = [0; 240];
    let mut next_chunk = 0;
    for at in (0..3000_u64).step_by(10) {
        while next_chunk < 30 {
            let arrival = next_chunk * 100 + if next_chunk == 20 { 20 } else { 0 };
            if arrival > at {
                break;
            }
            let start = next_chunk as usize * 2400;
            push(&mut p, ms(arrival), &source[start..start + 2400]);
            baseline.extend(source[start..start + 2400].iter().copied());
            next_chunk += 1;
        }
        let report = render(&mut p, ms(at), &mut block);
        account(&p, report);
        candidate.extend_from_slice(&block);
        untreated.extend((0..240).map(|_| baseline.pop_front().unwrap_or(0)));
    }
    for edge in [2000 * 24, 2020 * 24] {
        let hard = (i32::from(untreated[edge]) - i32::from(untreated[edge - 1])).abs();
        let smooth = (i32::from(candidate[edge]) - i32::from(candidate[edge - 1])).abs();
        assert!(smooth < hard / 4, "hard={hard} smooth={smooth}");
    }
    assert_eq!(p.statistics().concealed, 480);
}

#[test]
fn noisy_background_repeated_short_returns_drain_every_source_sample() {
    let mut p = plain();
    let mut seed = 734_u32;
    let mut output = [0; 240];
    let mut received = 0;
    let mut last_arrival = 0;
    for at in (0..5000).step_by(10) {
        if at % 130 == 0 {
            let input: Vec<_> = (0..2400)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    (seed as i16 / 100).saturating_add(50)
                })
                .collect();
            push(&mut p, ms(at), &input);
            received += input.len();
            last_arrival = at;
        }
        let report = render(&mut p, ms(at), &mut output);
        account(&p, report);
        assert!(output.iter().all(|s| i32::from(*s).abs() < 2000));
    }
    p.finish();
    let mut report = render(&mut p, ms(5000.max(last_arrival)), &mut output);
    for at in (5010..6000).step_by(10) {
        report = render(&mut p, ms(at), &mut output);
        account(&p, report);
    }
    assert!(report.drained);
    assert_eq!(p.statistics().source_consumed, received as u64);
    assert!(p.statistics().underruns > 20);
}

#[test]
fn recovery_state_is_cleared_by_reset_and_does_not_cross_epochs() {
    let mut p = plain();
    let mut out = [0; 240];
    push(&mut p, ms(0), &voice(2400, 23));
    for at in (0..150).step_by(10) {
        render(&mut p, ms(at), &mut out);
    }
    push(&mut p, ms(150), &voice(2400, 61));
    // Enter a partially completed recovery, then cancel it.
    render(&mut p, ms(150), &mut out[..1]);
    p.reset();
    push(&mut p, ms(0), &[1234; 2400]);
    render(&mut p, ms(0), &mut out);
    assert_eq!(out, [1234; 240]);
    assert_eq!(p.statistics().concealed, 0);
}

#[test]
fn concealment_and_recovery_are_invariant_to_render_block_boundaries() {
    fn run(size: usize) -> (Vec<i16>, Statistics) {
        let mut p = plain();
        let mut output = vec![0; size];
        let mut rendered = Vec::new();
        let count = 24_000;
        for position in (0..count).step_by(size) {
            let now = Duration::from_nanos(position as u64 * 1_000_000_000 / 24_000);
            // All tested render sizes divide the arrival positions exactly.
            if [0, 2400, 5280, 8640, 12000].contains(&position) {
                push(&mut p, now, &voice(2400, position + 23));
            }
            let report = render(&mut p, now, &mut output);
            account(&p, report);
            rendered.extend_from_slice(&output);
        }
        (rendered, p.statistics())
    }
    let reference = run(1);
    assert_eq!(reference, run(120));
    assert_eq!(reference, run(240));
    assert_eq!(reference, run(480));
}

#[test]
fn small_provider_chunks_build_reserve_before_a_later_delivery_stall() {
    let mut protected = Playout::continuous(Config {
        minimum_delay: Duration::ZERO,
        startup_delay: Duration::ZERO,
        maximum_stretch: 0.03,
        ..Config::default()
    })
    .unwrap();
    let mut untreated = plain();
    let mut output = [0; 240];
    let mut chunk = 0;
    for at in (0..4000_u64).step_by(10) {
        while chunk < 200 {
            let arrival = chunk * 20 + if chunk >= 150 { 20 } else { 0 };
            if arrival > at {
                break;
            }
            let input = voice(480, chunk as usize * 480);
            push(&mut protected, ms(arrival), &input);
            push(&mut untreated, ms(arrival), &input);
            chunk += 1;
        }
        let report = render(&mut protected, ms(at), &mut output);
        account(&protected, report);
        render(&mut untreated, ms(at), &mut output);
    }
    let s = protected.statistics();
    assert!(s.stretched >= 24 * 20, "{s:?}");
    assert_eq!(s.underruns, 0);
    assert_eq!(untreated.statistics().underruns, 1);
    assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
}

#[test]
fn severe_playout_backlog_bypasses_cooldown_but_not_the_rate_budget() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut output = [0; 240];
    for at in (0..1000_u64).step_by(10) {
        if at.is_multiple_of(100) {
            push(&mut p, ms(at), &voice(2400, at as usize * 24));
        }
        render(&mut p, ms(at), &mut output);
    }
    // The consumer stops rendering while delivery continues: a real 1.5s backlog, not a
    // delayed packet whose arrival envelope should legitimately retain extra reserve.
    for at in (1000..2500_u64).step_by(100) {
        push(&mut p, ms(at), &voice(2400, at as usize * 24));
    }
    p.stretch_after = u64::MAX;
    for at in (2500..2580).step_by(10) {
        let report = render(&mut p, ms(at), &mut output);
        account(&p, report);
        let s = p.statistics();
        assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
    }
    assert!(p.statistics().fast_accelerated >= dsp::PERIOD_MIN as u64, "{:?}", p.statistics());
}

#[test]
fn an_alignment_prefix_drains_with_eof_without_losing_source_or_accounting() {
    let mut p = plain();
    let mut output = [0; 240];
    push(&mut p, ms(0), &voice(2400, 7));
    for at in (0..120).step_by(10) {
        render(&mut p, ms(at), &mut output);
    }
    push(&mut p, ms(120), &voice(2400, 60));
    let report = render(&mut p, ms(120), &mut output[..1]);
    account(&p, report);
    assert_eq!(p.statistics().merge_inserted, 1);
    assert_eq!(p.statistics().source_consumed, 2400);
    p.finish();
    let mut report = render(&mut p, ms(120), &mut output[1..]);
    account(&p, report);
    for at in (130..300).step_by(10) {
        report = render(&mut p, ms(at), &mut output);
        account(&p, report);
    }
    assert!(report.drained);
    assert_eq!(p.statistics().source_consumed, 4800);
    assert!(p.statistics().merge_inserted < PERIOD_MAX as u64);
    assert_eq!(p.statistics().underruns, 1);
}

#[test]
fn faded_recovery_rebuilds_half_the_target_reserve_but_eof_never_waits() {
    for finish in [false, true] {
        let mut p = Playout::continuous(Config {
            minimum_delay: Duration::ZERO,
            startup_delay: Duration::ZERO,
            maximum_stretch: 0.0,
            ..Config::default()
        })
        .unwrap();
        let mut output = [0; 240];
        push(&mut p, ms(0), &voice(2400, 0));
        for at in (0..300).step_by(10) {
            render(&mut p, ms(at), &mut output);
        }
        push(&mut p, ms(300), &voice(480, 2400));
        let report = render(&mut p, ms(300), &mut output);
        account(&p, report);
        assert_eq!(p.statistics().source_consumed, 2400);
        assert_eq!(report.queued_source, 480);
        if finish {
            p.finish();
        } else {
            push(&mut p, ms(310), &voice(480, 2880));
        }
        let report = render(&mut p, ms(310), &mut output);
        account(&p, report);
        assert!(p.statistics().source_consumed > 2400);
        if finish {
            assert!(render(&mut p, ms(320), &mut output).drained);
        }
    }
}

#[test]
fn a_cold_start_budget_covers_the_delayed_second_chunk() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut out = [0; 240];
    for at in (0..500_u64).step_by(10) {
        if at == 0 {
            push(&mut p, ms(at), &voice(2400, 0));
        }
        if at >= 160 && (at - 60).is_multiple_of(100) {
            push(&mut p, ms(at), &voice(2400, (at - 60) as usize * 24));
        }
        let report = render(&mut p, ms(at), &mut out);
        account(&p, report);
        assert_eq!(report.started, at >= 80);
    }
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn queued_bursts_and_known_eof_release_startup_early() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut out = [0; 240];
    push(&mut p, ms(0), &voice(2400, 23));
    for at in [0, 10, 20, 30] {
        assert!(!render(&mut p, ms(at), &mut out).started);
    }
    push(&mut p, ms(40), &voice(2400, 2423));
    let report = render(&mut p, ms(40), &mut out);
    assert!(report.started);
    assert_eq!(report.startup_delay, ms(80));
    assert_eq!(out[0], voice(1, 23)[0]);
    p.reset();
    push(&mut p, ms(0), &voice(240, 23));
    p.finish();
    assert!(render(&mut p, ms(0), &mut out).drained);
    assert_ne!(out[0], 0);
}

#[test]
fn learned_startup_protects_later_answers_without_learning_idle_time() {
    let mut p = Playout::segmented(Config {
        startup_delay: ms(20),
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap();
    let mut out = [0; 240];
    for turn in 0..10_u64 {
        if turn > 0 {
            p.begin_segment();
        }
        let base = turn * 60_000; // Long conversation pauses must not train delay estimates.
        for tick in (0..1300_u64).step_by(10) {
            if tick == 0 {
                push(&mut p, ms(base), &voice(2400, 23));
            }
            if (160..=960).contains(&tick) && (tick - 60).is_multiple_of(100) {
                push(&mut p, ms(base + tick), &voice(2400, (tick - 60) as usize * 24 + 23));
                if tick == 960 {
                    p.end_segment();
                }
            }
            let report = render(&mut p, ms(base + tick), &mut out);
            account(&p, report);
            if tick == 0 {
                assert_eq!(report.startup_observations, turn);
                assert_eq!(report.startup_quantile, (turn > 0).then_some(ms(80)));
            }
            assert!(report.startup_delay <= ms(100));
            if turn > 0 {
                assert_eq!(report.startup_delay, ms(80));
            }
        }
        if turn == 0 {
            assert_eq!(p.statistics().underruns, 1);
        } else {
            assert_eq!(p.statistics().underruns, 0, "turn={turn}");
        }
    }
    p.reset();
    push(&mut p, ms(0), &voice(2400, 23));
    assert_eq!(render(&mut p, ms(0), &mut out).startup_delay, ms(20));
}

#[test]
fn beginning_a_segment_cancels_pending_audio_and_old_phonemes() {
    let mut p = Playout::segmented(Config {
        minimum_delay: Duration::ZERO,
        startup_delay: Duration::ZERO,
        maximum_delay: Duration::ZERO,
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap();
    let mut out = [0; 240];
    push(&mut p, ms(0), &voice(2400, 23));
    for at in (0..120).step_by(10) {
        render(&mut p, ms(at), &mut out);
    }
    push(&mut p, ms(120), &voice(2400, 61));
    render(&mut p, ms(120), &mut out[..1]);
    assert!(p.recovery.active());
    p.begin_segment();
    assert_eq!(p.statistics(), Statistics::default());
    push(&mut p, ms(0), &[1234; 2400]);
    render(&mut p, ms(0), &mut out);
    assert_eq!(out, [1234; 240]);
}

#[test]
fn hundred_ms_delivery_drift_builds_reserve_instead_of_only_concealing() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut out = [0; 240];
    let mut packet = 0_u64;
    for at in (0..20_000_u64).step_by(10) {
        while packet * 101 <= at {
            push(&mut p, ms(packet * 101), &voice(2400, packet as usize * 2400));
            packet += 1;
        }
        let report = render(&mut p, ms(at), &mut out);
        account(&p, report);
    }
    let s = p.statistics();
    assert_eq!(s.underruns, 0, "{s:?}");
    assert!(s.stretched > 24 * 100, "{s:?}");
    assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
}

#[test]
fn startup_reserve_is_not_accelerated_away_before_early_delivery_is_observed() {
    let mut p =
        Playout::continuous(Config { startup_delay: ms(100), ..Config::default() }).unwrap();
    let mut out = [0; 240];
    for at in (0..350_u64).step_by(10) {
        if at == 0 {
            push(&mut p, ms(at), &voice(2400, 23));
        }
        if at == 160 {
            push(&mut p, ms(at), &voice(2400, 2423));
        }
        if at == 300 {
            push(&mut p, ms(at), &voice(2400, 4823));
        }
        render(&mut p, ms(at), &mut out);
    }
    assert_eq!(p.statistics().accelerated, 0);
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn learned_severe_startup_respects_the_budget_instead_of_promising_impossible_recovery() {
    for maximum in [100, 400] {
        let mut p =
            Playout::segmented(Config { maximum_delay: ms(maximum), ..Config::default() }).unwrap();
        let mut out = [0; 240];
        for turn in 0..2_u64 {
            if turn > 0 {
                p.begin_segment();
            }
            let base = turn * 10_000;
            let arrivals = [0, 160, 300, 460, 700, 700, 700, 700, 800, 900];
            let mut packet = 0;
            let mut selected = Duration::ZERO;
            for tick in (0..1600_u64).step_by(10) {
                while packet < arrivals.len() && arrivals[packet] <= tick {
                    push(&mut p, ms(base + arrivals[packet]), &voice(2400, packet * 2400 + 23));
                    packet += 1;
                    if packet == arrivals.len() {
                        p.end_segment();
                    }
                }
                let report = render(&mut p, ms(base + tick), &mut out);
                selected = report.startup_delay;
                account(&p, report);
            }
            if turn > 0 {
                if maximum == 400 {
                    assert_eq!(selected, ms(320));
                    assert_eq!(p.statistics().underruns, 0);
                } else {
                    assert_eq!(selected, ms(100));
                    assert!(p.statistics().underruns > 0);
                }
            }
        }
    }
}

#[test]
fn a_clean_answer_does_not_raise_a_small_configured_startup_prior() {
    let mut p = Playout::segmented(Config { startup_delay: ms(20), ..Config::default() }).unwrap();
    let mut out = [0; 240];
    for turn in 0..4_u64 {
        if turn > 0 {
            p.begin_segment();
        }
        let base = turn * 10000;
        for at in (0..700_u64).step_by(10) {
            if at < 500 && at.is_multiple_of(100) {
                push(&mut p, ms(base + at), &voice(2400, at as usize * 24));
                if at == 400 {
                    p.end_segment();
                }
            }
            let report = render(&mut p, ms(base + at), &mut out);
            assert_eq!(report.startup_delay, ms(20));
        }
    }
}

#[test]
fn construction_is_bounded_and_lifecycle_transitions_reuse_storage() {
    let (mut p, bytes) =
        crate::test_alloc::allocated(|| Playout::continuous(Config::default()).unwrap());
    assert!(bytes <= 100_000, "default controller heap budget exceeded: {bytes}");
    push(&mut p, ms(0), &voice(2400, 23));
    render(&mut p, ms(100), &mut [0; 240]);
    let mut segments = Playout::segmented(Config::default()).unwrap();
    push(&mut segments, ms(0), &voice(2400, 23));
    render(&mut segments, ms(100), &mut [0; 240]);
    let (_, bytes) = crate::test_alloc::allocated(|| {
        p.finish();
        p.reset();
        segments.end_segment();
        segments.begin_segment();
        segments.reset();
    });
    assert_eq!(bytes, 0);
}

#[test]
fn continuous_pcm_silence_and_speech_resume_without_new_startup_or_boundaries() {
    let mut p = Playout::continuous(Config::default()).unwrap();
    let mut source = voice(120_000, 23);
    source[24_000..72_000].fill(0);
    let mut block = [0; 240];
    let mut sent = 0;
    let mut chunk = 0;
    for at in (0..5000_u64).step_by(10) {
        while sent < source.len() && sent as u64 <= at * 24 {
            let size = [480, 2400, 960, 4800, 1440][chunk % 5].min(source.len() - sent);
            push(
                &mut p,
                Duration::from_nanos(sent as u64 * 1_000_000_000 / 24_000),
                &source[sent..sent + size],
            );
            sent += size;
            chunk += 1;
        }
        let report = render(&mut p, ms(at), &mut block);
        account(&p, report);
        assert!(!report.drained);
        assert_eq!(report.started, at >= 80);
        if at >= 1000 {
            assert_eq!(report.startup_observations, 1);
        }
        if (1800..2600).contains(&at) {
            assert_eq!(block, [0; 240]);
        }
    }
    p.finish();
    let mut drained = false;
    for at in (5000..6000).step_by(10) {
        let report = render(&mut p, ms(at), &mut block);
        account(&p, report);
        drained = report.drained;
    }
    assert!(drained);
    assert_eq!(p.statistics().source_consumed, source.len() as u64);
    assert_eq!(p.statistics().underruns, 0);
}
