use super::*;

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}

fn plain() -> Playout {
    Playout::new(Config {
        minimum_delay: Duration::ZERO,
        maximum_delay: ms(100),
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

fn account(p: &Playout, render: Render) {
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
        p.push(ms(0), &source[offset..offset + length]).unwrap();
        offset += length;
    }
    assert_eq!(offset, source.len());
    let mut rendered = Vec::new();
    let mut block = [0; 137];
    for tick in 0..source.len() / block.len() {
        let r = p.render(ms(tick as u64 * 6), &mut block).unwrap();
        rendered.extend_from_slice(&block);
        account(&p, r);
    }
    assert_eq!(rendered, source[..rendered.len()]);
    assert_eq!(p.statistics().smallest_chunk, 1);
    assert_eq!(p.statistics().largest_chunk, 16_675);
}

#[test]
fn startup_delay_is_time_not_a_chunk_size_threshold() {
    let mut p = Playout::new(Config { maximum_stretch: 0.0, ..Config::default() }).unwrap();
    p.push(ms(0), &[4000; 2400]).unwrap();
    let mut output = [9; 240];
    for now in [0, 10] {
        p.render(ms(now), &mut output).unwrap();
        assert_eq!(output, [0; 240]);
    }
    p.render(ms(20), &mut output).unwrap();
    assert_eq!(output, [4000; 240]);
}

#[test]
fn chunk_size_can_change_during_a_stream() {
    let mut p = plain();
    let mut output = [0; 240];
    let mut source = 0;
    let mut consumed = 0;
    for (now, length) in [(0, 2400), (100, 480), (120, 1920), (200, 2400)] {
        p.push(ms(now), &voice(length, source)).unwrap();
        source += length;
        for at in (now..now + length as u64 / 24).step_by(10) {
            let r = p.render(ms(at), &mut output).unwrap();
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
    p.push(ms(0), &voice(2400, 0)).unwrap();
    let mut output = [0; 240];
    for at in (0..120).step_by(10) {
        let r = p.render(ms(at), &mut output).unwrap();
        account(&p, r);
    }
    assert_eq!(p.statistics().concealed, 480);
    p.push(ms(120), &voice(2400, 2400)).unwrap();
    let r = p.render(ms(120), &mut output).unwrap();
    account(&p, r);
    assert_eq!(p.statistics().underruns, 1);
    assert_eq!(p.statistics().source_received, 4800);
    assert_eq!(p.statistics().source_consumed, 2640);
}

#[test]
fn concealment_and_recovery_bound_discontinuities_and_fade_out() {
    let mut p = plain();
    p.push(ms(0), &voice(2400, 23)).unwrap();
    let mut output = [0; 240];
    let mut last = 0_i16;
    for at in (0..200).step_by(10) {
        if at == 120 {
            p.push(ms(at), &voice(2400, 61)).unwrap();
        }
        p.render(ms(at), &mut output).unwrap();
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
        p.render(ms(at), &mut output).unwrap();
    }
    assert_eq!(output, [0; 240]);
}

#[test]
fn unvoiced_outage_and_disabled_plc_ramp_to_silence() {
    let mut p = Playout::new(Config {
        minimum_delay: Duration::ZERO,
        maximum_concealment: Duration::ZERO,
        maximum_stretch: 0.0,
        ..Config::default()
    })
    .unwrap();
    p.push(ms(0), &[12_000; 240]).unwrap();
    let mut output = [0; 240];
    p.render(ms(0), &mut output).unwrap();
    p.render(ms(10), &mut output).unwrap();
    assert_eq!(output[0], 12_000);
    assert!(output[..120].windows(2).all(|pair| pair[1] <= pair[0]));
    assert_eq!(output[119], 0);
    assert_eq!(output[120..], [0; 120]);
}

#[test]
fn a_reserve_absorbs_short_delivery_stalls() {
    let mut p = Playout::new(Config { maximum_stretch: 0.0, ..Config::default() }).unwrap();
    p.push(ms(0), &voice(2400, 0)).unwrap();
    let mut output = [0; 240];
    for at in (0..220).step_by(10) {
        if at == 120 {
            p.push(ms(at), &voice(2400, 2400)).unwrap();
        }
        p.render(ms(at), &mut output).unwrap();
    }
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn queue_overflow_and_reversed_time_are_atomic() {
    let mut p = plain();
    p.push(ms(10), &[4; 240]).unwrap();
    let before = p.statistics();
    assert_eq!(p.push(ms(9), &[5]), Err(PlayoutError::TimeReversed));
    assert!(matches!(p.push(ms(10), &[0; 48_000]), Err(PlayoutError::Full { .. })));
    assert_eq!(p.statistics(), before);
    p.push(ms(10), &[5]).unwrap();
    let mut output = [0; 240];
    p.render(ms(10), &mut output).unwrap();
    assert_eq!(output, [4; 240]);
}

#[test]
fn finish_drains_without_inventing_speech_and_reset_clears_epoch() {
    let mut p = plain();
    p.push(ms(0), &[4000; 333]).unwrap();
    p.finish();
    assert_eq!(p.push(ms(0), &[0]), Err(PlayoutError::Finished));
    let mut output = [0; 240];
    assert!(!p.render(ms(0), &mut output).unwrap().drained);
    let r = p.render(ms(10), &mut output).unwrap();
    assert!(r.drained);
    assert_eq!(&output[93..], &[0; 147]);
    assert_eq!(output[92], 0);
    account(&p, r);
    assert_eq!(p.statistics().concealed, 0);
    p.reset();
    p.push(ms(0), &[900; 240]).unwrap();
    p.render(ms(0), &mut output).unwrap();
    assert_eq!(output, [900; 240]);
    assert_eq!(p.statistics().source_received, 240);
}

#[test]
fn voiced_time_stretching_accounts_for_every_source_sample() {
    let mut p = Playout::new(Config {
        minimum_delay: Duration::ZERO,
        maximum_stretch: 0.05,
        ..Config::default()
    })
    .unwrap();
    let mut output = [0; 240];
    // Forty-millisecond deliveries leave enough lookahead to build the initial target.
    for at in (0..500).step_by(10) {
        if at % 40 == 0 {
            p.push(ms(at), &voice(960, at as usize * 24)).unwrap();
        }
        let report = p.render(ms(at), &mut output).unwrap();
        account(&p, report);
    }
    assert!(p.statistics().stretched > 0);
    p.reset();
    p.push(ms(0), &voice(12_000, 0)).unwrap();
    // Delay pulling an already queued signal: this is real playout delay to remove.
    for at in (100..310).step_by(10) {
        let report = p.render(ms(at), &mut output).unwrap();
        account(&p, report);
    }
    assert!(p.statistics().accelerated > 0);
}

#[test]
fn deterministic_stalls_bursts_and_clock_drift_remain_bounded() {
    fn replay() -> (Vec<i16>, Statistics) {
        let mut p = Playout::new(Config::default()).unwrap();
        let mut output = [0; 240];
        let mut all = Vec::new();
        let mut source = 0;
        let mut next = 0;
        let mut seed = 734_u32;
        for at in (0..30_000).step_by(10) {
            if at >= next {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let count = [480, 960, 2400, 1200][seed as usize % 4];
                p.push(ms(at), &voice(count, source)).unwrap();
                source += count;
                next += count as u64 / 24 + u64::from(seed.is_multiple_of(19)) * 30;
            }
            let r = p.render(ms(at), &mut output).unwrap();
            account(&p, r);
            assert!(r.queued_source <= 48_000);
            assert!(r.pending_output <= FRAME);
            assert!(r.target_delay <= ms(100));
            all.extend_from_slice(&output);
        }
        (all, p.statistics())
    }
    assert_eq!(replay(), replay());
}

#[test]
fn invalid_configuration_and_render_leave_state_unchanged() {
    for config in [
        Config { maximum_stretch: f32::NAN, ..Config::default() },
        Config { minimum_delay: ms(101), ..Config::default() },
        Config { capacity: ms(1), ..Config::default() },
    ] {
        assert!(matches!(Playout::new(config), Err(PlayoutError::Configuration)));
    }
    let mut p = plain();
    assert_eq!(p.render(ms(100), &mut []), Err(PlayoutError::OutputSize));
    p.push(ms(0), &[0]).unwrap();
}

#[test]
fn a_permanent_delivery_shift_is_not_counted_twice_as_buffer_delay() {
    let mut p = Playout::new(Config { maximum_delay: ms(60), ..Config::default() }).unwrap();
    let mut output = [0; 240];
    let mut source = 0;
    for at in (0..5000_u64).step_by(10) {
        if at < 1000 && at.is_multiple_of(100) || at >= 1040 && (at - 40).is_multiple_of(100) {
            p.push(ms(at), &voice(2400, source)).unwrap();
            source += 2400;
        }
        let r = p.render(ms(at), &mut output).unwrap();
        account(&p, r);
    }
    let s = p.statistics();
    assert!(s.rendered - s.source_consumed <= 24 * 70);
    assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
}

#[test]
fn deterministic_pcm_is_unmodified_without_jitter_or_stretch() {
    let source = voice(192_000, 23);
    let mut p = plain();
    let mut output = [0; 240];
    for (tick, expected) in source.as_chunks::<240>().0.iter().enumerate() {
        let at = ms(tick as u64 * 10);
        if tick.is_multiple_of(10) {
            p.push(at, &source[tick * 240..tick * 240 + 2400]).unwrap();
        }
        let r = p.render(at, &mut output).unwrap();
        assert_eq!(output.as_slice(), expected);
        account(&p, r);
    }
    assert_eq!(p.statistics().underruns, 0);
}

#[test]
fn render_sizes_do_not_change_when_pcm_runs_out() {
    for block in [1, 120, 240, 480] {
        let mut p = plain();
        p.push(ms(0), &voice(2400, 0)).unwrap();
        let mut output = vec![0; block];
        for tick in 0..(4800 / block) {
            let r = p
                .render(
                    Duration::from_nanos(tick as u64 * block as u64 * 1_000_000_000 / 24_000),
                    &mut output,
                )
                .unwrap();
            account(&p, r);
        }
        assert_eq!(p.statistics().source_consumed, 2400);
        assert_eq!(p.statistics().concealed, 2400);
        assert_eq!(p.statistics().underruns, 1);
    }
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
            p.push(ms(arrival), &source[start..start + 2400]).unwrap();
            baseline.extend(source[start..start + 2400].iter().copied());
            next_chunk += 1;
        }
        let report = p.render(ms(at), &mut block).unwrap();
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
            p.push(ms(at), &input).unwrap();
            received += input.len();
            last_arrival = at;
        }
        let report = p.render(ms(at), &mut output).unwrap();
        account(&p, report);
        assert!(output.iter().all(|s| i32::from(*s).abs() < 2000));
    }
    p.finish();
    let mut report = p.render(ms(5000.max(last_arrival)), &mut output).unwrap();
    for at in (5010..6000).step_by(10) {
        report = p.render(ms(at), &mut output).unwrap();
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
    p.push(ms(0), &voice(2400, 23)).unwrap();
    for at in (0..150).step_by(10) {
        p.render(ms(at), &mut out).unwrap();
    }
    p.push(ms(150), &voice(2400, 61)).unwrap();
    // Enter a partially completed recovery, then cancel it.
    p.render(ms(150), &mut out[..1]).unwrap();
    p.reset();
    p.push(ms(0), &[1234; 2400]).unwrap();
    p.render(ms(0), &mut out).unwrap();
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
            if [0, 2400, 5280, 8400, 12000].contains(&position) {
                p.push(now, &voice(2400, position + 23)).unwrap();
            }
            let report = p.render(now, &mut output).unwrap();
            account(&p, report);
            rendered.extend_from_slice(&output);
        }
        (rendered, p.statistics())
    }
    let reference = run(1);
    assert_eq!(reference, run(120));
    assert_eq!(reference, run(240));
}

#[test]
fn small_provider_chunks_build_reserve_before_a_later_delivery_stall() {
    let mut protected = Playout::new(Config {
        minimum_delay: Duration::ZERO,
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
            protected.push(ms(arrival), &input).unwrap();
            untreated.push(ms(arrival), &input).unwrap();
            chunk += 1;
        }
        let report = protected.render(ms(at), &mut output).unwrap();
        account(&protected, report);
        untreated.render(ms(at), &mut output).unwrap();
    }
    let s = protected.statistics();
    assert!(s.stretched >= 24 * 20, "{s:?}");
    assert_eq!(s.underruns, 0);
    assert_eq!(untreated.statistics().underruns, 1);
    assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
}

#[test]
fn severe_playout_backlog_bypasses_cooldown_but_not_the_rate_budget() {
    let mut p = Playout::new(Config::default()).unwrap();
    let mut output = [0; 240];
    for at in (0..1000_u64).step_by(10) {
        if at.is_multiple_of(100) {
            p.push(ms(at), &voice(2400, at as usize * 24)).unwrap();
        }
        p.render(ms(at), &mut output).unwrap();
    }
    // The consumer stops rendering while delivery continues: a real 1.5s backlog, not a
    // delayed packet whose arrival envelope should legitimately retain extra reserve.
    for at in (1000..2500_u64).step_by(100) {
        p.push(ms(at), &voice(2400, at as usize * 24)).unwrap();
    }
    p.stretch_after = u64::MAX;
    for at in (2500..2580).step_by(10) {
        let report = p.render(ms(at), &mut output).unwrap();
        account(&p, report);
        let s = p.statistics();
        assert!(s.stretched + s.accelerated <= (s.rendered as f64 * 0.03) as u64);
    }
    assert!(p.statistics().fast_accelerated >= 720, "{:?}", p.statistics());
}

#[test]
fn an_alignment_prefix_drains_with_eof_without_losing_source_or_accounting() {
    let mut p = plain();
    let mut output = [0; 240];
    p.push(ms(0), &voice(2400, 7)).unwrap();
    for at in (0..120).step_by(10) {
        p.render(ms(at), &mut output).unwrap();
    }
    p.push(ms(120), &voice(2400, 60)).unwrap();
    let report = p.render(ms(120), &mut output[..1]).unwrap();
    account(&p, report);
    assert_eq!(p.statistics().merge_inserted, 1);
    assert_eq!(p.statistics().source_consumed, 2400);
    p.finish();
    let mut report = p.render(ms(120), &mut output[1..]).unwrap();
    account(&p, report);
    for at in (130..300).step_by(10) {
        report = p.render(ms(at), &mut output).unwrap();
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
        let mut p = plain();
        let mut output = [0; 240];
        p.push(ms(0), &voice(2400, 0)).unwrap();
        for at in (0..300).step_by(10) {
            p.render(ms(at), &mut output).unwrap();
        }
        p.push(ms(300), &voice(480, 2400)).unwrap();
        let report = p.render(ms(300), &mut output).unwrap();
        account(&p, report);
        assert_eq!(p.statistics().source_consumed, 2400);
        assert_eq!(report.queued_source, 480);
        if finish {
            p.finish();
        } else {
            p.push(ms(310), &voice(480, 2880)).unwrap();
        }
        let report = p.render(ms(310), &mut output).unwrap();
        account(&p, report);
        assert!(p.statistics().source_consumed > 2400);
        if finish {
            assert!(p.render(ms(320), &mut output).unwrap().drained);
        }
    }
}
