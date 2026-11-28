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
    for at in (200..400).step_by(10) {
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
    assert_eq!(output[119], 100);
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
    for expand in [false, true] {
        let mut p = Playout::new(Config {
            minimum_delay: if expand { ms(100) } else { Duration::ZERO },
            maximum_stretch: 0.05,
            ..Config::default()
        })
        .unwrap();
        p.push(ms(0), &voice(if expand { 2400 } else { 12_000 }, 0)).unwrap();
        let mut output = [0; 240];
        // Deliver a second chunk before the first has drained to give the stretcher lookahead.
        if expand {
            p.push(ms(100), &voice(2400, 2400)).unwrap();
        }
        for at in (100..310).step_by(10) {
            let r = p.render(ms(at), &mut output).unwrap();
            account(&p, r);
        }
        if expand {
            // With the next delivery withheld, the reserve is rebuilt gradually.
            assert!(p.statistics().stretched > 0);
        } else {
            assert!(p.statistics().accelerated > 0);
        }
    }
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
        assert_eq!(p.statistics().concealed, 960);
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
