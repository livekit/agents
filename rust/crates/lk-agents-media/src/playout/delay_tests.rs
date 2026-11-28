use super::*;

#[test]
fn packetization_is_not_a_reason_to_accelerate() {
    let mut delay = Delay::default();
    for packet in 0..30_u64 {
        let now = Duration::from_millis(packet * 100);
        delay.update(now, packet * 2400, 2400);
        assert_eq!(delay.max_delay_ms(), 0);
        let playout = delay.playout_delay(now + Duration::from_millis(20), packet * 2400);
        assert_eq!(playout, 480);
    }
}

#[test]
fn late_packet_envelope_prevents_immediate_catchup_into_another_gap() {
    let mut delay = Delay::default();
    for packet in 0..10_u64 {
        delay.update(Duration::from_millis(packet * 100), packet * 2400, 2400);
    }
    delay.update(Duration::from_millis(1300), 24000, 2400);
    assert_eq!(delay.max_delay_ms(), 300);
    let actual = delay.playout_delay(Duration::from_millis(1300), 24000);
    assert!(actual < (100 + delay.max_delay_ms() + 20) * 24);
}

#[test]
fn tiny_chunks_do_not_exhaust_the_time_window_or_allocate() {
    let mut delay = Delay::default();
    for sample in 0..24_000_u64 {
        delay.update(Duration::from_micros(sample * 1_000_000 / 24_000), sample, 1);
    }
    // Microsecond arrival timestamps quantize by at most one 24kHz sample.
    assert_eq!(delay.max_delay_ms(), 1);
    assert_eq!(delay.playout_delay(Duration::from_secs(1), 24_000), 1);
}

#[test]
fn sub_millisecond_reserve_is_not_rounded_up_to_the_target() {
    let mut delay = Delay::default();
    delay.update(Duration::ZERO, 0, 480);
    // 460 samples is below a 20ms (480 sample) target, despite straddling ms boundaries.
    let now = Duration::from_millis(1000);
    assert_eq!(delay.playout_delay(now, 24000 - 460), 460);
    delay.update(Duration::from_nanos(1000 * 1_000_000_000 / 24000), 1000, 1);
    assert!(delay.max_delay_ms() <= 1);
}

#[test]
fn clean_startup_relaxes_gradually_and_bad_startup_is_remembered() {
    let mut delay = Delay::default();
    for expected in [60, 40, 20] {
        delay.update(Duration::ZERO, 0, 2400);
        delay.update(Duration::from_millis(100), 2400, 2400);
        delay.end_utterance();
        assert_eq!(
            delay.startup(Duration::from_millis(80), Duration::ZERO, Duration::from_millis(100)),
            expected * 24
        );
        delay.begin_utterance();
    }
    delay.update(Duration::ZERO, 0, 2400);
    delay.update(Duration::from_millis(400), 2400, 2400);
    delay.end_utterance();
    assert_eq!(
        delay.startup(Duration::from_millis(80), Duration::ZERO, Duration::from_millis(100)),
        2400
    );
    delay.begin_utterance();
    delay.update(Duration::from_secs(3600), 0, 2400);
    delay.update(Duration::from_millis(3_600_100), 2400, 2400);
    assert_eq!(delay.max_delay_ms(), 0);
    assert!(
        delay.startup(Duration::from_millis(80), Duration::ZERO, Duration::from_millis(100))
            <= 2400
    );
}

#[test]
fn a_single_finished_chunk_is_not_evidence_of_good_packet_timing() {
    let mut delay = Delay::default();
    delay.update(Duration::ZERO, 0, 12000);
    delay.end_utterance();
    delay.begin_utterance();
    assert_eq!(delay.target(Duration::ZERO, Duration::from_millis(100)), 1920);
    assert_eq!(
        delay.startup(Duration::from_millis(80), Duration::ZERO, Duration::from_millis(100)),
        1920
    );
}
