use super::*;

#[test]
fn histogram_normalizes_and_remembers_repeated_stalls() {
    let mut delay = Delay::default();
    for _ in 0..8 {
        delay.add(15);
    }
    assert_eq!(delay.histogram.iter().sum::<i64>(), PROBABILITY);
    assert_eq!(delay.target_ms, 320);
    for _ in 0..20 {
        delay.add(1);
    }
    assert_eq!(delay.histogram.iter().sum::<i64>(), PROBABILITY);
    assert!(delay.target_ms >= 300);
}

#[test]
fn packetization_is_not_a_reason_to_accelerate() {
    let mut delay = Delay::default();
    for packet in 0..30_u64 {
        let now = Duration::from_millis(packet * 100);
        delay.update(now, packet * 2400, 2400);
        assert_eq!(delay.max_delay_ms(), 80);
        let playout = delay.playout_delay_ms(now + Duration::from_millis(20), packet * 2400);
        assert_eq!(playout, 100);
    }
}

#[test]
fn late_packet_envelope_prevents_immediate_catchup_into_another_gap() {
    let mut delay = Delay::default();
    for packet in 0..10_u64 {
        delay.update(Duration::from_millis(packet * 100), packet * 2400, 2400);
    }
    delay.update(Duration::from_millis(1300), 24000, 2400);
    assert_eq!(delay.max_delay_ms(), 380);
    let actual = delay.playout_delay_ms(Duration::from_millis(1300), 24000);
    assert!(actual < 100 + delay.max_delay_ms() + 20);
}

#[test]
fn tiny_chunks_do_not_exhaust_the_time_window_or_allocate() {
    let mut delay = Delay::default();
    for sample in 0..24_000_u64 {
        delay.update(Duration::from_micros(sample * 1_000_000 / 24_000), sample, 1);
    }
    assert_eq!(delay.max_delay_ms(), 0);
    assert_eq!(delay.playout_delay_ms(Duration::from_secs(1), 24_000), 0);
}
