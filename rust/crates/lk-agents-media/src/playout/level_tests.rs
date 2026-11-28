use super::*;

#[test]
fn a_short_queue_spike_does_not_become_a_time_scale_command() {
    let mut level = Level::default();
    assert_eq!(level.update(2400, 100, 0), 2400);
    let filtered = level.update(12000, 100, 240);
    assert!(filtered < 2600);
    assert!(level.update(2400, 100, 480) < filtered);
}

#[test]
fn successful_stretch_is_removed_from_filter_memory_immediately() {
    let mut level = Level::default();
    level.update(2400, 100, 0);
    level.correct(360);
    assert_eq!(level.update(2040, 100, 0), 2040);
    level.correct(-120);
    assert_eq!(level.update(2160, 100, 0), 2160);
    level.correct(10000);
    assert_eq!(level.update(0, 100, 0), 0);
}

#[test]
fn filter_cadence_does_not_depend_on_render_call_size() {
    fn run(block: u64) -> usize {
        let mut level = Level::default();
        level.update(2400, 100, 0);
        for rendered in (block..=24000).step_by(block as usize) {
            level.update(4800, 100, rendered);
        }
        level.update(4800, 100, 24000)
    }
    assert_eq!(run(1), run(120));
    assert_eq!(run(120), run(480));
    assert_eq!(Level::default().update(usize::MAX, 100, 0), 60 * SAMPLE_RATE as usize);
}
