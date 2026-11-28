use super::*;

#[test]
fn distribution_normalizes_and_remembers_repeated_stalls() {
    let mut histogram = Histogram::default();
    for _ in 0..8 {
        histogram.observe(300);
    }
    assert_eq!(histogram.histogram.iter().sum::<i64>(), PROBABILITY);
    assert_eq!(histogram.quantile_ms(), 320);
    for _ in 0..20 {
        histogram.observe(20);
    }
    assert_eq!(histogram.histogram.iter().sum::<i64>(), PROBABILITY);
    assert!(histogram.quantile_ms() >= 300);
    histogram.observe(usize::MAX);
    assert_eq!(histogram.histogram.iter().sum::<i64>(), PROBABILITY);
    assert_eq!(histogram.quantile_ms(), 2000);
}
