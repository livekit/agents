//! Measurement aggregation per destination: counter sums and explicit-bucket histograms, read out
//! with delta or cumulative temporality.

use super::proto::common::v1::KeyValue;
use super::proto::metrics::v1 as proto;
use lk_agents_abi::telemetry::record::{InstrumentKind, KeyValues, Measurement, Number};

use super::config::Temporality;
use super::encode::key_values;

/// Series one aggregator keeps; OpenTelemetry's default cardinality limit.
pub(crate) const MAX_SERIES: usize = 2000;

enum Point {
    Sum(Number),
    /// `buckets` has one count per bound, then the overflow bucket.
    Histogram {
        count: u64,
        sum: f64,
        min: f64,
        max: f64,
        buckets: Vec<u64>,
    },
}

struct Instrument {
    name: &'static str,
    kind: InstrumentKind,
    unit: &'static str,
    bounds: &'static [f64],
    series: Vec<(KeyValues, Point)>,
}

pub(crate) struct Aggregator {
    temporality: Temporality,
    /// The start of the current collection interval (delta) or of the aggregation (cumulative).
    start_unix_nanos: u64,
    instruments: Vec<Instrument>,
    series: usize,
}

impl Aggregator {
    pub fn new(temporality: Temporality, now_unix_nanos: u64) -> Self {
        Aggregator {
            temporality,
            start_unix_nanos: now_unix_nanos,
            instruments: Vec::new(),
            series: 0,
        }
    }

    /// False when the measurement is refused: a new series past [`MAX_SERIES`], or an instrument
    /// recorded with another kind.
    pub fn record(&mut self, m: &Measurement) -> bool {
        let index = match self.instruments.iter().position(|i| i.name == m.instrument) {
            Some(index) => index,
            None => {
                self.instruments.push(Instrument {
                    name: m.instrument,
                    kind: m.kind,
                    unit: m.unit,
                    bounds: m.bounds,
                    series: Vec::new(),
                });
                self.instruments.len() - 1
            }
        };
        let Some(instrument) = self.instruments.get_mut(index) else { return false };
        if instrument.kind != m.kind {
            return false;
        }
        let point = match instrument.series.iter().position(|(labels, _)| *labels == m.labels) {
            Some(i) => &mut instrument.series[i].1,
            None => {
                if self.series >= MAX_SERIES {
                    return false;
                }
                self.series += 1;
                let point = match m.kind {
                    InstrumentKind::Counter => Point::Sum(Number::Int(0)),
                    InstrumentKind::Histogram => Point::Histogram {
                        count: 0,
                        sum: 0.0,
                        min: f64::INFINITY,
                        max: f64::NEG_INFINITY,
                        buckets: vec![0; instrument.bounds.len() + 1],
                    },
                };
                instrument.series.push((m.labels.clone(), point));
                let Some((_, point)) = instrument.series.last_mut() else { return false };
                point
            }
        };
        let bounds = instrument.bounds;
        match point {
            Point::Sum(total) => {
                *total = match (*total, m.value) {
                    (Number::Int(a), Number::Int(b)) => Number::Int(a.saturating_add(b)),
                    (a, b) => Number::Double(as_f64(a) + as_f64(b)),
                }
            }
            Point::Histogram { count, sum, min, max, buckets } => {
                let v = as_f64(m.value);
                *count += 1;
                *sum += v;
                *min = min.min(v);
                *max = max.max(v);
                buckets[bounds.partition_point(|bound| *bound < v)] += 1;
            }
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.series == 0
    }

    /// The metrics accumulated so far, each data point stamped with `stamp`. Delta temporality
    /// starts a new interval.
    pub fn collect(&mut self, now_unix_nanos: u64, stamp: &KeyValues) -> Vec<proto::Metric> {
        let temporality = match self.temporality {
            Temporality::Delta => proto::AggregationTemporality::Delta,
            Temporality::Cumulative => proto::AggregationTemporality::Cumulative,
        } as i32;
        let interval = Interval { start: self.start_unix_nanos, now: now_unix_nanos, stamp };
        let metrics = self
            .instruments
            .iter()
            .filter(|i| !i.series.is_empty())
            .map(|instrument| proto::Metric {
                name: instrument.name.to_owned(),
                unit: instrument.unit.to_owned(),
                data: Some(instrument.data(temporality, &interval)),
                ..Default::default()
            })
            .collect();
        if self.temporality == Temporality::Delta {
            self.instruments.clear();
            self.series = 0;
            self.start_unix_nanos = now_unix_nanos;
        }
        metrics
    }
}

/// What every data point of one collection carries: its interval and the destination's stamp.
struct Interval<'a> {
    start: u64,
    now: u64,
    stamp: &'a KeyValues,
}

impl Interval<'_> {
    fn attributes(&self, labels: &KeyValues) -> Vec<KeyValue> {
        let mut all = labels.clone();
        super::encode::stamp(&mut all, self.stamp);
        key_values(&all)
    }
}

impl Instrument {
    fn data(&self, temporality: i32, interval: &Interval) -> proto::metric::Data {
        match self.kind {
            InstrumentKind::Counter => proto::metric::Data::Sum(proto::Sum {
                aggregation_temporality: temporality,
                is_monotonic: true,
                data_points: self.sum_points(interval),
            }),
            InstrumentKind::Histogram => proto::metric::Data::Histogram(proto::Histogram {
                aggregation_temporality: temporality,
                data_points: self.histogram_points(interval),
            }),
        }
    }

    fn sum_points(&self, interval: &Interval) -> Vec<proto::NumberDataPoint> {
        let points = self.series.iter().filter_map(|(labels, point)| {
            let Point::Sum(total) = point else { return None };
            let value = match *total {
                Number::Int(v) => proto::number_data_point::Value::AsInt(v),
                Number::Double(v) => proto::number_data_point::Value::AsDouble(v),
            };
            Some(proto::NumberDataPoint {
                attributes: interval.attributes(labels),
                start_time_unix_nano: interval.start,
                time_unix_nano: interval.now,
                value: Some(value),
                ..Default::default()
            })
        });
        points.collect()
    }

    fn histogram_points(&self, interval: &Interval) -> Vec<proto::HistogramDataPoint> {
        let points = self.series.iter().filter_map(|(labels, point)| {
            let Point::Histogram { count, sum, min, max, buckets } = point else { return None };
            Some(proto::HistogramDataPoint {
                attributes: interval.attributes(labels),
                start_time_unix_nano: interval.start,
                time_unix_nano: interval.now,
                count: *count,
                sum: Some(*sum),
                bucket_counts: buckets.clone(),
                explicit_bounds: self.bounds.to_vec(),
                min: Some(*min),
                max: Some(*max),
                ..Default::default()
            })
        });
        points.collect()
    }
}

fn as_f64(n: Number) -> f64 {
    match n {
        Number::Int(v) => v as f64,
        Number::Double(v) => v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(
        name: &'static str,
        kind: InstrumentKind,
        value: Number,
        model: &str,
    ) -> Measurement {
        Measurement {
            instrument: name,
            kind,
            unit: "s",
            bounds: &[0.0, 5.0, 10.0],
            value,
            labels: vec![("model_name".into(), model.into())],
            job: None,
        }
    }

    // Catches: a value on a bucket boundary counted in the next bucket (OTel buckets are
    // upper-inclusive), mixed int and double sums, or a delta interval that keeps old points.
    #[test]
    fn sums_and_histograms_follow_otel_aggregation() {
        let mut delta = Aggregator::new(Temporality::Delta, 1);
        for v in [0.0, 5.0, 5.5, 20000.0] {
            assert!(delta.record(&measurement(
                "h",
                InstrumentKind::Histogram,
                Number::Double(v),
                "a"
            )));
        }
        assert!(delta.record(&measurement("c", InstrumentKind::Counter, Number::Int(2), "a")));
        assert!(delta.record(&measurement("c", InstrumentKind::Counter, Number::Double(0.5), "a")));
        assert!(delta.record(&measurement("c", InstrumentKind::Counter, Number::Int(1), "b")));
        assert!(!delta.record(&measurement("c", InstrumentKind::Histogram, Number::Int(1), "a")));
        let metrics = delta.collect(10, &vec![("job_id".into(), "AJ_1".into())]);
        let Some(proto::metric::Data::Histogram(h)) = &metrics[0].data else { panic!() };
        let point = &h.data_points[0];
        assert_eq!(h.aggregation_temporality, proto::AggregationTemporality::Delta as i32);
        assert_eq!(
            (point.count, point.sum, point.min, point.max),
            (4, Some(20010.5), Some(0.0), Some(20000.0))
        );
        assert_eq!(point.bucket_counts, vec![1, 1, 1, 1]);
        assert_eq!(point.explicit_bounds, vec![0.0, 5.0, 10.0]);
        assert_eq!((point.start_time_unix_nano, point.time_unix_nano), (1, 10));
        assert_eq!(point.attributes.len(), 2);
        let Some(proto::metric::Data::Sum(s)) = &metrics[1].data else { panic!() };
        assert!(s.is_monotonic);
        assert_eq!(s.data_points[0].value, Some(proto::number_data_point::Value::AsDouble(2.5)));
        assert_eq!(s.data_points[1].value, Some(proto::number_data_point::Value::AsInt(1)));
        assert!(delta.is_empty());
        assert!(delta.collect(20, &vec![]).is_empty());

        let mut cumulative = Aggregator::new(Temporality::Cumulative, 1);
        cumulative.record(&measurement("c", InstrumentKind::Counter, Number::Int(2), "a"));
        cumulative.collect(10, &vec![]);
        cumulative.record(&measurement("c", InstrumentKind::Counter, Number::Int(3), "a"));
        let metrics = cumulative.collect(20, &vec![]);
        let Some(proto::metric::Data::Sum(s)) = &metrics[0].data else { panic!() };
        assert_eq!(s.aggregation_temporality, proto::AggregationTemporality::Cumulative as i32);
        assert_eq!(s.data_points[0].start_time_unix_nano, 1);
        assert_eq!(s.data_points[0].value, Some(proto::number_data_point::Value::AsInt(5)));
    }

    #[test]
    fn series_beyond_the_cardinality_limit_are_refused() {
        let mut aggregator = Aggregator::new(Temporality::Cumulative, 0);
        let labels: Vec<String> = (0..=MAX_SERIES).map(|i| i.to_string()).collect();
        let refused = labels
            .iter()
            .filter(|l| {
                !aggregator.record(&measurement("c", InstrumentKind::Counter, Number::Int(1), l))
            })
            .count();
        assert_eq!(refused, 1);
        assert!(aggregator.record(&measurement("c", InstrumentKind::Counter, Number::Int(1), "0")));
    }
}
