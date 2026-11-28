//! Transport timing in source-sample units. Each chunk contributes its actual first-sample
//! arrival, never fabricated RTP arrivals inside a WebSocket payload. Source time bounds the
//! recent path envelope; persistent distributions survive an explicit segment boundary.

use std::time::Duration;

use super::histogram::{BIN_MS, Histogram};
use super::{SAMPLE_RATE, samples};

const UNIT: u64 = SAMPLE_RATE as u64 / 50;
const WINDOW: u64 = 100;
const BINS: usize = 128;
const STARTUP_SAMPLES: u64 = SAMPLE_RATE as u64 / 2;

#[derive(Clone, Copy)]
struct Arrival {
    bucket: u64,
    minimum: i64,
    maximum: i64,
}
const EMPTY: Arrival = Arrival { bucket: u64::MAX, minimum: 0, maximum: 0 };

pub(super) struct Delay {
    arrivals: [Arrival; BINS],
    newest: u64,
    minimum: i64,
    maximum: i64,
    seen: bool,
    timing_observed: bool,
    histogram: Histogram,
    interval_start: Option<Duration>,
    interval_peak_ms: usize,
    startup_histogram: Histogram,
    startup_hint: Option<Duration>,
    first: Option<Duration>,
    startup_peak_ms: usize,
    startup_chunks: usize,
    startup_complete: bool,
}

impl Default for Delay {
    fn default() -> Self {
        Self {
            arrivals: [EMPTY; BINS],
            newest: 0,
            minimum: 0,
            maximum: 0,
            seen: false,
            timing_observed: false,
            histogram: Histogram::default(),
            interval_start: None,
            interval_peak_ms: 0,
            startup_histogram: Histogram::default(),
            startup_hint: None,
            first: None,
            startup_peak_ms: 0,
            startup_chunks: 0,
            startup_complete: false,
        }
    }
}

impl Delay {
    pub(super) fn update(&mut self, now: Duration, source_start: u64, length: usize) {
        let arrival = clock_samples(now);
        let source = source_start.min((i64::MAX / 4) as u64) as i64;
        let offset = arrival - source;
        let bucket = source_start / UNIT;
        let slot = &mut self.arrivals[bucket as usize % BINS];
        if slot.bucket != bucket {
            *slot = Arrival { bucket, minimum: offset, maximum: offset };
        } else {
            slot.minimum = slot.minimum.min(offset);
            slot.maximum = slot.maximum.max(offset);
        }
        self.newest = bucket;
        self.seen = true;
        self.minimum = i64::MAX;
        self.maximum = i64::MIN;
        for entry in &self.arrivals {
            if entry.bucket != u64::MAX && entry.bucket + WINDOW >= self.newest {
                self.minimum = self.minimum.min(entry.minimum);
                self.maximum = self.maximum.max(entry.maximum);
            }
        }
        if let Some(start) = self.interval_start {
            if now.saturating_sub(start) > Duration::from_millis(500) {
                if self.timing_observed {
                    self.histogram.observe(self.interval_peak_ms);
                }
                self.interval_start = Some(now);
                self.interval_peak_ms = 0;
            }
        } else {
            self.interval_start = Some(now);
        }
        self.interval_peak_ms = self.interval_peak_ms.max(self.max_delay_ms());
        self.timing_observed |= source_start > 0;

        let first = *self.first.get_or_insert(now);
        if !self.startup_complete && source_start <= STARTUP_SAMPLES {
            self.startup_chunks += 1;
            let elapsed = clock_samples(now.saturating_sub(first));
            let deficit = (elapsed - source).max(0) as usize;
            self.startup_peak_ms =
                self.startup_peak_ms.max(deficit.div_ceil(SAMPLE_RATE as usize / 1000));
            if source_start.saturating_add(length as u64) >= STARTUP_SAMPLES
                && self.startup_chunks >= 2
            {
                self.complete_startup();
            }
        } else if !self.startup_complete {
            self.complete_startup();
        }
    }

    pub(super) fn startup_observations(&self) -> u64 {
        self.startup_histogram.observations()
    }

    pub(super) fn startup_quantile(&self) -> Option<Duration> {
        (self.startup_observations() > 0)
            .then(|| Duration::from_millis(self.startup_histogram.quantile_ms() as u64))
    }

    pub(super) fn startup_observed(&self) -> bool {
        self.startup_complete
    }

    fn complete_startup(&mut self) {
        if self.startup_complete {
            return;
        }
        self.startup_complete = true;
        if self.startup_chunks < 2 {
            return;
        }
        self.startup_histogram.observe(self.startup_peak_ms);
        let previous = self.startup_hint.unwrap_or(Duration::from_millis(80));
        // React to worse startup coverage immediately. Release one histogram bin per
        // observed segment so a single clean response does not erase a cautious prior.
        self.startup_hint = Some(
            Duration::from_millis(self.startup_histogram.quantile_ms() as u64)
                .max(previous.saturating_sub(Duration::from_millis(BIN_MS as u64))),
        );
    }

    pub(super) fn end_segment(&mut self) {
        self.complete_startup();
    }

    pub(super) fn begin_segment(&mut self) {
        self.complete_startup();
        if self.timing_observed && self.interval_start.is_some() {
            self.histogram.observe(self.interval_peak_ms);
        }
        self.arrivals.fill(EMPTY);
        self.newest = 0;
        self.minimum = 0;
        self.maximum = 0;
        self.seen = false;
        self.timing_observed = false;
        self.interval_start = None;
        self.interval_peak_ms = 0;
        self.first = None;
        self.startup_peak_ms = 0;
        self.startup_chunks = 0;
        self.startup_complete = false;
    }

    pub(super) fn startup(
        &mut self,
        cold: Duration,
        minimum: Duration,
        maximum: Duration,
    ) -> usize {
        let hint = self.startup_hint.get_or_insert(cold.clamp(minimum, maximum));
        // Smooth the actually selected budget, rather than an invisible value above the
        // configured cap. A caller's cold prior is also the starting point for learning.
        *hint = (*hint).clamp(minimum, maximum);
        samples(*hint)
    }

    pub(super) fn target(&self, minimum: Duration, maximum: Duration) -> usize {
        samples(Duration::from_millis(self.histogram.quantile_ms() as u64).clamp(minimum, maximum))
    }

    pub(super) fn max_delay_ms(&self) -> usize {
        if self.seen {
            ((self.maximum - self.minimum).max(0) as usize).div_ceil(SAMPLE_RATE as usize / 1000)
        } else {
            0
        }
    }

    pub(super) fn playout_delay(&self, now: Duration, source_position: u64) -> usize {
        if !self.seen {
            return 0;
        }
        let source = source_position.min((i64::MAX / 4) as u64) as i64;
        (clock_samples(now) - source - self.minimum).max(0) as usize
    }
}

fn clock_samples(now: Duration) -> i64 {
    (now.as_nanos().saturating_mul(u128::from(SAMPLE_RATE)) / 1_000_000_000)
        .min((i64::MAX / 4) as u128) as i64
}

#[cfg(test)]
#[path = "delay_tests.rs"]
mod tests;
