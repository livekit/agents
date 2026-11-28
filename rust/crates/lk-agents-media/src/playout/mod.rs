//! Bounded playout of ordered, mono 24 kHz PCM. Transport chunks have arbitrary lengths.
//!
//! The caller supplies monotonic arrival times, renders on its own clock, and owns downstream
//! scheduling. No clock, task, I/O or allocation is used by `push` or `render`. Source samples
//! are never silently dropped: queue pressure rejects a whole chunk for the caller to handle.
//! `finish` drains real audio; `reset` begins a new timeline and discards the old epoch explicitly.

mod dsp;
#[cfg(test)]
mod playout_tests;

use std::collections::VecDeque;
use std::time::Duration;

use dsp::{Concealer, FRAME, OVERLAP, PERIOD_MAX, Recovery, Stretch, mix};

pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub minimum_delay: Duration,
    pub maximum_delay: Duration,
    pub capacity: Duration,
    /// Bounds extrapolated speech. Zero disables it but retains the short transition.
    /// Independently learned room tone can continue and fades out over two seconds.
    pub maximum_concealment: Duration,
    /// Maximum fraction of rendered samples inserted or removed by time stretching, 0..=0.1.
    pub maximum_stretch: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minimum_delay: Duration::from_millis(20),
            maximum_delay: Duration::from_millis(100),
            capacity: Duration::from_secs(2),
            maximum_concealment: Duration::from_millis(40),
            maximum_stretch: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PlayoutError {
    #[error("invalid playout configuration")]
    Configuration,
    #[error("arrival or render time moved backwards")]
    TimeReversed,
    #[error("input is already finished")]
    Finished,
    #[error("PCM queue is full; {available} samples available")]
    Full { available: usize },
    #[error("render blocks must contain 1..=480 samples")]
    OutputSize,
}

/// Cumulative accounting within one epoch. `source_consumed + queued_source == source_received`.
/// `rendered == source_consumed + stretched - accelerated + concealed + silent - pending_output`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Statistics {
    pub source_received: u64,
    pub source_consumed: u64,
    pub rendered: u64,
    pub stretched: u64,
    pub accelerated: u64,
    pub concealed: u64,
    pub silent: u64,
    pub underruns: u64,
    pub chunks: u64,
    pub smallest_chunk: usize,
    pub largest_chunk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Render {
    pub queued_source: usize,
    pub pending_output: usize,
    pub target_delay: Duration,
    /// All accepted source audio has been rendered and input was explicitly finished.
    pub drained: bool,
}

pub struct Playout {
    config: Config,
    queue: VecDeque<i16>,
    capacity: usize,
    pending: VecDeque<i16>,
    history: VecDeque<i16>,
    concealer: Concealer,
    statistics: Statistics,
    first_arrival: Option<Duration>,
    last_arrival: Option<Duration>,
    last_time: Option<Duration>,
    last_chunk: usize,
    // Relative arrival delay, in samples; fixed storage and bounded O(128 log 128) per chunk.
    delays: [i64; 128],
    delay_count: usize,
    target: usize,
    arrival_lateness: usize,
    started: bool,
    finished: bool,
    outage: bool,
    recovery: Recovery,
    last_output: i16,
    stretch_after: u64,
}

impl Playout {
    pub fn new(config: Config) -> Result<Self, PlayoutError> {
        let capacity = samples(config.capacity);
        if config.minimum_delay > config.maximum_delay
            || config.maximum_delay > config.capacity
            || capacity < FRAME + PERIOD_MAX
            || capacity > SAMPLE_RATE as usize * 60
            || config.maximum_concealment > Duration::from_millis(200)
            || !config.maximum_stretch.is_finite()
            || !(0.0..=0.1).contains(&config.maximum_stretch)
        {
            return Err(PlayoutError::Configuration);
        }
        Ok(Self {
            config,
            queue: VecDeque::with_capacity(capacity),
            capacity,
            pending: VecDeque::with_capacity(FRAME),
            history: VecDeque::with_capacity(dsp::HISTORY),
            concealer: Concealer::default(),
            statistics: Statistics::default(),
            first_arrival: None,
            last_arrival: None,
            last_time: None,
            last_chunk: 0,
            delays: [0; 128],
            delay_count: 0,
            target: samples(config.minimum_delay),
            arrival_lateness: 0,
            started: false,
            finished: false,
            outage: false,
            recovery: Recovery::default(),
            last_output: 0,
            stretch_after: 0,
        })
    }

    /// Accepts a complete chunk atomically. An error changes no state. Arrival times share the
    /// render clock's epoch; equal times are allowed, including several chunks in one burst.
    pub fn push(&mut self, now: Duration, pcm: &[i16]) -> Result<(), PlayoutError> {
        self.check_time(now)?;
        if self.finished {
            return Err(PlayoutError::Finished);
        }
        let available = self.capacity - self.queue.len();
        if pcm.len() > available {
            return Err(PlayoutError::Full { available });
        }
        self.last_time = Some(now);
        if pcm.is_empty() {
            return Ok(());
        }
        let first = *self.first_arrival.get_or_insert(now);
        let offset = (samples(now - first) as i128 - i128::from(self.statistics.source_received))
            .clamp(i128::from(i64::MIN / 2), i128::from(i64::MAX / 2)) as i64;
        self.delays[self.statistics.chunks as usize % self.delays.len()] = offset;
        self.delay_count = (self.delay_count + 1).min(self.delays.len());
        let mut sorted = self.delays;
        sorted[..self.delay_count].sort_unstable();
        let jitter = (sorted[(self.delay_count - 1) * 95 / 100] - sorted[0]).max(0) as usize;
        self.arrival_lateness = (offset - sorted[0]).max(0) as usize;
        self.target =
            (samples(self.config.minimum_delay) + jitter).min(samples(self.config.maximum_delay));
        self.queue.extend(pcm.iter().copied());
        self.last_arrival = Some(now);
        self.last_chunk = pcm.len();
        self.statistics.chunks += 1;
        self.statistics.source_received += pcm.len() as u64;
        self.statistics.smallest_chunk = if self.statistics.chunks == 1 {
            pcm.len()
        } else {
            self.statistics.smallest_chunk.min(pcm.len())
        };
        self.statistics.largest_chunk = self.statistics.largest_chunk.max(pcm.len());
        Ok(())
    }

    /// End of source audio, not a temporary delivery gap. Idempotent; accepted samples drain.
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Cancel the old epoch. Reuses allocated storage and clears DSP, estimates and counters.
    pub fn reset(&mut self) {
        self.queue.clear();
        self.pending.clear();
        self.history.clear();
        self.concealer = Concealer::default();
        self.statistics = Statistics::default();
        self.first_arrival = None;
        self.last_arrival = None;
        self.last_time = None;
        self.last_chunk = 0;
        self.delays.fill(0);
        self.delay_count = 0;
        self.target = samples(self.config.minimum_delay);
        self.arrival_lateness = 0;
        self.started = false;
        self.finished = false;
        self.outage = false;
        self.recovery = Recovery::default();
        self.last_output = 0;
        self.stretch_after = 0;
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    /// Writes every output sample, including startup silence and exhausted-stream silence.
    /// The caller chooses 1..=480 samples per call; this does not constrain chunk duration.
    pub fn render(&mut self, now: Duration, output: &mut [i16]) -> Result<Render, PlayoutError> {
        self.check_time(now)?;
        if output.is_empty() || output.len() > FRAME {
            return Err(PlayoutError::OutputSize);
        }
        self.last_time = Some(now);
        if !self.started {
            self.started = self
                .first_arrival
                .is_some_and(|first| self.finished || now - first >= self.config.minimum_delay);
        }
        if self.started
            && self.pending.is_empty()
            && !self.finished
            && !self.outage
            && !self.recovery.active()
        {
            self.adjust(now);
        }
        for out in output.iter_mut() {
            if self.outage && !self.queue.is_empty() {
                self.recovery.start(&self.concealer, &self.queue);
                self.outage = false;
            }
            let real = self.pending.pop_front().or_else(|| {
                if !self.started {
                    return None;
                }
                let sample = self.queue.pop_front()?;
                self.statistics.source_consumed += 1;
                self.concealer.observe(sample);
                Some(sample)
            });
            let sample = match real {
                Some(mut sample) => {
                    if self.recovery.active() {
                        sample = self.recovery.apply(sample);
                    }
                    if self.finished && self.pending.len() + self.queue.len() < OVERLAP {
                        sample = mix(0, sample, self.pending.len() + self.queue.len(), OVERLAP);
                    }
                    if self.history.len() == dsp::HISTORY {
                        self.history.pop_front();
                    }
                    self.history.push_back(sample);
                    sample
                }
                None if self.started && !self.finished => {
                    if !self.outage {
                        self.statistics.underruns += 1;
                        self.concealer.start(
                            &self.history,
                            samples(self.config.maximum_concealment),
                            self.last_output,
                        );
                        self.outage = true;
                        self.recovery = Recovery::default();
                    }
                    let sample = self.concealer.next();
                    if self.concealer.active() {
                        self.statistics.concealed += 1;
                    } else {
                        self.statistics.silent += 1;
                    }
                    sample
                }
                None => {
                    self.statistics.silent += 1;
                    0
                }
            };
            *out = sample;
            self.last_output = sample;
        }
        self.statistics.rendered += output.len() as u64;
        Ok(Render {
            queued_source: self.queue.len(),
            pending_output: self.pending.len(),
            target_delay: Duration::from_secs_f64(self.target as f64 / f64::from(SAMPLE_RATE)),
            drained: self.finished && self.queue.is_empty() && self.pending.is_empty(),
        })
    }

    fn check_time(&self, now: Duration) -> Result<(), PlayoutError> {
        if self.last_time.is_some_and(|last| now < last) {
            return Err(PlayoutError::TimeReversed);
        }
        Ok(())
    }

    fn adjust(&mut self, now: Duration) {
        if self.config.maximum_stretch == 0.0
            || self.statistics.rendered < self.stretch_after
            || self.queue.len() < FRAME + PERIOD_MAX
        {
            return;
        }
        // Estimate the trough just before the next delivery, rather than reacting to the normal
        // sawtooth from a large chunk. Past the expected arrival, treat remaining audio as reserve.
        let elapsed = samples(now - self.last_arrival.expect("started audio has an arrival"));
        let until_next = self.last_chunk.saturating_sub(elapsed);
        let reserve = self.queue.len().saturating_sub(until_next);
        // Arrival lateness already contributes to source-to-render delay. Do not add that
        // same delay a second time as reserve after a permanent transport delay change.
        let reserve_target = self.target.saturating_sub(self.arrival_lateness);
        let expand = reserve + OVERLAP < reserve_target;
        let accelerate = reserve > reserve_target + FRAME;
        if !expand && !accelerate {
            return;
        }
        let budget = (self.statistics.rendered as f64 * f64::from(self.config.maximum_stretch))
            .floor() as u64;
        let available_budget =
            budget.saturating_sub(self.statistics.stretched + self.statistics.accelerated);
        let mut output = [0; FRAME];
        let operation = if expand { Stretch::Expand } else { Stretch::Accelerate };
        let Some((consumed, period)) = dsp::stretch(
            &self.queue,
            operation,
            available_budget.min(PERIOD_MAX as u64) as usize,
            &mut output,
        ) else {
            return;
        };
        self.pending.extend(output);
        for _ in 0..consumed {
            let source = self.queue.pop_front().expect("stretch validated queue length");
            self.concealer.observe(source);
        }
        self.statistics.source_consumed += consumed as u64;
        if expand {
            self.statistics.stretched += period as u64;
        } else {
            self.statistics.accelerated += period as u64;
        }
        self.stretch_after = self.statistics.rendered
            + (period as f64 / f64::from(self.config.maximum_stretch)).ceil() as u64;
    }
}

fn samples(duration: Duration) -> usize {
    (duration.as_nanos().saturating_mul(u128::from(SAMPLE_RATE)) / 1_000_000_000)
        .min(usize::MAX as u128) as usize
}
