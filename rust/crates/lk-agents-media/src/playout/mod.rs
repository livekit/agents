//! Bounded playout of ordered, mono 24 kHz PCM. Transport chunks have arbitrary lengths.
//!
//! The caller supplies monotonic arrival times, renders on its own clock, and owns downstream
//! scheduling. No clock, task, I/O or allocation is used by `push` or `render`. Source samples
//! are never silently dropped: queue pressure rejects a whole chunk for the caller to handle.
//! `finish` drains real audio; `reset` begins a new timeline and discards the old epoch explicitly.

mod delay;
mod dsp;
mod histogram;
mod level;
#[cfg(test)]
mod playout_tests;

use std::collections::VecDeque;
use std::time::Duration;

use dsp::{Concealer, FRAME, OVERLAP, PERIOD_MAX, Recovery, Stretch, mix};

pub const SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Lower bound for the adaptive reserve target.
    pub minimum_delay: Duration,
    /// Cold-start waiting-time prior. Learned startup coverage replaces it across utterances;
    /// clamped to the minimum/maximum delay bounds. EOF can release a short reply sooner.
    pub startup_delay: Duration,
    /// Upper bound for the target, not a hard latency cap: accepted speech is never dropped.
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
            startup_delay: Duration::from_millis(80),
            maximum_delay: Duration::from_millis(100),
            capacity: Duration::from_secs(2),
            maximum_concealment: Duration::from_millis(200),
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
    /// Subset of inserted samples placed in low-energy source regions.
    pub passive_stretched: u64,
    pub reserve_requests: u64,
    pub reserve_budget_limited: u64,
    pub reserve_no_match: u64,
    pub accelerated: u64,
    /// Subset of `accelerated` removed by the severe-backlog mode.
    pub fast_accelerated: u64,
    /// Subset of `concealed` inserted to align a returning voiced waveform.
    pub merge_inserted: u64,
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
    /// Selected startup waiting-time budget; queued bursts/EOF may release it sooner.
    pub startup_delay: Duration,
    /// Completed answer-start observations retained across `begin_utterance()`.
    pub startup_observations: u64,
    /// Unclamped learned quantile; the selected budget also respects bounds and slow release.
    pub startup_quantile: Option<Duration>,
    pub started: bool,
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
    last_time: Option<Duration>,
    delay: delay::Delay,
    target: usize,
    startup_target: usize,
    level: level::Level,
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
            last_time: None,
            delay: delay::Delay::default(),
            target: samples(config.minimum_delay),
            startup_target: samples(
                config.startup_delay.clamp(config.minimum_delay, config.maximum_delay),
            ),
            level: level::Level::default(),
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
        if self.first_arrival.is_none() {
            self.startup_target = self.delay.startup(
                self.config.startup_delay,
                self.config.minimum_delay,
                self.config.maximum_delay,
            );
            self.first_arrival = Some(now);
        }
        self.delay.update(now, self.statistics.source_received, pcm.len());
        self.target = self.delay.target(self.config.minimum_delay, self.config.maximum_delay);
        self.queue.extend(pcm.iter().copied());
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
        self.delay.end_utterance();
    }

    /// Cancel the old utterance and start a new audio/clock epoch. Keep only normalized
    /// delivery/startup statistics: inter-answer silence never becomes an arrival delay.
    /// Clears queued PCM, old phonemes, render counters and the time-scaling budget.
    pub fn begin_utterance(&mut self) {
        self.delay.begin_utterance();
        self.clear_audio();
    }

    /// Full reset for a new provider/connection. Clears learned timing as well as audio.
    pub fn reset(&mut self) {
        self.delay = delay::Delay::default();
        self.clear_audio();
    }

    fn clear_audio(&mut self) {
        self.queue.clear();
        self.pending.clear();
        self.history.clear();
        self.concealer = Concealer::default();
        self.statistics = Statistics::default();
        self.first_arrival = None;
        self.last_time = None;
        self.target = self.delay.target(self.config.minimum_delay, self.config.maximum_delay);
        self.startup_target = self.delay.startup(
            self.config.startup_delay,
            self.config.minimum_delay,
            self.config.maximum_delay,
        );
        self.level = level::Level::default();
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
            self.started = self.first_arrival.is_some_and(|first| {
                self.finished
                    || samples(now - first) >= self.startup_target
                    || self.queue.len()
                        >= self.statistics.largest_chunk.saturating_add(self.startup_target)
            });
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
            let postpone = self.outage
                && !self.finished
                && self.concealer.gain() < 0.5
                && self.queue.len() < self.target / 2;
            if self.outage && !self.queue.is_empty() && !postpone {
                self.recovery.start(&self.concealer, &self.queue, !self.finished);
                self.outage = false;
                self.level = level::Level::default();
            }
            if let Some(sample) = self.recovery.next_prefix() {
                *out = sample;
                self.last_output = sample;
                self.statistics.concealed += 1;
                self.statistics.merge_inserted += 1;
                continue;
            }
            let real = self.pending.pop_front().or_else(|| {
                if !self.started || postpone {
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
            startup_delay: Duration::from_secs_f64(
                self.startup_target as f64 / f64::from(SAMPLE_RATE),
            ),
            startup_observations: self.delay.startup_observations(),
            startup_quantile: self.delay.startup_quantile(),
            started: self.started,
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
        if self.config.maximum_stretch == 0.0 {
            return;
        }
        let source_position =
            self.statistics.source_consumed.saturating_sub(self.pending.len() as u64);
        let playout_delay = self.delay.playout_delay(now, source_position);
        let target_ms = self.target * 1000 / SAMPLE_RATE as usize;
        // NetEq's decision window includes the recent arrival envelope and 20ms granularity.
        // A queue refilled by a late packet must not immediately be drained back to zero.
        let high_limit =
            self.target + (self.delay.max_delay_ms() + 20) * (SAMPLE_RATE as usize / 1000);
        let filtered = self.level.update(playout_delay, target_ms, self.statistics.rendered);
        let fast = playout_delay >= high_limit.saturating_mul(4);
        if !fast && self.statistics.rendered < self.stretch_after {
            return;
        }
        let expand = playout_delay < self.target && filtered < self.target;
        let accelerate = playout_delay >= high_limit && (filtered >= high_limit || fast);
        if !expand && !accelerate {
            return;
        }
        let budget = (self.statistics.rendered as f64 * f64::from(self.config.maximum_stretch))
            .floor() as u64;
        let available_budget =
            budget.saturating_sub(self.statistics.stretched + self.statistics.accelerated);
        if expand {
            self.statistics.reserve_requests += 1;
        }
        if available_budget < dsp::PERIOD_MIN as u64 {
            if expand {
                self.statistics.reserve_budget_limited += 1;
            }
            return;
        }
        let mut output = [0; FRAME];
        let operation = if expand {
            Stretch::Expand
        } else if fast {
            Stretch::FastAccelerate
        } else {
            Stretch::Accelerate
        };
        let mut limit = available_budget.min(PERIOD_MAX as u64) as usize;
        if !expand && !self.delay.startup_observed() {
            // Do not reclaim the selected startup reserve before early deliveries have
            // been observed. Genuine excess backlog can still be removed safely.
            limit = limit.min(playout_delay.saturating_sub(self.startup_target));
        }
        let result = if expand && self.queue.len() < FRAME + PERIOD_MAX {
            dsp::preemptive(
                &self.history,
                &self.queue,
                limit,
                self.concealer.passive_level(),
                &mut output,
            )
        } else {
            dsp::stretch(&self.queue, operation, limit, self.concealer.passive_level(), &mut output)
        };
        let Some(scale) = result else {
            if expand {
                self.statistics.reserve_no_match += 1;
            }
            return;
        };
        let consumed = scale.consumed;
        let period = scale.change;
        self.pending.extend(output);
        for _ in 0..consumed {
            let source = self.queue.pop_front().expect("stretch validated queue length");
            self.concealer.observe(source);
        }
        self.statistics.source_consumed += consumed as u64;
        self.level.correct(if expand { -(period as isize) } else { period as isize });
        if expand {
            self.statistics.stretched += period as u64;
            if scale.passive {
                self.statistics.passive_stretched += period as u64;
            }
        } else {
            self.statistics.accelerated += period as u64;
            if fast {
                self.statistics.fast_accelerated += period as u64;
            }
        }
        self.stretch_after = self.statistics.rendered
            + ((period as f64 / f64::from(self.config.maximum_stretch)).ceil() as u64)
                .max(u64::from(SAMPLE_RATE) / 20);
    }
}

fn samples(duration: Duration) -> usize {
    (duration.as_nanos().saturating_mul(u128::from(SAMPLE_RATE)) / 1_000_000_000)
        .min(usize::MAX as u128) as usize
}
