//! Bounded playout of ordered, mono 16/24/48 kHz PCM. Transport chunks have arbitrary lengths.
//!
//! The caller supplies monotonic arrival times, renders on its own clock, and owns downstream
//! scheduling. No clock, task, I/O or allocation is used by `push` or `render`. Source samples
//! are never silently dropped: queue pressure rejects a whole chunk for the caller to handle.
//!
//! Choose [`Playout::continuous`] for audio with no boundaries, or [`Playout::segmented`] for
//! explicit source segments. Continuous streams need only `push` and `render`; real PCM
//! silence and underruns never imply a new segment. `reset` explicitly discards the timeline.

mod delay;
mod dsp;
mod histogram;
#[cfg(test)]
mod playout_tests;
mod rate;

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::time::Duration;

use dsp::{Concealer, Recovery, Stretch, mix};
#[cfg(test)]
use dsp::{FRAME, PERIOD_MAX};
use rate::Rate;

pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Mono input and output rate: 16_000, 24_000 or 48_000 Hz. Fixed at construction.
    pub sample_rate: u32,
    /// Lower bound for the adaptive reserve target.
    pub minimum_delay: Duration,
    /// Cold-start waiting-time prior. Learned startup coverage replaces it across explicit segments;
    /// clamped to the minimum/maximum delay bounds. EOF can release a short reply sooner.
    pub startup_delay: Duration,
    /// Upper bound for the target, not a hard latency cap: accepted speech is never dropped.
    pub maximum_delay: Duration,
    pub capacity: Duration,
    /// Bounds extrapolated speech. Zero disables it but retains the short transition.
    /// Independently learned room tone can continue and fades out over two seconds.
    pub maximum_concealment: Duration,
    /// Enable pitch-preserving reserve insertion and acceleration, with NetEq-style
    /// ordinary spacing and emergency catch-up. No cumulative rate quota is imposed.
    pub time_stretching: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            minimum_delay: Duration::from_millis(20),
            startup_delay: Duration::from_millis(80),
            maximum_delay: Duration::from_millis(100),
            capacity: Duration::from_secs(2),
            maximum_concealment: Duration::from_millis(200),
            time_stretching: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PlayoutError {
    #[error("sample rate must be 16000, 24000 or 48000 Hz")]
    SampleRate,
    #[error("invalid playout configuration")]
    Configuration,
    #[error("arrival or render time moved backwards")]
    TimeReversed,
    #[error("input is already finished")]
    Finished,
    #[error("PCM queue is full; {available} samples available")]
    Full { available: usize },
    #[error("render blocks must contain 1 sample through 20 ms at the configured rate")]
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
    /// Completed stream/segment-start observations; retained across explicit segment boundaries.
    pub startup_observations: u64,
    /// Unclamped learned quantile; the selected budget also respects bounds and slow release.
    pub startup_quantile: Option<Duration>,
    pub started: bool,
    /// All accepted source audio has been rendered and input was explicitly finished.
    pub drained: bool,
}

/// Continuous source audio; no segment-boundary methods are exposed.
pub struct Continuous;

/// Explicit source segments with independent startup and shared learned delivery timing.
pub struct Segmented;

/// Bounded PCM playout. Choose `continuous` or `segmented` to match the provider's lifecycle.
/// Both modes share exactly the same buffering and DSP; the mode has no runtime state.
pub struct Playout<Mode = Continuous> {
    mode: PhantomData<Mode>,
    config: Config,
    rate: Rate,
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
    started: bool,
    finished: bool,
    outage: bool,
    recovery: Recovery,
    last_output: i16,
    stretch_after: u64,
}

impl Playout<Continuous> {
    /// Ready to accept a continuous source. Call `finish` only at its actual end, never
    /// during silence or a temporary delivery gap. No boundary events are needed.
    ///
    /// ```compile_fail
    /// use lk_agents_media::playout::{Config, Playout};
    /// let mut audio = Playout::continuous(Config::default()).unwrap();
    /// audio.begin_segment(); // Available only on segmented playout.
    /// ```
    pub fn continuous(config: Config) -> Result<Self, PlayoutError> {
        Self::new(config)
    }

    /// End of the continuous source. Idempotent; accepted samples drain before `drained`.
    pub fn finish(&mut self) {
        self.finish_input();
    }
}

impl Playout<Segmented> {
    /// Ready to accept the first explicit segment. Subsequent segments use `begin_segment`.
    /// Use this mode only when the caller knows where source boundaries occur.
    pub fn segmented(config: Config) -> Result<Self, PlayoutError> {
        Self::new(config)
    }

    /// Start a new segment and audio/clock epoch, retaining normalized delivery statistics.
    /// Clears queued audio, DSP, counters and scaling cadence. Drain the previous segment
    /// before calling this, unless deliberately cancelling it. Never infer this from silence.
    pub fn begin_segment(&mut self) {
        self.delay.begin_segment();
        self.clear_audio();
    }

    /// End the current segment. Keep rendering until `drained` before beginning the next.
    /// Rejects further input until `begin_segment` or `reset`. Idempotent.
    ///
    /// ```compile_fail
    /// use lk_agents_media::playout::{Config, Playout};
    /// let mut audio = Playout::segmented(Config::default()).unwrap();
    /// audio.finish(); // Segmented playout uses end_segment instead.
    /// ```
    pub fn end_segment(&mut self) {
        self.finish_input();
    }
}

impl<Mode> Playout<Mode> {
    fn new(config: Config) -> Result<Self, PlayoutError> {
        let rate = Rate::new(config.sample_rate).ok_or(PlayoutError::SampleRate)?;
        let capacity = rate.samples(config.capacity);
        if config.minimum_delay > config.maximum_delay
            || config.maximum_delay > config.capacity
            || capacity < rate.frame() + rate.period_max()
            || capacity > rate.hz() as usize * 60
            || config.maximum_concealment > Duration::from_millis(200)
        {
            return Err(PlayoutError::Configuration);
        }
        Ok(Self {
            mode: PhantomData,
            config,
            rate,
            queue: VecDeque::with_capacity(capacity),
            capacity,
            pending: VecDeque::with_capacity(rate.stretch_window() + rate.period_max()),
            history: VecDeque::with_capacity(rate.history()),
            concealer: Concealer::new(rate),
            statistics: Statistics::default(),
            first_arrival: None,
            last_time: None,
            delay: delay::Delay::new(rate),
            target: rate.samples(config.minimum_delay),
            startup_target: rate
                .samples(config.startup_delay.clamp(config.minimum_delay, config.maximum_delay)),
            started: false,
            finished: false,
            outage: false,
            recovery: Recovery::new(rate),
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

    fn finish_input(&mut self) {
        self.finished = true;
        self.delay.end_segment();
    }

    /// Full reset for a new provider/connection. Clears learned timing as well as audio.
    pub fn reset(&mut self) {
        self.delay = delay::Delay::new(self.rate);
        self.clear_audio();
    }

    fn clear_audio(&mut self) {
        self.queue.clear();
        self.pending.clear();
        self.history.clear();
        self.concealer = Concealer::new(self.rate);
        self.statistics = Statistics::default();
        self.first_arrival = None;
        self.last_time = None;
        self.target = self.delay.target(self.config.minimum_delay, self.config.maximum_delay);
        self.startup_target = self.delay.startup(
            self.config.startup_delay,
            self.config.minimum_delay,
            self.config.maximum_delay,
        );
        self.started = false;
        self.finished = false;
        self.outage = false;
        self.recovery = Recovery::new(self.rate);
        self.last_output = 0;
        self.stretch_after = 0;
    }

    /// The immutable rate shared by input, output and source-sample counters.
    pub fn sample_rate(&self) -> u32 {
        self.rate.hz()
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    /// Writes every output sample, including startup silence and exhausted-stream silence.
    /// The caller chooses 1 sample through 20 ms per call (320/480/960 samples at 16/24/48 kHz).
    /// This does not constrain chunk duration.
    pub fn render(&mut self, now: Duration, output: &mut [i16]) -> Result<Render, PlayoutError> {
        self.check_time(now)?;
        if output.is_empty() || output.len() > self.rate.frame() {
            return Err(PlayoutError::OutputSize);
        }
        self.last_time = Some(now);
        if !self.started {
            self.started = self.first_arrival.is_some_and(|first| {
                self.finished
                    || self.rate.samples(now - first) >= self.startup_target
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
                    if self.finished && self.pending.len() + self.queue.len() < self.rate.overlap()
                    {
                        sample = mix(
                            0,
                            sample,
                            self.pending.len() + self.queue.len(),
                            self.rate.overlap(),
                        );
                    }
                    if self.history.len() == self.rate.history() {
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
                            self.rate.samples(self.config.maximum_concealment),
                            self.last_output,
                        );
                        self.outage = true;
                        self.recovery = Recovery::new(self.rate);
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
            target_delay: Duration::from_secs_f64(self.target as f64 / f64::from(self.rate.hz())),
            startup_delay: Duration::from_secs_f64(
                self.startup_target as f64 / f64::from(self.rate.hz()),
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
        if !self.config.time_stretching {
            return;
        }
        let source_position =
            self.statistics.source_consumed.saturating_sub(self.pending.len() as u64);
        let playout_delay = self.delay.playout_delay(now, source_position);
        // NetEq's decision window includes the recent arrival envelope and 20ms granularity.
        // A queue refilled by a late packet must not immediately be drained back to zero.
        let high_limit =
            self.target + (self.delay.max_delay_ms() + 20) * (self.rate.hz() as usize / 1000);
        let fast = playout_delay >= high_limit.saturating_mul(4);
        if !fast && self.statistics.rendered < self.stretch_after {
            return;
        }
        let expand = playout_delay < self.target;
        let accelerate = playout_delay >= high_limit;
        if !expand && !accelerate {
            return;
        }
        if expand {
            self.statistics.reserve_requests += 1;
        }
        let mut output = [0; dsp::MAX_STRETCH];
        let operation = if expand {
            Stretch::Expand
        } else if fast {
            Stretch::FastAccelerate
        } else {
            Stretch::Accelerate
        };
        let mut limit = self.rate.period_max();
        if !expand && !self.delay.startup_observed() {
            // Do not reclaim the selected startup reserve before early deliveries have
            // been observed. Genuine excess backlog can still be removed safely.
            limit = limit.min(playout_delay.saturating_sub(self.startup_target));
        }
        let result = if expand && self.queue.len() < self.rate.stretch_window() {
            dsp::preemptive(
                self.rate,
                &self.history,
                &self.queue,
                limit,
                self.concealer.passive_level(),
                &mut output[..self.rate.frame()],
            )
        } else {
            dsp::stretch(
                self.rate,
                &self.queue,
                operation,
                limit,
                self.concealer.passive_level(),
                &mut output,
            )
        };
        let Some(scale) = result else {
            if expand {
                self.statistics.reserve_no_match += 1;
            }
            return;
        };
        let consumed = scale.consumed;
        let period = scale.change;
        self.pending.extend(output[..scale.produced].iter().copied());
        for _ in 0..consumed {
            let source = self.queue.pop_front().expect("stretch validated queue length");
            self.concealer.observe(source);
        }
        self.statistics.source_consumed += consumed as u64;
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
        // NetEq spaces ordinary successful operations by 5 x 10ms ticks. Its emergency
        // path bypasses that wait, bounded by one verified <=15ms removal per operation.
        self.stretch_after = self.statistics.rendered + u64::from(self.rate.hz()) / 20;
    }
}
