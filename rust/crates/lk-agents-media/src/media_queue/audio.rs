use super::core::{Consumer, Core, Producer, channel};
use super::metrics::Meter;
use super::{Head, Item, MERGED_FRAME, Policy, Stuck};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_abi::{AudioFrame, Clock};
use lk_agents_telemetry as telemetry;
use std::sync::Arc;
use std::time::Duration;

/// An audio queue feeding `consumer`, the label its drops and head drain are measured under;
/// `clock` times the drain.
pub fn audio<E>(
    policy: Policy,
    consumer: telemetry::AudioConsumer,
    clock: Arc<dyn Clock>,
) -> (AudioSender<E>, AudioReceiver<E>) {
    let queue = AudioQueue {
        core: Core::new(),
        policy,
        lag: Duration::ZERO,
        head_items: 0,
        head_received: Duration::ZERO,
    };
    let meter = Arc::new(Meter::new(consumer, policy, clock));
    let (producer, consumer) = channel(queue);
    let receiver = AudioReceiver { queue: consumer, rest: Vec::new(), meter: Arc::clone(&meter) };
    (AudioSender { queue: producer, meter }, receiver)
}

/// A queued frame, and whether it is filler.
struct Queued {
    frame: AudioFrame,
    filler: bool,
}

struct AudioQueue<E> {
    core: Core<Queued, E>,
    policy: Policy,
    /// The duration of the core's frames past the floor, which the policy bounds.
    lag: Duration,
    /// The items from the front up to the last head frame still queued.
    head_items: usize,
    /// The head audio received so far.
    head_received: Duration,
}

impl<E> AudioQueue<E> {
    fn push(&mut self, frame: AudioFrame, filler: bool) -> Duration {
        if self.core.frames() == 0 {
            // Settled or drained: nothing is measured.
            self.lag = Duration::ZERO;
        }
        self.lag += frame_duration(&frame);
        self.core.push(Queued { frame, filler });
        let mut dropped = Duration::ZERO;
        match self.policy {
            Policy::CatchUp(max) => {
                while self.lag > max && self.core.frames() > 1 {
                    dropped += self.drop_oldest();
                }
            }
            Policy::Now(max) => {
                if self.lag > max {
                    while self.core.frames() > 1 {
                        dropped += self.drop_oldest();
                    }
                }
            }
        }
        dropped
    }

    fn head(&mut self, frame: AudioFrame) {
        self.core.head(Queued { frame, filler: false });
        self.head_items = self.core.len();
    }

    fn flush(&mut self, event: E) -> Result<(), Stuck> {
        self.head_items = 0;
        self.core.flush(event)
    }

    /// Drops the oldest filler the policy may drop, which leaves no gap and loses nothing said;
    /// without one, the oldest frame, into the gap that starts the run. Returns what was lost.
    fn drop_oldest(&mut self) -> Duration {
        if let Some(filler) = self.core.drop_filler(|queued| queued.filler) {
            self.lag = self.lag.saturating_sub(frame_duration(&filler.frame));
            return Duration::ZERO;
        }
        let duration = frame_duration(&self.core.drop_oldest().frame);
        self.lag = self.lag.saturating_sub(duration);
        self.core.gap(duration);
        duration
    }

    /// The next item, leaving the frames that merge into a popped frame in `rest` for [`merge`],
    /// so the shared queue merges outside its lock. A frame that merges with nothing leaves `rest`
    /// empty and passes as pushed. Head and live audio never merge.
    fn pop(&mut self, rest: &mut Vec<AudioFrame>) -> Option<Item<AudioFrame, E>> {
        let head = self.head_items > 0;
        let item = self.pop_front()?;
        if let Item::Media(first) = &item {
            while let Some(Item::Media(next)) = self.core.front()
                && (self.head_items > 0) == head
                && joins(first, rest, &next.frame)
            {
                if let Some(Item::Media(next)) = self.pop_front() {
                    rest.push(next);
                }
            }
        }
        Some(item)
    }

    fn pop_front(&mut self) -> Option<Item<AudioFrame, E>> {
        let (item, counted) = self.core.pop()?;
        let head = self.head_items > 0;
        self.head_items = self.head_items.saturating_sub(1);
        Some(match item {
            Item::Media(Queued { frame, .. }) => {
                if counted {
                    self.lag = self.lag.saturating_sub(frame_duration(&frame));
                } else if head {
                    self.head_received += frame_duration(&frame);
                }
                Item::Media(frame)
            }
            Item::Gap(duration) => Item::Gap(duration),
            Item::Event(event) => Item::Event(event),
        })
    }
}

/// Whether `next` continues the merged frame `first` followed by `rest`.
fn joins(first: &AudioFrame, rest: &[AudioFrame], next: &AudioFrame) -> bool {
    let samples: u64 = std::iter::once(first)
        .chain(rest)
        .chain([next])
        .map(|frame| u64::from(frame.samples_per_channel))
        .sum();
    whole(first)
        && whole(next)
        && first.sample_rate == next.sample_rate
        && first.num_channels == next.num_channels
        && samples * 1000 <= u64::from(next.sample_rate) * MERGED_FRAME.as_millis() as u64
}

/// A frame whose payload is exactly its declared samples, so concatenating keeps both in step.
fn whole(frame: &AudioFrame) -> bool {
    frame.sample_rate > 0
        && frame.data.len()
            == frame.samples_per_channel as usize * usize::from(frame.num_channels) * 2
}

/// `first` then `rest` as one frame; `rest` is left empty for reuse.
fn merge(first: AudioFrame, rest: &mut Vec<AudioFrame>) -> AudioFrame {
    let mut data = Vec::with_capacity(
        first.data.len() + rest.iter().map(|frame| frame.data.len()).sum::<usize>(),
    );
    data.extend_from_slice(&first.data);
    let mut samples = first.samples_per_channel;
    for frame in rest.drain(..) {
        data.extend_from_slice(&frame.data);
        samples += frame.samples_per_channel;
    }
    AudioFrame::new(data, first.sample_rate, first.num_channels, samples)
}

/// The producer. Not `Clone`: one owner, so dropping it closes the queue.
pub struct AudioSender<E> {
    queue: Producer<AudioQueue<E>>,
    meter: Arc<Meter>,
}

impl<E> AudioSender<E> {
    /// Never fails and never blocks; the policy decides what is kept. Returns the audio it
    /// dropped, which the queue has measured.
    pub fn push(&self, frame: AudioFrame) -> Duration {
        let dropped = self.queue.apply(|queue| queue.push(frame, false)).unwrap_or_default();
        self.meter.dropped(dropped);
        dropped
    }

    /// Audio that stands in for nothing said (a recogniser's flush of silence): the bound drops
    /// it before any other audio, so it never pushes what was said out of the queue.
    pub fn filler(&self, frame: AudioFrame) -> Duration {
        let dropped = self.queue.apply(|queue| queue.push(frame, true)).unwrap_or_default();
        self.meter.dropped(dropped);
        dropped
    }

    /// Measures the queue under `cx`'s job from now on.
    pub fn observe(&self, cx: telemetry::Cx) {
        self.meter.observe(cx);
    }

    /// Pre-connect audio: kept in full, outside both policies.
    pub fn head(&self, frame: AudioFrame) {
        self.queue.apply(|queue| queue.head(frame));
    }

    /// In order with the audio pushed before it. `Stuck` when the consumer stopped draining.
    pub fn event(&self, event: E) -> Result<(), Stuck> {
        self.queue.apply(|queue| queue.core.event(event)).unwrap_or(Ok(()))
    }

    /// Drops all queued audio, then queues `event`: start over from here. No `Gap`.
    pub fn flush(&self, event: E) -> Result<(), Stuck> {
        self.queue.apply(|queue| queue.flush(event)).unwrap_or(Ok(()))
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        self.queue.is_closed()
    }
}

/// The consumer.
pub struct AudioReceiver<E> {
    queue: Consumer<AudioQueue<E>>,
    /// The frames merging into the one received, reused across calls.
    rest: Vec<AudioFrame>,
    meter: Arc<Meter>,
}

impl<E> AudioReceiver<E> {
    /// The next item, pending audio merged. `None` once the sender is gone and the queue is
    /// drained. Cancel-safe, no allocation in steady state. Measures the head's drain.
    pub async fn recv(&mut self) -> Option<Item<AudioFrame, E>> {
        self.meter.receiving(self.head());
        let rest = &mut self.rest;
        let item = self.queue.recv(|queue| queue.pop(rest)).await?;
        Some(match item {
            Item::Media(first) if !self.rest.is_empty() => {
                Item::Media(merge(first, &mut self.rest))
            }
            item => item,
        })
    }

    /// Frames the policy dropped so far, monotone.
    pub fn dropped(&self) -> u64 {
        self.queue.read(|queue| queue.core.dropped())
    }

    /// How far the receiver is through the head audio: a received item that grew
    /// [`Head::received`] was head audio.
    pub fn head(&self) -> Head {
        self.queue
            .read(|queue| Head { received: queue.head_received, pending: queue.head_items > 0 })
    }
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;
