//! Audio: bounded by its queued duration under a [`Policy`], what the bound drops reported as one
//! gap, a backlog merged into frames of up to [`MERGED_FRAME`].

use super::core::{Consumer, Core, Producer, channel};
use super::{Head, Item, MERGED_FRAME, Policy, Stuck};
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::rtc::frame_duration;
use std::time::Duration;

pub fn audio<E>(policy: Policy) -> (AudioSender<E>, AudioReceiver<E>) {
    let queue = AudioQueue {
        core: Core::new(),
        policy,
        lag: Duration::ZERO,
        head_items: 0,
        head_received: Duration::ZERO,
    };
    let (producer, consumer) = channel(queue);
    (AudioSender(producer), AudioReceiver { queue: consumer, rest: Vec::new() })
}

struct AudioQueue<E> {
    core: Core<AudioFrame, E>,
    policy: Policy,
    /// The duration of the core's frames past the floor, which the policy bounds.
    lag: Duration,
    /// The items from the front up to the last head frame still queued.
    head_items: usize,
    /// The head audio received so far.
    head_received: Duration,
}

impl<E> AudioQueue<E> {
    fn push(&mut self, frame: AudioFrame) -> Duration {
        if self.core.frames() == 0 {
            // Settled or drained: nothing is measured.
            self.lag = Duration::ZERO;
        }
        self.lag += frame_duration(&frame);
        self.core.push(frame);
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
        self.core.head(frame);
        self.head_items = self.core.len();
    }

    fn flush(&mut self, event: E) -> Result<(), Stuck> {
        self.head_items = 0;
        self.core.flush(event)
    }

    /// Drops the oldest frame the policy may drop into the gap that starts the run.
    fn drop_oldest(&mut self) -> Duration {
        let duration = frame_duration(&self.core.drop_oldest());
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
                && joins(first, rest, next)
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
        if let Item::Media(frame) = &item {
            if counted {
                self.lag = self.lag.saturating_sub(frame_duration(frame));
            } else if head {
                self.head_received += frame_duration(frame);
            }
        }
        Some(item)
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
pub struct AudioSender<E>(Producer<AudioQueue<E>>);

impl<E> AudioSender<E> {
    /// Never fails and never blocks; the policy decides what is kept. Returns the audio it
    /// dropped, for the owner to report.
    pub fn push(&self, frame: AudioFrame) -> Duration {
        self.0.apply(|queue| queue.push(frame)).unwrap_or_default()
    }

    /// Pre-connect audio: kept in full, outside both policies.
    pub fn head(&self, frame: AudioFrame) {
        self.0.apply(|queue| queue.head(frame));
    }

    /// In order with the audio pushed before it. `Stuck` when the consumer stopped draining.
    pub fn event(&self, event: E) -> Result<(), Stuck> {
        self.0.apply(|queue| queue.core.event(event)).unwrap_or(Ok(()))
    }

    /// Drops all queued audio, then queues `event`: start over from here. No `Gap`.
    pub fn flush(&self, event: E) -> Result<(), Stuck> {
        self.0.apply(|queue| queue.flush(event)).unwrap_or(Ok(()))
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }
}

/// The consumer.
pub struct AudioReceiver<E> {
    queue: Consumer<AudioQueue<E>>,
    /// The frames merging into the one received, reused across calls.
    rest: Vec<AudioFrame>,
}

impl<E> AudioReceiver<E> {
    /// The next item, pending audio merged. `None` once the sender is gone and the queue is
    /// drained. Cancel-safe, no allocation in steady state.
    pub async fn recv(&mut self) -> Option<Item<AudioFrame, E>> {
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
