//! Live media from a producer that never blocks to a consumer that may fall behind, after
//! GStreamer's leaky queue and its serialized events.
//!
//! - Audio is bounded by its queued duration under a [`Policy`]; past it the oldest frames are
//!   dropped as one [`Item::Gap`]. A backlog is merged into frames of up to [`MERGED_FRAME`].
//! - Video is bounded by frame count and keeps the newest; it never merges and never reports a gap.
//! - Events travel in the media at the point they apply to and are never dropped. The bound drops
//!   only media queued after the last event; [`MediaQueue::flush`] is the one way to discard media
//!   on purpose.
//!
//! [`MediaQueue`] is the queue itself, single-owner. [`audio`] and [`video`] share one between a
//! producer that never blocks and one async consumer: the lock is held only for the queue's own
//! short work, the receiver's waker is registered under it and woken after it is released, and a
//! merged frame is built outside it. No thread, no spawn, no clock.

use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::rtc::{VideoFrameRef, frame_duration};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Poll, Waker};
use std::time::Duration;

/// The longest frame merging builds: the frame size every consumer is tested against.
pub const MERGED_FRAME: Duration = Duration::from_millis(50);

/// Events pending past which the consumer counts as stuck.
pub const MAX_EVENTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item<M, E> {
    Media(M),
    /// Media dropped by the bound, as the duration it covered (audio only).
    Gap(Duration),
    Event(E),
}

/// What an audio queue keeps when its consumer falls behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Keep up to this much queued audio; drop the oldest beyond it.
    CatchUp(Duration),
    /// Once the consumer lags more than this, drop to the newest audio.
    Now(Duration),
}

/// More than [`MAX_EVENTS`] events are pending: the consumer stopped draining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("media consumer stopped draining its events")]
pub struct Stuck;

/// A frame a queue holds.
pub trait Media: Sized {
    /// The time it covers; zero for video, which is bounded by frame count.
    fn duration(&self) -> Duration;
    /// Whether `next` continues the merged frame `first` followed by `rest`.
    fn joins(first: &Self, rest: &[Self], next: &Self) -> bool;
    /// `first` then `rest` as one frame; `rest` is left empty for reuse.
    fn merge(first: Self, rest: &mut Vec<Self>) -> Self;
}

impl Media for AudioFrame {
    fn duration(&self) -> Duration {
        frame_duration(self)
    }

    fn joins(first: &Self, rest: &[Self], next: &Self) -> bool {
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

    fn merge(first: Self, rest: &mut Vec<Self>) -> Self {
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
}

/// A frame whose payload is exactly its declared samples, so concatenating keeps both in step.
fn whole(frame: &AudioFrame) -> bool {
    frame.sample_rate > 0
        && frame.data.len()
            == frame.samples_per_channel as usize * usize::from(frame.num_channels) * 2
}

impl Media for VideoFrameRef {
    fn duration(&self) -> Duration {
        Duration::ZERO
    }

    fn joins(_: &Self, _: &[Self], _: &Self) -> bool {
        false
    }

    fn merge(first: Self, _: &mut Vec<Self>) -> Self {
        first
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    Audio(Policy),
    Frames(usize),
}

pub struct MediaQueue<M, E> {
    items: VecDeque<Item<M, E>>,
    bound: Bound,
    /// Just after the last event or head frame: the bound drops only from here on.
    floor: usize,
    /// The media from `floor` on, which the bound measures.
    lag: Duration,
    frames: usize,
    events: usize,
    dropped: u64,
}

impl<E> MediaQueue<AudioFrame, E> {
    pub fn audio(policy: Policy) -> Self {
        Self::new(Bound::Audio(policy))
    }

    /// Pre-connect audio: kept in full, outside both policies.
    pub fn head(&mut self, frame: AudioFrame) {
        self.items.push_back(Item::Media(frame));
        self.settle();
    }
}

impl<E> MediaQueue<VideoFrameRef, E> {
    /// Keeps the newest `frames`.
    pub fn video(frames: usize) -> Self {
        assert!(frames > 0, "a zero-frame video queue would drop everything");
        Self::new(Bound::Frames(frames))
    }
}

impl<M: Media, E> MediaQueue<M, E> {
    fn new(bound: Bound) -> Self {
        Self {
            items: VecDeque::new(),
            bound,
            floor: 0,
            lag: Duration::ZERO,
            frames: 0,
            events: 0,
            dropped: 0,
        }
    }

    /// The audio policy; `None` for video.
    pub fn policy(&self) -> Option<Policy> {
        match self.bound {
            Bound::Audio(policy) => Some(policy),
            Bound::Frames(_) => None,
        }
    }

    /// Never fails; the bound decides what is kept. Returns the audio it dropped.
    pub fn push(&mut self, media: M) -> Duration {
        self.lag += media.duration();
        self.frames += 1;
        self.items.push_back(Item::Media(media));
        let mut dropped = Duration::ZERO;
        match self.bound {
            Bound::Audio(Policy::CatchUp(max)) => {
                while self.lag > max && self.frames > 1 {
                    dropped += self.drop_oldest();
                }
            }
            Bound::Audio(Policy::Now(max)) => {
                if self.lag > max {
                    while self.frames > 1 {
                        dropped += self.drop_oldest();
                    }
                }
            }
            Bound::Frames(max) => {
                while self.frames > max {
                    self.drop_oldest();
                }
            }
        }
        dropped
    }

    /// In order with the media pushed before it. `Stuck` when the consumer stopped draining.
    pub fn event(&mut self, event: E) -> Result<(), Stuck> {
        if self.events >= MAX_EVENTS {
            return Err(Stuck);
        }
        self.events += 1;
        self.items.push_back(Item::Event(event));
        self.settle();
        Ok(())
    }

    /// Drops all queued media (events stay, in order), then queues `event`: start over from here.
    /// No `Gap`. The media is dropped even when the consumer is `Stuck`.
    pub fn flush(&mut self, event: E) -> Result<(), Stuck> {
        self.items.retain(|item| matches!(item, Item::Event(_)));
        self.settle();
        self.event(event)
    }

    /// Frames dropped by the bound so far, monotone.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The next item, pending audio merged into frames of up to [`MERGED_FRAME`].
    pub fn pop(&mut self) -> Option<Item<M, E>> {
        let mut rest = Vec::new();
        match self.pop_unmerged(&mut rest)? {
            Item::Media(first) if !rest.is_empty() => Some(Item::Media(M::merge(first, &mut rest))),
            item => Some(item),
        }
    }

    /// [`pop`](Self::pop), leaving the frames that merge into a popped frame in `rest` for
    /// [`Media::merge`], so a shared queue merges outside its lock. A frame that merges with
    /// nothing leaves `rest` empty and passes as pushed.
    pub fn pop_unmerged(&mut self, rest: &mut Vec<M>) -> Option<Item<M, E>> {
        let item = self.pop_front()?;
        if let Item::Media(first) = &item {
            while let Some(Item::Media(next)) = self.items.front()
                && M::joins(first, rest, next)
            {
                if let Some(Item::Media(next)) = self.pop_front() {
                    rest.push(next);
                }
            }
        }
        Some(item)
    }

    fn pop_front(&mut self) -> Option<Item<M, E>> {
        let item = self.items.pop_front()?;
        match &item {
            _ if self.floor > 0 => self.floor -= 1,
            Item::Media(media) => {
                self.lag = self.lag.saturating_sub(media.duration());
                self.frames -= 1;
            }
            Item::Gap(_) | Item::Event(_) => {}
        }
        if matches!(item, Item::Event(_)) {
            self.events -= 1;
        }
        Some(item)
    }

    /// Nothing queued so far is the bound's to drop.
    fn settle(&mut self) {
        self.floor = self.items.len();
        self.lag = Duration::ZERO;
        self.frames = 0;
    }

    /// Drops the oldest frame the bound may drop; audio folds into the gap that starts the run.
    fn drop_oldest(&mut self) -> Duration {
        let gap = matches!(self.items.get(self.floor), Some(Item::Gap(_)));
        let Some(Item::Media(media)) = self.items.remove(self.floor + usize::from(gap)) else {
            unreachable!("only media follows the floor's gap");
        };
        let duration = media.duration();
        self.lag = self.lag.saturating_sub(duration);
        self.frames -= 1;
        self.dropped += 1;
        if let Bound::Audio(_) = self.bound {
            match self.items.get_mut(self.floor) {
                Some(Item::Gap(covered)) => *covered += duration,
                _ => self.items.insert(self.floor, Item::Gap(duration)),
            }
        }
        duration
    }
}

pub fn audio<E>(policy: Policy) -> (Sender<AudioFrame, E>, Receiver<AudioFrame, E>) {
    channel(MediaQueue::audio(policy))
}

/// Keeps the newest `frames`; never merges, never reports a gap.
pub fn video<E>(frames: usize) -> (Sender<VideoFrameRef, E>, Receiver<VideoFrameRef, E>) {
    channel(MediaQueue::video(frames))
}

fn channel<M, E>(queue: MediaQueue<M, E>) -> (Sender<M, E>, Receiver<M, E>) {
    let shared = Arc::new(Mutex::new(Shared { queue: Some(queue), sender: true, waker: None }));
    (Sender { shared: Arc::clone(&shared) }, Receiver { shared, rest: Vec::new() })
}

struct Shared<M, E> {
    /// `None` once the receiver is gone.
    queue: Option<MediaQueue<M, E>>,
    sender: bool,
    waker: Option<Waker>,
}

/// Every critical section leaves the queue consistent, so a poisoned lock is still sound.
fn lock<M, E>(shared: &Mutex<Shared<M, E>>) -> MutexGuard<'_, Shared<M, E>> {
    shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The producer. Not `Clone`: one owner, so dropping it closes the queue.
pub struct Sender<M, E> {
    shared: Arc<Mutex<Shared<M, E>>>,
}

impl<M: Media, E> Sender<M, E> {
    /// Never fails and never blocks; the bound decides what is kept. Returns the audio it dropped,
    /// for the owner to report.
    pub fn push(&self, media: M) -> Duration {
        self.apply(|queue| queue.push(media)).unwrap_or_default()
    }

    /// In order with the media pushed before it. `Stuck` when the consumer stopped draining.
    pub fn event(&self, event: E) -> Result<(), Stuck> {
        self.apply(|queue| queue.event(event)).unwrap_or(Ok(()))
    }

    /// Drops all queued media, then queues `event`: start over from here. No `Gap`.
    pub fn flush(&self, event: E) -> Result<(), Stuck> {
        self.apply(|queue| queue.flush(event)).unwrap_or(Ok(()))
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        lock(&self.shared).queue.is_none()
    }

    /// Applies `op` to an open queue, then wakes the receiver once the lock is released.
    fn apply<R>(&self, op: impl FnOnce(&mut MediaQueue<M, E>) -> R) -> Option<R> {
        let (out, waker) = {
            let mut shared = lock(&self.shared);
            let out = op(shared.queue.as_mut()?);
            (out, shared.waker.take())
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Some(out)
    }
}

impl<E> Sender<AudioFrame, E> {
    /// Pre-connect audio: kept in full, outside both policies.
    pub fn head(&self, frame: AudioFrame) {
        self.apply(|queue| queue.head(frame));
    }
}

impl<M, E> Drop for Sender<M, E> {
    fn drop(&mut self) {
        let waker = {
            let mut shared = lock(&self.shared);
            shared.sender = false;
            shared.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The consumer.
pub struct Receiver<M, E> {
    shared: Arc<Mutex<Shared<M, E>>>,
    /// The frames merging into the one received, reused across calls.
    rest: Vec<M>,
}

impl<M: Media, E> Receiver<M, E> {
    /// The next item, pending audio merged. `None` once the sender is gone and the queue is
    /// drained. Cancel-safe, no allocation in steady state.
    pub async fn recv(&mut self) -> Option<Item<M, E>> {
        let rest = &mut self.rest;
        let item = std::future::poll_fn(|cx| {
            let mut shared = lock(&self.shared);
            let shared = &mut *shared;
            let Some(queue) = shared.queue.as_mut() else { return Poll::Ready(None) };
            if let Some(item) = queue.pop_unmerged(rest) {
                Poll::Ready(Some(item))
            } else if !shared.sender {
                Poll::Ready(None)
            } else {
                if !shared.waker.as_ref().is_some_and(|waker| waker.will_wake(cx.waker())) {
                    shared.waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
        })
        .await?;
        Some(match item {
            Item::Media(first) if !self.rest.is_empty() => {
                Item::Media(M::merge(first, &mut self.rest))
            }
            item => item,
        })
    }

    /// Frames the bound dropped so far, monotone.
    pub fn dropped(&self) -> u64 {
        lock(&self.shared).queue.as_ref().map_or(0, MediaQueue::dropped)
    }
}

impl<M, E> Drop for Receiver<M, E> {
    fn drop(&mut self) {
        let queue = lock(&self.shared).queue.take();
        drop(queue);
    }
}

#[cfg(test)]
#[path = "media_queue_tests.rs"]
mod tests;
