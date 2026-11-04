//! Media into one consumer, after GStreamer's leaky queue.
//!
//! - [`audio_channel`]: old audio past the policy's bound leaks as a [`Item::Gap`], control items
//!   never do. Audio and control share one ordered queue, so a control is never reordered against
//!   the audio it applies to.
//! - [`video_channel`]: the newest few frames; the oldest is dropped past the capacity.

use crate::telemetry;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::rtc::frame_duration;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// The longest frame merging builds: the frame size every consumer is tested against.
pub const MERGED_FRAME: Duration = Duration::from_millis(50);

/// Controls pending past which the consumer counts as stuck.
pub const MAX_CONTROLS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item<C> {
    Audio(AudioFrame),
    /// Audio dropped by the policy, as the duration it covered.
    Gap(Duration),
    Control(C),
}

/// What the channel keeps when its consumer falls behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Everything up to `max` of audio; the oldest is dropped beyond it.
    CatchUp { max: Duration },
    /// Once the consumer lags more than `max_lag`, only the newest frame is kept.
    Now { max_lag: Duration },
}

/// More than [`MAX_CONTROLS`] controls are pending: the consumer stopped draining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("audio consumer stopped draining its controls")]
pub struct Stuck;

pub fn audio_channel<C>(policy: Policy) -> (Sender<C>, Receiver<C>) {
    let shared = Arc::new(Mutex::new(Queue {
        items: VecDeque::new(),
        barrier: 0,
        floor: 0,
        lag: Duration::ZERO,
        controls: 0,
        policy,
        sender: true,
        receiver: true,
        waker: None,
        dropped: Duration::ZERO,
        observer: None,
        warned: false,
    }));
    (Sender { shared: Arc::clone(&shared) }, Receiver { shared })
}

struct Queue<C> {
    items: VecDeque<Item<C>>,
    /// Just after the last control: `clear()` drops back to here.
    barrier: usize,
    /// Just after the last control or head frame: the policy drops only from here on.
    floor: usize,
    /// The audio from `floor` on, which the policy bounds.
    lag: Duration,
    controls: usize,
    policy: Policy,
    sender: bool,
    receiver: bool,
    waker: Option<Waker>,
    dropped: Duration,
    observer: Option<(telemetry::Cx, telemetry::AudioConsumer)>,
    warned: bool,
}

impl<C> Queue<C> {
    /// Appends `item` and hands back the receiver's waker, to wake once the lock is released.
    fn append(&mut self, item: Item<C>) -> Option<Waker> {
        self.items.push_back(item);
        self.waker.take()
    }

    /// Audio frames the policy may still drop.
    fn droppable(&self) -> usize {
        let gap = matches!(self.items.get(self.floor), Some(Item::Gap(_)));
        self.items.len() - self.floor - usize::from(gap)
    }

    /// Drops the oldest droppable frame into the gap that starts the droppable run.
    fn drop_oldest(&mut self) -> Duration {
        let gap = matches!(self.items.get(self.floor), Some(Item::Gap(_)));
        let Some(Item::Audio(frame)) = self.items.remove(self.floor + usize::from(gap)) else {
            unreachable!("only audio follows the floor's gap");
        };
        let duration = frame_duration(&frame);
        match self.items.get_mut(self.floor) {
            Some(Item::Gap(covered)) if gap => *covered += duration,
            _ => self.items.insert(self.floor, Item::Gap(duration)),
        }
        self.lag = self.lag.saturating_sub(duration);
        duration
    }

    /// Applies the policy after a push: what it dropped.
    fn enforce(&mut self) -> Duration {
        let mut dropped = Duration::ZERO;
        match self.policy {
            Policy::CatchUp { max } => {
                while self.lag > max && self.droppable() > 1 {
                    dropped += self.drop_oldest();
                }
            }
            Policy::Now { max_lag } => {
                if self.lag > max_lag {
                    while self.droppable() > 1 {
                        dropped += self.drop_oldest();
                    }
                }
            }
        }
        self.dropped += dropped;
        dropped
    }

    /// Everything pending, up to and including the first control, unmerged.
    fn take(&mut self) -> Vec<Item<C>> {
        let end = match self.items.iter().position(|item| matches!(item, Item::Control(_))) {
            Some(control) => control + 1,
            None => self.items.len(),
        };
        let taken: Vec<_> = self.items.drain(..end).collect();
        if self.items.is_empty() {
            self.lag = Duration::ZERO;
        }
        self.barrier = self.barrier.saturating_sub(end);
        self.floor = self.floor.saturating_sub(end);
        self.controls -= usize::from(matches!(taken.last(), Some(Item::Control(_))));
        taken
    }
}

/// Merges contiguous frames of one format into frames of up to [`MERGED_FRAME`]. A lone frame
/// passes through as pushed.
fn merge<C>(items: Vec<Item<C>>) -> Vec<Item<C>> {
    let mut merged = Vec::with_capacity(items.len());
    let mut run: Vec<AudioFrame> = Vec::new();
    for item in items {
        match item {
            Item::Audio(frame) if mergeable(&frame) => {
                let fits = run.first().is_some_and(|first| {
                    let samples: u32 = run.iter().map(|f| f.samples_per_channel).sum();
                    first.sample_rate == frame.sample_rate
                        && first.num_channels == frame.num_channels
                        && u64::from(samples + frame.samples_per_channel) * 1000
                            <= u64::from(frame.sample_rate) * MERGED_FRAME.as_millis() as u64
                });
                if !fits {
                    flush(&mut run, &mut merged);
                }
                run.push(frame);
            }
            item => {
                flush(&mut run, &mut merged);
                merged.push(item);
            }
        }
    }
    flush(&mut run, &mut merged);
    merged
}

/// A frame whose payload is exactly its declared samples, so concatenating keeps both in step.
fn mergeable(frame: &AudioFrame) -> bool {
    frame.sample_rate > 0
        && frame.data.len()
            == frame.samples_per_channel as usize * usize::from(frame.num_channels) * 2
}

fn flush<C>(run: &mut Vec<AudioFrame>, out: &mut Vec<Item<C>>) {
    match run.len() {
        0 => {}
        1 => out.extend(run.pop().map(Item::Audio)),
        _ => {
            let mut data = bytes::BytesMut::with_capacity(run.iter().map(|f| f.data.len()).sum());
            for frame in run.iter() {
                data.extend_from_slice(&frame.data);
            }
            let samples = run.iter().map(|f| f.samples_per_channel).sum();
            let (rate, channels) = (run[0].sample_rate, run[0].num_channels);
            out.push(Item::Audio(AudioFrame::new(data.freeze(), rate, channels, samples)));
            run.clear();
        }
    }
}

/// The producer. Not `Clone`: one owner, so dropping it closes the channel.
pub struct Sender<C> {
    shared: Arc<Mutex<Queue<C>>>,
}

impl<C> Sender<C> {
    /// Never fails and never blocks; the policy decides what is kept.
    pub fn push(&self, frame: AudioFrame) {
        self.push_audio(frame, false);
    }

    /// Pre-connect audio: kept in full, outside the policy.
    pub fn head(&self, frame: AudioFrame) {
        self.push_audio(frame, true);
    }

    fn push_audio(&self, frame: AudioFrame, head: bool) {
        let (waker, dropped, report) = {
            let mut q = self.shared.lock();
            if !q.receiver {
                return;
            }
            let duration = frame_duration(&frame);
            let waker = q.append(Item::Audio(frame));
            if head {
                q.floor = q.items.len();
                (waker, Duration::ZERO, None)
            } else {
                q.lag += duration;
                let dropped = q.enforce();
                let warn = !dropped.is_zero() && !std::mem::replace(&mut q.warned, true);
                let report = (!dropped.is_zero()).then(|| (q.observer.clone(), q.policy, warn));
                (waker, dropped, report)
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        if let Some((observer, policy, warn)) = report {
            report_drop(observer, policy, dropped, warn);
        }
    }

    /// Ordered after the audio pushed before it. `Stuck` when the consumer stopped draining.
    pub fn control(&self, control: C) -> Result<(), Stuck> {
        let waker = {
            let mut q = self.shared.lock();
            if !q.receiver {
                return Ok(());
            }
            if q.controls >= MAX_CONTROLS {
                return Err(Stuck);
            }
            q.controls += 1;
            let waker = q.append(Item::Control(control));
            q.barrier = q.items.len();
            q.floor = q.barrier;
            q.lag = Duration::ZERO;
            waker
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    /// Drops the audio not yet received, back to the last control item; no `Gap`.
    pub fn clear(&self) {
        let mut q = self.shared.lock();
        let barrier = q.barrier;
        q.items.truncate(barrier);
        q.floor = barrier;
        q.lag = Duration::ZERO;
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        !self.shared.lock().receiver
    }

    /// Reports drops as `consumer`'s, under `cx`'s job.
    pub fn observe(&self, cx: telemetry::Cx, consumer: telemetry::AudioConsumer) {
        self.shared.lock().observer = Some((cx, consumer));
    }
}

fn report_drop(
    observer: Option<(telemetry::Cx, telemetry::AudioConsumer)>,
    policy: Policy,
    dropped: Duration,
    warn: bool,
) {
    let label = match policy {
        Policy::CatchUp { .. } => telemetry::AudioPolicy::CatchUp,
        Policy::Now { .. } => telemetry::AudioPolicy::Now,
    };
    if warn {
        let consumer = observer.as_ref().map_or("unlabeled".into(), |(_, c)| format!("{c:?}"));
        log::warn!(
            target: "livekit.agents",
            "audio consumer {consumer} fell behind, dropping audio ({policy:?}); suppressing repeats"
        );
    }
    if let Some((cx, consumer)) = observer {
        let labels = telemetry::AudioDropped { policy: label, consumer };
        telemetry::measure(&cx, &labels, dropped.as_secs_f64());
    }
}

impl<C> Drop for Sender<C> {
    fn drop(&mut self) {
        let waker = {
            let mut q = self.shared.lock();
            q.sender = false;
            q.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The consumer.
pub struct Receiver<C> {
    shared: Arc<Mutex<Queue<C>>>,
}

impl<C> Receiver<C> {
    /// Everything pending, merged, up to and including the first control item. `None` once the
    /// sender is gone and everything is drained. Cancel-safe.
    pub async fn recv(&mut self) -> Option<Vec<Item<C>>> {
        Receive { shared: &self.shared }.await
    }

    /// Audio dropped so far, monotone.
    pub fn dropped(&self) -> Duration {
        self.shared.lock().dropped
    }
}

impl<C> Drop for Receiver<C> {
    fn drop(&mut self) {
        let mut q = self.shared.lock();
        q.receiver = false;
        q.items.clear();
    }
}

// The check and the waker registration share the producer's lock, so no wake can be lost.
struct Receive<'a, C> {
    shared: &'a Mutex<Queue<C>>,
}

impl<C> Future for Receive<'_, C> {
    type Output = Option<Vec<Item<C>>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut q = self.shared.lock();
        if !q.items.is_empty() {
            let taken = q.take();
            drop(q);
            // Merging copies, so it runs outside the producer's lock.
            Poll::Ready(Some(merge(taken)))
        } else if !q.sender {
            Poll::Ready(None)
        } else {
            q.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

impl<C> Drop for Receive<'_, C> {
    fn drop(&mut self) {
        self.shared.lock().waker = None;
    }
}

/// Drop-oldest by frame count, for video: each held frame pins a decoded host frame, and the
/// newest is the one worth keeping. Not `mpsc`: `try_send` drops the newest frame and
/// `send().await` backpressures a GIL-holding callback.
pub fn video_channel<T>(capacity: usize) -> (VideoSender<T>, VideoReceiver<T>) {
    assert!(capacity > 0, "a zero-capacity video channel would drop everything");
    let shared = Arc::new(Mutex::new(Frames {
        items: VecDeque::with_capacity(capacity),
        capacity,
        sender: true,
        receiver: true,
        waker: None,
        dropped: 0,
    }));
    (VideoSender { shared: Arc::clone(&shared) }, VideoReceiver { shared })
}

struct Frames<T> {
    items: VecDeque<T>,
    capacity: usize,
    sender: bool,
    receiver: bool,
    waker: Option<Waker>,
    dropped: u64,
}

/// The producer. Not `Clone`: one owner, so dropping it closes the channel.
pub struct VideoSender<T> {
    shared: Arc<Mutex<Frames<T>>>,
}

impl<T> VideoSender<T> {
    /// Never fails and never blocks; past the capacity the oldest frame is dropped.
    pub fn push(&self, frame: T) {
        let waker = {
            let mut q = self.shared.lock();
            if !q.receiver {
                return;
            }
            if q.items.len() >= q.capacity {
                q.items.pop_front();
                q.dropped += 1;
            }
            q.items.push_back(frame);
            q.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        !self.shared.lock().receiver
    }
}

impl<T> Drop for VideoSender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut q = self.shared.lock();
            q.sender = false;
            q.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The consumer.
pub struct VideoReceiver<T> {
    shared: Arc<Mutex<Frames<T>>>,
}

impl<T> VideoReceiver<T> {
    /// The oldest frame held; `None` once the sender is gone and everything is drained.
    /// Cancel-safe.
    pub async fn recv(&mut self) -> Option<T> {
        std::future::poll_fn(|cx| {
            let mut q = self.shared.lock();
            if let Some(frame) = q.items.pop_front() {
                Poll::Ready(Some(frame))
            } else if !q.sender {
                Poll::Ready(None)
            } else {
                q.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await
    }

    /// Frames dropped so far, monotone.
    pub fn dropped(&self) -> u64 {
        self.shared.lock().dropped
    }
}

impl<T> Drop for VideoReceiver<T> {
    fn drop(&mut self) {
        let mut q = self.shared.lock();
        q.receiver = false;
        q.items.clear();
    }
}

#[cfg(test)]
#[path = "leaky_tests.rs"]
mod tests;
