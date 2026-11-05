//! Video: bounded by frame count, keeping the newest. It never merges and never reports a gap.

use super::core::{Consumer, Core, Producer, channel};
use super::{Item, Stuck};
use lk_agents_abi::voice::rtc::VideoFrameRef;

/// Keeps the newest `frames`.
pub fn video<E>(frames: usize) -> (VideoSender<E>, VideoReceiver<E>) {
    assert!(frames > 0, "a zero-frame video queue would drop everything");
    let (producer, consumer) = channel(VideoQueue { core: Core::new(), capacity: frames });
    (VideoSender(producer), VideoReceiver(consumer))
}

struct VideoQueue<E> {
    core: Core<VideoFrameRef, E>,
    capacity: usize,
}

/// The producer. Not `Clone`: one owner, so dropping it closes the queue.
pub struct VideoSender<E>(Producer<VideoQueue<E>>);

impl<E> VideoSender<E> {
    /// Never fails and never blocks; past the capacity the oldest frame is dropped.
    pub fn push(&self, frame: VideoFrameRef) {
        self.0.apply(|queue| {
            queue.core.push(frame);
            while queue.core.frames() > queue.capacity {
                queue.core.drop_oldest();
            }
        });
    }

    /// In order with the frames pushed before it. `Stuck` when the consumer stopped draining.
    pub fn event(&self, event: E) -> Result<(), Stuck> {
        self.0.apply(|queue| queue.core.event(event)).unwrap_or(Ok(()))
    }

    /// Drops all queued frames, then queues `event`: start over from here.
    pub fn flush(&self, event: E) -> Result<(), Stuck> {
        self.0.apply(|queue| queue.core.flush(event)).unwrap_or(Ok(()))
    }

    /// The receiver is gone: what is pushed goes nowhere.
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }
}

/// The consumer.
pub struct VideoReceiver<E>(Consumer<VideoQueue<E>>);

impl<E> VideoReceiver<E> {
    /// The next item. `None` once the sender is gone and the queue is drained. Cancel-safe.
    pub async fn recv(&mut self) -> Option<Item<VideoFrameRef, E>> {
        self.0.recv(|queue| queue.core.pop().map(|(item, _)| item)).await
    }

    /// Frames dropped so far, monotone.
    pub fn dropped(&self) -> u64 {
        self.0.read(|queue| queue.core.dropped())
    }
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
