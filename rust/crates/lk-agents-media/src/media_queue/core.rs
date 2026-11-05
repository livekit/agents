//! What both kinds share: the queued items with their events, and the lock and waker between the
//! producer and the consumer.

use super::{Item, MAX_EVENTS, Stuck};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Poll, Waker};
use std::time::Duration;

/// The queued items. The bound drops only the media past the floor; the kind measures it.
pub(super) struct Core<M, E> {
    items: VecDeque<Item<M, E>>,
    /// Just after the last event or head frame: the bound drops only from here on.
    floor: usize,
    /// The media from `floor` on.
    frames: usize,
    events: usize,
    dropped: u64,
}

impl<M, E> Core<M, E> {
    pub(super) fn new() -> Self {
        Self { items: VecDeque::new(), floor: 0, frames: 0, events: 0, dropped: 0 }
    }

    /// Queues media the bound may drop.
    pub(super) fn push(&mut self, media: M) {
        self.frames += 1;
        self.items.push_back(Item::Media(media));
    }

    /// Queues media the bound never drops.
    pub(super) fn head(&mut self, media: M) {
        self.items.push_back(Item::Media(media));
        self.settle();
    }

    /// In order with the media pushed before it. `Stuck` when the consumer stopped draining.
    pub(super) fn event(&mut self, event: E) -> Result<(), Stuck> {
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
    pub(super) fn flush(&mut self, event: E) -> Result<(), Stuck> {
        self.items.retain(|item| matches!(item, Item::Event(_)));
        self.settle();
        self.event(event)
    }

    /// The media past the floor, which the bound may drop.
    pub(super) fn frames(&self) -> usize {
        self.frames
    }

    /// Frames dropped by the bound so far, monotone.
    pub(super) fn dropped(&self) -> u64 {
        self.dropped
    }

    pub(super) fn front(&self) -> Option<&Item<M, E>> {
        self.items.front()
    }

    /// The next item, and whether it was media past the floor.
    pub(super) fn pop(&mut self) -> Option<(Item<M, E>, bool)> {
        let item = self.items.pop_front()?;
        if matches!(item, Item::Event(_)) {
            self.events -= 1;
        }
        let counted = if self.floor > 0 {
            self.floor -= 1;
            false
        } else {
            matches!(item, Item::Media(_))
        };
        if counted {
            self.frames -= 1;
        }
        Some((item, counted))
    }

    /// Drops the oldest media past the floor, after the gap that may start the run.
    pub(super) fn drop_oldest(&mut self) -> M {
        let gap = matches!(self.items.get(self.floor), Some(Item::Gap(_)));
        let Some(Item::Media(media)) = self.items.remove(self.floor + usize::from(gap)) else {
            unreachable!("only media follows the floor's gap");
        };
        self.frames -= 1;
        self.dropped += 1;
        media
    }

    /// Widens the gap that starts the run past the floor by `duration`.
    pub(super) fn gap(&mut self, duration: Duration) {
        match self.items.get_mut(self.floor) {
            Some(Item::Gap(covered)) => *covered += duration,
            _ => self.items.insert(self.floor, Item::Gap(duration)),
        }
    }

    /// Nothing queued so far is the bound's to drop.
    fn settle(&mut self) {
        self.floor = self.items.len();
        self.frames = 0;
    }
}

pub(super) fn channel<Q>(queue: Q) -> (Producer<Q>, Consumer<Q>) {
    let shared = Arc::new(Mutex::new(Slot { queue: Some(queue), sender: true, waker: None }));
    (Producer(Arc::clone(&shared)), Consumer(shared))
}

struct Slot<Q> {
    /// `None` once the consumer is gone.
    queue: Option<Q>,
    sender: bool,
    waker: Option<Waker>,
}

/// Every critical section leaves the queue consistent, so a poisoned lock is still sound.
fn lock<Q>(shared: &Mutex<Slot<Q>>) -> MutexGuard<'_, Slot<Q>> {
    shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The producer's end. Not `Clone`: one owner, so dropping it closes the queue.
pub(super) struct Producer<Q>(Arc<Mutex<Slot<Q>>>);

impl<Q> Producer<Q> {
    /// Applies `op` to an open queue, then wakes the consumer once the lock is released.
    pub(super) fn apply<R>(&self, op: impl FnOnce(&mut Q) -> R) -> Option<R> {
        let (out, waker) = {
            let mut slot = lock(&self.0);
            let out = op(slot.queue.as_mut()?);
            (out, slot.waker.take())
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Some(out)
    }

    /// The consumer is gone: what is pushed goes nowhere.
    pub(super) fn is_closed(&self) -> bool {
        lock(&self.0).queue.is_none()
    }
}

impl<Q> Drop for Producer<Q> {
    fn drop(&mut self) {
        let waker = {
            let mut slot = lock(&self.0);
            slot.sender = false;
            slot.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The consumer's end.
pub(super) struct Consumer<Q>(Arc<Mutex<Slot<Q>>>);

impl<Q> Consumer<Q> {
    /// The next item `pop` takes. `None` once the producer is gone and the queue is drained.
    /// Cancel-safe.
    pub(super) async fn recv<T>(&self, mut pop: impl FnMut(&mut Q) -> Option<T>) -> Option<T> {
        std::future::poll_fn(|cx| {
            let mut slot = lock(&self.0);
            let slot = &mut *slot;
            let Some(queue) = slot.queue.as_mut() else { return Poll::Ready(None) };
            if let Some(item) = pop(queue) {
                Poll::Ready(Some(item))
            } else if !slot.sender {
                Poll::Ready(None)
            } else {
                if !slot.waker.as_ref().is_some_and(|waker| waker.will_wake(cx.waker())) {
                    slot.waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
        })
        .await
    }

    pub(super) fn read<R: Default>(&self, op: impl FnOnce(&Q) -> R) -> R {
        lock(&self.0).queue.as_ref().map_or_else(R::default, op)
    }
}

impl<Q> Drop for Consumer<Q> {
    fn drop(&mut self) {
        let queue = lock(&self.0).queue.take();
        drop(queue);
    }
}
