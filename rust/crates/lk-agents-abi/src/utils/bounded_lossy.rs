use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Notify;

/// One second at 20 ms.
pub const AUDIO_QUEUE_FRAMES: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushed {
    Ok,
    /// After evicting the oldest item; counted in [`BoundedLossy::dropped`].
    DroppedOldest,
    /// The queue is closed (receiver gone or input ended), so a pushing host learns it.
    Closed,
}

struct Shared<T> {
    q: Mutex<Queue<T>>,
    /// `Notify` stores one permit, so an early push is not lost.
    ready: Notify,
    dropped: AtomicU64,
}

struct Queue<T> {
    items: VecDeque<T>,
    cap: usize,
    closed: bool,
}

impl<T> Shared<T> {
    fn push(&self, v: T) -> Pushed {
        let outcome = {
            // A std `Mutex`, never held across an await: the host push is a synchronous method.
            let mut q = self.q.lock().unwrap();
            if q.closed {
                return Pushed::Closed;
            }
            if q.items.len() >= q.cap {
                q.items.pop_front();
                self.dropped.fetch_add(1, Ordering::Relaxed);
                q.items.push_back(v);
                Pushed::DroppedOldest
            } else {
                q.items.push_back(v);
                Pushed::Ok
            }
        };
        self.ready.notify_one();
        outcome
    }

    fn close(&self) {
        self.q.lock().unwrap().closed = true;
        self.ready.notify_one();
    }
}

/// Drop-oldest, for the audio-in legs. Not `mpsc`: `try_send` drops the newest frame (the
/// interruption) and `send().await` backpressures a GIL-holding callback. Never for text->TTS.
pub struct BoundedLossy<T> {
    shared: Arc<Shared<T>>,
}

/// Not `Clone`: one owner, so dropping it closes the queue (a kept clone means the stream never
/// ends).
pub struct LossyTx<T> {
    shared: Arc<Shared<T>>,
}

impl<T> BoundedLossy<T> {
    /// No separate sender, for a Rust producer; a host producer uses [`channel`](Self::channel).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "a zero-capacity lossy queue would drop everything");
        BoundedLossy {
            shared: Arc::new(Shared {
                q: Mutex::new(Queue {
                    items: VecDeque::with_capacity(capacity),
                    cap: capacity,
                    closed: false,
                }),
                ready: Notify::new(),
                dropped: AtomicU64::new(0),
            }),
        }
    }

    /// Dropping the sender ends [`recv`](Self::recv).
    pub fn channel(capacity: usize) -> (LossyTx<T>, BoundedLossy<T>) {
        let rx = Self::new(capacity);
        let tx = LossyTx { shared: Arc::clone(&rx.shared) };
        (tx, rx)
    }

    pub fn audio() -> (LossyTx<T>, BoundedLossy<T>) {
        Self::channel(AUDIO_QUEUE_FRAMES)
    }

    /// Never blocks, never awaits.
    pub fn push(&self, v: T) -> Pushed {
        self.shared.push(v)
    }

    /// `None` once closed and drained. Cancel-safe: a dropped future loses no item.
    pub async fn recv(&mut self) -> Option<T> {
        loop {
            {
                let mut q = self.shared.q.lock().unwrap();
                if let Some(v) = q.items.pop_front() {
                    return Some(v);
                }
                if q.closed {
                    return None;
                }
            }
            // Check, then park: `notify_one`'s single permit covers a push in between.
            self.shared.ready.notified().await;
        }
    }

    /// Monotone; the driver rate-limits the warning it drives.
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.shared.q.lock().unwrap().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Queued items still drain.
    pub fn close(&self) {
        self.shared.close();
    }
}

impl<T> LossyTx<T> {
    pub fn push(&self, v: T) -> Pushed {
        self.shared.push(v)
    }

    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }

    /// Equivalent to dropping, and clearer at a call site that ends a turn.
    pub fn close(self) {}
}

impl<T> Drop for LossyTx<T> {
    fn drop(&mut self) {
        // The one sender going away is end-of-input.
        self.shared.close();
    }
}

impl<T> Drop for BoundedLossy<T> {
    fn drop(&mut self) {
        // Receiver gone: stop accepting and free what is queued.
        let mut q = self.shared.q.lock().unwrap();
        q.closed = true;
        q.items.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Real-time guard: these hazards fail as hangs.
    const GUARD: Duration = Duration::from_secs(2);

    async fn within<F: std::future::Future>(f: F) -> F::Output {
        tokio::time::timeout(GUARD, f).await.expect("the queue hung instead of answering")
    }

    // Catches `mpsc::try_send`'s behaviour: rejecting the newest frame when full.
    #[tokio::test]
    async fn a_full_queue_evicts_the_oldest_frame_not_the_newest() {
        let mut q = BoundedLossy::new(3);
        assert_eq!(q.push(1), Pushed::Ok);
        assert_eq!(q.push(2), Pushed::Ok);
        assert_eq!(q.push(3), Pushed::Ok);
        assert_eq!(q.push(4), Pushed::DroppedOldest);
        assert_eq!(q.push(5), Pushed::DroppedOldest);

        assert_eq!(q.dropped(), 2, "the loss must be observable to the driver's warn");
        assert_eq!(q.len(), 3, "eviction, not growth");
        assert_eq!(within(q.recv()).await, Some(3), "the three most recent frames survive");
        assert_eq!(within(q.recv()).await, Some(4));
        assert_eq!(within(q.recv()).await, Some(5));
    }

    // Catches: nothing observing hang-up, so `recv` never returns `None`; and hang-up discarding the queue.
    #[tokio::test]
    async fn recv_drains_then_ends_when_the_sender_hangs_up() {
        let (tx, mut rx) = BoundedLossy::audio();
        tx.push("hello");
        tx.push("world");
        drop(tx);

        assert_eq!(within(rx.recv()).await, Some("hello"));
        assert_eq!(within(rx.recv()).await, Some("world"));
        assert_eq!(within(rx.recv()).await, None, "recv must end when the only sender is gone");
    }

    // Catches a lost wakeup: checking before registering for notification. Real OS thread: the
    // push side is synchronous host code.
    #[tokio::test]
    async fn a_push_racing_the_receiver_still_wakes_it() {
        let (tx, mut rx) = BoundedLossy::audio();
        let start = std::sync::Arc::new(std::sync::Barrier::new(2));

        // A real OS thread: a pusher with no reactor of its own.
        let producer = {
            let start = std::sync::Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for i in 0..200 {
                    tx.push(i);
                }
            })
        };

        start.wait();
        let mut seen = 0;
        while seen < 200 {
            match tokio::time::timeout(GUARD, rx.recv()).await {
                Ok(Some(_)) => seen += 1,
                Ok(None) => break, // sender finished; the rest were evicted, which is legal
                Err(_) => panic!("the receiver slept through a push (lost wakeup) after {seen}"),
            }
        }
        producer.join().unwrap();
        assert!(seen + rx.dropped() as usize >= 200, "every push either arrived or was counted");
    }

    // Catches: a `select!` arm dropping the in-flight `recv` and taking a frame with it.
    #[tokio::test]
    async fn an_abandoned_recv_does_not_consume_a_frame() {
        let (tx, mut rx) = BoundedLossy::audio();

        {
            // Empty, so this parks -- the state a losing `select!` arm is in.
            let mut waiting = Box::pin(rx.recv());
            assert!(tokio::time::timeout(Duration::from_millis(5), &mut waiting).await.is_err());
            // The frame lands while that arm is parked, and the arm is then dropped unpolled.
            tx.push(7);
        }

        let got = within(rx.recv()).await;
        assert_eq!(got, Some(7), "the abandoned recv consumed the frame, or ate its wakeup");
        assert_eq!(rx.dropped(), 0, "an abandoned recv is not a drop");
    }

    // Catches: pushes into a torn-down leg getting no `Closed` answer.
    #[tokio::test]
    async fn pushing_after_the_receiver_is_gone_reports_closed() {
        let (tx, rx) = BoundedLossy::audio();
        assert_eq!(tx.push(1), Pushed::Ok);
        drop(rx);
        assert_eq!(tx.push(2), Pushed::Closed);
        assert_eq!(tx.push(3), Pushed::Closed);
    }
}
