//! The audio-in legs' drop-oldest queue between a synchronous producer and the runtime.

use lk_agents_abi::AUDIO_QUEUE_FRAMES;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

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
    dropped: AtomicU64,
}

struct Queue<T> {
    items: VecDeque<T>,
    cap: usize,
    closed: bool,
    receiver: Option<Waker>,
}

impl<T> Shared<T> {
    fn push(&self, v: T) -> Pushed {
        let (outcome, receiver) = {
            // Never held across an await: the host push is a synchronous method.
            let mut q = self.q.lock();
            if q.closed {
                return Pushed::Closed;
            }
            let outcome = if q.items.len() >= q.cap {
                q.items.pop_front();
                self.dropped.fetch_add(1, Ordering::Relaxed);
                q.items.push_back(v);
                Pushed::DroppedOldest
            } else {
                q.items.push_back(v);
                Pushed::Ok
            };
            (outcome, q.receiver.take())
        };
        if let Some(receiver) = receiver {
            receiver.wake();
        }
        outcome
    }

    fn close(&self) {
        let receiver = {
            let mut q = self.q.lock();
            q.closed = true;
            q.receiver.take()
        };
        if let Some(receiver) = receiver {
            receiver.wake();
        }
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
                    receiver: None,
                }),
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
        Receive { shared: &self.shared }.await
    }

    /// Monotone; the driver rate-limits the warning it drives.
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.shared.q.lock().items.len()
    }
}

// The queue check and registration share the producer's lock, so no wake can be lost.
struct Receive<'a, T> {
    shared: &'a Shared<T>,
}

impl<T> Future for Receive<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut q = self.shared.q.lock();
        if let Some(item) = q.items.pop_front() {
            Poll::Ready(Some(item))
        } else if q.closed {
            Poll::Ready(None)
        } else {
            q.receiver = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

impl<T> Drop for Receive<'_, T> {
    fn drop(&mut self) {
        self.shared.q.lock().receiver = None;
    }
}

impl<T> LossyTx<T> {
    pub fn push(&self, v: T) -> Pushed {
        self.shared.push(v)
    }
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
        let mut q = self.shared.q.lock();
        q.closed = true;
        q.items.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::pin;
    use std::time::Duration;

    /// Real-time guard: a lost wakeup fails instead of hanging.
    const GUARD: Duration = Duration::from_secs(2);

    /// One poll: every wait below is on a queue that already holds its answer.
    fn ready<F: Future>(f: F) -> F::Output {
        match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("the queue hung instead of answering"),
        }
    }

    /// Polls `f` on this thread, parked between polls; a wait no wake ends fails.
    fn woken<F: Future>(f: F) -> F::Output {
        struct Unpark(std::thread::Thread, std::sync::OnceLock<()>);
        impl std::task::Wake for Unpark {
            fn wake(self: Arc<Self>) {
                let _ = self.1.set(());
                self.0.unpark();
            }
        }
        let mut f = pin!(f);
        loop {
            let unpark = Arc::new(Unpark(std::thread::current(), std::sync::OnceLock::new()));
            let waker = Waker::from(unpark.clone());
            if let Poll::Ready(v) = f.as_mut().poll(&mut Context::from_waker(&waker)) {
                return v;
            }
            std::thread::park_timeout(GUARD);
            assert!(unpark.1.get().is_some(), "the receiver slept through a push (lost wakeup)");
        }
    }

    // Catches `mpsc::try_send`'s behaviour: rejecting the newest frame when full.
    #[test]
    fn a_full_queue_evicts_the_oldest_frame_not_the_newest() {
        let mut q = BoundedLossy::new(3);
        assert_eq!(q.push(1), Pushed::Ok);
        assert_eq!(q.push(2), Pushed::Ok);
        assert_eq!(q.push(3), Pushed::Ok);
        assert_eq!(q.push(4), Pushed::DroppedOldest);
        assert_eq!(q.push(5), Pushed::DroppedOldest);

        assert_eq!(q.dropped(), 2, "the loss must be observable to the driver's warn");
        assert_eq!(q.len(), 3, "eviction, not growth");
        assert_eq!(ready(q.recv()), Some(3), "the three most recent frames survive");
        assert_eq!(ready(q.recv()), Some(4));
        assert_eq!(ready(q.recv()), Some(5));
    }

    // Catches: nothing observing hang-up, so `recv` never returns `None`; and hang-up discarding the queue.
    #[test]
    fn recv_drains_then_ends_when_the_sender_hangs_up() {
        let (tx, mut rx) = BoundedLossy::audio();
        tx.push("hello");
        tx.push("world");
        drop(tx);

        assert_eq!(ready(rx.recv()), Some("hello"));
        assert_eq!(ready(rx.recv()), Some("world"));
        assert_eq!(ready(rx.recv()), None, "recv must end when the only sender is gone");
    }

    // Catches a lost wakeup: checking before registering for notification. Real OS thread: the
    // push side is synchronous host code.
    #[test]
    fn a_push_racing_the_receiver_still_wakes_it() {
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
            match woken(rx.recv()) {
                Some(_) => seen += 1,
                None => break, // sender finished; the rest were evicted, which is legal
            }
        }
        producer.join().unwrap();
        assert!(seen + rx.dropped() as usize >= 200, "every push either arrived or was counted");
    }

    // Catches: a `select!` arm dropping the in-flight `recv` and taking a frame with it.
    #[test]
    fn an_abandoned_recv_does_not_consume_a_frame() {
        let (tx, mut rx) = BoundedLossy::audio();

        {
            // Empty, so this parks -- the state a losing `select!` arm is in.
            let waiting = pin!(rx.recv());
            assert!(waiting.poll(&mut Context::from_waker(Waker::noop())).is_pending());
            // The frame lands while that arm is parked, and the arm is then dropped unpolled.
            tx.push(7);
        }

        let got = ready(rx.recv());
        assert_eq!(got, Some(7), "the abandoned recv consumed the frame, or ate its wakeup");
        assert_eq!(rx.dropped(), 0, "an abandoned recv is not a drop");
    }

    // Catches: pushes into a torn-down leg getting no `Closed` answer.
    #[test]
    fn pushing_after_the_receiver_is_gone_reports_closed() {
        let (tx, rx) = BoundedLossy::audio();
        assert_eq!(tx.push(1), Pushed::Ok);
        drop(rx);
        assert_eq!(tx.push(2), Pushed::Closed);
        assert_eq!(tx.push(3), Pushed::Closed);
    }
}
