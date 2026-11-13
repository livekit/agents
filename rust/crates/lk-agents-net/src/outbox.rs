//! A bounded channel to a receiver that may leave at any time.

use tokio::sync::mpsc;

/// A bounded channel whose receiver owes the sender nothing: it reads while it cares (a caller
/// awaiting a run's updates, a client reading a response) and may leave at any time, after
/// which nobody needs what is sent. `send` waits for room and never fails.
pub fn outbox<T>(capacity: usize) -> (Outbox<T>, mpsc::Receiver<T>) {
    let (tx, rx) = mpsc::channel(capacity);
    (Outbox(tx), rx)
}

/// The sending side of [`outbox`].
pub struct Outbox<T>(mpsc::Sender<T>);

/// A channel whose receiver turned out to owe the sender nothing for what is left to send.
impl<T> From<mpsc::Sender<T>> for Outbox<T> {
    fn from(sender: mpsc::Sender<T>) -> Self {
        Outbox(sender)
    }
}

impl<T> Clone for Outbox<T> {
    fn clone(&self) -> Self {
        Outbox(self.0.clone())
    }
}

impl<T> std::fmt::Debug for Outbox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Outbox").field("closed", &self.0.is_closed()).finish()
    }
}

impl<T> Outbox<T> {
    /// Hands `value` on once there is room; once the receiver left, nobody reads it.
    pub async fn send(&self, value: T) {
        match self.0.send(value).await {
            Ok(()) | Err(mpsc::error::SendError(_)) => {}
        }
    }

    /// Hands `value` on if there is room now: `false` if the queue is full and the receiver is
    /// still there.
    pub fn try_send(&self, value: T) -> bool {
        !matches!(self.0.try_send(value), Err(mpsc::error::TrySendError::Full(_)))
    }

    /// The receiver left.
    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Until the receiver leaves. Cancel-safe.
    pub async fn closed(&self) {
        self.0.closed().await;
    }

    /// Room the queue has now.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }
}
