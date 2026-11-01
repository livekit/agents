//! Explicit cancellation of an owned host operation, acknowledged by its normal result.

use lk_agents_abi::BoxFuture;
use tokio::sync::watch;

pub struct Cancellation(watch::Sender<bool>);
impl Default for Cancellation {
    fn default() -> Self {
        Self(watch::channel(false).0)
    }
}
impl Cancellation {
    pub fn cancel(&self) {
        self.0.send_replace(true);
    }
}
impl lk_agents_abi::invoke::Cancellation for Cancellation {
    fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }
    fn cancelled(&self) -> BoxFuture<'_, ()> {
        let mut changes = self.0.subscribe();
        Box::pin(async move {
            while !*changes.borrow_and_update() {
                if changes.changed().await.is_err() {
                    return;
                }
            }
        })
    }
}
