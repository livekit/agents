//! A cursor owns consumer admission even before its first `next()` is polled.

use super::*;
use lk_agents_abi::agent::stream::PresentedEvent;

pub(super) struct Cursor {
    inner: Arc<Inner>,
    consumer: u64,
    closed: watch::Sender<bool>,
}
impl Cursor {
    pub(super) fn new(inner: Arc<Inner>, consumer: u64) -> Self {
        Self { inner, consumer, closed: watch::channel(false).0 }
    }
    fn close(&self) {
        self.closed.send_replace(true);
        self.inner.release(self.consumer);
    }
    fn begin_read(&self) -> Result<bool, StreamError> {
        if *self.closed.borrow() {
            return Ok(false);
        }
        self.inner.policy.lock().begin_read(self.consumer)?;
        Ok(true)
    }
    fn end_read(&self, accepted: bool) {
        self.inner.projection.lock().acknowledge(self.consumer, accepted);
        self.inner.policy.lock().end_read(self.consumer);
    }
    fn close_idle(&self) -> Result<(), StreamError> {
        self.inner.policy.lock().close_cursor(self.consumer)?;
        self.closed.send_replace(true);
        Ok(())
    }
    async fn next(&self) -> Result<Option<Item>, StreamError> {
        if *self.closed.borrow() {
            return Ok(None);
        }
        let item = self.inner.pull(self.consumer, Some(&self.closed)).await;
        if !matches!(item, Ok(Some(_))) {
            self.close();
        }
        item
    }
}
impl Drop for Cursor {
    fn drop(&mut self) {
        self.close();
    }
}
pub struct Events(pub(super) Cursor);
pub struct Text(pub(super) Cursor);
impl stream::StreamEvents for Events {
    fn begin_read(&self) -> Result<bool, StreamError> {
        self.0.begin_read()
    }
    fn end_read(&self, accepted: bool) {
        self.0.end_read(accepted);
    }
    async fn recv(&self) -> Result<Option<PresentedEvent>, StreamError> {
        match self.0.next().await? {
            Some(Item::Event(event)) => Ok(Some(*event)),
            _ => Ok(None),
        }
    }
    async fn aclose(&self) -> Result<(), StreamError> {
        self.0.close_idle()
    }
}
impl stream::StreamText for Text {
    fn begin_read(&self) -> Result<bool, StreamError> {
        self.0.begin_read()
    }
    fn end_read(&self, accepted: bool) {
        self.0.end_read(accepted);
    }
    async fn recv(&self) -> Result<Option<String>, StreamError> {
        match self.0.next().await? {
            Some(Item::Text(text)) => Ok(Some(text)),
            _ => Ok(None),
        }
    }
    async fn aclose(&self) -> Result<(), StreamError> {
        self.0.close_idle()
    }
}
