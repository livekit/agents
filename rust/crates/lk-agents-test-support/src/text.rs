//! A text processor whose segment completion is controlled by the test.
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::voice::nodes::TextNode;
use lk_agents_abi::{BoxFuture, Monotone, TimedString};
use std::sync::Mutex;

pub struct BufferedTextNode {
    completed: Monotone<bool>,
    retained: Mutex<Option<Box<dyn Send>>>,
}

impl Default for BufferedTextNode {
    fn default() -> Self {
        Self { completed: Monotone::new(false), retained: Mutex::new(None) }
    }
}

impl BufferedTextNode {
    pub fn complete(&self) {
        let owner = {
            let mut retained = self.retained.lock().unwrap();
            self.completed.set(true);
            retained.take()
        };
        drop(owner);
    }
}

impl TextNode for BufferedTextNode {
    fn capture_text(&self, _: &str) {}
    fn capture_timed(&self, _: &TimedString) {}
    fn flush(&self) {}
    fn cancel(&self) {
        self.complete();
    }
    fn retain_until_drained(&self, owner: Box<dyn Send>) {
        let mut retained = self.retained.lock().unwrap();
        if !self.completed.get() {
            assert!(retained.is_none(), "a segment has one binding owner");
            *retained = Some(owner);
        } else {
            drop(retained);
            drop(owner);
        }
    }
    fn drain(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.completed.wait_for(|done| *done).await;
            Ok(())
        })
    }
}
