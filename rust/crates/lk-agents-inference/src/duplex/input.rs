use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::llm::duplex::{self as abi, DuplexInputGap, DuplexInputItem, DuplexInputMute};
use lk_agents_abi::{AudioFrame, Clock};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy, Stuck};
use lk_agents_net::env::ProcessEnv;
use lk_agents_telemetry as telemetry;

/// A duplex model talks over the caller in real time: old audio is worse than missing audio.
pub const INPUT_POLICY: Policy = Policy::Now(Duration::from_millis(250));

enum Event {
    Mute(bool),
    Flush,
}

pub struct DuplexInput {
    tx: AudioSender<Event>,
    rx: tokio::sync::Mutex<AudioReceiver<Event>>,
}

fn stuck(_: Stuck) -> ApiError {
    ApiError::new(ErrorKind::InvalidRequest, "the duplex plugin stopped pulling its input")
}

impl DuplexInput {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        let (tx, rx) = media_queue::audio(INPUT_POLICY, telemetry::AudioConsumer::Duplex, clock);
        Self { tx, rx: tokio::sync::Mutex::new(rx) }
    }

    /// The next item, pending audio merged; one puller at a time.
    pub async fn recv(&self) -> Option<DuplexInputItem> {
        let mut rx = self.rx.lock().await;
        loop {
            return Some(match rx.recv().await? {
                Item::Media(frame) => DuplexInputItem::Audio(frame),
                Item::Gap(gap) => {
                    DuplexInputItem::Gap(DuplexInputGap { duration: gap.as_secs_f64() })
                }
                Item::Event(Event::Mute(muted)) => DuplexInputItem::Mute(DuplexInputMute { muted }),
                Item::Event(Event::Flush) => continue,
            });
        }
    }
}

impl abi::DuplexInputQueue for DuplexInput {
    fn push(&self, frame: AudioFrame) {
        self.tx.push(frame);
    }

    fn head(&self, frame: AudioFrame) {
        self.tx.head(frame);
    }

    fn observe(&self, cx: &telemetry::Cx) {
        self.tx.observe(cx.clone());
    }
}

impl abi::DuplexInput for DuplexInput {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv) -> Self {
        DuplexInput::new(env.clock.clone())
    }

    fn push(&self, frame: AudioFrame) {
        abi::DuplexInputQueue::push(self, frame);
    }

    fn mute(&self, muted: bool) -> Result<(), ApiError> {
        self.tx.event(Event::Mute(muted)).map_err(stuck)
    }

    fn flush(&self) -> Result<(), ApiError> {
        self.tx.flush(Event::Flush).map_err(stuck)
    }

    async fn next(&self) -> Result<Option<DuplexInputItem>, ApiError> {
        Ok(self.recv().await)
    }
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod tests;
