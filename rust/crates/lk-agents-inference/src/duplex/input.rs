//! `_core.DuplexInput`: a duplex session's input. The engine pushes the caller's audio into a `Now`
//! media queue and the plugin pulls it, so a plugin that falls behind skips to the newest audio
//! after a gap instead of answering audio that is seconds old. Mute and unmute travel in the audio
//! at the point they were asked; a flush (a reconnect) drops what is queued.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::duplex::{self as abi, DuplexInputGap, DuplexInputItem, DuplexInputMute};
use lk_agents_abi::{AudioFrame, Clock};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy, Stuck};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::telemetry;

/// A duplex model talks over the caller in real time: old audio is worse than missing audio.
pub const INPUT_POLICY: Policy = Policy::Now(Duration::from_millis(250));

enum Event {
    Mute(bool),
    Flush,
}

pub struct DuplexInput {
    tx: AudioSender<Event>,
    rx: tokio::sync::Mutex<AudioReceiver<Event>>,
    drops: AudioDrops,
    clock: Arc<dyn Clock>,
}

fn stuck(_: Stuck) -> ApiError {
    ApiError::new("the duplex plugin stopped pulling its input")
}

impl DuplexInput {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        let (tx, rx) = media_queue::audio(INPUT_POLICY);
        let drops = AudioDrops::new(telemetry::AudioConsumer::Duplex, INPUT_POLICY);
        Self { tx, rx: tokio::sync::Mutex::new(rx), drops, clock }
    }

    /// The next item, pending audio merged; one puller at a time.
    pub async fn recv(&self) -> Option<DuplexInputItem> {
        let mut rx = self.rx.lock().await;
        loop {
            self.drops.catching_up(rx.head(), self.clock.now());
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
        self.drops.record(self.tx.push(frame));
    }

    fn head(&self, frame: AudioFrame) {
        self.tx.head(frame);
    }

    fn observe(&self, cx: &telemetry::Cx) {
        self.drops.observe(cx.clone());
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
