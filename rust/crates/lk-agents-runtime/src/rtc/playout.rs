//! Serialized SDK calls into the voice playout ledger.

use lk_agents_abi::voice::rtc::PlaybackInfo;
use lk_agents_voice::room_io::playout::PlayoutLedger as Ledger;
use lk_agents_net::lock::Mutex;
use std::time::Duration;

#[derive(Default)]
pub struct PlayoutLedger(Mutex<Ledger>);

impl lk_agents_abi::voice::rtc::playout::PlayoutLedger for PlayoutLedger {
    fn new() -> Self {
        Self::default()
    }
    fn add_pushed_samples(&self, samples_per_channel: u32, sample_rate: u32) {
        self.0.lock().add_pushed_samples(samples_per_channel, sample_rate);
    }
    fn add_discarded(&self, duration: Duration) {
        self.0.lock().add_discarded(duration);
    }
    fn snapshot(&self, queued: Duration) -> PlaybackInfo {
        self.0.lock().snapshot(queued)
    }
    fn settle(&self) -> PlaybackInfo {
        self.0.lock().settle()
    }
}
