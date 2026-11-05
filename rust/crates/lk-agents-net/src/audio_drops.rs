//! What a `media_queue` consumer's bound dropped, measured as `lk.agents.audio.dropped` under the
//! observed job and warned about once.

use crate::telemetry;
use lk_agents_media::media_queue::Policy;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub struct AudioDrops {
    consumer: telemetry::AudioConsumer,
    policy: Policy,
    cx: Mutex<Option<telemetry::Cx>>,
    warned: AtomicBool,
}

impl AudioDrops {
    pub fn new(consumer: telemetry::AudioConsumer, policy: Policy) -> Self {
        Self { consumer, policy, cx: Mutex::new(None), warned: AtomicBool::new(false) }
    }

    /// Measures drops under `cx`'s job from now on.
    pub fn observe(&self, cx: telemetry::Cx) {
        *self.cx.lock() = Some(cx);
    }

    /// Reports what one push dropped; nothing for zero.
    pub fn record(&self, dropped: Duration) {
        if dropped.is_zero() {
            return;
        }
        let (consumer, policy) = (self.consumer, self.policy);
        if !self.warned.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "livekit.agents",
                "audio consumer {consumer:?} fell behind, dropping audio ({policy:?}); suppressing repeats"
            );
        }
        let Some(cx) = self.cx.lock().clone() else { return };
        let policy = match policy {
            Policy::CatchUp(_) => telemetry::AudioPolicy::CatchUp,
            Policy::Now(_) => telemetry::AudioPolicy::Now,
        };
        telemetry::measure(
            &cx,
            &telemetry::AudioDropped { policy, consumer },
            dropped.as_secs_f64(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::AudioFrame;
    use lk_agents_abi::telemetry::record::{Record, RecordSink};
    use lk_agents_media::media_queue;
    use std::sync::Arc;

    // Catches: drops that never reach the `audio_dropped` metric, or a measurement per frame
    // pushed.
    #[test]
    fn drops_are_measured_as_seconds_of_audio_for_the_observed_consumer() {
        #[derive(Default)]
        struct Collect(Mutex<Vec<Record>>);
        impl RecordSink for Collect {
            fn record(&self, record: Record) {
                self.0.lock().push(record);
            }
        }
        let sink = Arc::new(Collect::default());
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let tracer = telemetry::Tracer::new(sink.clone(), clock, 7);
        let policy = Policy::CatchUp(Duration::from_millis(100));
        let drops = AudioDrops::new(telemetry::AudioConsumer::RoomInput, policy);
        drops.observe(tracer.root());
        let (tx, _rx) = media_queue::audio::<()>(policy);
        for _ in 0..12 {
            drops.record(tx.push(AudioFrame::new(vec![0; 320], 16_000, 1, 160)));
        }
        let measured: Vec<_> = sink
            .0
            .lock()
            .iter()
            .filter_map(|r| match r {
                Record::Measurement(m) => Some(m.instrument),
                _ => None,
            })
            .collect();
        assert_eq!(measured, ["lk.agents.audio.dropped"; 2], "one per push that dropped");
    }
}
