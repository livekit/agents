//! What a `media_queue` consumer's bound dropped, measured as `lk.agents.audio.dropped` under the
//! observed job and warned about once, and how fast it drained its pre-connect head, measured as
//! `lk.agents.audio.catch_up_time` and logged.

use crate::telemetry;
use lk_agents_abi::Ts;
use lk_agents_media::media_queue::{Head, Policy};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub struct AudioDrops {
    consumer: telemetry::AudioConsumer,
    policy: Policy,
    cx: Mutex<Option<telemetry::Cx>>,
    warned: AtomicBool,
    catch_up: Mutex<CatchUp>,
}

/// The head being drained: since when, and the head received when the last one was measured.
#[derive(Default)]
struct CatchUp {
    since: Option<Ts>,
    measured: Duration,
}

impl AudioDrops {
    pub fn new(consumer: telemetry::AudioConsumer, policy: Policy) -> Self {
        Self {
            consumer,
            policy,
            cx: Mutex::new(None),
            warned: AtomicBool::new(false),
            catch_up: Mutex::new(CatchUp::default()),
        }
    }

    /// Measures how fast the consumer drains its head: called before each receive, with the
    /// receiver's [`Head`].
    pub fn catching_up(&self, head: Head, now: Ts) {
        let mut state = self.catch_up.lock();
        if !head.pending && head.received == state.measured {
            state.since = None;
            return;
        }
        let since = *state.since.get_or_insert(now);
        if head.pending {
            return;
        }
        let audio = head.received - std::mem::replace(&mut state.measured, head.received);
        state.since = None;
        drop(state);
        let took = now.saturating_duration_since(since);
        let (consumer, seconds) = (self.consumer, audio.as_secs_f64());
        let speed = seconds / took.as_secs_f64().max(1e-6);
        if took > audio {
            log::warn!(
                target: "livekit.agents",
                "audio consumer {consumer:?} took {:.3}s to drain {seconds:.2}s of pre-connect audio ({speed:.1}x real time): it falls behind the call",
                took.as_secs_f64()
            );
        } else {
            log::info!(
                target: "livekit.agents",
                "audio consumer {consumer:?} drained {seconds:.2}s of pre-connect audio in {:.3}s ({speed:.0}x real time)",
                took.as_secs_f64()
            );
        }
        let Some(cx) = self.cx.lock().clone() else { return };
        telemetry::measure(&cx, &telemetry::AudioCatchUp { consumer }, took.as_secs_f64());
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
    use lk_agents_media::media_queue;
    use std::sync::Arc;

    #[derive(Default)]
    struct Collect(Mutex<Vec<lk_agents_abi::telemetry::record::Record>>);

    impl lk_agents_abi::telemetry::record::RecordSink for Collect {
        fn record(&self, record: lk_agents_abi::telemetry::record::Record) {
            self.0.lock().push(record);
        }
    }

    impl Collect {
        fn measured(&self) -> Vec<(&'static str, f64)> {
            let records = self.0.lock();
            let measurements = records.iter().filter_map(|r| match r {
                lk_agents_abi::telemetry::record::Record::Measurement(m) => match m.value {
                    lk_agents_abi::telemetry::record::Number::Double(v) => Some((m.instrument, v)),
                    lk_agents_abi::telemetry::record::Number::Int(v) => {
                        Some((m.instrument, v as f64))
                    }
                },
                _ => None,
            });
            measurements.collect()
        }
    }

    fn observed(drops: &AudioDrops) -> Arc<Collect> {
        let sink = Arc::new(Collect::default());
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        drops.observe(telemetry::Tracer::new(sink.clone(), clock, 7).root());
        sink
    }

    // Catches: drops that never reach the `audio_dropped` metric, or a measurement per frame
    // pushed.
    #[test]
    fn drops_are_measured_as_seconds_of_audio_for_the_observed_consumer() {
        let policy = Policy::CatchUp(Duration::from_millis(100));
        let drops = AudioDrops::new(telemetry::AudioConsumer::RoomInput, policy);
        let sink = observed(&drops);
        let (tx, _rx) = media_queue::audio::<()>(policy);
        for _ in 0..12 {
            drops.record(tx.push(AudioFrame::new(vec![0; 320], 16_000, 1, 160)));
        }
        let measured: Vec<_> = sink.measured().into_iter().map(|(name, _)| name).collect();
        assert_eq!(measured, ["lk.agents.audio.dropped"; 2], "one per push that dropped");
    }

    // Catches: a head drain measured from before the head arrived, or measured again on live
    // audio after it.
    #[test]
    fn a_drained_head_is_measured_once_from_its_first_receive() {
        let drops = AudioDrops::new(telemetry::AudioConsumer::Vad, Policy::Now(Duration::ZERO));
        let sink = observed(&drops);
        let at = |ms: u64| Ts::from_nanos(ms * 1_000_000);
        let head = |ms: u64, pending: bool| Head { received: Duration::from_millis(ms), pending };
        drops.catching_up(head(0, false), at(0));
        drops.catching_up(head(50, true), at(100));
        drops.catching_up(head(5_000, true), at(150));
        drops.catching_up(head(10_000, false), at(200));
        drops.catching_up(head(10_000, false), at(300));
        assert_eq!(sink.measured(), [("lk.agents.audio.catch_up_time", 0.1)]);
    }
}
