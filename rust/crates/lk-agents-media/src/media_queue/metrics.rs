use super::{Head, Policy};
use lk_agents_abi::{Clock, Ts};
use lk_agents_telemetry as telemetry;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Shared by the queue's two ends: the sender reports drops, the receiver its head's drain.
pub(super) struct Meter {
    consumer: telemetry::AudioConsumer,
    policy: Policy,
    clock: Arc<dyn Clock>,
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

impl Meter {
    pub(super) fn new(
        consumer: telemetry::AudioConsumer,
        policy: Policy,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            consumer,
            policy,
            clock,
            cx: Mutex::new(None),
            warned: AtomicBool::new(false),
            catch_up: Mutex::new(CatchUp::default()),
        }
    }

    /// Measures under `cx`'s job from now on.
    pub(super) fn observe(&self, cx: telemetry::Cx) {
        *lock(&self.cx) = Some(cx);
    }

    fn cx(&self) -> Option<telemetry::Cx> {
        lock(&self.cx).clone()
    }

    /// Reports what one push dropped; nothing for zero.
    pub(super) fn dropped(&self, dropped: Duration) {
        if dropped.is_zero() {
            return;
        }
        let (consumer, policy) = (self.consumer, self.policy);
        if !self.warned.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: lk_agents_abi::LOG_TARGET,
                "audio consumer {consumer:?} fell behind, dropping audio ({policy:?}); suppressing repeats"
            );
        }
        let Some(cx) = self.cx() else { return };
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

    /// Measures how fast the consumer drains its head: called before each receive.
    pub(super) fn receiving(&self, head: Head) {
        self.catching_up(head, self.clock.now());
    }

    fn catching_up(&self, head: Head, now: Ts) {
        let mut state = lock(&self.catch_up);
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
                target: lk_agents_abi::LOG_TARGET,
                "audio consumer {consumer:?} took {:.3}s to drain {seconds:.2}s of pre-connect audio ({speed:.1}x real time): it falls behind the call",
                took.as_secs_f64()
            );
        } else {
            log::info!(
                target: lk_agents_abi::LOG_TARGET,
                "audio consumer {consumer:?} drained {seconds:.2}s of pre-connect audio in {:.3}s ({speed:.0}x real time)",
                took.as_secs_f64()
            );
        }
        let Some(cx) = self.cx() else { return };
        telemetry::measure(&cx, &telemetry::AudioCatchUp { consumer }, took.as_secs_f64());
    }
}

/// Every critical section leaves its state consistent, so a poisoned lock is still sound.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::super::audio;
    use super::*;
    use lk_agents_abi::AudioFrame;
    use lk_agents_abi::telemetry::record::{Number, Record, RecordSink};

    #[derive(Default)]
    struct Collect(Mutex<Vec<Record>>);

    impl RecordSink for Collect {
        fn record(&self, record: Record) {
            lock(&self.0).push(record);
        }
    }

    impl Collect {
        fn measured(&self) -> Vec<(&'static str, f64)> {
            let records = lock(&self.0);
            let measurements = records.iter().filter_map(|r| match r {
                Record::Measurement(m) => match m.value {
                    Number::Double(v) => Some((m.instrument, v)),
                    Number::Int(v) => Some((m.instrument, v as f64)),
                },
                _ => None,
            });
            measurements.collect()
        }
    }

    fn observed(clock: Arc<dyn Clock>) -> (Arc<Collect>, telemetry::Cx) {
        let sink = Arc::new(Collect::default());
        let cx = telemetry::Tracer::new(sink.clone(), clock, 7).root();
        (sink, cx)
    }

    // Catches: drops that never reach the `audio_dropped` metric, or a measurement per frame
    // pushed.
    #[test]
    fn drops_are_measured_as_seconds_of_audio_for_the_observed_consumer() {
        let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::VirtualClock::new());
        let (sink, cx) = observed(clock.clone());
        let policy = Policy::CatchUp(Duration::from_millis(100));
        let (tx, _rx) = audio::<()>(policy, telemetry::AudioConsumer::RoomInput, clock);
        tx.observe(cx);
        for _ in 0..12 {
            tx.push(AudioFrame::new(vec![0; 320], 16_000, 1, 160));
        }
        let measured: Vec<_> = sink.measured().into_iter().map(|(name, _)| name).collect();
        assert_eq!(measured, ["lk.agents.audio.dropped"; 2], "one per push that dropped");
    }

    // Catches: a head drain measured from before the head arrived, or measured again on live
    // audio after it.
    #[test]
    fn a_drained_head_is_measured_once_from_its_first_receive() {
        let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::VirtualClock::new());
        let (sink, cx) = observed(clock.clone());
        let meter = Meter::new(telemetry::AudioConsumer::Vad, Policy::Now(Duration::ZERO), clock);
        meter.observe(cx);
        let at = |ms: u64| Ts::from_nanos(ms * 1_000_000);
        let head = |ms: u64, pending: bool| Head { received: Duration::from_millis(ms), pending };
        meter.catching_up(head(0, false), at(0));
        meter.catching_up(head(50, true), at(100));
        meter.catching_up(head(5_000, true), at(150));
        meter.catching_up(head(10_000, false), at(200));
        meter.catching_up(head(10_000, false), at(300));
        assert_eq!(sink.measured(), [("lk.agents.audio.catch_up_time", 0.1)]);
    }
}
