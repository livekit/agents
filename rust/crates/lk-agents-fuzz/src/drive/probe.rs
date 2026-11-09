use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

use super::*;
use crate::input::SpeechWait;
use crate::record::Wait;

// Host waits armed at random points, each a task of the world's that stamps the instant it
// resolves: `every_wait_resolves` reads when, never the run's polling.

/// How long past the close a wait may take: the session's close watchdog (`CLOSE_WATCHDOG` in
/// the runtime's session, private there) and the host teardown bound after it.
pub const WAIT_BOUND_MS: u64 = 30_000 + lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_millis() as u64;

/// Waiters still parked on a runtime signal, by where the wait and its signal were made.
pub(super) fn parked_waiters() -> Vec<String> {
    let parked = lk_agents_runtime::net::signal::park::parked().into_iter();
    parked.map(|w| format!("{} (signal made at {})", w.waiter, w.producer)).collect()
}

#[derive(Default)]
pub(super) struct Probes {
    /// What the stressed run draws its own probes from; `None` arms only the case's.
    rng: Option<fastrand::Rng>,
    /// `(probe, resolved at)`, left by the world's tasks.
    resolved: Arc<Mutex<Vec<(usize, Observed)>>>,
    /// The host's await of `aclose()`, which the run polls itself (`Run::closing`).
    pub(super) close: Option<usize>,
}

impl Probes {
    pub(super) fn new(stress: Option<u64>) -> Self {
        let rng = stress.map(|seed| fastrand::Rng::with_seed(seed ^ 0x0070_726f_6265));
        Probes { rng, resolved: Arc::default(), close: None }
    }
}

impl Run {
    /// The host starts awaiting `wait`, kept until it resolves.
    pub(super) fn probe(&mut self, wait: Wait, future: BoxFuture<'static, ()>) -> usize {
        let index = self.record.record_probe_armed(self.stamps.next(), wait);
        let (stamps, resolved) = (self.stamps.clone(), self.probes.resolved.clone());
        self.registry.world.spawn_detached(async move {
            future.await;
            resolved.lock().unwrap().push((index, stamps.next()));
        });
        index
    }

    /// One of a speech handle's waits, created or not.
    pub(super) fn probe_speech(&mut self, handle: Arc<SpeechHandle>, wait: SpeechWait) {
        let id = handle.reserved().map_or(0, |id| id.0);
        let (wait, future): (Wait, BoxFuture<'static, ()>) = match wait {
            SpeechWait::Created => (
                Wait::Created(id),
                Box::pin(async move {
                    let _ = HostSpeech::wait_until_created(&*handle).await;
                }),
            ),
            SpeechWait::Scheduled => (
                Wait::Scheduled(id),
                Box::pin(async move {
                    let _ = handle.wait_scheduled().await;
                }),
            ),
            SpeechWait::Playout => {
                (Wait::Playout(id), Box::pin(async move { handle.wait_for_playout().await }))
            }
            SpeechWait::Generation => (
                Wait::Generation(id),
                Box::pin(async move {
                    let _ = handle.wait_for_generation(0).await;
                }),
            ),
            SpeechWait::LatestGeneration => (
                Wait::LatestGeneration(id),
                Box::pin(async move {
                    let _ = handle.wait_for_latest_generation().await;
                }),
            ),
            SpeechWait::StepAuthorized => (
                Wait::StepAuthorized(id),
                Box::pin(async move { handle.wait_for_step_authorized(0).await }),
            ),
            SpeechWait::Reported => (
                Wait::Reported(id),
                Box::pin(async move { HostSpeech::wait_reported(&*handle).await }),
            ),
        };
        self.probe(wait, future);
    }

    /// In a stressed run, one wait drawn on a speech the host just asked for, before it exists.
    pub(super) fn probe_new_speech(&mut self, handle: &Arc<SpeechHandle>) {
        let Some(rng) = &mut self.probes.rng else { return };
        let waits = [
            SpeechWait::Created,
            SpeechWait::Scheduled,
            SpeechWait::Playout,
            SpeechWait::Generation,
            SpeechWait::LatestGeneration,
            SpeechWait::StepAuthorized,
            SpeechWait::Reported,
        ];
        let wait = waits[rng.usize(0..waits.len())];
        self.probe_speech(handle.clone(), wait);
    }

    /// The host's `aclose()` await resolved now.
    pub(super) fn closed_probe(&mut self) {
        if let Some(probe) = self.probes.close.take() {
            self.record.record_probe_resolved(probe, self.stamps.next());
        }
    }

    /// Stamps what the world's probe tasks saw resolve.
    pub(super) fn absorb_probes(&mut self) {
        let resolved = std::mem::take(&mut *self.probes.resolved.lock().unwrap());
        for (probe, at) in resolved {
            self.record.record_probe_resolved(probe, at);
        }
    }

    /// Lets every probe still pending run to its bound past the close, so a wait that would end
    /// in time is not judged early.
    pub(super) async fn await_probes(&mut self) {
        self.absorb_probes();
        let Some(close) = self.record.close_begun() else { return };
        let latest = self
            .record
            .probes()
            .iter()
            .filter(|p| p.resolved.is_none())
            .map(|p| p.armed.at_ms.max(close.at_ms))
            .max();
        let Some(from) = latest else { return };
        let until = Ts::ZERO.saturating_add(Duration::from_millis(from + WAIT_BOUND_MS + 1));
        if until > self.time.now() {
            self.time.wait_until(until).await;
        }
        self.absorb_probes();
    }
}
