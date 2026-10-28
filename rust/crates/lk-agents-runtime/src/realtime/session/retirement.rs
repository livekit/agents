//! One admitted disposal owner per realtime session, independent of socket/control dispatch.

use super::{Pump, Shared};
use lk_agents_abi::{Clock, SpawnerExt};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Notify, watch};

// A failed process-service startup retains ownership and retries without blocking controls.
const SERVICE_RETRY: Duration = Duration::from_millis(2);

impl Pump {
    pub(super) fn wake_retirement(&self) {
        let start = {
            let mut retirement = self.shared.retirement.lock().unwrap_or_else(|e| e.into_inner());
            if retirement.is_some() {
                None
            } else {
                let (settled, done) = watch::channel(false);
                *retirement = Some(done);
                Some(settled)
            }
        };
        if let Some(settled) = start {
            let shared = Arc::downgrade(&self.shared);
            let wake = self.shared.retirement_wake.clone();
            let clock = self.clock.clone();
            let label = self.label.clone();
            let _owner = self.spawner.spawn_unboxed(run(shared, wake, clock, label, settled, None));
        }
        self.shared.retirement_wake.notify_one();
    }
}

pub(super) async fn run(
    shared: Weak<Shared>,
    wake: Arc<Notify>,
    clock: Arc<dyn Clock>,
    label: String,
    settled: watch::Sender<bool>,
    mut workers: Option<crate::media_workers::MediaWorkers>,
) {
    let mut retained = None;
    loop {
        let notified = wake.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let Some(shared) = retained.take().or_else(|| shared.upgrade()) else {
            settled.send_replace(true);
            return;
        };
        let needed = shared.machine.lock().await.has_context_retirement();
        if !needed {
            if shared.retirement_closing.load(Ordering::Acquire) {
                settled.send_replace(true);
                return;
            }
            drop(shared);
            tokio::select! {
                () = notified => {},
                () = settled.closed() => return,
            }
            continue;
        }
        if workers.is_none() {
            match crate::process::workers() {
                Ok(service) => workers = Some(service),
                Err(_) => {
                    let generation = shared.machine.lock().await.connection_generation();
                    shared.incomplete_cleanup(generation, &label);
                    clock.wait_until(clock.now().saturating_add(SERVICE_RETRY)).await;
                    retained = Some(shared);
                    continue;
                }
            }
        }
        let workers = workers.as_ref().expect("worker service initialized");
        let permit = match workers.try_admit(crate::media_workers::MediaWorkerClass::Formatting) {
            Ok(permit) => permit,
            Err(_) => {
                // Do not move a destructor-bearing owner into a closure until admission succeeds.
                workers.wait_for_capacity(crate::media_workers::MediaWorkerClass::Formatting).await;
                retained = Some(shared);
                continue;
            }
        };
        let retirement = shared.machine.lock().await.take_context_retirement();
        let Some(mut retirement) = retirement else {
            retained = Some(shared);
            continue;
        };
        if workers.spawn_admitted(permit, move || while !retirement.advance(64) {}).await.is_err() {
            let generation = shared.machine.lock().await.connection_generation();
            shared.incomplete_cleanup(generation, &label);
        }
        retained = Some(shared);
    }
}
