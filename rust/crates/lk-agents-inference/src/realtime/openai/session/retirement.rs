use super::{Pump, Shared};
use lk_agents_net::signal::{self, Cell, Producer};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

impl Pump {
    pub(super) fn wake_retirement(&self) {
        let start = {
            let mut retirement = self.shared.retirement.lock();
            if retirement.is_some() {
                None
            } else {
                let (settled, done) = signal::once();
                *retirement = Some(done);
                Some(settled)
            }
        };
        if let Some(settled) = start {
            let shared = Arc::downgrade(&self.shared);
            let wake = self.shared.retirement_wake.cell();
            let label = self.label.clone();
            self.shared.disposal.spawn(self.spawner.as_ref(), "realtime retirement", async move {
                run(shared, wake, label, settled, lk_agents_media::pool::media_pool()).await;
            });
        }
        self.shared.retirement_wake.modify(|n| *n += 1);
    }
}

/// Drops the session's retired context on the media pool until the session closes or is gone;
/// `settled` ends with it. `wake` ends with the session.
pub(super) async fn run(
    shared: Weak<Shared>,
    mut wake: Cell<u64>,
    label: String,
    settled: Producer<Option<()>>,
    media_pool: lk_agents_media::pool::MediaPool,
) {
    let _settled = settled;
    let mut retained = None;
    loop {
        let Some(shared) = retained.take().or_else(|| shared.upgrade()) else { return };
        let needed = shared.machine.lock().await.has_context_retirement();
        if !needed {
            if shared.retirement_closing.load(Ordering::Acquire) {
                return;
            }
            drop(shared);
            if wake.changed().await.is_err() {
                return;
            }
            continue;
        }
        let permit = match media_pool.try_admit(lk_agents_media::pool::JobClass::Formatting) {
            Ok(permit) => permit,
            Err(_) => {
                // Do not move a destructor-bearing owner into a closure until admission succeeds.
                media_pool.wait_for_capacity(lk_agents_media::pool::JobClass::Formatting).await;
                retained = Some(shared);
                continue;
            }
        };
        let retirement = shared.machine.lock().await.take_context_retirement();
        let Some(retirement) = retirement else {
            retained = Some(shared);
            continue;
        };
        shared.context_ready.modify(|n| *n += 1);
        if media_pool.spawn_admitted(permit, move || drop(retirement)).await.is_err() {
            let generation = shared.machine.lock().await.connection_generation();
            shared.incomplete_cleanup(generation, &label);
        }
        retained = Some(shared);
    }
}

#[cfg(test)]
#[path = "retirement_tests.rs"]
mod tests;
