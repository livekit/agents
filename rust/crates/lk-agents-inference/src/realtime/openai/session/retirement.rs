//! One admitted disposal owner per realtime session, independent of socket/control dispatch.

use super::{Pump, Shared};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use tokio::sync::{Notify, watch};

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
            let label = self.label.clone();
            let media_pool = self.shared.media_pool.clone();
            self.shared.disposal.spawn(self.spawner.as_ref(), "realtime retirement", async move {
                run(shared, wake, label, settled, media_pool).await;
            });
        }
        self.shared.retirement_wake.notify_one();
    }
}

pub(super) async fn run(
    shared: Weak<Shared>,
    wake: Arc<Notify>,
    label: String,
    settled: watch::Sender<bool>,
    media_pool: lk_agents_media::pool::MediaPool,
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
        shared.context_ready.notify_one();
        if media_pool.spawn_admitted(permit, move || drop(retirement)).await.is_err() {
            let generation = shared.machine.lock().await.connection_generation();
            shared.incomplete_cleanup(generation, &label);
        }
        retained = Some(shared);
    }
}
