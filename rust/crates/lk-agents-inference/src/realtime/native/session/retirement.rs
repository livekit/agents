//! One admitted disposal owner per realtime session, independent of socket/control dispatch.

use super::{Pump, Shared};
use lk_agents_abi::{Clock, SpawnerExt};
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
            let clock = self.clock.clone();
            let label = self.label.clone();
            let events = self.events.clone();
            let workers = self.shared.workers;
            let _owner = self.spawner.spawn_unboxed(async move {
                run(
                    shared,
                    wake,
                    clock,
                    label,
                    events,
                    settled,
                    workers().map(lk_agents_media::workers::MediaWorkers::from),
                )
                .await;
            });
        }
        self.shared.retirement_wake.notify_one();
    }
}

pub(super) async fn run(
    shared: Weak<Shared>,
    wake: Arc<Notify>,
    clock: Arc<dyn Clock>,
    label: String,
    events: super::super::queue::Delivery,
    settled: watch::Sender<bool>,
    workers: Result<lk_agents_media::workers::MediaWorkers, String>,
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
        let workers = match &workers {
            Ok(service) => service,
            Err(error) => {
                let (generation, step, retired) = {
                    let mut machine = shared.machine.lock().await;
                    let generation = machine.connection_generation();
                    let step = machine.close(clock.now());
                    let mut retired = Vec::new();
                    while let Some(owner) = machine.take_context_retirement() {
                        retired.push(owner);
                    }
                    (generation, step, retired)
                };
                shared.retirement_unavailable(generation, &label, error);
                shared.retirement_closing.store(true, Ordering::Release);
                for event in step.events {
                    if let Err(error) = events.send(event) {
                        events.fail(clock.now(), &label, error);
                    }
                }
                let error = lk_agents_abi::errors::ApiError::new(format!(
                    "realtime context cleanup service is unavailable: {error}"
                ))
                .with_retryable(false);
                let event = lk_agents_abi::llm::realtime::Event::Error(
                    lk_agents_abi::llm::realtime::RealtimeModelError {
                        timestamp: clock.now(),
                        label: label.clone(),
                        message: error.message.clone(),
                        recoverable: false,
                        req_id: None,
                        code: None,
                        provider_error: Some(error),
                    },
                );
                if let Err(error) = events.send(event) {
                    events.fail(clock.now(), &label, error);
                }
                shared.closing.notify_one();
                // Failed process startup cannot admit a replacement worker. Refuse this session
                // and reclaim once outside its state lock instead of retaining and retrying.
                drop(retired);
                shared.context_ready.notify_one();
                settled.send_replace(true);
                return;
            }
        };
        let permit = match workers.try_admit(lk_agents_media::workers::MediaWorkerClass::Formatting)
        {
            Ok(permit) => permit,
            Err(_) => {
                // Do not move a destructor-bearing owner into a closure until admission succeeds.
                workers
                    .wait_for_capacity(lk_agents_media::workers::MediaWorkerClass::Formatting)
                    .await;
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
        if workers.spawn_admitted(permit, move || drop(retirement)).await.is_err() {
            let generation = shared.machine.lock().await.connection_generation();
            shared.incomplete_cleanup(generation, &label);
        }
        retained = Some(shared);
    }
}
