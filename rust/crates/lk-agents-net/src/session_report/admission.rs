//! The session report's process admission (decision 88): transcripts in preparation, recordings
//! in flight and the retained session events of every session. The hosting process owns one and
//! lends it through [`Env::report_admission`](crate::env::Env::report_admission); clones share it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Transcripts in preparation, each retaining at most its transcript, event and console report
/// bounds. A refused preparation fails at once.
const PREPARATIONS: usize = 4;
/// Recordings in flight; a refused upload fails at once rather than waiting.
const UPLOADS: usize = 2;
/// The retained session events of every session in the process (`budgets.json`).
const PROCESS_EVENT_ITEMS: usize = 65_536;
const PROCESS_EVENT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct ReportAdmission {
    preparations: Arc<Semaphore>,
    uploads: Arc<Semaphore>,
    pub(super) events: Arc<EventQuota>,
}

impl Default for ReportAdmission {
    fn default() -> Self {
        Self {
            preparations: Arc::new(Semaphore::new(PREPARATIONS)),
            uploads: Arc::new(Semaphore::new(UPLOADS)),
            events: Arc::new(EventQuota::default()),
        }
    }
}

impl ReportAdmission {
    pub(super) fn preparation(&self) -> Option<OwnedSemaphorePermit> {
        self.preparations.clone().try_acquire_owned().ok()
    }

    pub(super) fn upload(&self) -> Option<OwnedSemaphorePermit> {
        self.uploads.clone().try_acquire_owned().ok()
    }

    #[cfg(test)]
    pub(super) fn uploads_available(&self) -> usize {
        self.uploads.available_permits()
    }
}

/// Retained event items and bytes. Quotas publish no payload; in-flight reservations may
/// conservatively refuse optional data.
#[derive(Default)]
pub(super) struct EventQuota {
    items: AtomicUsize,
    bytes: AtomicUsize,
}

impl EventQuota {
    pub(super) fn admit(&self, bytes: usize) -> bool {
        if self
            .items
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |items| {
                (items < PROCESS_EVENT_ITEMS).then_some(items + 1)
            })
            .is_err()
        {
            return false;
        }
        if self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                retained.checked_add(bytes).filter(|total| *total <= PROCESS_EVENT_BYTES)
            })
            .is_err()
        {
            self.items.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    pub(super) fn release(&self, items: usize, bytes: usize) {
        self.bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.items.fetch_sub(items, Ordering::Relaxed);
    }
}
