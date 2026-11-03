//! The session report's process admission: recordings in flight and the retained session events
//! of every session. One per process; clones share it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Recordings in flight; further uploads wait, within their own deadline.
const UPLOADS: usize = 2;
/// The retained session events of every session in the process.
pub(super) const PROCESS_EVENT_ITEMS: usize = 65_536;
pub(super) const PROCESS_EVENT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct ReportAdmission {
    uploads: Arc<Semaphore>,
    pub(super) events: Arc<EventQuota>,
}

impl Default for ReportAdmission {
    fn default() -> Self {
        Self { uploads: Arc::new(Semaphore::new(UPLOADS)), events: Arc::new(EventQuota::default()) }
    }
}

impl ReportAdmission {
    /// The process's admission.
    pub fn process() -> Self {
        static ADMISSION: lk_agents_media::pool::PerProcess<ReportAdmission> =
            lk_agents_media::pool::PerProcess::new(ReportAdmission::default);
        ADMISSION.get()
    }

    /// Waits for a recording upload slot.
    pub(super) async fn upload(&self) -> OwnedSemaphorePermit {
        self.uploads.clone().acquire_owned().await.expect("the upload semaphore is never closed")
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
