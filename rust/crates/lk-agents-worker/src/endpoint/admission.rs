//! Reservations for endpoint exchange and inbox payload ownership.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use super::WsExchange;

use super::budget::{EndpointBudget, EndpointOccupancy, EndpointPayloadBudget};
use lk_agents_abi::worker::endpoint::EndpointLimits;

/// The endpoint's handler and payload budget: what an exchange holds while it runs.
#[derive(Clone)]
pub struct EndpointAdmission {
    pub limits: EndpointLimits,
    budget: Arc<Mutex<EndpointBudget>>,
}

impl Default for EndpointAdmission {
    fn default() -> Self {
        Self::new(EndpointLimits::default())
    }
}

impl EndpointAdmission {
    pub fn new(limits: EndpointLimits) -> Self {
        Self { limits, budget: Arc::new(Mutex::new(EndpointBudget::new(limits))) }
    }

    pub fn occupancy(&self) -> EndpointOccupancy {
        self.budget.lock().occupancy()
    }

    pub fn try_exchange(&self, bytes: usize) -> Option<EndpointReservation> {
        let mut budget = self.budget.lock();
        if !budget.admit_handler() {
            return None;
        }
        if !budget.admit_payload(bytes) {
            budget.release_handler();
            return None;
        }
        Some(EndpointReservation { budget: self.budget.clone(), bytes, handler: true })
    }

    pub fn try_payload(&self, bytes: usize) -> Option<EndpointReservation> {
        let mut budget = self.budget.lock();
        if !budget.admit_payload(bytes) {
            return None;
        }
        Some(EndpointReservation { budget: self.budget.clone(), bytes, handler: false })
    }
}

pub struct EndpointReservation {
    budget: Arc<Mutex<EndpointBudget>>,
    bytes: usize,
    handler: bool,
}

impl EndpointReservation {
    pub(super) fn release_payload(&mut self) {
        self.budget.lock().release_payload(std::mem::take(&mut self.bytes));
    }
}

impl Drop for EndpointReservation {
    fn drop(&mut self) {
        let mut budget = self.budget.lock();
        budget.release_payload(self.bytes);
        if self.handler {
            budget.release_handler();
        }
    }
}

pub(super) struct MessageAdmission {
    budget: Arc<Mutex<EndpointPayloadBudget>>,
}

impl MessageAdmission {
    pub fn new(bytes: usize) -> Self {
        Self { budget: Arc::new(Mutex::new(EndpointPayloadBudget::new(bytes))) }
    }

    pub fn try_message(
        &self,
        worker: &EndpointAdmission,
        bytes: usize,
    ) -> Option<MessageReservation> {
        let mut budget = self.budget.lock();
        if !budget.reserve(bytes) {
            return None;
        }
        let Some(worker) = worker.try_payload(bytes) else {
            budget.release(bytes);
            return None;
        };
        Some(MessageReservation { budget: self.budget.clone(), bytes, _worker: worker })
    }
}

pub(super) struct MessageReservation {
    budget: Arc<Mutex<EndpointPayloadBudget>>,
    bytes: usize,
    _worker: EndpointReservation,
}

impl Drop for MessageReservation {
    fn drop(&mut self) {
        self.budget.lock().release(self.bytes);
    }
}

/// The open WebSockets, for draining.
#[derive(Default)]
pub struct Registry {
    pub(super) admission: EndpointAdmission,
    pub(super) next: AtomicU64,
    pub(super) open: Mutex<HashMap<u64, Weak<WsExchange>>>,
}

impl Registry {
    pub fn new(limits: EndpointLimits) -> Self {
        Self {
            admission: EndpointAdmission::new(limits),
            next: AtomicU64::new(0),
            open: Mutex::new(HashMap::new()),
        }
    }

    /// Logical payload and admission observations for this worker's endpoint interface.
    pub fn occupancy(&self) -> super::budget::EndpointOccupancy {
        self.admission.occupancy()
    }

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    pub async fn close_all(&self, code: u16, reason: &str) {
        let open: Vec<_> = self.open.lock().values().filter_map(Weak::upgrade).collect();
        futures_util::future::join_all(open.iter().map(|ws| ws.close(code, reason))).await;
    }
}
