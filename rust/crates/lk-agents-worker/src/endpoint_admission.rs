//! Reservations for endpoint exchange and inbox payload ownership.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, Weak};

use super::WsExchange;

use crate::endpoint_budget::{EndpointBudget, EndpointOccupancy, EndpointPayloadBudget};
use lk_agents_abi::worker::endpoint::EndpointLimits;

#[derive(Clone)]
pub(super) struct EndpointAdmission {
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
        lk_agents_abi::lock(&self.budget).occupancy()
    }

    pub fn try_exchange(&self, bytes: usize) -> Option<EndpointReservation> {
        let mut budget = lk_agents_abi::lock(&self.budget);
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
        let mut budget = lk_agents_abi::lock(&self.budget);
        if !budget.admit_payload(bytes) {
            return None;
        }
        Some(EndpointReservation { budget: self.budget.clone(), bytes, handler: false })
    }
}

pub(super) struct EndpointReservation {
    budget: Arc<Mutex<EndpointBudget>>,
    bytes: usize,
    handler: bool,
}

impl EndpointReservation {
    pub(super) fn release_payload(&mut self) {
        lk_agents_abi::lock(&self.budget).release_payload(std::mem::take(&mut self.bytes));
    }
}

impl Drop for EndpointReservation {
    fn drop(&mut self) {
        let mut budget = lk_agents_abi::lock(&self.budget);
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
        let mut budget = lk_agents_abi::lock(&self.budget);
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
        lk_agents_abi::lock(&self.budget).release(self.bytes);
    }
}

/// The open WebSockets, for draining.
#[derive(Default)]
pub struct Registry {
    pub(super) admission: EndpointAdmission,
    pub(super) next: AtomicU64,
    open: std::sync::Mutex<HashMap<u64, Weak<WsExchange>>>,
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
    pub fn occupancy(&self) -> crate::endpoint_budget::EndpointOccupancy {
        self.admission.occupancy()
    }

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    pub async fn close_all(&self, code: u16, reason: &str) {
        let open: Vec<_> = self.lock().values().filter_map(Weak::upgrade).collect();
        futures_util::future::join_all(open.iter().map(|ws| ws.close(code, reason))).await;
    }

    pub(super) fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u64, Weak<WsExchange>>> {
        lk_agents_abi::lock(&self.open)
    }
}
