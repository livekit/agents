//! Endpoint admission decisions. Runtime permits own each reservation until its payload is released.

use lk_agents_abi::worker::endpoint::EndpointLimits;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndpointOccupancy {
    pub handlers: usize,
    pub payload_bytes: usize,
    pub rejected_handlers: u64,
    pub rejected_payloads: u64,
}

pub struct EndpointBudget {
    limits: EndpointLimits,
    occupancy: EndpointOccupancy,
    payload: EndpointPayloadBudget,
}

/// A byte reservation owner used by both the worker and each WebSocket inbox.
pub struct EndpointPayloadBudget {
    capacity: usize,
    bytes: usize,
}

impl EndpointPayloadBudget {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, bytes: 0 }
    }

    pub fn reserve(&mut self, bytes: usize) -> bool {
        match self.bytes.checked_add(bytes) {
            Some(total) if total <= self.capacity => {
                self.bytes = total;
                true
            }
            _ => false,
        }
    }

    pub fn release(&mut self, bytes: usize) {
        self.bytes = self.bytes.checked_sub(bytes).expect("owned endpoint payload");
    }
}

impl EndpointBudget {
    pub fn new(limits: EndpointLimits) -> Self {
        Self {
            limits,
            occupancy: EndpointOccupancy::default(),
            payload: EndpointPayloadBudget::new(limits.worker_payload_bytes),
        }
    }

    pub fn occupancy(&self) -> EndpointOccupancy {
        EndpointOccupancy { payload_bytes: self.payload.bytes, ..self.occupancy }
    }

    pub fn admit_handler(&mut self) -> bool {
        if self.occupancy.handlers >= self.limits.pending_handlers {
            self.occupancy.rejected_handlers += 1;
            return false;
        }
        self.occupancy.handlers += 1;
        true
    }

    pub fn release_handler(&mut self) {
        self.occupancy.handlers = self.occupancy.handlers.checked_sub(1).expect("owned handler");
    }

    pub fn admit_payload(&mut self, bytes: usize) -> bool {
        if !self.payload.reserve(bytes) {
            self.occupancy.rejected_payloads += 1;
            return false;
        }
        true
    }

    pub fn release_payload(&mut self, bytes: usize) {
        self.payload.release(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_worker_ceiling_is_shared_across_handlers_and_released_exactly() {
        let mut budget = EndpointBudget::new(EndpointLimits {
            pending_handlers: 2,
            worker_payload_bytes: 16,
            ..Default::default()
        });
        assert!(budget.admit_handler());
        assert!(budget.admit_handler());
        assert!(!budget.admit_handler());
        assert!(budget.admit_payload(9));
        assert!(!budget.admit_payload(8));
        assert!(!budget.admit_payload(usize::MAX));
        budget.release_payload(9);
        assert!(budget.admit_payload(16));
        budget.release_handler();
        assert!(budget.admit_handler());
        budget.release_handler();
        budget.release_handler();
        budget.release_payload(16);
        assert_eq!(
            budget.occupancy(),
            EndpointOccupancy { rejected_handlers: 1, rejected_payloads: 2, ..Default::default() }
        );
    }
}
