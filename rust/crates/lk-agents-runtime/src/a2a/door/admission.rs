//! Admission for decoded A2A operations, independent of worker stream transport.

use std::sync::Arc;

use parking_lot::Mutex;

#[derive(Default)]
struct Budget {
    handlers: usize,
    bytes: usize,
}

#[derive(Clone)]
pub(super) struct Admission {
    budget: Arc<Mutex<Budget>>,
    max_handlers: usize,
    max_bytes: usize,
}

impl Default for Admission {
    fn default() -> Self {
        Self { budget: Arc::default(), max_handlers: 256, max_bytes: 512 << 20 }
    }
}

impl Admission {
    pub(super) fn try_exchange(&self, bytes: usize) -> Option<Reservation> {
        let mut budget = self.budget.lock();
        let total = budget.bytes.checked_add(bytes)?;
        if budget.handlers >= self.max_handlers || total > self.max_bytes {
            return None;
        }
        budget.handlers += 1;
        budget.bytes = total;
        Some(Reservation { budget: self.budget.clone(), bytes })
    }
}

pub(super) struct Reservation {
    budget: Arc<Mutex<Budget>>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut budget = self.budget.lock();
        budget.bytes -= self.bytes;
        budget.handlers -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_and_payload_capacity_recover_when_operations_finish() {
        let admission = Admission { max_handlers: 2, max_bytes: 16, ..Admission::default() };
        let first = admission.try_exchange(9).unwrap();
        assert!(admission.try_exchange(8).is_none());
        assert!(admission.try_exchange(usize::MAX).is_none());
        let second = admission.try_exchange(7).unwrap();
        assert!(admission.try_exchange(0).is_none());
        drop(first);
        assert!(admission.try_exchange(9).is_some());
        drop(second);
        assert_eq!(admission.budget.lock().handlers, 0);
        assert_eq!(admission.budget.lock().bytes, 0);
    }
}
