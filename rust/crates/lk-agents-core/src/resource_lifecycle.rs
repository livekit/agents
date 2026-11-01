//! Shared resource-lifetime admission. Executors own work; a cancelled waiter cannot abandon it.

/// Admission for the lifetime surrounding a voice session or text request. Repeated callers join the
/// operation already admitted; closing before start never acquires resources.
#[derive(Default)]
pub struct OwnerLifecycle {
    starting: bool,
    closing: bool,
}

pub enum StartOwner {
    Start,
    Join,
    Closed,
}

pub enum CloseOwner {
    Close { wait_start: bool },
    Join,
}

impl OwnerLifecycle {
    pub fn start(&mut self) -> StartOwner {
        if self.closing {
            StartOwner::Closed
        } else if self.starting {
            StartOwner::Join
        } else {
            self.starting = true;
            StartOwner::Start
        }
    }

    pub fn close(&mut self) -> CloseOwner {
        if self.closing {
            CloseOwner::Join
        } else {
            self.closing = true;
            CloseOwner::Close { wait_start: self.starting }
        }
    }
}

/// Storage and context ownership are independent: a managed context may be supplied even when
/// the invocation needs its own temporary thread. A namespace and thread identity travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoragePlan {
    pub own_namespace: bool,
    pub own_activation: bool,
}
impl StoragePlan {
    pub fn new(namespace: bool, thread: bool, activation: bool) -> Option<Self> {
        (namespace == thread)
            .then_some(Self { own_namespace: !namespace, own_activation: !activation })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn starts_and_closes_are_admitted_once() {
        let mut lifetime = OwnerLifecycle::default();
        assert!(matches!(lifetime.start(), StartOwner::Start));
        assert!(matches!(lifetime.start(), StartOwner::Join));
        assert!(matches!(lifetime.close(), CloseOwner::Close { wait_start: true }));
        assert!(matches!(lifetime.close(), CloseOwner::Join));
        assert!(matches!(lifetime.start(), StartOwner::Closed));
    }

    #[test]
    fn close_before_start_acquires_nothing() {
        let mut lifetime = OwnerLifecycle::default();
        assert!(matches!(lifetime.close(), CloseOwner::Close { wait_start: false }));
        assert!(matches!(lifetime.start(), StartOwner::Closed));
    }

    #[test]
    fn borrowed_context_and_storage_are_independent() {
        assert_eq!(
            StoragePlan::new(false, false, true),
            Some(StoragePlan { own_namespace: true, own_activation: false })
        );
        assert_eq!(
            StoragePlan::new(true, true, false),
            Some(StoragePlan { own_namespace: false, own_activation: true })
        );
        assert_eq!(
            StoragePlan::new(true, true, true),
            Some(StoragePlan { own_namespace: false, own_activation: false })
        );
        assert!(StoragePlan::new(true, false, false).is_none());
        assert!(StoragePlan::new(false, true, false).is_none());
    }
}
