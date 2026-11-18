//! Live transport handles are indexed by operation; saved state alone associates tasks.

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use lk_agents_abi::a2a::remote::{RemoteCallAdmission, RemoteResponseClaim, RemoteTaskDriver};
use lk_agents_abi::delegation::remote::RemoteTaskRef;
use lk_agents_abi::delegation::routing::CallAdmission;
use lk_agents_abi::session::{InputRequestRef, snapshot::SessionSnapshot};
use lk_agents_abi::storage::StorageError;
use parking_lot::Mutex;

type Operation = (String, String, String);
type Task = (String, String, RemoteTaskRef);
type ConfiguredDriver = (String, Weak<dyn RemoteTaskDriver>);

#[derive(Default)]
pub(crate) struct Drivers(Mutex<Registry>);

#[derive(Default)]
struct Registry {
    operations: BTreeMap<Operation, Weak<dyn RemoteTaskDriver>>,
    /// Native candidates are published before connect/Prepare; no credential is saved here.
    configurations: BTreeMap<String, Vec<ConfiguredDriver>>,
    tasks: BTreeMap<Task, Owner>,
    cancellations: BTreeMap<Task, Weak<dyn lk_agents_abi::session::control::RemoteCancellation>>,
}

struct Owner {
    driver: Weak<dyn RemoteTaskDriver>,
    responses: BTreeMap<InputRequestRef, Response>,
}

struct Response {
    /// Admission validated the full canonical message. These are exactly the source identity
    /// and words forwarded on this text response path; no host media object lives in the lock.
    message_id: String,
    words: String,
    claim: Weak<RemoteResponseClaim>,
}

fn live(driver: &Weak<dyn RemoteTaskDriver>) -> Option<Arc<dyn RemoteTaskDriver>> {
    driver.upgrade().filter(|driver| driver.is_live())
}

fn operations(
    remote: &lk_agents_abi::delegation::remote::RemoteAgentState,
    remote_id: &str,
    endpoint: &str,
    task: &RemoteTaskRef,
) -> Vec<Operation> {
    let mut selected: Vec<_> = remote
        .requests
        .iter()
        .filter(|(_, operation)| operation.task.as_ref() == Some(task))
        .collect();
    selected.sort_by_key(|(_, operation)| operation.request_event);
    selected.into_iter().map(|(id, _)| (remote_id.into(), endpoint.into(), id.clone())).collect()
}

impl Registry {
    fn driver(
        &mut self,
        key: &Task,
        operations: &[Operation],
    ) -> Option<Arc<dyn RemoteTaskDriver>> {
        if let Some(driver) = self.tasks.get(key).and_then(|owner| live(&owner.driver)) {
            return Some(driver);
        }
        self.tasks.remove(key);
        let driver = operations.iter().find_map(|key| self.operations.get(key).and_then(live));
        if let Some(driver) = &driver {
            self.tasks.insert(
                key.clone(),
                Owner { driver: Arc::downgrade(driver), responses: BTreeMap::new() },
            );
        }
        driver
    }
}

impl Drivers {
    /// One live or retained control per exact task across every managed store attached to
    /// this conversation. The producer stays with actual queued/running work, never this map.
    pub(crate) fn claim_cancellation(
        &self,
        target: &lk_agents_abi::session::control::RemoteControlRef,
        candidate: Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
    ) -> Result<Arc<dyn lk_agents_abi::session::control::RemoteCancellation>, StorageError> {
        if !candidate.pending() {
            return Err(StorageError::Invalid(
                "the cancellation submitter ended before admission".into(),
            ));
        }
        let mut inspected = Vec::new();
        let mut registry = self.0.lock();
        registry.cancellations.retain(|_, receipt| {
            receipt.upgrade().is_some_and(|receipt| {
                let pending = receipt.pending();
                inspected.push(receipt);
                pending
            })
        });
        let key = (target.remote_id.clone(), target.endpoint.clone(), target.task.clone());
        if let Some(receipt) = registry.cancellations.get(&key).and_then(Weak::upgrade) {
            return Ok(receipt);
        }
        registry.cancellations.insert(key, Arc::downgrade(&candidate));
        drop(registry);
        drop(inspected);
        Ok(candidate)
    }

    pub(crate) fn register(
        &self,
        remote_id: &str,
        endpoint: &str,
        request_id: &str,
        driver: Weak<dyn RemoteTaskDriver>,
    ) {
        let mut drivers = self.0.lock();
        drivers.operations.retain(|_, driver| live(driver).is_some());
        drivers.tasks.retain(|_, owner| live(&owner.driver).is_some());
        drivers.operations.insert((remote_id.into(), endpoint.into(), request_id.into()), driver);
    }

    pub(crate) fn get(
        &self,
        state: &SessionSnapshot,
        remote_id: &str,
        endpoint: &str,
        task: &RemoteTaskRef,
    ) -> Result<Option<Arc<dyn RemoteTaskDriver>>, StorageError> {
        let Some(remote) = lk_agents_core::session::remote::state(state, remote_id, endpoint)?
        else {
            return Ok(None);
        };
        let operations = operations(remote, remote_id, endpoint, task);
        let mut drivers = self.0.lock();
        let key = (remote_id.into(), endpoint.into(), task.clone());
        Ok(drivers.driver(&key, &operations))
    }

    /// Stable identity cannot silently select another endpoint or custom credential while
    /// configured work is alive. Equality is checked on native resource configuration only.
    fn admit_configuration(
        &self,
        remote_id: &str,
        endpoint: &str,
        candidate: &Weak<dyn RemoteTaskDriver>,
    ) -> Result<(), StorageError> {
        let candidate = live(candidate).ok_or_else(|| {
            StorageError::Invalid("the remote configuration owner ended before admission".into())
        })?;
        let mut inspected = Vec::new();
        let mut registry = self.0.lock();
        let configured = registry.configurations.entry(remote_id.into()).or_default();
        configured.retain(|(_, driver)| {
            let Some(driver) = live(driver) else { return false };
            inspected.push(driver);
            true
        });
        for ((saved_endpoint, _), driver) in configured.iter().zip(&inspected) {
            if saved_endpoint != endpoint || !driver.same_configuration(candidate.as_ref()) {
                return Err(StorageError::Invalid(
                    "this remote id already has a different live endpoint or credential; use distinct remote ids".into(),
                ));
            }
        }
        if !inspected.iter().any(|driver| Arc::ptr_eq(driver, &candidate)) {
            configured.push((endpoint.into(), Arc::downgrade(&candidate)));
        }
        drop(registry);
        drop(inspected);
        Ok(())
    }

    /// Claim native work in the same serialized owner step that validated its input. A
    /// response is coalesced only while a receipt or actual submission retains its token.
    pub(crate) fn admit(
        &self,
        state: &SessionSnapshot,
        remote_id: &str,
        endpoint: &str,
        admission: CallAdmission,
        candidate: Weak<dyn RemoteTaskDriver>,
    ) -> Result<RemoteCallAdmission, StorageError> {
        self.admit_configuration(remote_id, endpoint, &candidate)?;
        let task = match &admission {
            CallAdmission::Continue(task) => task.clone(),
            CallAdmission::Respond { request, .. } => request
                .remote()
                .ok_or_else(|| StorageError::Invalid("a remote response needs its task".into()))?
                .1
                .clone(),
            CallAdmission::New | CallAdmission::AlreadyForwarded { .. } => {
                return Ok(RemoteCallAdmission { admission, driver: None, response_claim: None });
            }
        };
        let remote = lk_agents_core::session::remote::state(state, remote_id, endpoint)?
            .ok_or_else(|| StorageError::Invalid("this remote has no known tasks".into()))?;
        let operations = operations(remote, remote_id, endpoint, &task);
        let key = (remote_id.into(), endpoint.into(), task.clone());
        let mut registry = self.0.lock();
        let driver =
            registry.driver(&key, &operations).or_else(|| live(&candidate)).ok_or_else(|| {
                StorageError::Invalid("the remote task owner ended before admission".into())
            })?;
        let configured = live(&candidate).ok_or_else(|| {
            StorageError::Invalid("the remote configuration owner ended before admission".into())
        })?;
        if !driver.same_configuration(configured.as_ref()) {
            return Err(StorageError::Invalid(
                "this remote id has a different live configuration; use distinct remote ids".into(),
            ));
        }
        let owner = registry.tasks.entry(key).or_insert_with(|| Owner {
            driver: Arc::downgrade(&driver),
            responses: BTreeMap::new(),
        });
        owner.responses.retain(|_, response| response.claim.strong_count() != 0);
        let mut response_claim = None;
        if let CallAdmission::Respond { request, message } = &admission {
            let words = message.raw_text_content().unwrap_or_default();
            let tracked = owner
                .responses
                .get(request)
                .filter(|response| response.message_id == message.id && response.words == words)
                .and_then(|response| response.claim.upgrade());
            if let Some(_tracked) = tracked {
                return Ok(RemoteCallAdmission {
                    admission: CallAdmission::AlreadyForwarded { task },
                    driver: None,
                    response_claim: None,
                });
            }
            let claim = Arc::new(RemoteResponseClaim);
            owner.responses.insert(
                request.clone(),
                Response { message_id: message.id.clone(), words, claim: Arc::downgrade(&claim) },
            );
            response_claim = Some(claim);
        }
        Ok(RemoteCallAdmission { admission, driver: Some(driver), response_claim })
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
