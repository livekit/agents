use super::resource::{ActiveBinding, ResourceScope};
use lk_agents_abi::{Closed, HOST_TEARDOWN_TIMEOUT};
use lk_agents_net::owned::{Background, Closer};
use lk_agents_net::signal;
use lk_agents_voice::room_io::bindings::effects::RoomBindingEffect;
use lk_agents_voice::room_io::bindings::{BindingId, BindingKey, BindingRefused};
use lk_agents_voice::room_io::legacy::effects::LegacyPrimaryEffect;
use lk_agents_voice::room_io::legacy::{LegacyProjection, LegacyTargetUnavailable};
use lk_agents_voice::room_io::shared::BindingCapabilities;
use lk_agents_voice::room_io::shared::effects::SharedRoomEffect;
use lk_agents_voice::room_io::shared::machine::SharedRoom;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_ROOM: AtomicU64 = AtomicU64::new(1);

enum Slot {
    Declared(signal::Producer<Option<Closed>>),
    Starting(signal::Producer<Option<Closed>>),
    Constructing,
    Attached(Weak<dyn ResourceScope>),
}

struct State {
    machine: SharedRoom,
    next_id: u64,
    slots: BTreeMap<BindingId, Slot>,
    legacy: Vec<LegacyPrimaryEffect>,
    cleanup: Closed,
}

/// One room's resource registry, independent of any session engine.
pub struct BindingOwner {
    identity: u64,
    state: Mutex<State>,
    pub(super) background: Background,
    /// The table owns progress: declarations retire synchronously, acquired resources
    /// retire from their cleanup task, including when that task unwinds.
    pending: signal::Producer<usize>,
}

impl BindingOwner {
    pub fn new(background: Background, legacy_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            identity: NEXT_ROOM.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(State {
                machine: SharedRoom::new(legacy_enabled),
                next_id: 0,
                slots: BTreeMap::new(),
                legacy: Vec::new(),
                cleanup: Closed::Clean,
            }),
            background,
            pending: signal::signal(0).0,
        })
    }

    pub fn bind(
        self: &Arc<Self>,
        key: String,
        capabilities: BindingCapabilities,
    ) -> Result<DeclaredBinding, BindingRefused> {
        self.declare(Some(key), capabilities)
    }

    pub fn automatic(
        self: &Arc<Self>,
        capabilities: BindingCapabilities,
    ) -> Result<DeclaredBinding, BindingRefused> {
        self.declare(None, capabilities)
    }

    fn declare(
        self: &Arc<Self>,
        key: Option<String>,
        capabilities: BindingCapabilities,
    ) -> Result<DeclaredBinding, BindingRefused> {
        let mut state = self.state.lock();
        state.next_id = state.next_id.checked_add(1).ok_or(BindingRefused::Unavailable)?;
        let id = BindingId(state.next_id);
        let key = key.map_or(BindingKey::Automatic(id), BindingKey::Named);
        state.machine.declare(id, key, capabilities)?;
        let (done, closed) = signal::once();
        state.slots.insert(id, Slot::Declared(done));
        self.pending.set(state.slots.len());
        Ok(DeclaredBinding { owner: self.clone(), id, held: true, closed })
    }

    pub fn legacy_projection(&self) -> Option<LegacyProjection> {
        self.state.lock().machine.legacy_projection()
    }

    pub(super) fn pin_legacy(&self, id: BindingId) -> Result<(), LegacyTargetUnavailable> {
        let scopes = {
            let mut state = self.state.lock();
            let effects = state.machine.pin_legacy(id)?;
            self.apply(&mut state, effects)
        };
        for scope in scopes {
            scope.request_close();
        }
        Ok(())
    }

    pub fn automatic_legacy(&self) {
        let scopes = {
            let mut state = self.state.lock();
            let effects = state.machine.automatic_legacy();
            self.apply(&mut state, effects)
        };
        for scope in scopes {
            scope.request_close();
        }
    }

    /// The transport performer drains ordered cutovers; it must settle the existing
    /// wire projection rather than interpreting selection as an audio-floor change.
    pub fn take_legacy_effects(&self) -> Vec<LegacyPrimaryEffect> {
        std::mem::take(&mut self.state.lock().legacy)
    }

    pub fn request_close(&self) {
        let scopes = {
            let mut state = self.state.lock();
            let effects = state.machine.close();
            self.apply(&mut state, effects)
        };
        for scope in scopes {
            scope.request_close();
        }
    }

    /// Bounds only this wait. Each admitted resource close retains its original
    /// deadline and keeps running if this waiter is cancelled or reaches its deadline.
    pub async fn aclose(&self) -> Closed {
        self.request_close();
        let deadline = self.background.deadline();
        let pending = self.pending.cell();
        match lk_agents_abi::timeout(
            self.background.clock.as_ref(),
            HOST_TEARDOWN_TIMEOUT,
            pending.until(|&pending| pending == 0),
        )
        .await
        {
            Ok(()) => {}
            Err(_) => return Closed::TimedOut,
        }
        if !self.background.tasks.wait_empty(self.background.clock.as_ref(), Some(deadline)).await {
            return Closed::TimedOut;
        }
        self.state.lock().cleanup
    }

    fn apply(
        &self,
        state: &mut State,
        effects: Vec<SharedRoomEffect>,
    ) -> Vec<Arc<dyn ResourceScope>> {
        let mut scopes = Vec::new();
        for effect in effects {
            match effect {
                SharedRoomEffect::Legacy(effect) => state.legacy.push(effect),
                SharedRoomEffect::Binding(RoomBindingEffect::Close(id)) => {
                    match state.slots.get(&id) {
                        Some(Slot::Declared(_)) => {
                            // A declaration has nothing asynchronous to release.
                            if let Some(Slot::Declared(done)) = state.slots.remove(&id) {
                                done.finish(Some(Closed::Clean));
                            }
                            state.machine.retired(id);
                        }
                        Some(Slot::Attached(scope)) => {
                            if let Some(scope) = scope.upgrade() {
                                scopes.push(scope);
                            }
                        }
                        // A constructor or an already admitted cleanup retains retirement.
                        Some(Slot::Starting(_) | Slot::Constructing) | None => {}
                    }
                }
            }
        }
        self.pending.set(state.slots.len());
        scopes
    }

    pub(super) fn started(&self, id: BindingId) -> Result<(), BindingRefused> {
        let scopes = {
            let mut state = self.state.lock();
            let effects = state.machine.started(id)?;
            self.apply(&mut state, effects)
        };
        for scope in scopes {
            scope.request_close();
        }
        Ok(())
    }

    pub(super) fn closing_resource(&self, id: BindingId) {
        let scopes = {
            let mut state = self.state.lock();
            let effects = state.machine.close_binding(id);
            self.apply(&mut state, effects)
        };
        // The calling resource has already taken its Owned value for close; no
        // second close is launched while executing the registry's effects.
        for scope in scopes {
            scope.request_close();
        }
    }

    pub(super) fn retired_resource(&self, id: BindingId, outcome: Closed) {
        let mut state = self.state.lock();
        if state.slots.remove(&id).is_none() {
            return;
        }
        state.cleanup = state.cleanup.and(outcome);
        if outcome == Closed::Clean {
            state.machine.retired(id);
        }
        // An uncertain/failed cleanup leaves its key reserved: a replacement must
        // not publish over a resource whose retirement could not be established.
        self.pending.set(state.slots.len());
    }

    fn abandon_declaration(&self, id: BindingId) {
        let mut state = self.state.lock();
        let effects = state.machine.close_binding(id);
        self.apply(&mut state, effects);
        if let Some(Slot::Declared(done) | Slot::Starting(done)) = state.slots.remove(&id) {
            done.finish(Some(Closed::Clean));
        }
        state.machine.retired(id);
        self.pending.set(state.slots.len());
    }
}

/// Non-cloneable activation authority. Dropping an unused declaration releases
/// its key; activating it transfers retirement to the resource's Owned close.
#[must_use = "retain the declaration until activating or releasing it"]
pub struct DeclaredBinding {
    owner: Arc<BindingOwner>,
    id: BindingId,
    held: bool,
    closed: signal::Cell<Option<Closed>>,
}

impl DeclaredBinding {
    pub fn id(&self) -> BindingId {
        self.id
    }

    pub fn identity(&self) -> (u64, u64) {
        (self.owner.identity, self.id.0)
    }

    pub fn closed(&self) -> signal::Cell<Option<Closed>> {
        self.closed.clone()
    }

    /// Construction is synchronous and follows activation admission. Resources
    /// are held in Owned before the caller can perform its next await. The closer
    /// must settle by its supplied deadline; an outer wait never cuts it short.
    pub fn activate<R: Send + Sync + 'static>(
        self,
        create: impl FnOnce() -> R,
        close: Closer<Arc<R>>,
    ) -> Result<ActiveBinding<R>, BindingRefused> {
        self.activate_with_callback(create, close, None)
    }

    pub fn activate_with_callback<R: Send + Sync + 'static>(
        mut self,
        create: impl FnOnce() -> R,
        close: Closer<Arc<R>>,
        on_closed: Option<Box<dyn FnOnce(Closed) + Send>>,
    ) -> Result<ActiveBinding<R>, BindingRefused> {
        {
            let mut state = self.owner.state.lock();
            state.machine.begin_start(self.id)?;
            let Some(Slot::Declared(done)) = state.slots.remove(&self.id) else {
                return Err(BindingRefused::Unavailable);
            };
            state.slots.insert(self.id, Slot::Starting(done));
        }
        let resource = Arc::new(create());
        let done = {
            let mut state = self.owner.state.lock();
            let Some(Slot::Starting(done)) = state.slots.remove(&self.id) else {
                return Err(BindingRefused::Unavailable);
            };
            state.slots.insert(self.id, Slot::Constructing);
            done
        };
        let active =
            ActiveBinding::new(self.owner.clone(), self.id, resource, close, done, on_closed);
        self.held = false;
        let accepted = {
            let mut state = self.owner.state.lock();
            let accepted = state.machine.phase(self.id)
                == Some(lk_agents_voice::room_io::bindings::BindingPhase::Starting);
            state.slots.insert(self.id, Slot::Attached(active.downgrade()));
            accepted
        };
        if !accepted {
            // Owner close raced construction; dropping active admits its cleanup.
            return Err(BindingRefused::Unavailable);
        }
        Ok(active)
    }
}

impl Drop for DeclaredBinding {
    fn drop(&mut self) {
        if self.held {
            self.owner.abandon_declaration(self.id);
        }
    }
}
