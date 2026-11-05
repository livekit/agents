//! What the host's events have delivered: `session.history` as the host's own item objects. Each
//! event reaches the host with the state it leaves, so a callback reads the session as of its
//! event, and a read outside one reads the latest.

use lk_agents_abi::voice::events::HostWire;
use lk_agents_abi::voice::session_effects::{DeliveredState, SessionEvents};
use lk_agents_abi::{ChatItem, HostRef};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Clone)]
struct Item {
    id: String,
    created_at: SystemTime,
    host: HostRef,
}

/// One delivered state. Never changed: each change publishes a new one.
#[derive(Clone, Default)]
pub struct SessionState {
    history: Vec<Item>,
}

impl SessionState {
    /// `ChatContext.insert`: after the last item created no later. An output told again (an
    /// answered question's) replaces its earlier telling in place.
    fn add(&mut self, item: &ChatItem, host: HostRef) {
        let id = item.id().to_owned();
        if matches!(item, ChatItem::FunctionCallOutput(_))
            && let Some(told) = self.history.iter_mut().find(|told| told.id == id)
        {
            told.host = host;
            return;
        }
        let created_at = item.created_at();
        let at = self.history.iter().rposition(|i| i.created_at <= created_at).map_or(0, |i| i + 1);
        self.history.insert(at, Item { id, created_at, host });
    }

    fn reload(&mut self, items: &[ChatItem], hosts: Vec<HostRef>) {
        if hosts.len() != items.len() {
            return;
        }
        self.history = items
            .iter()
            .zip(hosts)
            .map(|(item, host)| Item {
                id: item.id().to_owned(),
                created_at: item.created_at(),
                host,
            })
            .collect();
    }
}

impl DeliveredState for SessionState {}

impl lk_agents_abi::voice::session_effects::SessionState for SessionState {
    fn history(&self) -> Vec<HostRef> {
        self.history.iter().map(|item| item.host.clone()).collect()
    }
}

/// The latest delivered state, advanced by the host drain and read by the session.
#[derive(Clone, Default)]
pub struct Delivered(Arc<Mutex<Arc<SessionState>>>);

impl Delivered {
    pub fn latest(&self) -> Arc<SessionState> {
        Arc::clone(&self.0.lock())
    }

    /// The history the session starts on, before its first event.
    pub(super) fn open(&self, events: &dyn SessionEvents, items: &[ChatItem]) {
        let hosts = host_items(events, items);
        self.publish(|state| state.reload(items, hosts));
    }

    /// Applies `wire`, filling in the host's object for an added item, and returns the state it
    /// leaves.
    pub(super) fn apply(
        &self,
        events: &dyn SessionEvents,
        wire: &mut HostWire,
    ) -> Arc<SessionState> {
        match wire {
            HostWire::ItemAdded(added) => {
                if let Some(host) = host_items(events, std::slice::from_ref(&added.item)).pop() {
                    added.host_item = Some(host.clone());
                    self.publish(|state| state.add(&added.item, host));
                }
            }
            HostWire::HistoryReloaded(reloaded) => {
                let hosts = host_items(events, &reloaded.items);
                self.publish(|state| state.reload(&reloaded.items, hosts));
            }
            _ => {}
        }
        self.latest()
    }

    fn publish(&self, change: impl FnOnce(&mut SessionState)) {
        let mut next = SessionState::clone(&self.latest());
        change(&mut next);
        // The replaced state's host objects are released after the guard.
        let replaced = std::mem::replace(&mut *self.0.lock(), Arc::new(next));
        drop(replaced);
    }
}

/// The host's objects for `items`; none when the host made none, and the history keeps what it
/// had.
fn host_items(events: &dyn SessionEvents, items: &[ChatItem]) -> Vec<HostRef> {
    if items.is_empty() {
        return Vec::new();
    }
    match events.history_items(items.to_vec()) {
        Ok(hosts) if hosts.len() == items.len() => hosts,
        // A host that keeps no objects (a simulation's) has no history to read.
        Ok(_) => Vec::new(),
        Err(error) => {
            log::warn!(target: "livekit.agents", error = error.to_string().as_str(); "the host made no history items");
            Vec::new()
        }
    }
}
