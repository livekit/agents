use super::BindingOwner;
use lk_agents_abi::{Closed, Ts};
use lk_agents_net::owned::{Closer, Owned};
use lk_agents_net::signal;
use lk_agents_voice::room_io::bindings::{BindingId, BindingRefused};
use lk_agents_voice::room_io::legacy::LegacyTargetUnavailable;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};

pub(super) trait ResourceScope: Send + Sync {
    fn request_close(&self);
}

struct Resource<R: Send + Sync + 'static> {
    value: Arc<R>,
    owned: Mutex<Option<Owned<Arc<R>>>>,
    closed: signal::Cell<Option<Closed>>,
}

impl<R: Send + Sync + 'static> ResourceScope for Resource<R> {
    fn request_close(&self) {
        self.close_by(None);
    }
}

impl<R: Send + Sync + 'static> Resource<R> {
    fn close_by(&self, deadline: Option<Ts>) {
        let owned = self.owned.lock().take();
        if let Some(owned) = owned {
            // Dropping Closing runs this same close as an owned cleanup, without
            // requiring the session or a caller to keep polling it.
            drop(match deadline {
                Some(deadline) => owned.close_by(deadline),
                None => owned.close(),
            });
        }
    }
}

/// Single resource ownership for an activated binding. Borrowed access does not
/// transfer the aggregate or the authority to release its registry entry.
#[must_use = "retain the binding until its resources should close"]
pub struct ActiveBinding<R: Send + Sync + 'static> {
    owner: Arc<BindingOwner>,
    id: BindingId,
    resource: Arc<Resource<R>>,
}

impl<R: Send + Sync + 'static> ActiveBinding<R> {
    pub(super) fn new(
        owner: Arc<BindingOwner>,
        id: BindingId,
        value: Arc<R>,
        close: Closer<Arc<R>>,
        done: signal::Producer<Option<Closed>>,
        on_closed: Option<Box<dyn FnOnce(Closed) + Send>>,
    ) -> Self {
        let closed = done.cell();
        let completion =
            Completion { owner: Arc::downgrade(&owner), id, done: Some(done), on_closed };
        let owned = Owned::new(
            &owner.background,
            value.clone(),
            Box::new(move |value, deadline: Ts| {
                if let Some(owner) = completion.owner.upgrade() {
                    owner.closing_resource(id);
                }
                Box::pin(async move {
                    let outcome = close(value, deadline).await;
                    completion.finish(outcome);
                    outcome
                })
            }),
        );
        Self {
            owner,
            id,
            resource: Arc::new(Resource { value, owned: Mutex::new(Some(owned)), closed }),
        }
    }

    pub fn id(&self) -> BindingId {
        self.id
    }

    pub fn started(&self) -> Result<(), BindingRefused> {
        self.owner.started(self.id)
    }

    /// The owning binding supplies both owner and incarnation; a caller cannot
    /// accidentally pin a different room's binding with the same numeric ID.
    pub fn pin_legacy(&self) -> Result<(), LegacyTargetUnavailable> {
        self.owner.pin_legacy(self.id)
    }

    pub(super) fn downgrade(&self) -> Weak<dyn ResourceScope> {
        Arc::downgrade(&(self.resource.clone() as Arc<dyn ResourceScope>))
    }

    pub async fn aclose(self) -> Closed {
        self.request_close();
        self.closed().await
    }

    pub fn request_close(&self) {
        self.resource.request_close();
    }

    pub fn request_close_by(&self, deadline: Ts) {
        self.resource.close_by(Some(deadline));
    }

    pub async fn closed(&self) -> Closed {
        self.resource.closed.wait().await.unwrap_or(Closed::Failed)
    }
}

impl<R: Send + Sync + 'static> std::ops::Deref for ActiveBinding<R> {
    type Target = R;

    fn deref(&self) -> &R {
        &self.resource.value
    }
}

impl<R: Send + Sync + 'static> Drop for ActiveBinding<R> {
    fn drop(&mut self) {
        self.resource.request_close();
    }
}

/// Even a panicking closer reports its death. Pending owner-close waits cannot
/// retain an entry whose cleanup task has already ended.
struct Completion {
    owner: Weak<BindingOwner>,
    id: BindingId,
    done: Option<signal::Producer<Option<Closed>>>,
    on_closed: Option<Box<dyn FnOnce(Closed) + Send>>,
}

impl Completion {
    fn finish(mut self, outcome: Closed) {
        self.report(outcome);
    }

    fn report(&mut self, outcome: Closed) {
        if let Some(on_closed) = self.on_closed.take() {
            on_closed(outcome);
        }
        if let Some(done) = self.done.take() {
            if let Some(owner) = self.owner.upgrade() {
                owner.retired_resource(self.id, outcome);
            }
            done.finish(Some(outcome));
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        self.report(Closed::Failed);
    }
}
