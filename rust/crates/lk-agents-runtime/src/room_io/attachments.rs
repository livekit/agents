use super::bindings::BindingOwner;
use super::bound::BoundRoomIo;
use super::session_binding::{RoomBinding, close_result};
use lk_agents_abi::voice::room_binding::{BindingCapabilities, BindingRefused};
use lk_agents_abi::voice::room_io::RoomOptionsError;
use lk_agents_abi::voice::room_owner::{RoomAttachments, RoomAttachmentsPort, RoomIo, SessionRoom};
use lk_agents_abi::{Clock, HostRef, Spawner};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::Background;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Default)]
enum State {
    #[default]
    Fresh,
    Live(Arc<BindingOwner>),
    Closed(Option<Arc<BindingOwner>>),
}

/// Cached by SDK room identity, without a strong reference back to the SDK room.
#[derive(Default)]
pub struct AttachmentRegistry {
    state: Mutex<State>,
}

impl AttachmentRegistry {
    fn owner(
        &self,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Result<Arc<BindingOwner>, BindingRefused> {
        let mut state = self.state.lock();
        match &*state {
            State::Closed(_) => Err(BindingRefused::OwnerClosed),
            State::Live(owner) => Ok(owner.clone()),
            State::Fresh => {
                let owner =
                    BindingOwner::new(Background::new(Arc::default(), spawner, clock), true);
                *state = State::Live(owner.clone());
                Ok(owner)
            }
        }
    }
}

impl RoomAttachments for AttachmentRegistry {
    type Env = ProcessEnv;
    type Binding = RoomBinding;

    fn new() -> Self {
        Self::default()
    }

    fn bind(
        &self,
        env: &ProcessEnv,
        key: String,
        room: Arc<dyn SessionRoom>,
        sdk_room: HostRef,
    ) -> Result<RoomBinding, RoomOptionsError> {
        if room.rtc().is_none() {
            return Err(RoomOptionsError("RoomIO requires an RTC room".into()));
        }
        let spawner = (env.spawner)().map_err(|error| RoomOptionsError(error.to_string()))?;
        let owner = self
            .owner(room.clock(), spawner.clone())
            .map_err(|error| RoomOptionsError(error.to_string()))?;
        let declared = owner
            .bind(key.clone(), BindingCapabilities::default())
            .map_err(|error| RoomOptionsError(error.to_string()))?;
        Ok(RoomBinding::new(key, room, sdk_room, spawner, declared))
    }

    async fn aclose(&self) -> Result<(), RoomOptionsError> {
        let owner = {
            let mut state = self.state.lock();
            match &*state {
                State::Fresh => {
                    *state = State::Closed(None);
                    None
                }
                State::Live(owner) => {
                    let owner = owner.clone();
                    *state = State::Closed(Some(owner.clone()));
                    Some(owner)
                }
                State::Closed(owner) => owner.clone(),
            }
        };
        match owner {
            Some(owner) => close_result(owner.aclose().await),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "attachments_tests.rs"]
mod tests;

impl RoomAttachmentsPort for AttachmentRegistry {
    fn attach(
        &self,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        create: Box<dyn FnOnce() -> Arc<dyn RoomIo> + Send>,
    ) -> Result<Arc<dyn RoomIo>, BindingRefused> {
        let owner = self.owner(clock, spawner)?;
        let binding =
            owner.automatic(BindingCapabilities::default())?.activate::<Arc<dyn RoomIo>>(
                create,
                Box::new(|io, deadline| io.as_ref().clone().aclose_by(deadline)),
            )?;
        Ok(Arc::new(BoundRoomIo(binding)))
    }
}
