use super::bindings::DeclaredBinding;
use super::bound::BoundRoomIo;
use lk_agents_abi::voice::room_binding::BindingRefused;
use lk_agents_abi::voice::room_io::{RoomIoConfig, RoomOptionsError};
use lk_agents_abi::voice::room_owner::{self, RoomIo, SessionRoom};
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::{Clock, Closed, HostRef, Spawner, VirtualClock};
use lk_agents_net::signal;
use parking_lot::Mutex;
use std::sync::Arc;

enum State {
    Declared(DeclaredBinding),
    Preparing,
    Ready(Arc<dyn RoomIo>),
    Running(Arc<dyn RoomIo>),
    Closed,
}

/// Native room binding; each activation transfers its declaration exactly once.
pub struct RoomBinding {
    key: String,
    context: Arc<dyn SessionRoom>,
    sdk_room: HostRef,
    spawner: Arc<dyn Spawner>,
    state: Mutex<State>,
    closed: signal::Cell<Option<Closed>>,
}

impl RoomBinding {
    pub(super) fn new(
        key: String,
        context: Arc<dyn SessionRoom>,
        sdk_room: HostRef,
        spawner: Arc<dyn Spawner>,
        declared: DeclaredBinding,
    ) -> Self {
        let closed = declared.closed();
        Self {
            key,
            context,
            sdk_room,
            spawner,
            state: Mutex::new(State::Declared(declared)),
            closed,
        }
    }
}

impl room_owner::RoomBinding for RoomBinding {
    fn key(&self) -> String {
        self.key.clone()
    }
    fn room(&self) -> HostRef {
        self.sdk_room.clone()
    }

    fn prepare(
        &self,
        options: RoomIoConfig,
        host_audio_input: bool,
        host_video_input: bool,
        host_audio_output: bool,
    ) -> Result<(), RoomOptionsError> {
        let declared = {
            let mut state = self.state.lock();
            let previous = std::mem::replace(&mut *state, State::Preparing);
            match previous {
                State::Declared(declared) => declared,
                other => {
                    *state = other;
                    return Err(RoomOptionsError(BindingRefused::Unavailable.to_string()));
                }
            }
        };
        let options = lk_agents_voice::room_io::resolve_room_io(
            options,
            host_audio_input,
            host_video_input,
            host_audio_output,
            true,
        );
        let context = self.context.clone();
        let spawner = self.spawner.clone();
        let active = declared
            .activate::<Arc<dyn RoomIo>>(
                move || context.create_io(spawner, options),
                Box::new(|io, deadline| io.as_ref().clone().aclose_by(deadline)),
            )
            .map_err(|error| RoomOptionsError(error.to_string()))?;
        let io: Arc<dyn RoomIo> = Arc::new(BoundRoomIo(active));
        let accepted = {
            let mut state = self.state.lock();
            if matches!(*state, State::Preparing) {
                *state = State::Ready(io.clone());
                true
            } else {
                false
            }
        };
        if !accepted {
            drop(io.aclose());
            return Err(RoomOptionsError(BindingRefused::Unavailable.to_string()));
        }
        Ok(())
    }

    async fn aclose(&self) -> Result<(), RoomOptionsError> {
        let previous = std::mem::replace(&mut *self.state.lock(), State::Closed);
        match previous {
            State::Ready(io) | State::Running(io) => drop(io.aclose()),
            State::Declared(declared) => drop(declared),
            State::Preparing | State::Closed => {}
        }
        close_result(self.closed.wait().await.unwrap_or(Closed::Failed))
    }
}

impl SessionRoom for RoomBinding {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        self.context.rtc()
    }
    fn clock(&self) -> Arc<dyn Clock> {
        self.context.clock()
    }
    fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        self.context.virtual_clock()
    }
    fn create_io(&self, spawner: Arc<dyn Spawner>, config: RoomIoConfig) -> Arc<dyn RoomIo> {
        self.context.create_io(spawner, config)
    }
    fn room_io(
        &self,
        _spawner: Arc<dyn Spawner>,
        _config: RoomIoConfig,
    ) -> Result<Arc<dyn RoomIo>, BindingRefused> {
        let mut state = self.state.lock();
        let previous = std::mem::replace(&mut *state, State::Closed);
        match previous {
            State::Ready(io) => {
                *state = State::Running(io.clone());
                Ok(io)
            }
            other => {
                *state = other;
                Err(BindingRefused::Unavailable)
            }
        }
    }
}

pub(super) fn close_result(outcome: Closed) -> Result<(), RoomOptionsError> {
    match outcome {
        Closed::Clean => Ok(()),
        Closed::Failed | Closed::TimedOut => {
            Err(RoomOptionsError(format!("room I/O cleanup failed: {outcome:?}")))
        }
    }
}
