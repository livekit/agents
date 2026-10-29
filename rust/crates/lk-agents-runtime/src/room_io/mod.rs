//! Runtime room performers and their supplied session capability.

pub mod input;
pub mod output;
mod owner;
pub mod pre_connect_audio;

use self::output::transcription::SegmentIds;
use lk_agents_abi::voice::room_io::RoomIoConfig;
use lk_agents_abi::voice::room_owner;
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::{Clock, Spawner};
use lk_agents_net::env::ProcessEnv;
use std::sync::Arc;

pub use owner::RoomIo;

pub struct FixedRoom {
    pub rtc: Arc<dyn RtcRoom>,
    pub time: lk_agents_voice::agent_session::Timebase,
    /// Where the room's CPU work (a pre-connect head's decode) runs; `None` runs it on the
    /// executor's own task.
    pub pool: Option<lk_agents_media::workers::MediaWorkers>,
}

impl room_owner::SessionRoom for FixedRoom {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        Some(crate::rtc::room::Room::wrap(self.rtc.clone()))
    }
    fn virtual_clock(&self) -> Option<Arc<lk_agents_abi::VirtualClock>> {
        self.time.virtual_clock()
    }
    fn clock(&self) -> Arc<dyn Clock> {
        self.time.clock()
    }
    fn room_io(
        &self,
        spawner: Arc<dyn Spawner>,
        config: RoomIoConfig,
    ) -> Arc<dyn room_owner::RoomIo> {
        RoomIo::new(
            crate::rtc::room::Room::wrap(self.rtc.clone()),
            self.time.clock(),
            spawner,
            Arc::new(SegmentIds::default()),
            config.clone(),
            None,
            self.pool.clone(),
        )
    }
}

pub struct RoomFactory;

impl room_owner::RoomRuntime for RoomFactory {
    type Env = ProcessEnv;

    async fn wait_avatar_join(
        room: room_owner::RoomSession,
        identity: String,
    ) -> std::result::Result<(f64, f64), lk_agents_abi::voice::rtc::RtcError> {
        let clock = room.0.clock();
        let room = room.0.rtc().ok_or(lk_agents_abi::voice::rtc::RtcError::NotConnected)?;
        crate::avatar::receiver::wait_avatar_join_room(room, identity, clock).await
    }

    fn create_room_session(
        env: &ProcessEnv,
        room: Option<Arc<dyn RtcRoom>>,
    ) -> room_owner::RoomSession {
        let time = lk_agents_voice::agent_session::Timebase::real();
        match room {
            Some(room) => {
                let pool = (env.workers)().ok().map(lk_agents_media::workers::MediaWorkers::from);
                room_owner::RoomSession(Arc::new(FixedRoom { rtc: room, time, pool }))
            }
            None => room_owner::RoomSession(Arc::new(DetachedSession(time))),
        }
    }
}

/// A session without RTC has no SDK resources to publish, subscribe or retire.
struct DetachedSession(lk_agents_voice::agent_session::Timebase);

impl room_owner::SessionRoom for DetachedSession {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        None
    }
    fn clock(&self) -> Arc<dyn Clock> {
        self.0.clock()
    }
    fn room_io(&self, _: Arc<dyn Spawner>, _: RoomIoConfig) -> Arc<dyn room_owner::RoomIo> {
        lk_agents_voice::room_io::detached::room_io()
    }
}

#[cfg(test)]
mod session_tests;
