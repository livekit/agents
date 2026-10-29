//! Runtime room performers and their supplied session capability.

pub mod input;
pub mod output;
mod owner;
pub mod pre_connect_audio;

use self::output::transcription::SegmentIds;
use crate::rtc::wait;
use lk_agents_abi::voice::room_io::RoomIoConfig;
use lk_agents_abi::voice::room_owner;
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RtcError, TrackSid};
use lk_agents_abi::{Clock, Spawner};
use lk_agents_net::env::ProcessEnv;
use std::sync::Arc;

pub use owner::RoomIo;

pub struct FixedRoom {
    pub rtc: Arc<dyn RtcRoom>,
    pub time: lk_agents_voice::agent_session::Timebase,
    /// Where the room's CPU work (a pre-connect head's decode) runs; `None` runs it on the
    /// executor's own task.
    pub pool: Option<lk_agents_media::pool::MediaPool>,
}

impl room_owner::SessionRoom for FixedRoom {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        Some(self.rtc.clone())
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
            self.rtc.clone(),
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
    ) -> std::result::Result<(f64, f64), RtcError> {
        let clock = room.0.clock();
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        crate::avatar::receiver::wait_avatar_join_room(room, identity, clock).await
    }

    async fn wait_for_participant(
        room: room_owner::RoomSession,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    ) -> std::result::Result<Option<ParticipantIdentity>, RtcError> {
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        wait::participant(room.as_ref(), identity, kinds, include_local).await
    }

    async fn wait_for_agent(
        room: room_owner::RoomSession,
        name: Option<String>,
    ) -> std::result::Result<ParticipantIdentity, RtcError> {
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        wait::agent(room.as_ref(), name).await
    }

    async fn wait_for_participant_attribute(
        room: room_owner::RoomSession,
        identity: String,
        attribute: String,
        value: String,
    ) -> std::result::Result<(), RtcError> {
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        wait::participant_attribute(room.as_ref(), identity, attribute, value).await
    }

    async fn wait_for_track_publication(
        room: room_owner::RoomSession,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    ) -> std::result::Result<(Option<ParticipantIdentity>, TrackSid), RtcError> {
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        wait::track_publication(room.as_ref(), identity, kinds, include_local, subscribed).await
    }

    fn create_room_session(
        env: &ProcessEnv,
        room: Option<Arc<dyn RtcRoom>>,
    ) -> room_owner::RoomSession {
        let time = lk_agents_voice::agent_session::Timebase::real();
        match room {
            Some(room) => {
                let pool = (env.media_pool)().ok().map(lk_agents_media::pool::MediaPool::from);
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

