//! Runtime room performers and their supplied session capability.

pub mod input;
pub mod output;
mod owner;
pub mod pre_connect_audio;

use self::output::transcription::SegmentIds;
use crate::rtc::wait;
use lk_agents_abi::voice::io;
use lk_agents_abi::voice::room_io::RoomIoConfig;
use lk_agents_abi::voice::room_owner;
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RtcError, TrackSid};
use lk_agents_abi::{Clock, Spawner};
use lk_agents_net::clock::RealClock;
use lk_agents_net::env::ProcessEnv;
use std::sync::Arc;

pub use owner::RoomIo;

/// `_core_RoomSession`: the session's room, or none for a session without RTC.
pub struct FixedRoom {
    pub rtc: Option<Arc<dyn RtcRoom>>,
    pub time: crate::voice::session::Timebase,
    /// Where the room's CPU work (a pre-connect head's decode) runs; `None` runs it on the
    /// executor's own task.
    pub pool: Option<lk_agents_media::pool::MediaPool>,
}

impl room_owner::SessionRoom for FixedRoom {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        self.rtc.clone()
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
        let Some(rtc) = self.rtc.clone() else {
            // A session without RTC has no SDK resources to publish, subscribe or retire.
            return lk_agents_voice::room_io::detached::room_io();
        };
        RoomIo::new(
            rtc,
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
    type Room = FixedRoom;
    type Env = ProcessEnv;

    async fn wait_avatar_join(
        room: Arc<dyn room_owner::SessionRoom>,
        identity: String,
    ) -> std::result::Result<(f64, f64), RtcError> {
        let clock = room.clock();
        let room = room.rtc().ok_or(RtcError::NotConnected)?;
        crate::avatar::receiver::wait_avatar_join_room(room, identity, clock).await
    }

    async fn wait_for_participant(
        room: Arc<dyn room_owner::SessionRoom>,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    ) -> std::result::Result<Option<ParticipantIdentity>, RtcError> {
        let room = room.rtc().ok_or(RtcError::NotConnected)?;
        wait::participant(room.as_ref(), identity, kinds, include_local).await
    }

    async fn wait_for_agent(
        room: Arc<dyn room_owner::SessionRoom>,
        name: Option<String>,
    ) -> std::result::Result<ParticipantIdentity, RtcError> {
        let room = room.rtc().ok_or(RtcError::NotConnected)?;
        wait::agent(room.as_ref(), name).await
    }

    async fn wait_for_track_publication(
        room: Arc<dyn room_owner::SessionRoom>,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    ) -> std::result::Result<(Option<ParticipantIdentity>, TrackSid), RtcError> {
        let room = room.rtc().ok_or(RtcError::NotConnected)?;
        wait::track_publication(room.as_ref(), identity, kinds, include_local, subscribed).await
    }

    fn create_room_session(env: &ProcessEnv, room: Option<Arc<dyn RtcRoom>>) -> FixedRoom {
        let time = crate::voice::session::Timebase::of_clock(Arc::new(RealClock::new()));
        let pool = room
            .as_ref()
            .and_then(|_| (env.media_pool)().ok())
            .map(lk_agents_media::pool::MediaPool::from);
        FixedRoom { rtc: room, time, pool }
    }
}

impl room_owner::RoomSession for FixedRoom {}

/// `_core.RoomAudioInput`: the room's microphone leg as the host holds it.
pub struct RoomAudioInput(pub Arc<dyn room_owner::RoomAudioInputPort>);

impl room_owner::RoomAudioInput for RoomAudioInput {
    fn reading(&self) -> Option<TrackSid> {
        self.0.reading()
    }

    async fn read_frame(&self) -> io::Result<Option<lk_agents_abi::AudioFrame>> {
        self.0.next_frame().await
    }

    async fn aclose(&self) -> io::Result<()> {
        input_closed(self.0.aclose().await)
    }
}

/// Taken back as the session's audio input: the leg itself.
impl io::AudioInput for RoomAudioInput {
    fn label(&self) -> &str {
        self.0.label()
    }

    fn next_frame(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, io::Result<Option<lk_agents_abi::AudioFrame>>> {
        self.0.next_frame()
    }

    fn on_attached(&self) {
        self.0.on_attached()
    }

    fn on_detached(&self) {
        self.0.on_detached()
    }
}

/// `_core.RoomVideoInput`: the room's video leg as the host holds it.
pub struct RoomVideoInput(pub Arc<dyn room_owner::RoomVideoInputPort>);

impl room_owner::RoomVideoInput for RoomVideoInput {
    fn reading(&self) -> Option<TrackSid> {
        self.0.reading()
    }

    async fn read_frame(&self) -> io::Result<Option<lk_agents_abi::MediaRef>> {
        self.0.next_frame().await.map(|frame| frame.map(|frame| frame.media))
    }

    async fn aclose(&self) -> io::Result<()> {
        input_closed(self.0.aclose().await)
    }
}

/// Taken back as the session's video input: the leg itself.
impl io::VideoInput for RoomVideoInput {
    fn label(&self) -> &str {
        self.0.label()
    }

    fn next_frame(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, io::Result<Option<lk_agents_abi::voice::rtc::VideoFrameRef>>>
    {
        self.0.next_frame()
    }

    fn on_attached(&self) {
        self.0.on_attached()
    }

    fn on_detached(&self) {
        self.0.on_detached()
    }
}

fn input_closed(outcome: lk_agents_abi::Closed) -> io::Result<()> {
    match outcome {
        lk_agents_abi::Closed::Clean => Ok(()),
        lk_agents_abi::Closed::Failed => Err(io::IoError::Rtc("room input cleanup failed".into())),
        lk_agents_abi::Closed::TimedOut => {
            Err(io::IoError::Rtc("room input cleanup deadline expired".into()))
        }
    }
}
