//! Runtime room performers and their supplied session capability.

pub mod input;
pub mod output;
mod owner;
pub mod pre_connect_audio;

use self::output::transcription::SegmentIds;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, Result, TextOutput};
use lk_agents_abi::voice::room_io::RoomIoConfig;
use lk_agents_abi::voice::room_owner::{self, RoomAudioInput, RoomVideoInput};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, capabilities::RtcRoom};
use lk_agents_abi::{Clock, Spawner};
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
    async fn wait_avatar_join(
        room: room_owner::RoomSession,
        identity: String,
    ) -> std::result::Result<(f64, f64), lk_agents_abi::voice::rtc::RtcError> {
        let clock = room.0.clock();
        let room = room.0.rtc().ok_or(lk_agents_abi::voice::rtc::RtcError::NotConnected)?;
        crate::avatar::receiver::wait_avatar_join_room(room, identity, clock).await
    }

    fn create_room_session(room: Option<Arc<dyn RtcRoom>>) -> room_owner::RoomSession {
        let time = lk_agents_voice::agent_session::Timebase::real();
        match room {
            Some(room) => {
                let pool = crate::process::workers().ok();
                room_owner::RoomSession(Arc::new(FixedRoom { rtc: room, time, pool }))
            }
            None => room_owner::RoomSession(Arc::new(DetachedSession(time))),
        }
    }
}

impl room_owner::RoomIo for RoomIo {
    fn forward_events(
        self: Arc<Self>,
        listener: Arc<dyn Fn(room_owner::RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(lk_agents_abi::voice::rtc::RoomEvent) -> bool + Send + Sync>,
    ) {
        RoomIo::forward_events(&self, listener, rtc_listener);
    }
    fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        RoomIo::audio_input(self)
    }
    fn audio_input_leg(&self) -> Option<Arc<RoomAudioInput>> {
        RoomIo::audio_input_leg(self).map(|input| Arc::new(RoomAudioInput(input)))
    }
    fn video_input(&self) -> Option<Arc<RoomVideoInput>> {
        RoomIo::video_input(self).map(|input| Arc::new(RoomVideoInput(input)))
    }
    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        RoomIo::audio_output(self)
    }
    fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        RoomIo::transcription_output(self)
    }
    fn wants_transcript_sync(&self, external_audio: bool) -> bool {
        RoomIo::wants_transcript_sync(self, external_audio)
    }
    fn linked_participant(&self) -> Option<ParticipantIdentity> {
        RoomIo::linked_participant(self)
    }
    fn start(self: Arc<Self>) {
        RoomIo::start(&self);
    }
    fn on_agent_state_changed(self: Arc<Self>, state: &str) {
        RoomIo::on_agent_state_changed(&self, state);
    }
    fn wait_for_ready<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { RoomIo::wait_for_ready(self).await })
    }
    fn set_participant<'a>(
        &'a self,
        identity: Option<ParticipantIdentity>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { RoomIo::set_participant(self, identity).await })
    }
    fn unset_participant<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { RoomIo::unset_participant(self).await })
    }
    fn on_user_input_transcribed<'a>(
        &'a self,
        transcript: &'a str,
        is_final: bool,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { RoomIo::on_user_input_transcribed(self, transcript, is_final).await })
    }
    fn aclose(self: Arc<Self>) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move { RoomIo::aclose(&self).await })
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
