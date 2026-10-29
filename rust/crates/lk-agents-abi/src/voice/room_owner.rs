//! Room ownership ports. Live SDK objects, queues and cleanup belong to the runtime.

use crate::voice::io::{AudioInput, AudioOutput, Result, TextOutput, VideoInput};
use crate::voice::room_io::{
    DeleteRoomRequested, ParticipantLinked, ParticipantUnlinked, RoomIoConfig, TextInput,
};
use crate::voice::rtc::capabilities::RtcRoom;
use crate::voice::rtc::{ParticipantIdentity, RtcError, TrackSid};
use crate::{AudioFrame, Clock, Closed, Spawner, VirtualClock};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomIoEvent {
    ParticipantLinked(ParticipantLinked),
    ParticipantUnlinked(ParticipantUnlinked),
    TextInput(TextInput),
    DeleteRoomRequested(DeleteRoomRequested),
    KeyPressed,
}

pub trait RoomAudioInputPort: AudioInput {
    fn reading(&self) -> Option<TrackSid>;
    fn aclose<'a>(&'a self) -> crate::BoxFuture<'a, Closed>;
}

pub trait RoomVideoInputPort: VideoInput {
    fn reading(&self) -> Option<TrackSid>;
    fn aclose<'a>(&'a self) -> crate::BoxFuture<'a, Closed>;
}

pub struct RoomAudioInput(pub Arc<dyn RoomAudioInputPort>);

#[crate::host(object(cell = "arc", as_dyn(AudioInput = as_audio_input)))]
impl RoomAudioInput {
    pub fn as_audio_input(self: &Arc<Self>) -> Arc<dyn AudioInput> {
        self.0.clone()
    }
    #[host(getter)]
    pub fn reading(&self) -> Option<TrackSid> {
        self.0.reading()
    }
    #[host(method)]
    pub async fn next_frame(&self) -> Result<Option<AudioFrame>> {
        self.0.next_frame().await
    }
    #[host(method)]
    pub async fn aclose(&self) -> Result<()> {
        input_closed(self.0.aclose().await)
    }
}

pub struct RoomVideoInput(pub Arc<dyn RoomVideoInputPort>);

#[crate::host(object(cell = "arc", as_dyn(VideoInput = as_video_input)))]
impl RoomVideoInput {
    pub fn as_video_input(self: &Arc<Self>) -> Arc<dyn VideoInput> {
        self.0.clone()
    }
    #[host(getter)]
    pub fn reading(&self) -> Option<TrackSid> {
        self.0.reading()
    }
    #[host(method)]
    pub async fn next_frame(&self) -> Result<Option<crate::MediaRef>> {
        self.0.next_frame().await.map(|frame| frame.map(|frame| frame.media))
    }
    #[host(method)]
    pub async fn aclose(&self) -> Result<()> {
        input_closed(self.0.aclose().await)
    }
}

pub trait RoomIo: Send + Sync {
    fn forward_events(
        self: Arc<Self>,
        listener: Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(crate::voice::rtc::RoomEvent) -> bool + Send + Sync>,
    );
    fn audio_input(&self) -> Option<Arc<dyn AudioInput>>;
    fn audio_input_leg(&self) -> Option<Arc<RoomAudioInput>>;
    fn video_input(&self) -> Option<Arc<RoomVideoInput>>;
    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>>;
    fn transcription_output(&self) -> Option<Arc<dyn TextOutput>>;
    fn wants_transcript_sync(&self, external_audio: bool) -> bool;
    fn linked_participant(&self) -> Option<ParticipantIdentity>;
    fn start(self: Arc<Self>);
    fn on_agent_state_changed(self: Arc<Self>, state: &str);
    fn wait_for_ready<'a>(&'a self) -> crate::BoxFuture<'a, Result<()>>;
    fn set_participant<'a>(
        &'a self,
        identity: Option<ParticipantIdentity>,
    ) -> crate::BoxFuture<'a, Result<()>>;
    fn unset_participant<'a>(&'a self) -> crate::BoxFuture<'a, Result<()>>;
    fn on_user_input_transcribed<'a>(
        &'a self,
        transcript: &'a str,
        is_final: bool,
    ) -> crate::BoxFuture<'a, Result<()>>;
    fn aclose(self: Arc<Self>) -> crate::BoxFuture<'static, Closed>;
}

/// A supplied runtime capability constructs the room performer; the core never locates a runtime.
#[crate::host(native(refused = "expected a native room session"))]
pub trait SessionRoom: Send + Sync {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>>;
    fn clock(&self) -> Arc<dyn Clock>;
    fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        None
    }
    fn room_io(&self, spawner: Arc<dyn Spawner>, config: RoomIoConfig) -> Arc<dyn RoomIo>;
}

#[derive(Clone)]
pub struct RoomSession(pub Arc<dyn SessionRoom>);

/// Runtime capability connecting a session to its room, clock, and owned I/O.
#[crate::host(object(name = "_core_RoomSession", cell = "value", pyclass(from_py_object), derive(Clone), as_dyn(SessionRoom = session_room)))]
impl RoomSession {
    pub fn session_room(&self) -> Arc<dyn SessionRoom> {
        self.0.clone()
    }
}

#[crate::host(functions)]
pub trait RoomRuntime {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(function, name = "_core_create_room_session")]
    fn create_room_session(env: &Self::Env, room: Option<Arc<dyn RtcRoom>>) -> RoomSession;

    /// Wait for the supplied room's connection, avatar participant and subscribed video track.
    #[host(function, name = "wait_avatar_join")]
    async fn wait_avatar_join(
        room: RoomSession,
        identity: String,
    ) -> std::result::Result<(f64, f64), crate::voice::rtc::RtcError>;

    #[host(function)]
    async fn wait_for_participant(
        room: RoomSession,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    ) -> std::result::Result<Option<ParticipantIdentity>, RtcError>;

    #[host(function)]
    async fn wait_for_agent(
        room: RoomSession,
        name: Option<String>,
    ) -> std::result::Result<ParticipantIdentity, RtcError>;

    #[host(function)]
    async fn wait_for_track_publication(
        room: RoomSession,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    ) -> std::result::Result<(Option<ParticipantIdentity>, TrackSid), RtcError>;
}

fn input_closed(outcome: Closed) -> Result<()> {
    match outcome {
        Closed::Clean => Ok(()),
        Closed::Failed => Err(crate::voice::io::IoError::Rtc("room input cleanup failed".into())),
        Closed::TimedOut => {
            Err(crate::voice::io::IoError::Rtc("room input cleanup deadline expired".into()))
        }
    }
}
