//! Room ownership ports. Live SDK objects, queues and cleanup belong to the runtime.

use crate::voice::io::{AudioInput, AudioOutput, Result, TextOutput, VideoInput};
use crate::voice::room_io::{
    DeleteRoomRequested, ParticipantLinked, ParticipantUnlinked, RoomIoConfig, TextInput,
};
use crate::voice::rtc::capabilities::RtcRoom;
use crate::voice::rtc::{ParticipantIdentity, ProcessorFailure, RtcError, TrackSid};
use crate::{AudioFrame, Clock, Closed, Spawner, VirtualClock};
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomIoEvent {
    /// `outbound_sip`: a SIP participant the agent dialed (no dispatch rule admitted them).
    ParticipantLinked {
        participant: ParticipantLinked,
        outbound_sip: bool,
    },
    ParticipantUnlinked(ParticipantUnlinked),
    TextInput(TextInput),
    DeleteRoomRequested(DeleteRoomRequested),
    /// The linked participant pressed a keypad key: its symbol.
    KeyPressed(String),
    /// An input processor's failing span started; the input goes on without it.
    InputProcessorFailed(ProcessorFailure),
}

pub trait RoomAudioInputPort: AudioInput {
    fn reading(&self) -> Option<TrackSid>;
    fn aclose<'a>(&'a self) -> crate::BoxFuture<'a, Closed>;
}

pub trait RoomVideoInputPort: VideoInput {
    fn reading(&self) -> Option<TrackSid>;
    fn aclose<'a>(&'a self) -> crate::BoxFuture<'a, Closed>;
}

/// `_core.RoomAudioInput`: the room's microphone leg, taken as itself wherever the host passes
/// an audio input.
#[crate::host(class(name = "RoomAudioInput", native = AudioInput))]
pub trait RoomAudioInput: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn reading(&self) -> Option<TrackSid>;
    #[host(await, name = "next_frame")]
    async fn read_frame(&self) -> Result<Option<AudioFrame>>;
    #[host(await)]
    async fn aclose(&self) -> Result<()>;
}

/// `_core.RoomVideoInput`: the room's video leg, taken as itself wherever the host passes a video
/// input.
#[crate::host(class(name = "RoomVideoInput", native = VideoInput))]
pub trait RoomVideoInput: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn reading(&self) -> Option<TrackSid>;
    #[host(await, name = "next_frame")]
    async fn read_frame(&self) -> Result<Option<crate::MediaRef>>;
    #[host(await)]
    async fn aclose(&self) -> Result<()>;
}

pub trait RoomIo: Send + Sync {
    fn forward_events(
        self: Arc<Self>,
        listener: Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(crate::voice::rtc::RoomEvent) -> bool + Send + Sync>,
    );
    fn audio_input(&self) -> Option<Arc<dyn AudioInput>>;
    fn audio_input_leg(&self) -> Option<Arc<dyn RoomAudioInputPort>>;
    fn video_input(&self) -> Option<Arc<dyn RoomVideoInputPort>>;
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
pub trait SessionRoom: Any + Send + Sync {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>>;
    fn clock(&self) -> Arc<dyn Clock>;
    fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        None
    }
    fn room_io(&self, spawner: Arc<dyn Spawner>, config: RoomIoConfig) -> Arc<dyn RoomIo>;
}

/// `_core_RoomSession`: what a session's room I/O is built from, handed to the host and taken
/// back as itself.
#[crate::host(class(name = "_core_RoomSession", native = SessionRoom))]
pub trait RoomSession: Send + Sync + Sized + 'static {}

#[crate::host(functions)]
pub trait RoomRuntime {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "_core_RoomSession"))]
    type Room;

    #[host(function, name = "_core_create_room_session")]
    fn create_room_session(env: &Self::Env, room: Option<Arc<dyn RtcRoom>>) -> Self::Room;

    /// Wait for the supplied room's connection, avatar participant and subscribed video track.
    #[host(function, name = "wait_avatar_join")]
    async fn wait_avatar_join(
        room: Arc<dyn SessionRoom>,
        identity: String,
    ) -> std::result::Result<(f64, f64), crate::voice::rtc::RtcError>;

    #[host(function)]
    async fn wait_for_participant(
        room: Arc<dyn SessionRoom>,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    ) -> std::result::Result<Option<ParticipantIdentity>, RtcError>;

    #[host(function)]
    async fn wait_for_agent(
        room: Arc<dyn SessionRoom>,
        name: Option<String>,
    ) -> std::result::Result<ParticipantIdentity, RtcError>;

    #[host(function)]
    async fn wait_for_track_publication(
        room: Arc<dyn SessionRoom>,
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    ) -> std::result::Result<(Option<ParticipantIdentity>, TrackSid), RtcError>;
}
