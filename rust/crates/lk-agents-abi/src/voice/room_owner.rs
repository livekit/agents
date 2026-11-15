use crate::voice::io::{AudioInput, AudioOutput, Result, TranscriptionOutput, VideoInput};
use crate::voice::room_io::{
    DeleteRoomRequested, ParticipantInfo, ParticipantLinked, ParticipantUnlinked, RoomIoConfig,
    TextInput,
};
use crate::voice::rtc::capabilities::RtcRoom;
use crate::voice::rtc::{
    ConnectionState, ParticipantIdentity, ProcessorFailure, RtcError, TrackSid,
};
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
    /// Who a participant is, for the session's spans: the linked one before its
    /// `ParticipantLinked`, the agent's own (`local`) once the room connects.
    ParticipantInfo {
        participant: ParticipantInfo,
        local: bool,
    },
    /// The room's connection state changed.
    ConnectionStateChanged(ConnectionState),
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
    async fn read_frame(
        &self,
    ) -> Result<Option<std::sync::Arc<dyn crate::utils::video::RtcVideoFrame>>>;
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
    fn transcription_output(&self) -> Option<Arc<dyn TranscriptionOutput>>;
    fn wants_transcript_sync(&self, external_audio: bool) -> bool;
    fn linked_participant(&self) -> Option<ParticipantIdentity>;
    /// Starts the room's IO; its startup spans (the participant, its track, the agent's
    /// publication) open under `startup`, the session's `session_start`.
    fn start(self: Arc<Self>, startup: crate::telemetry::Cx);
    /// The consumer and its transports finished startup successfully. A cancelled
    /// or failed startup must not activate its binding or a legacy projection.
    fn session_started(&self) -> Result<()> {
        Ok(())
    }
    fn on_agent_state_changed(self: Arc<Self>, state: crate::voice::events::AgentState);
    fn wait_for_ready<'a>(&'a self) -> crate::BoxFuture<'a, Result<()>>;
    fn set_participant<'a>(
        &'a self,
        identity: Option<ParticipantIdentity>,
    ) -> crate::BoxFuture<'a, Result<()>>;
    fn unset_participant<'a>(&'a self) -> crate::BoxFuture<'a, Result<()>>;
    /// The user's transcript, in the segment the session names.
    fn on_user_input_transcribed<'a>(
        &'a self,
        transcript: &'a str,
        is_final: bool,
        segment: &'a str,
    ) -> crate::BoxFuture<'a, Result<()>>;
    /// The user's transcript `segment`, already final, now reads `text`.
    fn on_user_input_corrected(&self, segment: &str, text: &str);
    /// Admit cleanup with its owner's original deadline. A later poll or waiter
    /// must not reset the time allowed for retiring the same resources.
    fn aclose_by(self: Arc<Self>, deadline: crate::Ts) -> crate::BoxFuture<'static, Closed>;
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
    fn room_io(
        &self,
        spawner: Arc<dyn Spawner>,
        config: RoomIoConfig,
    ) -> std::result::Result<Arc<dyn RoomIo>, crate::voice::room_binding::BindingRefused>;
    /// Construct the consumer's ports after its binding has admitted activation.
    fn create_io(&self, spawner: Arc<dyn Spawner>, config: RoomIoConfig) -> Arc<dyn RoomIo>;
}

/// `_core_RoomSession`: what a session's room I/O is built from, handed to the host and taken
/// back as itself.
#[crate::host(class(name = "_core_RoomSession", native = SessionRoom))]
pub trait RoomSession: Send + Sync + Sized + 'static {}

/// Native attachment ownership, cached independently of SDK room references.
#[crate::host(native(refused = "expected native room attachments"))]
pub trait RoomAttachmentsPort: Send + Sync {
    /// The factory runs after admission and its resource is owned before the next await.
    fn attach(
        &self,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        create: Box<dyn FnOnce() -> Arc<dyn RoomIo> + Send>,
    ) -> std::result::Result<Arc<dyn RoomIo>, crate::voice::room_binding::BindingRefused>;
}

/// Room-local ownership without a strong reference back to the SDK room.
#[crate::host(class(name = "_core_RoomAttachments", native = RoomAttachmentsPort))]
pub trait RoomAttachments: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "_core_RoomBinding"))]
    type Binding;

    #[host(new)]
    fn new() -> Self;

    #[host(call)]
    fn bind(
        &self,
        env: &Self::Env,
        key: String,
        #[host(kw)] room: Arc<dyn SessionRoom>,
        #[host(kw)] sdk_room: crate::HostRef,
    ) -> std::result::Result<Self::Binding, crate::voice::room_io::RoomOptionsError>;

    #[host(await)]
    async fn aclose(&self) -> std::result::Result<(), crate::voice::room_io::RoomOptionsError>;
}

/// A declared room binding. Its only activation authority is transferred to a consumer.
#[crate::host(class(name = "_core_RoomBinding", native = SessionRoom))]
pub trait RoomBinding: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn key(&self) -> String;
    #[host(getter)]
    fn room(&self) -> crate::HostRef;
    #[host(call)]
    fn prepare(
        &self,
        options: RoomIoConfig,
        host_audio_input: bool,
        host_video_input: bool,
        host_audio_output: bool,
    ) -> std::result::Result<(), crate::voice::room_io::RoomOptionsError>;
    #[host(await)]
    async fn aclose(&self) -> std::result::Result<(), crate::voice::room_io::RoomOptionsError>;
}

/// Shared room I/O. Ordinary session startup obtains the same service automatically.
#[crate::host(facade(path = "livekit.agents.voice.room_io", name = "RoomIO", native = RoomAttachments, extended))]
pub trait RoomIOFacade {
    /// Declare a binding without subscribing to tracks or publishing outputs.
    #[host(
        bind(room = "_room_native", sdk_room = "_room"),
        returns = "..voice.room_io.RoomBinding"
    )]
    fn bind(&self, key: String);
    /// Close this room's bindings without disconnecting the SDK room.
    async fn aclose(&self);
}

/// One consumer's room input/output ports and lifetime. Obtain it with `RoomIO.bind(key)`.
#[crate::host(facade(path = "livekit.agents.voice.room_io", name = "RoomBinding", native = RoomBinding, from_native))]
pub trait RoomBindingFacade {
    #[host(getter)]
    fn key(&self) -> String;
    #[host(getter, annotation(ret = "livekit.rtc.Room"))]
    fn room(&self) -> crate::HostRef;
    async fn aclose(&self);
}

#[crate::host(functions)]
pub trait RoomRuntime {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "_core_RoomSession"))]
    type Room;

    #[host(function, name = "_core_create_room_session")]
    fn create_room_session(env: &Self::Env, room: Option<Arc<dyn RtcRoom>>) -> Self::Room;

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
    ) -> std::result::Result<TrackPublished, RtcError>;
}

/// A publication found in the room: its participant (`None`, the local one) and its track.
#[crate::host(record(module = "room_runtime", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackPublished {
    pub publisher: Option<ParticipantIdentity>,
    pub sid: TrackSid,
}
