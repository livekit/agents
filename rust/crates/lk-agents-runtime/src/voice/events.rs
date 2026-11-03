//! VoiceRuntime messages retain resources until converted into pure decision inputs.

use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::{SpeechEvent, VadEvent};
use lk_agents_voice::agent_session::effects::StreamKind;
use lk_agents_voice::agent_session::{Command, RequestId, Sent, VoiceCommandOrigin, VoiceEvent};
use lk_agents_voice::audio::ChannelEnd;
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::trace::RoomEventKind;

pub struct StreamOpened {
    pub open: u64,
    pub leg: StreamKind,
    pub outcome: std::result::Result<Opened, lk_agents_abi::errors::ProviderError>,
}

/// Opaque runtime ownership: dropping even an unread/refused result closes its stream.
pub use super::legs::OpeningLeg as Opened;

/// What the runtime queue owner handles: an event in arrival order, or time reaching
/// [`lk_agents_voice::agent_session::VoiceDriver::wake_at`].
pub enum VoiceRuntimeEvent {
    /// For the decision core.
    Input(VoiceEvent),
    /// For the decision core, from a sender awaiting the answer to the call it names: the
    /// runtime keeps the call's reply until the core answers it.
    Awaited(VoiceEvent, RequestId),
    /// A provider completion, adopted into runtime resources before entering the core.
    LegOpened(StreamOpened),
    /// For the data plane, from the decision core.
    AudioInputRouting(lk_agents_voice::data_plane::AudioInputRouterCommand),
    Wake,
    /// The host asked something of its speeches: the requests wait in their own list, which the
    /// runtime steps before anything else it takes.
    HostRequests,
    /// The session's report, read from the decision core now.
    Report(tokio::sync::oneshot::Sender<lk_agents_abi::telemetry::report::SessionReport>),
}

/// What each source sends, as the loop receives it.
macro_rules! queue_events {
    ($($ty:ty => |$v:pat_param| $event:expr;)*) => {$(
        impl From<$ty> for VoiceRuntimeEvent {
            fn from($v: $ty) -> VoiceRuntimeEvent {
                $event
            }
        }
    )*};
}
queue_events! {
    VoiceEvent => |input| VoiceRuntimeEvent::Input(input);
    lk_agents_voice::data_plane::AudioInputRouterCommand => |command| VoiceRuntimeEvent::AudioInputRouting(command);
    Command => |command| VoiceRuntimeEvent::Input(VoiceEvent::Host(Sent { origin: VoiceCommandOrigin::Host, command }));
    lk_agents_voice::audio::Tagged<SpeechEvent> => |ev| VoiceRuntimeEvent::Input(VoiceEvent::Stt(ev, None));
    VadEvent => |ev| VoiceRuntimeEvent::Input(VoiceEvent::Vad(ev));
    RoomEvent => |ev| VoiceRuntimeEvent::Input(VoiceEvent::Room(room_event_kind(&ev)));
    lk_agents_abi::voice::room_owner::RoomIoEvent => |ev| VoiceRuntimeEvent::Input(VoiceEvent::RoomIo(ev));
    GenEvent => |ev| VoiceRuntimeEvent::Input(VoiceEvent::Gen(ev));
    lk_agents_voice::agent_activity::realtime::RealtimeWake => |wake| VoiceRuntimeEvent::Input(VoiceEvent::Realtime(wake));
    lk_agents_abi::turn::detector::LegEvent<lk_agents_voice::audio_recognition::OverlapVerdict> =>
        |ev| VoiceRuntimeEvent::Input(VoiceEvent::Overlap(ev));
    ChannelEnd => |end| VoiceRuntimeEvent::Input(VoiceEvent::ChannelEnded(end));
    StreamOpened => |opened| VoiceRuntimeEvent::LegOpened(opened);
    lk_agents_abi::voice::rtc::VideoFrameRef => |frame| VoiceRuntimeEvent::Input(VoiceEvent::Video(frame));
}

fn room_event_kind(ev: &lk_agents_abi::voice::rtc::RoomEvent) -> RoomEventKind {
    use lk_agents_abi::voice::rtc::RoomEvent;
    match ev {
        RoomEvent::TrackSubscribed { .. } => RoomEventKind::TrackSubscribed,
        RoomEvent::TrackUnsubscribed { .. } => RoomEventKind::TrackUnsubscribed,
        RoomEvent::TrackPublished { .. } => RoomEventKind::TrackPublished,
        RoomEvent::TrackUnpublished { .. } => RoomEventKind::TrackUnpublished,
        RoomEvent::LocalTrackPublished { .. } => RoomEventKind::LocalTrackPublished,
        RoomEvent::ParticipantConnected { .. } => RoomEventKind::ParticipantConnected,
        RoomEvent::ParticipantActive { .. } => RoomEventKind::ParticipantActive,
        RoomEvent::ParticipantAttributesChanged { .. } => {
            RoomEventKind::ParticipantAttributesChanged
        }
        RoomEvent::ParticipantDisconnected { .. } => RoomEventKind::ParticipantDisconnected,
        RoomEvent::ConnectionStateChanged { .. } => RoomEventKind::ConnectionStateChanged,
        RoomEvent::Close { .. } => RoomEventKind::Close,
        RoomEvent::SipDtmf { .. } => RoomEventKind::SipDtmf,
    }
}
