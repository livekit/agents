//! Runtime messages retain resources until converted into pure decision inputs.

use lk_agents_abi::voice::rtc::RoomEvent;
use lk_agents_abi::{SpeechEvent, VadEvent};
use lk_agents_voice::agent_session::effects::StreamKind;
use lk_agents_voice::agent_session::{Command, Input, Origin, RequestId, Sent};
use lk_agents_voice::audio::ChannelEnd;
use lk_agents_voice::generation::GenEvent;
use lk_agents_voice::trace::RoomEventKind;

pub struct StreamOpened {
    pub open: u64,
    pub leg: StreamKind,
    pub outcome: std::result::Result<Opened, lk_agents_abi::errors::ProviderError>,
}

pub enum Opened {
    /// Runtime owns both sides until the core accepts or discards the prepared resource.
    Stt(lk_agents_abi::SttSink, lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSink, lk_agents_abi::VadSource),
}

/// What the runtime queue owner handles: an event in arrival order, or time reaching
/// [`lk_agents_voice::agent_session::Driver::wake_at`].
pub enum Event {
    /// For the decision core.
    Input(Input),
    /// For the decision core, from a sender awaiting the answer to the call it names: the
    /// runtime keeps the call's reply until the core answers it.
    Awaited(Input, RequestId),
    /// A provider completion, adopted into runtime resources before entering the core.
    LegOpened(StreamOpened),
    /// For the data plane, from the decision core.
    Forwarder(lk_agents_voice::data_plane::ForwarderCommand),
    Wake,
    /// The host asked something of its speeches: the requests wait in their own list, which the
    /// runtime steps before anything else it takes.
    HostRequests,
}

/// What each source sends, as the loop receives it.
macro_rules! queue_events {
    ($($ty:ty => |$v:pat_param| $event:expr;)*) => {$(
        impl From<$ty> for Event {
            fn from($v: $ty) -> Event {
                $event
            }
        }
    )*};
}
queue_events! {
    Input => |input| Event::Input(input);
    lk_agents_voice::data_plane::ForwarderCommand => |command| Event::Forwarder(command);
    Command => |command| Event::Input(Input::Host(Sent { origin: Origin::Host, command }));
    lk_agents_voice::audio::Tagged<SpeechEvent> => |ev| Event::Input(Input::Stt(ev, None));
    VadEvent => |ev| Event::Input(Input::Vad(ev));
    RoomEvent => |ev| Event::Input(Input::Room(room_event_kind(&ev)));
    lk_agents_abi::voice::room_owner::RoomIoEvent => |ev| Event::Input(Input::RoomIo(ev));
    GenEvent => |ev| Event::Input(Input::Gen(ev));
    lk_agents_voice::agent_activity::realtime::RealtimeWake => |wake| Event::Input(Input::Realtime(wake));
    lk_agents_abi::turn::detector::LegEvent<lk_agents_voice::audio_recognition::OverlapVerdict> =>
        |ev| Event::Input(Input::Overlap(ev));
    ChannelEnd => |end| Event::Input(Input::ChannelEnded(end));
    StreamOpened => |opened| Event::LegOpened(opened);
    lk_agents_abi::voice::rtc::VideoFrameRef => |frame| Event::Input(Input::Video(frame));
}

fn room_event_kind(ev: &lk_agents_abi::voice::rtc::RoomEvent) -> RoomEventKind {
    use lk_agents_abi::voice::rtc::RoomEvent as E;
    match ev {
        E::TrackSubscribed { .. } => RoomEventKind::TrackSubscribed,
        E::TrackUnsubscribed { .. } => RoomEventKind::TrackUnsubscribed,
        E::TrackPublished { .. } => RoomEventKind::TrackPublished,
        E::TrackUnpublished { .. } => RoomEventKind::TrackUnpublished,
        E::LocalTrackPublished { .. } => RoomEventKind::LocalTrackPublished,
        E::ParticipantConnected { .. } => RoomEventKind::ParticipantConnected,
        E::ParticipantActive { .. } => RoomEventKind::ParticipantActive,
        E::ParticipantAttributesChanged { .. } => RoomEventKind::ParticipantAttributesChanged,
        E::ParticipantDisconnected { .. } => RoomEventKind::ParticipantDisconnected,
        E::ConnectionStateChanged { .. } => RoomEventKind::ConnectionStateChanged,
        E::Close { .. } => RoomEventKind::Close,
        E::SipDtmf { .. } => RoomEventKind::SipDtmf,
    }
}
