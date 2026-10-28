//! Typed room observations and the native SDK event callback port.

use std::fmt;
use std::sync::Arc;

use super::capabilities::{RemoteParticipant, TrackPublication};
use super::ports::Receiver;
use super::{ConnectionState, DisconnectReason, ParticipantIdentity};

/// One of the twelve.
///
/// Handles rather than identifiers in the payloads, so that "which participant, which publication" and "this publication is gone" are expressible.
/// An event carrying only a `TrackSid` would send every consumer back to the room to look the
/// handle up, and that lookup races the very disconnection the event reports.
#[derive(Clone, crate::Vocabulary)]
pub enum RoomEvent {
    /// A remote track is subscribed and its frames can be read.
    /// `_input.py` reacts by opening an `audio_stream`.
    TrackSubscribed {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// A remote track is no longer subscribed: its stream ended. Re-subscribing the same
    /// publication (a reconnect, `set_subscribed(true)`) arrives as a new `TrackSubscribed`.
    TrackUnsubscribed {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// A remote participant published a track (not necessarily subscribed).
    /// The transcription outputs use this to learn the user's microphone `sid`.
    TrackPublished {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// A remote track went away. `_input.py` closes its stream and falls back to the
    /// participant's next available track.
    TrackUnpublished {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// The agent's own track was published; carries the `sid` the agent's transcription is
    /// attached to.
    LocalTrackPublished {
        publication: Arc<dyn TrackPublication>,
    },
    ParticipantConnected {
        participant: Arc<dyn RemoteParticipant>,
    },
    /// A remote participant is active: connected and ready to exchange media
    /// (`utils.wait_for_participant` waits on it).
    ParticipantActive {
        participant: Arc<dyn RemoteParticipant>,
    },
    /// A remote participant's attributes changed; the participant holds the new ones.
    ParticipantAttributesChanged {
        participant: Arc<dyn RemoteParticipant>,
    },
    /// Carries the reason **on the event**.
    ///
    /// The reference reads `participant.disconnect_reason` off the participant object at handler
    /// time (`room_io.RoomIO._on_participant_connected`), which only works because Python holds the SDK's own mutable
    /// object. A handle whose `disconnect_reason` is `None` until some later mutation would be a
    /// race across the FFI, so the reason travels with the event that reports it.
    ParticipantDisconnected {
        participant: Arc<dyn RemoteParticipant>,
        reason: DisconnectReason,
    },
    ConnectionStateChanged {
        state: ConnectionState,
    },
    /// `rtc.Room`'s `disconnected` event.
    Close {
        reason: DisconnectReason,
    },
    /// A key pressed on a SIP participant's keypad (`sip_dtmf_received`): its symbol, and who
    /// pressed it.
    SipDtmf {
        digit: String,
        participant: Option<ParticipantIdentity>,
    },
}

// Hand-written so `RemoteParticipant`/`TrackPublication` need not be `Debug`.
impl fmt::Debug for RoomEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoomEvent::TrackSubscribed { participant, publication } => f
                .debug_struct("TrackSubscribed")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::TrackUnsubscribed { participant, publication } => f
                .debug_struct("TrackUnsubscribed")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::TrackPublished { participant, publication } => f
                .debug_struct("TrackPublished")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::TrackUnpublished { participant, publication } => f
                .debug_struct("TrackUnpublished")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::LocalTrackPublished { publication } => f
                .debug_struct("LocalTrackPublished")
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::ParticipantConnected { participant } => f
                .debug_struct("ParticipantConnected")
                .field("participant", participant.identity())
                .finish(),
            RoomEvent::ParticipantActive { participant } => f
                .debug_struct("ParticipantActive")
                .field("participant", participant.identity())
                .finish(),
            RoomEvent::ParticipantAttributesChanged { participant } => f
                .debug_struct("ParticipantAttributesChanged")
                .field("participant", participant.identity())
                .finish(),
            RoomEvent::ParticipantDisconnected { participant, reason } => f
                .debug_struct("ParticipantDisconnected")
                .field("participant", participant.identity())
                .field("reason", reason)
                .finish(),
            RoomEvent::ConnectionStateChanged { state } => {
                f.debug_struct("ConnectionStateChanged").field("state", state).finish()
            }
            RoomEvent::Close { reason } => f.debug_struct("Close").field("reason", reason).finish(),
            RoomEvent::SipDtmf { digit, participant } => f
                .debug_struct("SipDtmf")
                .field("digit", digit)
                .field("participant", participant)
                .finish(),
        }
    }
}

/// The runtime's native event fan-out. This port contains no queue or synchronization state.
pub trait RoomEventSink: Send + Sync {
    fn subscribe(&self) -> Receiver<RoomEvent>;
    fn emit(&self, event: RoomEvent) -> usize;
    fn subscriber_count(&self) -> usize;
}

/// The SDK's callback handle; the one live fan-out remains runtime-owned.
#[crate::class(from_py_object)]
#[derive(Clone)]
pub struct RoomEventBus(pub Arc<dyn RoomEventSink>);

// Register the native value class; SDK callbacks use the modern function boundary below.
#[crate::host(object)]
impl RoomEventBus {
    #[host(getter)]
    pub fn subscriber_count(&self) -> usize {
        self.0.subscriber_count()
    }
}

impl RoomEventBus {
    pub fn subscribe(&self) -> Receiver<RoomEvent> {
        self.0.subscribe()
    }
    pub fn emit(&self, event: RoomEvent) -> usize {
        self.0.emit(event)
    }
}

impl RoomEventBus {
    pub fn track_subscribed(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.0.emit(RoomEvent::TrackSubscribed { participant, publication });
    }

    pub fn track_unsubscribed(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.0.emit(RoomEvent::TrackUnsubscribed { participant, publication });
    }

    pub fn track_published(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.0.emit(RoomEvent::TrackPublished { participant, publication });
    }

    pub fn track_unpublished(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.0.emit(RoomEvent::TrackUnpublished { participant, publication });
    }

    pub fn local_track_published(&self, publication: Arc<dyn TrackPublication>) {
        self.0.emit(RoomEvent::LocalTrackPublished { publication });
    }

    pub fn participant_connected(&self, participant: Arc<dyn RemoteParticipant>) {
        self.0.emit(RoomEvent::ParticipantConnected { participant });
    }

    pub fn participant_active(&self, participant: Arc<dyn RemoteParticipant>) {
        self.0.emit(RoomEvent::ParticipantActive { participant });
    }

    pub fn participant_attributes_changed(&self, participant: Arc<dyn RemoteParticipant>) {
        self.0.emit(RoomEvent::ParticipantAttributesChanged { participant });
    }

    pub fn participant_disconnected(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        reason: DisconnectReason,
    ) {
        self.0.emit(RoomEvent::ParticipantDisconnected { participant, reason });
    }

    pub fn connection_state_changed(&self, state: ConnectionState) {
        self.0.emit(RoomEvent::ConnectionStateChanged { state });
    }

    pub fn disconnected(&self, reason: DisconnectReason) {
        self.0.emit(RoomEvent::Close { reason });
    }

    pub fn sip_dtmf_received(&self, digit: String, participant: Option<ParticipantIdentity>) {
        self.0.emit(RoomEvent::SipDtmf { digit, participant });
    }
}

/// Native construction and SDK callbacks for the runtime-owned event fan-out.
#[crate::host(functions)]
pub trait RtcRuntime: Send + Sync + Sized + 'static {
    #[host(function, name = "_core_create_room_event_bus")]
    fn create_room_event_bus() -> RoomEventBus;

    #[host(function, name = "_core_room_track_subscribed")]
    fn track_subscribed(
        bus: RoomEventBus,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        bus.track_subscribed(participant, publication);
    }

    #[host(function, name = "_core_room_track_unsubscribed")]
    fn track_unsubscribed(
        bus: RoomEventBus,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        bus.track_unsubscribed(participant, publication);
    }

    #[host(function, name = "_core_room_track_published")]
    fn track_published(
        bus: RoomEventBus,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        bus.track_published(participant, publication);
    }

    #[host(function, name = "_core_room_track_unpublished")]
    fn track_unpublished(
        bus: RoomEventBus,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        bus.track_unpublished(participant, publication);
    }

    #[host(function, name = "_core_room_local_track_published")]
    fn local_track_published(bus: RoomEventBus, publication: Arc<dyn TrackPublication>) {
        bus.local_track_published(publication);
    }

    #[host(function, name = "_core_room_participant_connected")]
    fn participant_connected(bus: RoomEventBus, participant: Arc<dyn RemoteParticipant>) {
        bus.participant_connected(participant);
    }

    #[host(function, name = "_core_room_participant_active")]
    fn participant_active(bus: RoomEventBus, participant: Arc<dyn RemoteParticipant>) {
        bus.participant_active(participant);
    }

    #[host(function, name = "_core_room_participant_attributes_changed")]
    fn participant_attributes_changed(bus: RoomEventBus, participant: Arc<dyn RemoteParticipant>) {
        bus.participant_attributes_changed(participant);
    }

    #[host(function, name = "_core_room_participant_disconnected")]
    fn participant_disconnected(
        bus: RoomEventBus,
        participant: Arc<dyn RemoteParticipant>,
        reason: DisconnectReason,
    ) {
        bus.participant_disconnected(participant, reason);
    }

    #[host(function, name = "_core_room_connection_state_changed")]
    fn connection_state_changed(bus: RoomEventBus, state: ConnectionState) {
        bus.connection_state_changed(state);
    }

    #[host(function, name = "_core_room_disconnected")]
    fn disconnected(bus: RoomEventBus, reason: DisconnectReason) {
        bus.disconnected(reason);
    }

    #[host(function, name = "_core_room_sip_dtmf_received")]
    fn sip_dtmf_received(
        bus: RoomEventBus,
        digit: String,
        participant: Option<ParticipantIdentity>,
    ) {
        bus.sip_dtmf_received(digit, participant);
    }
}
