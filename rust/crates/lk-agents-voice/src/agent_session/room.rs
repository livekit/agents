//! Session consequences of room attachment and departure.

use super::{Close, VoiceDriver, VoiceHostEvent};
use crate::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::room_io::DeleteRoomRequested;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_abi::voice::room_owner::RoomIoEvent;
use lk_agents_abi::voice::rtc::DisconnectReason;

/// `RoomOptions`' two session-level switches and the room they apply to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomLinkConfig {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub room: String,
}

pub(crate) struct RoomLink {
    config: RoomLinkConfig,
    /// A later disconnect closes nothing, a second close deletes nothing.
    deletion_requested: bool,
    /// The participants the room described, by identity.
    participants: std::collections::BTreeMap<String, ParticipantInfo>,
    /// The linked participant's identity, and its description.
    linked: Option<String>,
    linked_info: Option<ParticipantInfo>,
    /// The agent's own participant, and whether the room is connected.
    local: Option<ParticipantInfo>,
    connected: bool,
}

impl RoomLink {
    pub(crate) fn new(config: Option<RoomLinkConfig>) -> Self {
        let config = config.unwrap_or(RoomLinkConfig {
            close_on_disconnect: true,
            delete_room_on_close: false,
            room: String::new(),
        });
        RoomLink {
            config,
            deletion_requested: false,
            participants: Default::default(),
            linked: None,
            linked_info: None,
            local: None,
            connected: false,
        }
    }

    /// The linked participant: the one the user turns are heard from.
    pub(crate) fn linked_participant(&self) -> Option<&ParticipantInfo> {
        self.linked_info.as_ref()
    }

    /// The agent's own participant, while the room is connected.
    pub(crate) fn local_participant(&self) -> Option<&ParticipantInfo> {
        self.local.as_ref().filter(|_| self.connected)
    }

    /// The room described a participant: the linked one, or the agent's own (`local`).
    fn describe(&mut self, info: &ParticipantInfo, local: bool) {
        if local {
            self.local = Some(info.clone());
            return;
        }
        let identity = info.identity.as_str();
        if self.linked.as_deref() == Some(identity) {
            self.linked_info = Some(info.clone());
        }
        self.participants.insert(identity.to_owned(), info.clone());
    }

    /// The participant `identity` linked (`left: false`) or left: its description, when the room
    /// gave one before.
    fn link(&mut self, identity: &str, left: bool) -> Option<ParticipantInfo> {
        let known = self.participants.get(identity).cloned();
        self.linked = (!left).then(|| identity.to_owned());
        self.linked_info = known.clone().filter(|_| !left);
        known
    }

    /// Does this disconnect end the session? A deletion in flight is the end of the call already
    /// (`not self._delete_room_task`).
    fn disconnect_closes(&self, reason: DisconnectReason) -> bool {
        !self.deletion_requested
            && self.config.close_on_disconnect
            && reason.closes_session_by_default()
    }

    /// `delete_room_on_close`, once: the room to delete.
    fn request_deletion(&mut self) -> Option<String> {
        if self.deletion_requested || !self.config.delete_room_on_close {
            return None;
        }
        self.deletion_requested = true;
        Some(self.config.room.clone())
    }
}

impl VoiceDriver {
    pub(crate) fn on_room_io(&mut self, now: Ts, ev: RoomIoEvent) {
        let closing = self.session.closing_past_edge();
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                let identity = unlinked.identity.as_str();
                let known = self.session.room.link(identity, true);
                self.participant_span_event(identity, known, Some(unlinked.reason));
                {
                    let updates = self.session.states.on_link(now, false, closing);
                    self.apply_user_state_updates(updates);
                };
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.room.disconnect_closes(reason);
                self.record_trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    let reason = CloseReason::ParticipantDisconnected;
                    self.begin_closing(now, Close { reason, drain: false, error: None });
                }
            }
            RoomIoEvent::ParticipantLinked { participant, outbound_sip } => {
                self.record_trace(RoomStep::Link(RoomLinkStep::Linked));
                let identity = participant.identity.as_str();
                let known = self.session.room.link(identity, false);
                self.participant_span_event(identity, known, None);
                {
                    let updates = self.session.states.on_link(now, true, closing);
                    self.apply_user_state_updates(updates);
                };
                // `_on_room_io_participant_linked`: an outbound SIP call has no warmup, unless the
                // application set `aec_warmup_duration` itself.
                if *outbound_sip && !self.session.config.aec_warmup_given() {
                    let patch = crate::agent_activity::options::aec_warmup_layer(None);
                    let diff = self.session.config.configure(&patch);
                    let layer = crate::agent_activity::options::ConfigLayer::Session;
                    self.apply_config(now, layer, diff);
                }
            }
            RoomIoEvent::KeyPressed(digit) => {
                let key = crate::workflows::dtmf::DtmfEvent::parse(digit);
                self.record_trace(RoomStep::Link(RoomLinkStep::KeyPressed { key }));
                {
                    let updates = self.session.states.reset_away_timer(now, closing);
                    self.apply_user_state_updates(updates);
                };
                let digit = crate::workflows::dtmf::DtmfTaskEvent::Digit(digit.clone());
                return self.dtmf_step(digit);
            }
            RoomIoEvent::InputProcessorFailed(failure) => {
                let error = lk_agents_abi::errors::ApiError::new(failure.message.clone());
                let ev = crate::events::ErrorEvent {
                    source: crate::events::ErrorSource::AudioProcessing,
                    recoverable: true,
                    error: error.into(),
                    speech: None,
                    processor_failure: Some(failure.clone()),
                };
                self.publish_event(crate::events::SessionEvent::Error(Box::new(ev)));
                return;
            }
            // For the spans only.
            RoomIoEvent::ParticipantInfo { participant, local } => {
                return self.session.room.describe(participant, *local);
            }
            RoomIoEvent::ConnectionStateChanged(state) => {
                self.session.room.connected = state.is_connected();
                return self.connection_span_event(*state);
            }
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        self.record_effect(crate::agent_session::effects::VoiceEffect::Host(VoiceHostEvent::Room(
            ev,
        )));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.room.request_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            self.record_effect(crate::agent_session::effects::VoiceEffect::Host(
                VoiceHostEvent::Room(RoomIoEvent::DeleteRoomRequested(DeleteRoomRequested {
                    room,
                })),
            ));
        }
    }
}
