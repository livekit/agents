//! Session consequences of room attachment and departure.

use super::effects::Out;
use super::{Close, Driver, HostEvent};
use crate::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::room_io::DeleteRoomRequested;
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
}

impl RoomLink {
    pub(crate) fn new(config: Option<RoomLinkConfig>) -> Self {
        let config = config.unwrap_or(RoomLinkConfig {
            close_on_disconnect: true,
            delete_room_on_close: false,
            room: String::new(),
        });
        RoomLink { config, deletion_requested: false }
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

impl Driver {
    pub(crate) fn on_room_io(&mut self, now: Ts, ev: RoomIoEvent, out: &mut Out) {
        let closing = self.session.closing_past_edge();
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                out.machine().append(self.session.states.on_link(now, false, closing));
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.room.disconnect_closes(reason);
                out.trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    let reason = CloseReason::ParticipantDisconnected;
                    self.begin_closing(now, Close { reason, drain: false, error: None }, out);
                }
            }
            RoomIoEvent::ParticipantLinked { outbound_sip, .. } => {
                out.trace(RoomStep::Link(RoomLinkStep::Linked));
                out.machine().append(self.session.states.on_link(now, true, closing));
                // `_on_room_io_participant_linked`: an outbound SIP call has no warmup, unless the
                // application set `aec_warmup_duration` itself.
                if *outbound_sip && !self.session.config.aec_warmup_given() {
                    let patch = crate::agent_activity::options::aec_warmup_layer(None);
                    let diff = self.session.config.configure(&patch);
                    let layer = crate::agent_activity::options::ConfigLayer::Session;
                    self.apply_config(now, layer, diff, out);
                }
            }
            RoomIoEvent::KeyPressed(digit) => {
                let key = crate::workflows::dtmf::DtmfEvent::parse(digit);
                out.trace(RoomStep::Link(RoomLinkStep::KeyPressed { key }));
                out.machine().append(self.session.states.reset_away_timer(now, closing));
                let digit = crate::workflows::dtmf::DtmfTaskEvent::Digit(digit.clone());
                return self.dtmf_step(digit, out);
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
                out.machine().emit(crate::events::SessionEvent::Error(Box::new(ev)));
                return;
            }
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        out.machine().push(crate::agent_session::effects::Effect::Host(HostEvent::Room(ev)));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.room.request_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            self.outbox.push(crate::agent_session::effects::Effect::Host(HostEvent::Room(
                RoomIoEvent::DeleteRoomRequested(DeleteRoomRequested { room }),
            )));
        }
    }
}
