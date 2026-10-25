//! The session's link to its room (`RoomOptions`): whether the linked participant leaving closes
//! the session, and the room's deletion when it closes.

use crate::agent_session::effects::Out;
use crate::agent_session::{Close, Driver, HostEvent};
use crate::room_io::{DeleteRoomRequested, RoomIoEvent};
use crate::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;
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
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.room.disconnect_closes(reason);
                out.trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    let reason = CloseReason::ParticipantDisconnected;
                    self.begin_closing(now, Close { reason, drain: false, error: None }, out);
                }
            }
            // Journalled so a re-link is a step the rules can read; no driver state changes.
            RoomIoEvent::ParticipantLinked(_) => out.trace(RoomStep::Link(RoomLinkStep::Linked)),
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        let _ = self.events.send(HostEvent::Room(ev));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.room.request_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            let _ = self.events.send(HostEvent::Room(RoomIoEvent::DeleteRoomRequested(
                DeleteRoomRequested { room },
            )));
        }
    }
}
