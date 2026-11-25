//! AMD's callee leg (`AMD._setup_listening` and `_on_disconnected`): the participant's audio
//! track within [`TRACK_WAIT`], then, for a SIP participant with `wait_until_answered`, the
//! call answered, with no timeout. Then it watches for the participant to leave.
//!
//! Without a room the run listens at once, as Python does without room I/O.

use crate::rtc::wait;
use crate::voice::session::Timebase;
use lk_agents_abi::voice::rtc::{
    ConnectionState, ParticipantIdentity, ParticipantKind, RoomEvent, RtcRoom, TRACK_KIND_AUDIO,
};
use lk_agents_voice::amd::machine::Callee;
use std::sync::Arc;
use std::time::Duration;

/// The setup's wait for the participant's audio track (`_TRACK_PUBLICATION_TIMEOUT`).
pub const TRACK_WAIT: Duration = Duration::from_secs(5);
/// The participant attribute a SIP call reports its status in.
pub const SIP_CALL_STATUS: &str = "sip.callStatus";
/// The status of an answered call.
pub const SIP_CALL_ACTIVE: &str = "active";

/// Where the callee leg's progress goes. A `false` return means nobody listens any more.
pub type Report = Box<dyn Fn(Callee) -> bool + Send + Sync>;

/// Whom the leg waits for.
#[derive(Clone)]
pub struct CalleeSetup {
    pub room: Option<Arc<dyn RtcRoom>>,
    /// The participant to wait for; any audio publisher when `None`.
    pub participant: Option<String>,
    pub wait_until_answered: bool,
}

impl std::fmt::Debug for CalleeSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalleeSetup")
            .field("room", &self.room.is_some())
            .field("participant", &self.participant)
            .field("wait_until_answered", &self.wait_until_answered)
            .finish()
    }
}

/// Reports `Answered`, `Missing` or `Disconnected` once for the setup, and `Disconnected` once
/// more if the participant leaves after it.
pub async fn watch(setup: CalleeSetup, time: Timebase, report: Report) {
    let Some(room) = setup.room.clone() else {
        report(Callee::Answered);
        return;
    };
    // Subscribed before any read, so a departure during setup is seen.
    let mut events = room.events();
    let mut identity = setup.participant.clone();
    let outcome = {
        let listen = listen(room.clone(), setup.clone(), time.clone());
        tokio::pin!(listen);
        loop {
            tokio::select! {
                biased;
                event = events.next() => match event {
                    Ok(Some(event)) => {
                        if departs(&event, identity.as_deref()) {
                            break Err(Callee::Disconnected);
                        }
                    }
                    // The room's events ended: it is gone.
                    Ok(None) | Err(_) => break Err(Callee::Disconnected),
                },
                outcome = &mut listen => break outcome,
            }
        }
    };
    let answered = match outcome {
        Ok(publisher) => {
            identity = Some(publisher);
            report(Callee::Answered)
        }
        Err(callee) => {
            report(callee);
            false
        }
    };
    if answered {
        loop {
            match events.next().await {
                Ok(Some(event)) if !departs(&event, identity.as_deref()) => {}
                Ok(Some(_)) | Ok(None) | Err(_) => {
                    report(Callee::Disconnected);
                    break;
                }
            }
        }
    }
    let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    let closed = events.close(deadline).await;
    if closed != lk_agents_abi::Closed::Clean {
        log::debug!(target: lk_agents_abi::LOG_TARGET, outcome:? = closed; "AMD room events cleanup incomplete");
    }
}

/// Whether `event` ends the call for `identity`: its departure, or the room's.
fn departs(event: &RoomEvent, identity: Option<&str>) -> bool {
    match event {
        RoomEvent::ParticipantDisconnected { participant, .. } => {
            identity.is_some_and(|identity| participant.identity().as_str() == identity)
        }
        RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected }
        | RoomEvent::Close { .. } => true,
        _ => false,
    }
}

/// The publisher's identity once the run may listen.
async fn listen(
    room: Arc<dyn RtcRoom>,
    setup: CalleeSetup,
    time: Timebase,
) -> Result<String, Callee> {
    let missing = |room: &dyn RtcRoom| match room.connection_state() {
        Ok(ConnectionState::Connected | ConnectionState::Reconnecting) => Callee::Missing,
        Ok(ConnectionState::Disconnected) | Err(_) => Callee::Disconnected,
    };
    let track = wait::track_publication(
        room.as_ref(),
        setup.participant.clone(),
        Some(vec![TRACK_KIND_AUDIO]),
        wait::TrackOptions { subscribed: true, ..Default::default() },
    );
    let publisher = match time.timeout(TRACK_WAIT, track).await {
        Ok(Ok((Some(publisher), _))) => publisher,
        Ok(Ok((None, _))) | Ok(Err(_)) | Err(_) => return Err(missing(room.as_ref())),
    };
    let participant = match room.participant(&publisher) {
        Ok(Some(participant)) => participant,
        Ok(None) | Err(_) => return Err(missing(room.as_ref())),
    };
    if setup.wait_until_answered && participant.kind() == ParticipantKind::Sip {
        let ParticipantIdentity(identity) = publisher.clone();
        let answered = wait::participant_attribute(
            room.as_ref(),
            identity,
            SIP_CALL_STATUS.to_owned(),
            SIP_CALL_ACTIVE.to_owned(),
        )
        .await;
        if answered.is_err() {
            // The attribute wait ends when the participant or the room leaves.
            return Err(Callee::Disconnected);
        }
    }
    Ok(publisher.0)
}

#[cfg(test)]
#[path = "callee_tests.rs"]
mod tests;
