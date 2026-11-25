//! AMD's callee leg (`AMD._setup_listening` and `_on_disconnected`): the participant's audio
//! track within [`TRACK_WAIT`], then, for a SIP participant with `wait_until_answered`, the
//! call answered, with no timeout. Then it watches for the participant to leave.
//!
//! The participant is the one given, else the one the session's room I/O links, waited for
//! within the same [`TRACK_WAIT`] (`_wait_for_linked_participant`): never another audio
//! publisher. Without a room the run listens at once, as Python does without room I/O.
//!
//! The leg stops when its `stop` token is cancelled, and always closes its room events within
//! the teardown bound before it ends.

use crate::rtc::wait;
use crate::voice::session::Timebase;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::voice::rtc::{
    ConnectionState, ParticipantIdentity, ParticipantKind, RoomEvent, RtcRoom, TRACK_KIND_AUDIO,
};
use lk_agents_voice::amd::machine::Callee;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// The setup's wait for the participant's audio track (`_TRACK_PUBLICATION_TIMEOUT`).
pub const TRACK_WAIT: Duration = Duration::from_secs(5);
/// The participant attribute a SIP call reports its status in.
pub const SIP_CALL_STATUS: &str = "sip.callStatus";
/// The status of an answered call.
pub const SIP_CALL_ACTIVE: &str = "active";

/// Where the callee leg's progress goes. A `false` return means nobody listens any more.
pub type Report = Box<dyn Fn(Callee) -> bool + Send + Sync>;

/// The participant the session's room I/O links now, or the next one it links; `None` once no
/// link can come.
pub type LinkedWait = Arc<dyn Fn() -> BoxFuture<'static, Option<String>> + Send + Sync>;

/// Whom the leg waits for.
#[derive(Clone)]
pub struct CalleeSetup {
    pub room: Option<Arc<dyn RtcRoom>>,
    /// The participant to wait for; the linked one when `None`.
    pub participant: Option<String>,
    /// The room I/O's link, waited for when no participant is given. `None` (no room I/O)
    /// waits for any audio publisher.
    pub linked: Option<LinkedWait>,
    pub wait_until_answered: bool,
}

impl std::fmt::Debug for CalleeSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalleeSetup")
            .field("room", &self.room.is_some())
            .field("participant", &self.participant)
            .field("linked", &self.linked.is_some())
            .field("wait_until_answered", &self.wait_until_answered)
            .finish()
    }
}

/// Reports `Answered`, `Missing` or `Disconnected` once for the setup, and `Disconnected` once
/// more if the participant leaves after it, until `stop` is cancelled. Either way it closes the
/// room events it subscribed to before it returns.
pub async fn watch(setup: CalleeSetup, time: Timebase, stop: CancellationToken, report: Report) {
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
                () = stop.cancelled() => break None,
                event = events.next() => match event {
                    Ok(Some(event)) => {
                        if departs(&event, identity.as_deref()) {
                            break Some(Err(Callee::Disconnected));
                        }
                    }
                    // The room's events ended: it is gone.
                    Ok(None) | Err(_) => break Some(Err(Callee::Disconnected)),
                },
                outcome = &mut listen => break Some(outcome),
            }
        }
    };
    let answered = match outcome {
        None => false,
        Some(Ok(publisher)) => {
            identity = Some(publisher);
            report(Callee::Answered)
        }
        Some(Err(callee)) => {
            report(callee);
            false
        }
    };
    if answered {
        loop {
            let event = tokio::select! {
                biased;
                () = stop.cancelled() => break,
                event = events.next() => event,
            };
            match event {
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
    // The participant AgentSession hears, not any other audio publisher; the link and the track
    // share one limit, as `asyncio.wait_for(_wait_for_audio(), ...)` does.
    let audio = async {
        let identity = match (setup.participant.clone(), &setup.linked) {
            (Some(identity), _) => Some(identity),
            (None, Some(linked)) => Some(linked().await.ok_or(())?),
            (None, None) => None,
        };
        let track = wait::track_publication(
            room.as_ref(),
            identity,
            Some(vec![TRACK_KIND_AUDIO]),
            wait::TrackOptions { subscribed: true, ..Default::default() },
        );
        track.await.map_err(drop)
    };
    let publisher = match time.timeout(TRACK_WAIT, audio).await {
        Ok(Ok((Some(publisher), _))) => publisher,
        Ok(Ok((None, _))) | Ok(Err(())) | Err(_) => return Err(missing(room.as_ref())),
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
