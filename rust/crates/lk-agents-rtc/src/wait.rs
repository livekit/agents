//! Waiting on a room for someone or something to show up (`utils/participant.py`): each wait
//! subscribes to the room's events before it looks at the room, so nothing that happens between
//! the two is missed, and ends with the room's disconnection or the awaited participant's.

use std::sync::Arc;

use crate::traits::{RemoteParticipant, RtcRoom, TrackPublication};
use crate::{ConnectionState, ParticipantIdentity, ParticipantKind, Result, RoomEvent, RtcError};

use lk_agents_abi::ATTRIBUTE_AGENT_NAME;

/// Python's `repr()` of a string, as the reference's messages quote identities.
fn repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn ended(msg: String) -> RtcError {
    RtcError::WaitEnded(msg)
}

/// `rtc.Room.isconnected()`: anything but disconnected.
fn connected(room: &dyn RtcRoom) -> Result<()> {
    match room.connection_state() {
        ConnectionState::Disconnected => Err(RtcError::NotConnected),
        _ => Ok(()),
    }
}

fn is_disconnect(event: &RoomEvent) -> bool {
    matches!(event, RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected })
}

/// Kinds as the SDK's ints; one the core does not know matches only itself, never a known one.
fn kind_in(kinds: &Option<Vec<i32>>, kind: ParticipantKind) -> bool {
    kinds.as_ref().is_none_or(|kinds| kinds.contains(&kind.wire()))
}

/// `wait_for_participant`: the first participant of `identity` (any when `None`) and of one of
/// `kinds` (any when `None`) to be active, or the agent itself when `include_local` and it
/// matches. Returns the participant's identity, `None` for the agent itself.
#[lk_agents_abi::host(export)]
pub async fn wait_for_participant(
    room: Arc<dyn RtcRoom>,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
) -> Result<Option<ParticipantIdentity>> {
    connected(&*room)?;
    let mut events = room.events();
    let matches = |p: &dyn RemoteParticipant| {
        identity.as_deref().is_none_or(|id| p.identity().as_str() == id) && kind_in(&kinds, p.kind())
    };
    if include_local {
        let local = room.local();
        if identity.as_deref().is_none_or(|id| local.identity().as_str() == id) && kind_in(&kinds, local.kind()) {
            return Ok(None);
        }
    }
    if let Some(p) = room.remote_participants().into_iter().find(|p| p.active() && matches(&**p)) {
        return Ok(Some(p.identity().clone()));
    }
    while let Some(event) = events.recv().await {
        match event {
            RoomEvent::ParticipantActive { participant } if matches(&*participant) => {
                return Ok(Some(participant.identity().clone()));
            }
            e if is_disconnect(&e) => {
                return Err(ended("room disconnected while waiting for participant".into()));
            }
            _ => {}
        }
    }
    Err(ended("room disconnected while waiting for participant".into()))
}

/// `wait_for_agent`: the first agent participant, named `agent_name` (its `lk.agent.name`) when
/// given, in the room or joining it.
#[lk_agents_abi::host(export)]
pub async fn wait_for_agent(room: Arc<dyn RtcRoom>, agent_name: Option<String>) -> Result<ParticipantIdentity> {
    connected(&*room)?;
    let mut events = room.events();
    let matches = |p: &dyn RemoteParticipant| {
        p.kind() == ParticipantKind::Agent
            && agent_name.as_deref().is_none_or(|name| p.attributes().get(ATTRIBUTE_AGENT_NAME).map(String::as_str) == Some(name))
    };
    if let Some(p) = room.remote_participants().into_iter().find(|p| matches(&**p)) {
        return Ok(p.identity().clone());
    }
    let gone = || ended("room disconnected while waiting for agent participant".into());
    while let Some(event) = events.recv().await {
        match event {
            RoomEvent::ParticipantConnected { participant } | RoomEvent::ParticipantAttributesChanged { participant }
                if matches(&*participant) =>
            {
                return Ok(participant.identity().clone());
            }
            e if is_disconnect(&e) => return Err(gone()),
            _ => {}
        }
    }
    Err(gone())
}

/// `wait_for_participant_attribute`: until the participant's `attribute` is `value`; fails when
/// they are not in the room or leave it, or the room disconnects, first.
#[lk_agents_abi::host(export)]
pub async fn wait_for_participant_attribute(
    room: Arc<dyn RtcRoom>,
    identity: String,
    attribute: String,
    value: String,
) -> Result<()> {
    connected(&*room)?;
    let id = ParticipantIdentity(identity.clone());
    let absent = || ended(format!("participant {} is not in the room", repr(&identity)));
    let mut events = room.events();
    let participant = room.participant(&id).ok_or_else(absent)?;
    let set = |p: &dyn RemoteParticipant| p.attributes().get(&attribute) == Some(&value);
    if set(&*participant) {
        return Ok(());
    }
    let gone = || ended(format!("room disconnected while waiting for {} {attribute}", repr(&identity)));
    while let Some(event) = events.recv().await {
        match event {
            RoomEvent::ParticipantAttributesChanged { participant } if participant.identity() == &id && set(&*participant) => {
                return Ok(());
            }
            RoomEvent::ParticipantDisconnected { participant, .. } if participant.identity() == &id => {
                return Err(ended(format!("participant {} disconnected while waiting for {attribute}", repr(&identity))));
            }
            e if is_disconnect(&e) => return Err(gone()),
            _ => {}
        }
    }
    Err(gone())
}

/// `wait_for_track_publication`: the first publication of `identity`'s (anyone's when `None`) of
/// one of `kinds` (any when `None`), subscribed when `wait_for_subscription`; the agent's own
/// too when `include_local` (on publish, subscription aside). Returns the publisher's identity
/// (`None` for the agent itself) and the publication's sid; fails when `identity` leaves first.
#[lk_agents_abi::host(export)]
pub async fn wait_for_track_publication(
    room: Arc<dyn RtcRoom>,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
    wait_for_subscription: bool,
) -> Result<(Option<ParticipantIdentity>, crate::TrackSid)> {
    connected(&*room)?;
    let mut events = room.events();
    let kind_ok = |publication: &dyn TrackPublication| kinds.as_ref().is_none_or(|k| k.contains(&publication.kind()));
    let ours = |id: &ParticipantIdentity| identity.as_deref().is_none_or(|want| id.as_str() == want);
    let matches = |publication: &dyn TrackPublication, participant: &dyn RemoteParticipant| {
        ours(participant.identity())
            && kind_ok(publication)
            && (!wait_for_subscription || publication.is_subscribed())
    };
    if include_local {
        let local = room.local();
        if ours(&local.identity()) {
            if let Some(p) = local.track_publications().into_iter().find(|p| kind_ok(&**p)) {
                return Ok((None, p.sid().clone()));
            }
        }
    }
    for participant in room.remote_participants() {
        if let Some(p) = participant.track_publications().into_iter().find(|p| matches(&**p, &*participant)) {
            return Ok((Some(participant.identity().clone()), p.sid().clone()));
        }
    }
    let gone = || ended("room disconnected while waiting for track publication".into());
    while let Some(event) = events.recv().await {
        match event {
            RoomEvent::TrackSubscribed { participant, publication } if wait_for_subscription && matches(&*publication, &*participant) => {
                return Ok((Some(participant.identity().clone()), publication.sid().clone()));
            }
            RoomEvent::TrackPublished { participant, publication } if !wait_for_subscription && matches(&*publication, &*participant) => {
                return Ok((Some(participant.identity().clone()), publication.sid().clone()));
            }
            RoomEvent::LocalTrackPublished { publication } if include_local && ours(&room.local().identity()) && kind_ok(&*publication) => {
                return Ok((None, publication.sid().clone()));
            }
            RoomEvent::ParticipantDisconnected { participant, .. }
                if identity.as_deref() == Some(participant.identity().as_str()) =>
            {
                let who = repr(participant.identity().as_str());
                return Err(ended(format!("participant {who} disconnected while waiting for track publication")));
            }
            e if is_disconnect(&e) => return Err(gone()),
            _ => {}
        }
    }
    Err(gone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DisconnectReason, FakeRtc, TrackSource};
    use std::time::Duration;

    fn room() -> Arc<crate::FakeRoom> {
        let (rtc, _clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("r");
        room.set_connection_state(ConnectionState::Connected);
        room
    }

    async fn soon<T>(f: impl std::future::Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), f).await.expect("the wait never ended")
    }

    fn pending<F: std::future::Future + Unpin>(f: &mut F) -> bool {
        let waker = std::task::Waker::noop();
        std::pin::Pin::new(f).poll(&mut std::task::Context::from_waker(waker)).is_pending()
    }

    // Catches: a wait that misses who is already in the room, or someone of another kind taken
    // for the awaited one; a refused wait on a room that is not connected.
    #[tokio::test]
    async fn a_participant_already_in_the_room_or_joining_is_found() {
        let room = room();
        room.add_participant("bob", ParticipantKind::Sip);
        let dyn_room: Arc<dyn RtcRoom> = room.clone();
        let sip = Some(vec![ParticipantKind::Sip.wire()]);
        let got = soon(wait_for_participant(dyn_room.clone(), None, sip.clone(), false)).await.unwrap();
        assert_eq!(got.as_ref().map(|p| p.as_str()), Some("bob"));
        assert_eq!(soon(wait_for_participant(dyn_room.clone(), None, None, true)).await.unwrap(), None);

        let agents = Some(vec![ParticipantKind::Agent.wire()]);
        let mut wait = Box::pin(wait_for_participant(dyn_room.clone(), None, agents, false));
        assert!(pending(&mut wait));
        room.add_participant("helper", ParticipantKind::Agent);
        assert_eq!(soon(wait).await.unwrap().map(|p| p.0), Some("helper".to_owned()));

        room.close(DisconnectReason::ClientInitiated);
        assert!(matches!(soon(wait_for_participant(dyn_room, None, None, false)).await, Err(RtcError::NotConnected)));
    }

    // Catches: an attribute change missed, or a wait that outlives the participant it waits on.
    #[tokio::test]
    async fn an_attribute_wait_ends_with_the_value_or_the_participant() {
        let room = room();
        let bob = room.add_participant("bob", ParticipantKind::Sip);
        let dyn_room: Arc<dyn RtcRoom> = room.clone();
        let mut wait = Box::pin(wait_for_participant_attribute(dyn_room.clone(), "bob".into(), "status".into(), "active".into()));
        assert!(pending(&mut wait));
        bob.set_attribute("status", "dialing");
        assert!(pending(&mut wait));
        bob.set_attribute("status", "active");
        soon(wait).await.unwrap();

        let wait = wait_for_participant_attribute(dyn_room.clone(), "bob".into(), "x".into(), "y".into());
        let mut wait = Box::pin(wait);
        assert!(pending(&mut wait));
        room.remove_participant(&ParticipantIdentity("bob".into()), DisconnectReason::ClientInitiated);
        let err = soon(wait).await.unwrap_err();
        assert_eq!(err.to_string(), "participant 'bob' disconnected while waiting for x");
        let err = soon(wait_for_participant_attribute(dyn_room, "bob".into(), "x".into(), "y".into())).await.unwrap_err();
        assert_eq!(err.to_string(), "participant 'bob' is not in the room");
    }

    // Catches: a publication of another kind, or an unsubscribed one when subscription is asked.
    #[tokio::test]
    async fn a_publication_wait_matches_kind_and_subscription() {
        let room = room();
        room.set_auto_subscribe(false);
        let bob = room.add_participant("bob", ParticipantKind::Standard);
        let dyn_room: Arc<dyn RtcRoom> = room.clone();
        let audio = Some(vec![crate::TRACK_KIND_AUDIO]);
        let mut wait = Box::pin(wait_for_track_publication(dyn_room.clone(), Some("bob".into()), audio.clone(), false, true));
        assert!(pending(&mut wait));
        let mic = bob.publish_track("TR_mic", TrackSource::Microphone);
        assert!(pending(&mut wait), "published, not subscribed");
        mic.set_subscribed(true).await.unwrap();
        let (who, sid) = soon(wait).await.unwrap();
        assert_eq!((who.map(|p| p.0), sid.0), (Some("bob".to_owned()), "TR_mic".to_owned()));

        let got = soon(wait_for_track_publication(dyn_room, None, audio, false, false)).await.unwrap();
        assert_eq!(got.1 .0, "TR_mic");
    }

    #[test]
    fn identities_are_quoted_as_python_does() {
        assert_eq!(repr("bob"), "'bob'");
        assert_eq!(repr("o'neil"), "\"o'neil\"");
        assert_eq!(repr("a'b\"c"), "'a\\'b\"c'");
    }
}
