//! Wait effects are SDK snapshots and event subscriptions; decisions live in the voice core.

use super::*;
use lk_agents_voice::room_io::machine::wait::{
    Participant, Ready, RoomWait, RoomWaitEvent, Snapshot, Track, WaitFor,
};
use std::collections::BTreeMap;
use std::sync::Arc;

fn tracks(publications: Vec<Arc<dyn TrackPublication>>) -> Result<Vec<Track>> {
    publications
        .into_iter()
        .map(|publication| {
            Ok(Track {
                sid: publication.sid().clone(),
                kind: publication.kind(),
                subscribed: publication.is_subscribed()?,
            })
        })
        .collect()
}

fn observe(value: &dyn RemoteParticipant, request: &WaitFor) -> Result<Participant> {
    Ok(Participant {
        identity: value.identity().clone(),
        kind: value.kind(),
        active: if matches!(request, WaitFor::Participant { .. }) { value.active()? } else { true },
        attributes: if matches!(request, WaitFor::Agent { .. } | WaitFor::Attribute { .. }) {
            value.attributes()?
        } else {
            BTreeMap::new()
        },
        tracks: if matches!(request, WaitFor::Track { .. }) {
            tracks(value.track_publications()?)?
        } else {
            Vec::new()
        },
    })
}

fn snapshot(room: &dyn RtcRoom, request: &WaitFor) -> Result<Snapshot> {
    let connection = room.connection_state()?;
    RoomWait::check_connection(connection)?;
    let local = room.local();
    Ok(Snapshot {
        connection,
        local: Participant {
            identity: local.identity()?,
            kind: local.kind()?,
            active: true,
            attributes: BTreeMap::new(),
            tracks: if matches!(request, WaitFor::Track { include_local: true, .. }) {
                tracks(local.track_publications()?)?
            } else {
                Vec::new()
            },
        },
        remote: room
            .remote_participants()?
            .iter()
            .map(|value| observe(value.as_ref(), request))
            .collect::<Result<_>>()?,
    })
}

fn event(room: &dyn RtcRoom, value: RoomEvent, request: &WaitFor) -> Result<Option<RoomWaitEvent>> {
    let track = |publication: &dyn TrackPublication| -> Result<Track> {
        Ok(Track {
            sid: publication.sid().clone(),
            kind: publication.kind(),
            subscribed: publication.is_subscribed()?,
        })
    };
    Ok(match value {
        RoomEvent::ParticipantActive { participant: value } => {
            Some(RoomWaitEvent::Active(observe(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantConnected { participant: value } => {
            Some(RoomWaitEvent::Connected(observe(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantAttributesChanged { participant: value } => {
            Some(RoomWaitEvent::Attributes(observe(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantDisconnected { participant, .. } => {
            Some(RoomWaitEvent::Disconnected(participant.identity().clone()))
        }
        RoomEvent::TrackPublished { participant, publication } => Some(RoomWaitEvent::Published {
            owner: participant.identity().clone(),
            local: false,
            track: track(publication.as_ref())?,
        }),
        RoomEvent::TrackSubscribed { participant, publication } => {
            Some(RoomWaitEvent::Subscribed {
                owner: participant.identity().clone(),
                track: track(publication.as_ref())?,
            })
        }
        RoomEvent::LocalTrackPublished { publication } => Some(RoomWaitEvent::Published {
            owner: room.local().identity()?,
            local: true,
            track: track(publication.as_ref())?,
        }),
        RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected }
        | RoomEvent::Close { .. } => Some(RoomWaitEvent::RoomDisconnected),
        _ => None,
    })
}

async fn perform(room: &dyn RtcRoom, request: WaitFor) -> Result<Ready> {
    // Subscribe before reading SDK views so a transition during the snapshot remains observable.
    let mut events = room.events();
    let wait = RoomWait::new(request.clone());
    if let Some(ready) = wait.start(&snapshot(room, &request)?)? {
        return Ok(ready);
    }
    while let Some(value) = events.next().await? {
        if let Some(event) = event(room, value, &request)?
            && let Some(outcome) = wait.handle(event)
        {
            return outcome;
        }
    }
    Err(wait.ended())
}

pub async fn participant(
    room: &dyn RtcRoom,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
) -> Result<Option<ParticipantIdentity>> {
    match perform(room, WaitFor::Participant { identity, kinds, include_local }).await? {
        Ready::Participant(identity) => Ok(identity),
        _ => unreachable!("a participant wait resolves with a participant"),
    }
}

pub async fn agent(room: &dyn RtcRoom, name: Option<String>) -> Result<ParticipantIdentity> {
    match perform(room, WaitFor::Agent { name }).await? {
        Ready::Agent(identity) => Ok(identity),
        _ => unreachable!("an agent wait resolves with an agent"),
    }
}

pub async fn participant_attribute(
    room: &dyn RtcRoom,
    identity: String,
    attribute: String,
    value: String,
) -> Result<()> {
    match perform(room, WaitFor::Attribute { identity, attribute, value }).await? {
        Ready::Attribute => Ok(()),
        _ => unreachable!("an attribute wait resolves with the attribute"),
    }
}

pub async fn track_publication(
    room: &dyn RtcRoom,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
    subscribed: bool,
) -> Result<(Option<ParticipantIdentity>, TrackSid)> {
    match perform(room, WaitFor::Track { identity, kinds, include_local, subscribed }).await? {
        Ready::Track(identity, sid) => Ok((identity, sid)),
        _ => unreachable!("a track wait resolves with a track"),
    }
}
