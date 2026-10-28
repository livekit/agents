//! Supplied-clock peer observations drive the production synchronous room wait decisions.

use super::*;
use lk_agents_voice::room_io::wait::{
    Event, Participant, Ready, RoomWait, Snapshot, Track, WaitFor,
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

fn participant(value: &dyn RemoteParticipant, request: &WaitFor) -> Result<Participant> {
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
            .map(|value| participant(value.as_ref(), request))
            .collect::<Result<_>>()?,
    })
}

fn event(room: &dyn RtcRoom, value: RoomEvent, request: &WaitFor) -> Result<Option<Event>> {
    let track = |publication: &dyn TrackPublication| -> Result<Track> {
        Ok(Track {
            sid: publication.sid().clone(),
            kind: publication.kind(),
            subscribed: publication.is_subscribed()?,
        })
    };
    Ok(match value {
        RoomEvent::ParticipantActive { participant: value } => {
            Some(Event::Active(participant(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantConnected { participant: value } => {
            Some(Event::Connected(participant(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantAttributesChanged { participant: value } => {
            Some(Event::Attributes(participant(value.as_ref(), request)?))
        }
        RoomEvent::ParticipantDisconnected { participant, .. } => {
            Some(Event::Disconnected(participant.identity().clone()))
        }
        RoomEvent::TrackPublished { participant, publication } => Some(Event::Published {
            owner: participant.identity().clone(),
            local: false,
            track: track(publication.as_ref())?,
        }),
        RoomEvent::TrackSubscribed { participant, publication } => Some(Event::Subscribed {
            owner: participant.identity().clone(),
            track: track(publication.as_ref())?,
        }),
        RoomEvent::LocalTrackPublished { publication } => Some(Event::Published {
            owner: room.local().identity()?,
            local: true,
            track: track(publication.as_ref())?,
        }),
        RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected }
        | RoomEvent::Close { .. } => Some(Event::RoomDisconnected),
        _ => None,
    })
}

pub async fn perform(room: &dyn RtcRoom, request: WaitFor) -> Result<Ready> {
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
