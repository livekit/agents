//! Participant/publication wait decisions over supplied room snapshots and events.

use lk_agents_abi::voice::rtc::{
    ConnectionState, ParticipantIdentity, ParticipantKind, RtcError, TrackSid,
};
use std::collections::BTreeMap;

#[derive(Clone)]
pub enum WaitFor {
    Participant {
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
    },
    Agent {
        name: Option<String>,
    },
    Attribute {
        identity: String,
        attribute: String,
        value: String,
    },
    Track {
        identity: Option<String>,
        kinds: Option<Vec<i32>>,
        include_local: bool,
        subscribed: bool,
    },
}

pub struct Participant {
    pub identity: ParticipantIdentity,
    pub kind: ParticipantKind,
    pub active: bool,
    pub attributes: BTreeMap<String, String>,
    pub tracks: Vec<Track>,
}

pub struct Track {
    pub sid: TrackSid,
    pub kind: i32,
    pub subscribed: bool,
}

pub struct Snapshot {
    pub connection: ConnectionState,
    pub local: Participant,
    pub remote: Vec<Participant>,
}

pub enum RoomWaitEvent {
    Active(Participant),
    Connected(Participant),
    Attributes(Participant),
    Disconnected(ParticipantIdentity),
    Published { owner: ParticipantIdentity, local: bool, track: Track },
    Subscribed { owner: ParticipantIdentity, track: Track },
    RoomDisconnected,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Ready {
    Participant(Option<ParticipantIdentity>),
    Agent(ParticipantIdentity),
    Attribute,
    Track(Option<ParticipantIdentity>, TrackSid),
}

pub struct RoomWait {
    request: WaitFor,
}

impl RoomWait {
    pub fn new(request: WaitFor) -> Self {
        Self { request }
    }

    /// Refuse observations whose participant handles do not exist yet in the SDK.
    pub fn check_connection(connection: ConnectionState) -> Result<(), RtcError> {
        if connection == ConnectionState::Disconnected {
            return Err(RtcError::NotConnected);
        }
        Ok(())
    }

    pub fn start(&self, snapshot: &Snapshot) -> Result<Option<Ready>, RtcError> {
        Self::check_connection(snapshot.connection)?;
        match &self.request {
            WaitFor::Participant { identity, kinds, include_local } => {
                let matches = |participant: &Participant| {
                    identity_matches(identity, &participant.identity)
                        && kinds
                            .as_ref()
                            .is_none_or(|kinds| kinds.contains(&participant.kind.wire()))
                };
                if *include_local && matches(&snapshot.local) {
                    return Ok(Some(Ready::Participant(None)));
                }
                Ok(snapshot
                    .remote
                    .iter()
                    .find(|participant| matches(participant) && participant.active)
                    .map(|participant| Ready::Participant(Some(participant.identity.clone()))))
            }
            WaitFor::Agent { name } => Ok(snapshot
                .remote
                .iter()
                .find(|participant| agent_matches(name, participant))
                .map(|participant| Ready::Agent(participant.identity.clone()))),
            WaitFor::Attribute { identity, attribute, value } => {
                let participant = snapshot
                    .remote
                    .iter()
                    .find(|participant| participant.identity.as_str() == identity)
                    .ok_or_else(|| {
                        RtcError::WaitEnded(format!(
                            "participant {} is not in the room",
                            repr(identity)
                        ))
                    })?;
                Ok((participant.attributes.get(attribute) == Some(value))
                    .then_some(Ready::Attribute))
            }
            WaitFor::Track { identity, kinds, include_local, subscribed } => {
                if *include_local
                    && identity_matches(identity, &snapshot.local.identity)
                    && let Some(track) =
                        snapshot.local.tracks.iter().find(|track| kind_matches(kinds, track))
                {
                    return Ok(Some(Ready::Track(None, track.sid.clone())));
                }
                Ok(snapshot
                    .remote
                    .iter()
                    .filter(|participant| identity_matches(identity, &participant.identity))
                    .find_map(|participant| {
                        participant
                            .tracks
                            .iter()
                            .find(|track| {
                                kind_matches(kinds, track) && (!subscribed || track.subscribed)
                            })
                            .map(|track| {
                                Ready::Track(Some(participant.identity.clone()), track.sid.clone())
                            })
                    }))
            }
        }
    }

    pub fn handle(&self, event: RoomWaitEvent) -> Option<Result<Ready, RtcError>> {
        let ready = match (&self.request, event) {
            (_, RoomWaitEvent::RoomDisconnected) => return Some(Err(self.ended())),
            (WaitFor::Participant { identity, kinds, .. }, RoomWaitEvent::Active(participant))
                if identity_matches(identity, &participant.identity)
                    && kinds
                        .as_ref()
                        .is_none_or(|kinds| kinds.contains(&participant.kind.wire())) =>
            {
                Some(Ready::Participant(Some(participant.identity)))
            }
            (
                WaitFor::Agent { name },
                RoomWaitEvent::Connected(participant) | RoomWaitEvent::Attributes(participant),
            ) if agent_matches(name, &participant) => Some(Ready::Agent(participant.identity)),
            (
                WaitFor::Attribute { identity, attribute, value },
                RoomWaitEvent::Attributes(participant),
            ) if participant.identity.as_str() == identity
                && participant.attributes.get(attribute) == Some(value) =>
            {
                Some(Ready::Attribute)
            }
            (
                WaitFor::Attribute { identity, attribute, .. },
                RoomWaitEvent::Disconnected(participant),
            ) if participant.as_str() == identity => {
                return Some(Err(RtcError::WaitEnded(format!(
                    "participant {} disconnected while waiting for {attribute}",
                    repr(identity)
                ))));
            }
            (
                WaitFor::Track { identity, kinds, include_local, subscribed },
                RoomWaitEvent::Published { owner, local, track },
            ) if identity_matches(identity, &owner)
                && ((!local && !subscribed) || (local && *include_local))
                && kind_matches(kinds, &track) =>
            {
                Some(Ready::Track((!local).then_some(owner), track.sid))
            }
            (
                WaitFor::Track { identity, kinds, subscribed: true, .. },
                RoomWaitEvent::Subscribed { owner, track },
            ) if identity_matches(identity, &owner)
                && kind_matches(kinds, &track)
                && track.subscribed =>
            {
                Some(Ready::Track(Some(owner), track.sid))
            }
            (
                WaitFor::Track { identity: Some(identity), .. },
                RoomWaitEvent::Disconnected(participant),
            ) if participant.as_str() == identity => {
                return Some(Err(RtcError::WaitEnded(format!(
                    "participant {} disconnected while waiting for track publication",
                    repr(identity)
                ))));
            }
            _ => None,
        };
        ready.map(Ok)
    }

    pub fn ended(&self) -> RtcError {
        let subject = match &self.request {
            WaitFor::Participant { .. } => "participant".to_owned(),
            WaitFor::Agent { .. } => "agent participant".to_owned(),
            WaitFor::Attribute { identity, attribute, .. } => {
                format!("{} {attribute}", repr(identity))
            }
            WaitFor::Track { .. } => "track publication".to_owned(),
        };
        RtcError::WaitEnded(format!("room disconnected while waiting for {subject}"))
    }
}

fn identity_matches(identity: &Option<String>, candidate: &ParticipantIdentity) -> bool {
    identity.as_deref().is_none_or(|identity| identity == candidate.as_str())
}
fn kind_matches(kinds: &Option<Vec<i32>>, track: &Track) -> bool {
    kinds.as_ref().is_none_or(|kinds| kinds.contains(&track.kind))
}
fn agent_matches(name: &Option<String>, participant: &Participant) -> bool {
    participant.kind == ParticipantKind::Agent
        && name.as_deref().is_none_or(|name| {
            participant.attributes.get(lk_agents_abi::ATTRIBUTE_AGENT_NAME).map(String::as_str)
                == Some(name)
        })
}

fn repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn participant(identity: &str, active: bool) -> Participant {
        Participant {
            identity: identity.into(),
            kind: ParticipantKind::Standard,
            active,
            attributes: BTreeMap::new(),
            tracks: Vec::new(),
        }
    }

    #[test]
    fn participant_wait_requires_activity_and_keeps_the_requested_identity() {
        let wait = RoomWait::new(WaitFor::Participant {
            identity: Some("alice".into()),
            kinds: None,
            include_local: false,
        });
        let snapshot = Snapshot {
            connection: ConnectionState::Connected,
            local: participant("agent", true),
            remote: vec![participant("alice", false)],
        };
        assert_eq!(wait.start(&snapshot).unwrap(), None);
        assert!(wait.handle(RoomWaitEvent::Connected(participant("alice", true))).is_none());
        assert!(wait.handle(RoomWaitEvent::Active(participant("bob", true))).is_none());
        assert_eq!(
            wait.handle(RoomWaitEvent::Active(participant("alice", true))).unwrap().unwrap(),
            Ready::Participant(Some("alice".into()))
        );
    }

    #[test]
    fn track_wait_ignores_other_participants_and_unsubscribed_remote_tracks() {
        let wait = RoomWait::new(WaitFor::Track {
            identity: Some("alice".into()),
            kinds: Some(vec![1]),
            include_local: true,
            subscribed: true,
        });
        let track = || Track { sid: "track".into(), kind: 1, subscribed: true };
        assert!(
            wait.handle(RoomWaitEvent::Published {
                owner: "alice".into(),
                local: false,
                track: track()
            })
            .is_none()
        );
        assert!(
            wait.handle(RoomWaitEvent::Subscribed { owner: "bob".into(), track: track() })
                .is_none()
        );
        assert!(
            wait.handle(RoomWaitEvent::Published {
                owner: "agent".into(),
                local: true,
                track: track()
            })
            .is_none()
        );
        assert_eq!(
            wait.handle(RoomWaitEvent::Subscribed { owner: "alice".into(), track: track() })
                .unwrap()
                .unwrap(),
            Ready::Track(Some("alice".into()), "track".into())
        );
    }
}
