//! Room attachment and startup decisions over supplied participant records.

use lk_agents_abi::Closed;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{
    ParticipantIdentity, ParticipantKind, RemoteParticipant, RoomEvent,
};
use std::sync::Arc;

/// The remote participant a room event is about, whose lifetime
/// [`RoomCoordinator::current_event`] checks.
pub fn event_participant(event: &RoomEvent) -> Option<&Arc<dyn RemoteParticipant>> {
    match event {
        RoomEvent::TrackSubscribed { participant, .. }
        | RoomEvent::TrackUnsubscribed { participant, .. }
        | RoomEvent::TrackPublished { participant, .. }
        | RoomEvent::TrackUnpublished { participant, .. }
        | RoomEvent::ParticipantConnected { participant }
        | RoomEvent::ParticipantActive { participant }
        | RoomEvent::ParticipantAttributesChanged { participant }
        | RoomEvent::ParticipantDisconnected { participant, .. } => Some(participant),
        _ => None,
    }
}

use super::decide_link;

#[derive(Clone)]
pub struct RoomState {
    pub linked: Option<ParticipantIdentity>,
    pub ready: Option<Result<()>>,
    pub closed: Option<Closed>,
}

enum Phase {
    New,
    Started,
    Closing,
    Closed(Closed),
}

/// What `set_participant` does after the selection.
#[derive(Debug, PartialEq, Eq)]
pub struct Relink {
    /// Emit `ParticipantLinked`: a present participant other than the one linked.
    pub announce: bool,
    /// Detach the media of the previous attachment first, even for the same participant.
    pub detach: bool,
}

pub struct RoomCoordinator {
    target: Option<ParticipantIdentity>,
    linked: Option<ParticipantIdentity>,
    linked_sid: Option<String>,
    attachment: u64,
    phase: Phase,
    ready: Option<Result<()>>,
}

impl RoomCoordinator {
    pub fn new(target: Option<ParticipantIdentity>) -> Self {
        Self {
            target,
            linked: None,
            linked_sid: None,
            attachment: 0,
            phase: Phase::New,
            ready: None,
        }
    }

    pub fn state(&self) -> RoomState {
        RoomState {
            linked: self.linked.clone(),
            ready: self.ready.clone(),
            closed: match self.phase {
                Phase::Closed(outcome) => Some(outcome),
                _ => None,
            },
        }
    }

    pub fn is_open(&self) -> bool {
        !matches!(self.phase, Phase::Closing | Phase::Closed(_))
    }

    pub fn start(&mut self) -> bool {
        if !matches!(self.phase, Phase::New) {
            return false;
        }
        self.phase = Phase::Started;
        true
    }

    pub fn consider(
        &mut self,
        candidate: &ParticipantIdentity,
        sid: &str,
        kind: ParticipantKind,
        on_behalf_of: Option<&str>,
        agent: &ParticipantIdentity,
        accepted_kinds: &[ParticipantKind],
    ) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_))
            || !decide_link(
                self.linked.is_some()
                    && !(self.linked.as_ref() == Some(candidate)
                        && self.linked_sid.as_deref() != Some(sid)),
                self.target.as_ref(),
                candidate,
                kind,
                on_behalf_of,
                agent,
                accepted_kinds,
            )
        {
            return false;
        }
        self.attachment += 1;
        self.linked = Some(candidate.clone());
        self.linked_sid = Some(sid.to_owned());
        true
    }

    /// A selection is a new attachment even if the same participant is selected again.
    pub fn select(&mut self, target: Option<ParticipantIdentity>, sid: Option<&str>) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_)) {
            return false;
        }
        self.attachment += 1;
        self.target = target.clone();
        self.linked = target.filter(|_| sid.is_some());
        self.linked_sid = sid.map(str::to_owned);
        true
    }

    /// `set_participant`: `listed_sid` is the participant's SDK sid when it is in the room.
    /// `None` once closing.
    pub fn set_participant(
        &mut self,
        identity: Option<ParticipantIdentity>,
        listed_sid: Option<&str>,
    ) -> Option<Relink> {
        if !self.is_open() {
            return None;
        }
        let relink = Relink {
            announce: listed_sid.is_some() && identity.is_some() && self.linked != identity,
            detach: identity.is_none() || self.linked == identity,
        };
        self.select(identity, listed_sid);
        Some(relink)
    }

    /// Whether a room event about this participant lifetime is current: the SDK still lists that
    /// lifetime, or it is the linked one departing. A delayed event of an earlier lifetime of
    /// the same identity is dropped.
    pub fn current_event(
        &self,
        identity: &ParticipantIdentity,
        sid: &str,
        listed_sid: Option<&str>,
        departure: bool,
    ) -> bool {
        match listed_sid {
            Some(listed) => listed == sid,
            None => departure && self.accepts_participant(identity, sid),
        }
    }

    pub fn disconnect(&mut self, identity: &ParticipantIdentity, sid: &str) -> bool {
        if !self.accepts_participant(identity, sid)
            || matches!(self.phase, Phase::Closing | Phase::Closed(_))
        {
            return false;
        }
        self.attachment += 1;
        self.linked = None;
        self.linked_sid = None;
        true
    }

    pub fn accepts_participant(&self, participant: &ParticipantIdentity, sid: &str) -> bool {
        self.is_open()
            && self.linked.as_ref() == Some(participant)
            && self.linked_sid.as_deref() == Some(sid)
    }

    pub fn attachment(&self, participant: &ParticipantIdentity) -> Option<u64> {
        (!matches!(self.phase, Phase::Closing | Phase::Closed(_))
            && self.linked.as_ref() == Some(participant))
        .then_some(self.attachment)
    }

    /// `_on_sip_dtmf_received`: a key the linked participant pressed is activity.
    pub fn accepts_key_press(&self, participant: &ParticipantIdentity) -> bool {
        self.linked.as_ref() == Some(participant)
    }

    /// A chat stream is read only from the attached participant while the room still lists
    /// it (`listed` is asked only for that participant); the returned attachment is rechecked
    /// with [`Self::accepts`] once the text is read.
    pub fn chat_attachment<E>(
        &self,
        sender: &ParticipantIdentity,
        listed: impl FnOnce() -> std::result::Result<bool, E>,
    ) -> std::result::Result<Option<u64>, E> {
        match self.attachment(sender) {
            Some(attachment) if listed()? => Ok(Some(attachment)),
            _ => Ok(None),
        }
    }

    pub fn accepts(&self, attachment: u64, participant: &ParticipantIdentity) -> bool {
        self.attachment(participant) == Some(attachment)
    }

    pub fn finish_start(&mut self, outcome: Result<()>) {
        if !matches!(self.phase, Phase::Closing | Phase::Closed(_)) && self.ready.is_none() {
            self.ready = Some(outcome);
        }
    }

    pub fn begin_close(&mut self) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_)) {
            return false;
        }
        self.phase = Phase::Closing;
        self.attachment += 1;
        self.linked = None;
        self.linked_sid = None;
        if self.ready.is_none() {
            self.ready = Some(Err(IoError::Rtc("room I/O is closed".into())));
        }
        true
    }

    pub fn finish_close(&mut self, outcome: Closed) {
        self.phase = Phase::Closed(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reattachment_rejects_old_work_even_for_the_same_identity() {
        let person = ParticipantIdentity::from("person");
        let mut room = RoomCoordinator::new(None);
        room.start();
        room.select(Some(person.clone()), Some("PA_person"));
        let first = room.attachment(&person).unwrap();
        room.select(Some(person.clone()), Some("PA_person"));
        assert!(!room.accepts(first, &person));
        let second = room.attachment(&person).unwrap();
        assert!(room.accepts(second, &person));
        room.begin_close();
        assert!(!room.accepts(second, &person));
    }

    #[test]
    fn rejoining_identity_retires_the_previous_participant_lifetime() {
        let identity = ParticipantIdentity::from("person");
        let mut room = RoomCoordinator::new(None);
        room.start();
        assert!(room.consider(
            &identity,
            "PA_old",
            ParticipantKind::Standard,
            None,
            &"agent".into(),
            &[ParticipantKind::Standard]
        ));
        let old_attachment = room.attachment(&identity).unwrap();
        assert!(room.consider(
            &identity,
            "PA_new",
            ParticipantKind::Standard,
            None,
            &"agent".into(),
            &[ParticipantKind::Standard]
        ));
        assert!(!room.accepts(old_attachment, &identity));
        assert!(!room.disconnect(&identity, "PA_old"));
        assert!(room.accepts_participant(&identity, "PA_new"));
    }

    #[test]
    fn delayed_departure_cannot_unlink_a_rejoined_identity() {
        let identity = ParticipantIdentity::from("person");
        let mut room = RoomCoordinator::new(None);
        room.start();
        room.select(Some(identity.clone()), Some("PA_old"));
        room.select(Some(identity.clone()), Some("PA_new"));
        assert!(!room.disconnect(&identity, "PA_old"));
        assert!(room.accepts_participant(&identity, "PA_new"));
        assert!(room.disconnect(&identity, "PA_new"));
        assert!(room.state().linked.is_none());
    }

    #[test]
    fn selecting_the_linked_participant_again_reattaches_without_announcing() {
        let person = ParticipantIdentity::from("person");
        let mut room = RoomCoordinator::new(None);
        room.start();
        let first = room.set_participant(Some(person.clone()), Some("PA_1")).unwrap();
        assert_eq!(first, Relink { announce: true, detach: false });
        let again = room.set_participant(Some(person.clone()), Some("PA_1")).unwrap();
        assert_eq!(again, Relink { announce: false, detach: true });
        let absent = room.set_participant(Some("absent".into()), None).unwrap();
        assert!(!absent.announce, "nobody to announce until they join");
        room.begin_close();
        assert!(room.set_participant(None, None).is_none());
    }

    #[test]
    fn an_event_of_an_earlier_participant_lifetime_is_dropped() {
        let person = ParticipantIdentity::from("person");
        let mut room = RoomCoordinator::new(None);
        room.start();
        room.select(Some(person.clone()), Some("PA_new"));
        assert!(!room.current_event(&person, "PA_old", Some("PA_new"), false));
        assert!(room.current_event(&person, "PA_new", Some("PA_new"), false));
        assert!(room.current_event(&person, "PA_new", None, true), "the linked one departs");
        assert!(!room.current_event(&person, "PA_old", None, true));
    }

    #[test]
    fn late_start_success_cannot_replace_close_failure() {
        let mut room = RoomCoordinator::new(None);
        room.start();
        room.begin_close();
        room.finish_start(Ok(()));
        assert!(room.state().ready.unwrap().is_err());
        assert!(!room.start());
    }
}
