//! RoomIO attachment, participant selection, readiness and retirement decisions.

use lk_agents_abi::Closed;
use lk_agents_abi::voice::io::{IoError, Result};
pub use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, DeleteRoomRequested, ParticipantLinked,
    ParticipantUnlinked, RoomIoConfig, TextInput, TextOutputConfig,
};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, ParticipantKind};
use std::collections::BTreeMap;
use std::time::Duration;

/// `job.DEFAULT_PARTICIPANT_KINDS`: no `Agent` or `Egress`, so agents never auto-link to each other.
pub const DEFAULT_PARTICIPANT_KINDS: &[ParticipantKind] =
    &[ParticipantKind::Connector, ParticipantKind::Sip, ParticipantKind::Standard];

/// `_pre_connect_audio.PRE_CONNECT_AUDIO_BUFFER_STREAM`.
pub const PRE_CONNECT_AUDIO_BUFFER_STREAM: &str = "lk.agent.pre-connect-audio-buffer";

/// `PreConnectAudioHandler._max_delta_s`: a buffer older than this is stale and dropped.
pub const PRE_CONNECT_MAX_DELTA: Duration = Duration::from_secs(1);

/// The kinds a [`RoomIoConfig`] auto-links: `participant_kinds`, or the defaults when empty.
pub fn accepted_kinds(config: &RoomIoConfig) -> &[ParticipantKind] {
    if config.participant_kinds.is_empty() {
        DEFAULT_PARTICIPANT_KINDS
    } else {
        &config.participant_kinds
    }
}

/// Whether the agent transcript follows audio playout: `sync_transcription` unset (`NOT_GIVEN`)
/// means yes when there is an audio output.
pub fn wants_transcript_sync(config: &RoomIoConfig, has_audio_output: bool) -> bool {
    let Some(text) = &config.text_output else { return false };
    text.sync_transcription != Some(false) && has_audio_output
}

pub use lk_agents_abi::voice::room_owner::RoomIoEvent;

/// `RoomIO._on_participant_connected`. Not when someone is already linked, when an explicit target
/// is configured and this is not it, or when the candidate is an avatar publishing for the agent.
/// The publish-on-behalf filter applies only without an explicit target (the reference's `elif`).
pub fn decide_link(
    already_linked: bool,
    configured_target: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_kind: ParticipantKind,
    candidate_publishes_on_behalf_of: Option<&str>,
    agent_identity: &ParticipantIdentity,
    accepted_kinds: &[ParticipantKind],
) -> bool {
    let target_ok = match configured_target {
        Some(target) => target == candidate,
        None => candidate_publishes_on_behalf_of != Some(agent_identity.as_str()),
    };
    !already_linked && target_ok && accepted_kinds.contains(&candidate_kind)
}

pub fn publishes_on_behalf_of(attributes: &BTreeMap<String, String>) -> Option<&str> {
    attributes.get(lk_agents_abi::ATTRIBUTE_PUBLISH_ON_BEHALF).map(String::as_str)
}

/// A SIP participant no dispatch rule admitted (`sip.ruleID` unset): a call the agent dialed.
pub fn outbound_sip(
    kind: lk_agents_abi::voice::rtc::ParticipantKind,
    attributes: &BTreeMap<String, String>,
) -> bool {
    kind == lk_agents_abi::voice::rtc::ParticipantKind::Sip
        && !attributes.contains_key("sip.ruleID")
}

/// `_ParticipantLegacyTranscriptionOutput._is_local_proxy_participant`: publishing on behalf of the
/// agent is a reason to accept (the caption belongs on the avatar's track).
pub fn is_local_proxy_participant(
    subject: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_publishes_on_behalf_of: Option<&str>,
) -> bool {
    let Some(subject) = subject else { return false };
    subject == candidate || candidate_publishes_on_behalf_of == Some(subject.as_str())
}

#[derive(Clone)]
pub struct RoomView {
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

pub struct RoomIo {
    target: Option<ParticipantIdentity>,
    linked: Option<ParticipantIdentity>,
    linked_sid: Option<String>,
    attachment: u64,
    phase: Phase,
    ready: Option<Result<()>>,
}

impl RoomIo {
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

    pub fn state(&self) -> RoomView {
        RoomView {
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
        let mut room = RoomIo::new(None);
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
        let mut room = RoomIo::new(None);
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
        let mut room = RoomIo::new(None);
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
        let mut room = RoomIo::new(None);
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
        let mut room = RoomIo::new(None);
        room.start();
        room.select(Some(person.clone()), Some("PA_new"));
        assert!(!room.current_event(&person, "PA_old", Some("PA_new"), false));
        assert!(room.current_event(&person, "PA_new", Some("PA_new"), false));
        assert!(room.current_event(&person, "PA_new", None, true), "the linked one departs");
        assert!(!room.current_event(&person, "PA_old", None, true));
    }

    #[test]
    fn late_start_success_cannot_replace_close_failure() {
        let mut room = RoomIo::new(None);
        room.start();
        room.begin_close();
        room.finish_start(Ok(()));
        assert!(room.state().ready.unwrap().is_err());
        assert!(!room.start());
    }
}
