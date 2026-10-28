//! Room attachment and startup decisions over supplied participant records.

use lk_agents_abi::Closed;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, ParticipantKind};

use super::decide_link;

#[derive(Clone)]
pub(super) struct RoomState {
    pub(super) linked: Option<ParticipantIdentity>,
    pub(super) ready: Option<Result<()>>,
    pub(super) closed: Option<Closed>,
}

enum Phase {
    New,
    Started,
    Closing,
    Closed(Closed),
}

pub(super) struct RoomCoordinator {
    target: Option<ParticipantIdentity>,
    linked: Option<ParticipantIdentity>,
    attachment: u64,
    phase: Phase,
    ready: Option<Result<()>>,
}

impl RoomCoordinator {
    pub(super) fn new(target: Option<ParticipantIdentity>) -> Self {
        Self { target, linked: None, attachment: 0, phase: Phase::New, ready: None }
    }

    pub(super) fn state(&self) -> RoomState {
        RoomState {
            linked: self.linked.clone(),
            ready: self.ready.clone(),
            closed: match self.phase {
                Phase::Closed(outcome) => Some(outcome),
                _ => None,
            },
        }
    }

    pub(super) fn is_open(&self) -> bool {
        !matches!(self.phase, Phase::Closing | Phase::Closed(_))
    }

    pub(super) fn start(&mut self) -> bool {
        if !matches!(self.phase, Phase::New) {
            return false;
        }
        self.phase = Phase::Started;
        true
    }

    pub(super) fn consider(
        &mut self,
        candidate: &ParticipantIdentity,
        kind: ParticipantKind,
        on_behalf_of: Option<&str>,
        agent: &ParticipantIdentity,
        accepted_kinds: &[ParticipantKind],
    ) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_))
            || !decide_link(
                self.linked.is_some(),
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
        true
    }

    /// A selection is a new attachment even if the same participant is selected again.
    pub(super) fn select(&mut self, target: Option<ParticipantIdentity>, present: bool) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_)) {
            return false;
        }
        self.attachment += 1;
        self.target = target.clone();
        self.linked = target.filter(|_| present);
        true
    }

    pub(super) fn disconnect(&mut self, identity: &ParticipantIdentity) -> bool {
        if self.linked.as_ref() != Some(identity)
            || matches!(self.phase, Phase::Closing | Phase::Closed(_))
        {
            return false;
        }
        self.attachment += 1;
        self.linked = None;
        true
    }

    pub(super) fn attachment(&self, participant: &ParticipantIdentity) -> Option<u64> {
        (!matches!(self.phase, Phase::Closing | Phase::Closed(_))
            && self.linked.as_ref() == Some(participant))
        .then_some(self.attachment)
    }

    pub(super) fn accepts(&self, attachment: u64, participant: &ParticipantIdentity) -> bool {
        self.attachment(participant) == Some(attachment)
    }

    pub(super) fn finish_start(&mut self, outcome: Result<()>) {
        if !matches!(self.phase, Phase::Closing | Phase::Closed(_)) && self.ready.is_none() {
            self.ready = Some(outcome);
        }
    }

    pub(super) fn begin_close(&mut self) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed(_)) {
            return false;
        }
        self.phase = Phase::Closing;
        self.attachment += 1;
        self.linked = None;
        if self.ready.is_none() {
            self.ready = Some(Err(IoError::Rtc("room I/O is closed".into())));
        }
        true
    }

    pub(super) fn finish_close(&mut self, outcome: Closed) {
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
        room.select(Some(person.clone()), true);
        let first = room.attachment(&person).unwrap();
        room.select(Some(person.clone()), true);
        assert!(!room.accepts(first, &person));
        let second = room.attachment(&person).unwrap();
        assert!(room.accepts(second, &person));
        room.begin_close();
        assert!(!room.accepts(second, &person));
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
