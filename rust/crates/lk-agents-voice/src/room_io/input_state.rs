//! Selection and attachment identity for a participant's media input.

use std::collections::BTreeSet;

use lk_agents_abi::voice::rtc::{ParticipantIdentity, TrackSid, TrackSource};

/// A frame retains both the track selection and the attachment that admitted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputIdentity {
    pub selection: u64,
    pub attachment: u64,
}

pub struct TrackSelection {
    linked: Option<ParticipantIdentity>,
    reading: Option<TrackSid>,
    generation: u64,
    attachment: u64,
    attached: bool,
    closed: bool,
    pre_connect_done: BTreeSet<(ParticipantIdentity, String, TrackSid)>,
}

impl Default for TrackSelection {
    fn default() -> Self {
        Self {
            linked: None,
            reading: None,
            generation: 0,
            attachment: 0,
            attached: true,
            closed: false,
            pre_connect_done: BTreeSet::new(),
        }
    }
}

impl TrackSelection {
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn reading(&self) -> Option<&TrackSid> {
        self.reading.as_ref()
    }

    /// `true` retires the previous reader before the runtime starts another one.
    pub fn link(&mut self, identity: Option<ParticipantIdentity>) -> bool {
        if self.closed || self.linked == identity {
            return false;
        }
        self.linked = identity;
        self.retire();
        true
    }

    pub fn claim(
        &mut self,
        participant: &ParticipantIdentity,
        source: TrackSource,
        sid: &TrackSid,
        sources: &[TrackSource],
    ) -> Option<u64> {
        if self.closed
            || self.linked.as_ref() != Some(participant)
            || !sources.contains(&source)
            || self.reading.as_ref() == Some(sid)
        {
            return None;
        }
        self.retire();
        self.reading = Some(sid.clone());
        Some(self.generation)
    }

    /// A publication event may retire only the selection it belongs to.
    pub fn release(&mut self, participant: &ParticipantIdentity, sid: &TrackSid) -> bool {
        if self.reading.as_ref() != Some(sid) || self.linked.as_ref() != Some(participant) {
            return false;
        }
        self.retire();
        true
    }

    /// EOF stops production while preserving accepted frames for this selection.
    pub fn ended(&mut self, generation: u64) -> bool {
        if self.generation != generation {
            return false;
        }
        self.reading = None;
        true
    }

    pub fn producing(&self, generation: u64) -> bool {
        !self.closed && self.generation == generation && self.reading.is_some()
    }

    pub fn admit(&self, generation: u64) -> Option<InputIdentity> {
        (self.producing(generation) && self.attached)
            .then_some(InputIdentity { selection: self.generation, attachment: self.attachment })
    }

    pub fn accepts(&self, identity: InputIdentity) -> bool {
        !self.closed
            && self.attached
            && self.generation == identity.selection
            && self.attachment == identity.attachment
    }

    pub fn set_attached(&mut self, attached: bool) {
        if !self.closed && self.attached != attached {
            self.attached = attached;
            self.attachment += 1;
        }
    }

    /// A publication's head can enter this session once, even after a fallback to its track.
    pub fn replay_pre_connect(
        &mut self,
        participant: ParticipantIdentity,
        participant_sid: String,
        sid: TrackSid,
        capacity: usize,
    ) -> bool {
        !self.closed
            && self.pre_connect_done.len() < capacity
            && self.pre_connect_done.insert((participant, participant_sid, sid))
    }

    pub fn close(&mut self) {
        self.closed = true;
        self.retire();
        self.pre_connect_done.clear();
    }

    fn retire(&mut self) {
        self.generation += 1;
        self.reading = None;
    }
}
