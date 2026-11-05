//! Which of the linked participant's tracks a media input reads, and which frames it admits.
//!
//! The executor reports the SDK's tracks and its readers' lifecycle; this decides which track to
//! open, when a reader is obsolete, and when the frames already queued no longer belong to the
//! selection and attachment the session reads.

use std::collections::BTreeSet;

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, TrackSid, TrackSource, silence_frame};
use lk_agents_abi::{AudioFrame, Closed};

/// Readers opened and not yet retired, including obsolete ones still closing their stream.
pub const MAX_READERS: usize = 4;

pub const AUDIO_SOURCES: &[TrackSource] = &[TrackSource::Microphone];
pub const VIDEO_SOURCES: &[TrackSource] = &[TrackSource::Camera, TrackSource::Screenshare];

/// The track selection and attachment frames are admitted under: when it changes, the frames
/// queued before it no longer reach the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputIdentity {
    pub selection: u64,
    pub attachment: u64,
}

/// A participant's publication as the SDK lists it.
pub struct Publication {
    pub sid: TrackSid,
    pub source: TrackSource,
    pub subscribed: bool,
}

/// The participant a publication belongs to.
#[derive(Clone, Copy)]
pub struct Owner<'a> {
    pub identity: &'a ParticipantIdentity,
    pub sid: &'a str,
}

/// Open this track for the reader of `generation`.
#[derive(Debug, PartialEq, Eq)]
pub struct Open {
    pub generation: u64,
    pub sid: TrackSid,
}

/// What the executor does with a frame its reader produced.
#[derive(Debug, PartialEq, Eq)]
pub enum Admit {
    Deliver,
    /// Detached: the frame is dropped, the reader keeps reading.
    Drop,
    /// The reader's selection is obsolete or the input is closed.
    Stop,
}

pub struct ParticipantInput {
    sources: &'static [TrackSource],
    linked: Option<ParticipantIdentity>,
    reading: Option<TrackSid>,
    generation: u64,
    attachment: u64,
    attached: bool,
    closed: bool,
    readers: usize,
    cleanup: Closed,
    pre_connect_done: BTreeSet<(ParticipantIdentity, String, TrackSid)>,
}

impl ParticipantInput {
    pub fn new(sources: &'static [TrackSource]) -> Self {
        Self {
            sources,
            linked: None,
            reading: None,
            generation: 0,
            attachment: 0,
            attached: true,
            closed: false,
            readers: 0,
            cleanup: Closed::Clean,
            pre_connect_done: BTreeSet::new(),
        }
    }

    pub fn reading(&self) -> Option<&TrackSid> {
        self.reading.as_ref()
    }

    /// `set_participant`: a new link retires the current reader and reads the participant's first
    /// subscribed track; the same link keeps a track already read.
    pub fn link(
        &mut self,
        identity: Option<ParticipantIdentity>,
        owner: Option<Owner<'_>>,
        publications: &[Publication],
    ) -> Option<Open> {
        let changed = !self.closed && self.linked != identity;
        if changed {
            self.linked = identity;
            self.retire();
        } else if self.reading.is_some() {
            return None;
        }
        self.first(owner?, publications, None)
    }

    pub fn subscribed(&mut self, owner: Owner<'_>, publication: &Publication) -> Option<Open> {
        self.claim(owner, publication)
    }

    /// The track read is gone: fall back to another of the participant's tracks. An unsubscribed
    /// track may still be listed as subscribed, so it is skipped.
    pub fn removed(
        &mut self,
        owner: Owner<'_>,
        sid: &TrackSid,
        unsubscribed: bool,
        publications: &[Publication],
    ) -> Option<Open> {
        if self.reading.as_ref() != Some(sid) || self.linked.as_ref() != Some(owner.identity) {
            return None;
        }
        self.retire();
        self.first(owner, publications, unsubscribed.then_some(sid))
    }

    fn first(
        &mut self,
        owner: Owner<'_>,
        publications: &[Publication],
        skip: Option<&TrackSid>,
    ) -> Option<Open> {
        publications
            .iter()
            .filter(|publication| publication.subscribed && Some(&publication.sid) != skip)
            .find_map(|publication| self.claim(owner, publication))
    }

    fn claim(&mut self, owner: Owner<'_>, publication: &Publication) -> Option<Open> {
        if self.closed
            || self.readers >= MAX_READERS
            || self.linked.as_ref() != Some(owner.identity)
            || !self.sources.contains(&publication.source)
            || self.reading.as_ref() == Some(&publication.sid)
        {
            return None;
        }
        self.retire();
        self.reading = Some(publication.sid.clone());
        self.readers += 1;
        Some(Open { generation: self.generation, sid: publication.sid.clone() })
    }

    /// Whether to replay an opened track's pre-connect head (the publisher buffered one): once per
    /// publication, even after a fallback to its track.
    pub fn replay_pre_connect(&mut self, owner: Owner<'_>, sid: &TrackSid) -> bool {
        !self.closed
            && self.pre_connect_done.len() < super::pre_connect_state::MAX_TRACKS
            && self.pre_connect_done.insert((
                owner.identity.clone(),
                owner.sid.to_owned(),
                sid.clone(),
            ))
    }

    pub fn admit(&self, generation: u64) -> Admit {
        if !self.producing(generation) {
            Admit::Stop
        } else if !self.attached {
            Admit::Drop
        } else {
            Admit::Deliver
        }
    }

    pub fn producing(&self, generation: u64) -> bool {
        !self.closed && self.generation == generation && self.reading.is_some()
    }

    /// What admitted frames belong to; a change drops the frames queued before it.
    pub fn source(&self) -> InputIdentity {
        InputIdentity { selection: self.generation, attachment: self.attachment }
    }

    pub fn set_attached(&mut self, attached: bool) {
        if !self.closed && self.attached != attached {
            self.attached = attached;
            self.attachment += 1;
        }
    }

    /// The track ended (EOF, error or a failed open): production stops, accepted frames stay.
    pub fn ended(&mut self, generation: u64) {
        if self.generation == generation {
            self.reading = None;
        }
    }

    /// A reader retired its stream with this outcome; a failed open retires with none.
    pub fn retired(&mut self, generation: u64, cleanup: Closed) {
        self.ended(generation);
        self.readers -= 1;
        self.cleanup = self.cleanup.and(cleanup);
    }

    pub fn close(&mut self) {
        self.closed = true;
        self.retire();
        self.pre_connect_done.clear();
    }

    /// Every reader has retired its stream: the input's cleanup outcome.
    pub fn settled(&self) -> Option<Closed> {
        (self.closed && self.readers == 0).then_some(self.cleanup)
    }

    fn retire(&mut self) {
        self.generation += 1;
        self.reading = None;
    }
}

/// `_ParticipantAudioInputStream`: an ended track leaves half a second of silence, so a turn in
/// progress can end.
pub fn end_of_track(sample_rate: u32, num_channels: u16) -> AudioFrame {
    silence_frame(sample_rate / 2, sample_rate, num_channels)
}

/// The pre-connect head a reader replays ahead of the live frames. A missing or expired head is
/// no head, so the live track plays on; any other failure stops the reader with it.
pub fn replayed_head(head: Result<Vec<AudioFrame>>) -> Result<Vec<AudioFrame>> {
    match head {
        Err(IoError::PreConnectTimeout(_)) => Ok(Vec::new()),
        head => head,
    }
}

/// Which of a stream's input processors are failing. A failure is reported when it opens its
/// processor's span; the span ends at the first frame that processor handles cleanly, so each
/// processor reports on its own and again after it recovers.
#[derive(Debug, Default)]
pub struct ProcessorSpans {
    failing: BTreeSet<String>,
    /// The processors that failed on the frame still to arrive.
    failed: BTreeSet<String>,
}

impl ProcessorSpans {
    /// `label` failed on the next frame: whether that opens its span, so is reported.
    pub fn failed(&mut self, label: &str) -> bool {
        self.failed.insert(label.to_owned());
        !self.failing.contains(label)
    }

    /// A frame arrived: the processors that did not fail on it recovered.
    pub fn frame(&mut self) {
        self.failing = std::mem::take(&mut self.failed);
    }
}

#[cfg(test)]
mod tests {
    // Catches: a span kept per stream rather than per processor (AGC failing while noise
    // cancellation's span is open went unreported) and a recovered processor never reporting again.
    #[test]
    fn each_processor_reports_once_per_failing_span() {
        let mut spans = super::ProcessorSpans::default();
        assert!(spans.failed("nc"));
        spans.frame();
        assert!(!spans.failed("nc"), "a span reports once");
        assert!(spans.failed("agc"), "another processor's failure is its own span");
        spans.frame();
        spans.frame();
        assert!(spans.failed("nc"), "a recovered processor reports its next span");
        assert!(spans.failed("agc"));
    }

    use super::*;

    fn owner(identity: &ParticipantIdentity) -> Owner<'_> {
        Owner { identity, sid: "PA_user" }
    }

    fn microphone(sid: &str) -> Publication {
        Publication { sid: sid.into(), source: TrackSource::Microphone, subscribed: true }
    }

    fn reading(readers: &mut ParticipantInput, user: &ParticipantIdentity, sid: &str) -> u64 {
        readers.subscribed(owner(user), &microphone(sid)).unwrap().generation
    }

    #[test]
    fn replacing_a_track_rejects_its_buffered_input() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        let old = reading(&mut readers, &user, "old");
        let queued = readers.source();
        assert_eq!(readers.admit(old), Admit::Deliver);
        let new = reading(&mut readers, &user, "new");
        assert_ne!(readers.source(), queued);
        assert_eq!(readers.admit(old), Admit::Stop);
        readers.retired(old, Closed::Clean);
        assert_eq!(readers.reading(), Some(&"new".into()));
        assert_eq!(readers.admit(new), Admit::Deliver);
    }

    #[test]
    fn reattaching_the_same_input_rejects_its_previous_buffer() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        let track = reading(&mut readers, &user, "mic");
        let previous = readers.source();
        readers.set_attached(false);
        assert_eq!(readers.admit(track), Admit::Drop);
        readers.set_attached(true);
        assert_eq!(readers.admit(track), Admit::Deliver);
        assert_ne!(readers.source(), previous);
    }

    #[test]
    fn a_track_ending_preserves_its_accepted_audio() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        let track = reading(&mut readers, &user, "mic");
        let accepted = readers.source();
        readers.ended(track);
        assert_eq!(readers.source(), accepted);
        assert_eq!(readers.reading(), None);
    }

    #[test]
    fn readers_still_closing_bound_new_subscriptions() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        let generations: Vec<u64> =
            (0..MAX_READERS).map(|n| reading(&mut readers, &user, &format!("mic-{n}"))).collect();
        assert!(readers.subscribed(owner(&user), &microphone("one-too-many")).is_none());
        readers.retired(generations[0], Closed::Clean);
        assert!(readers.subscribed(owner(&user), &microphone("after-retire")).is_some());
    }

    #[test]
    fn losing_the_track_read_falls_back_to_another_subscribed_track() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        let tracks = [microphone("first"), microphone("second")];
        let first = readers.link(Some(user.clone()), Some(owner(&user)), &tracks).unwrap();
        assert_eq!(first.sid, "first".into());
        let fallback = readers.removed(owner(&user), &"first".into(), true, &tracks).unwrap();
        assert_eq!(fallback.sid, "second".into());
        assert!(readers.removed(owner(&user), &"first".into(), true, &tracks).is_none());
    }

    #[test]
    fn a_pre_connect_head_is_replayed_once_per_publication() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        reading(&mut readers, &user, "mic");
        assert!(readers.replay_pre_connect(owner(&user), &"mic".into()));
        readers.removed(owner(&user), &"mic".into(), false, &[]);
        reading(&mut readers, &user, "mic");
        assert!(!readers.replay_pre_connect(owner(&user), &"mic".into()));
    }

    #[test]
    fn close_settles_once_every_reader_retired() {
        let user = ParticipantIdentity::from("user");
        let mut readers = ParticipantInput::new(AUDIO_SOURCES);
        readers.link(Some(user.clone()), None, &[]);
        let track = reading(&mut readers, &user, "mic");
        readers.close();
        assert_eq!(readers.settled(), None, "a reader may still be opening its stream");
        readers.retired(track, Closed::Failed);
        assert_eq!(readers.settled(), Some(Closed::Failed));
    }
}
