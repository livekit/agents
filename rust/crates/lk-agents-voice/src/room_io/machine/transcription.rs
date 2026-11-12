use super::{is_local_proxy_participant, publishes_on_behalf_of};
use crate::room_io::MAX_STREAM_BYTES;
use crate::room_io::effects::RoomTranscriptionEffect;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, TrackSid, TrackSource};
use lk_agents_abi::{
    ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID, TranscriptChunk,
};
use lk_agents_core::tts::markup::delivery_attribute;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The microphone view observed by the performer when a transcript's subject changes.
pub struct ParticipantTracks {
    pub identity: ParticipantIdentity,
    pub attributes: BTreeMap<String, String>,
    pub microphone: Option<TrackSid>,
}

/// Wrap a visible chunk for the wire: protojson of `agent_pb.TimedString`, default fields omitted,
/// newline-delimited.
fn encode(clean_text: &str, json_format: bool, timing: Option<&TranscriptChunk>) -> String {
    if !json_format {
        return clean_text.to_owned();
    }
    let mut map = serde_json::Map::new();
    if !clean_text.is_empty() {
        map.insert("text".into(), serde_json::Value::String(clean_text.to_owned()));
    }
    if let Some(t) = timing {
        for (key, value) in [
            ("start_time", t.start_time),
            ("end_time", t.end_time),
            ("confidence", t.confidence),
            ("start_time_offset", t.start_time_offset),
        ] {
            if let Some(v) = value
                && let Some(n) = serde_json::Number::from_f64(v)
            {
                map.insert(key.into(), serde_json::Value::Number(n));
            }
        }
    }
    let mut out = serde_json::Value::Object(map).to_string();
    out.push('\n');
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Writing {
    /// No segment open; the next visible text starts one.
    Idle,
    /// A segment has visible text; in delta mode its stream is open.
    Segment,
    Closed,
}

/// Room publication state: one segment's visible text, tags, and ordered publication work.
pub struct RoomTranscriptionWriter {
    /// Deltas accumulate into one stream per segment (the agent's transcript); otherwise each text
    /// is a full replacement in its own short stream (the user's).
    is_delta_stream: bool,
    json_format: bool,
    writing: Writing,
    segment: String,
    /// The segment's visible text: accumulated for deltas, the latest for replacements.
    text: String,
    expression: Option<(&'static str, String)>,
    /// A host's plain strings, read for markup.
    host: crate::transcription::HostText,
    pending: Vec<RoomTranscriptionEffect>,
    raw_bytes: usize,
}

impl RoomTranscriptionWriter {
    /// Reserves the first segment before text arrives; every later boundary supplies its successor.
    pub fn new(is_delta_stream: bool, json_format: bool, segment: String) -> Self {
        Self {
            is_delta_stream,
            json_format,
            writing: Writing::Idle,
            segment,
            text: String::new(),
            expression: None,
            host: Default::default(),
            pending: Vec::new(),
            raw_bytes: 0,
        }
    }

    pub fn closed(&self) -> bool {
        self.writing == Writing::Closed
    }
    pub fn take_effects(&mut self) -> Vec<RoomTranscriptionEffect> {
        std::mem::take(&mut self.pending)
    }
    pub fn clear_effects(&mut self) {
        self.pending.clear();
    }
    pub fn expression_bytes(&self) -> usize {
        self.expression.as_ref().map_or(0, |(_, value)| value.capacity())
    }
    pub fn admit_text(&mut self, text: &str) -> Result<()> {
        if self.closed() {
            return Ok(());
        }
        let prior = if self.is_delta_stream { self.raw_bytes } else { 0 };
        if text.len() > MAX_STREAM_BYTES.saturating_sub(prior) {
            return Err(IoError::TranscriptionCapacity("segment text bytes".into()));
        }
        self.raw_bytes = prior + text.len();
        Ok(())
    }

    fn send(&mut self, effect: RoomTranscriptionEffect) {
        self.pending.push(effect);
    }

    /// A host's plain string: its markup taken out, its delivery kept. The chunk it was read as.
    pub fn capture_text(&mut self, text: &str) -> TranscriptChunk {
        let chunk = self.host.push(text);
        self.capture(&chunk);
        chunk
    }

    /// A piece of the transcript: the first of a segment opens it, the rest extend (or replace)
    /// it. A piece with no text (a delivery alone) opens nothing.
    pub fn capture(&mut self, chunk: &TranscriptChunk) {
        if self.writing == Writing::Closed {
            return;
        }
        if self.expression.is_none() {
            self.expression = chunk.expression.as_deref().and_then(delivery_attribute);
        }
        if chunk.text.is_empty() {
            return;
        }
        let opening = self.writing != Writing::Segment;
        if opening {
            self.text.clear();
        }
        let payload = encode(&chunk.text, self.json_format, Some(chunk));
        let segment = self.segment.clone();
        if self.is_delta_stream {
            self.text.push_str(&chunk.text);
            if opening {
                // The leading expression also goes on the opening header, so a frontend can
                // colour the turn at once.
                let expression = self.expression.clone();
                self.send(RoomTranscriptionEffect::OpenStream {
                    segment,
                    is_final: false,
                    expression,
                });
            }
            self.send(RoomTranscriptionEffect::Write { payload });
        } else {
            self.text.clone_from(&chunk.text);
            self.send(RoomTranscriptionEffect::OpenStream {
                segment,
                is_final: false,
                expression: None,
            });
            self.send(RoomTranscriptionEffect::Write { payload });
            self.send(RoomTranscriptionEffect::CloseStream { is_final: false, expression: None });
        }
        let (segment, text) = (self.segment.clone(), self.text.clone());
        self.send(RoomTranscriptionEffect::PublishLegacy { segment, text, is_final: false });
        self.writing = Writing::Segment;
    }

    /// The segment's text is complete. `next_segment` belongs to the following text, including
    /// when this segment was empty.
    pub fn end_segment(&mut self, next_segment: String) {
        match self.writing {
            Writing::Closed => {}
            Writing::Idle => self.reset(Writing::Idle),
            Writing::Segment => self.end(Writing::Idle),
        }
        if !self.closed() {
            self.segment = next_segment;
        }
    }

    /// The room disconnected: an open segment is closed now, not leaked (a dropped host stream
    /// stays open server-side).
    pub fn disconnected(&mut self, next_segment: String) {
        if self.writing == Writing::Segment {
            let expression = self.expression.clone();
            self.send(RoomTranscriptionEffect::CloseStream { is_final: true, expression });
            self.reset(Writing::Idle);
            self.segment = next_segment;
        }
    }

    pub fn close(&mut self) {
        match self.writing {
            Writing::Closed => {}
            Writing::Idle => self.reset(Writing::Closed),
            Writing::Segment => self.end(Writing::Closed),
        }
    }

    fn end(&mut self, next: Writing) {
        let expression = self.expression.clone();
        let segment = self.segment.clone();
        if self.is_delta_stream {
            self.send(RoomTranscriptionEffect::CloseStream { is_final: true, expression });
        } else {
            // The final replacement opens already final with the whole latest text.
            let payload = encode(&self.text, self.json_format, None);
            self.send(RoomTranscriptionEffect::OpenStream {
                segment: segment.clone(),
                is_final: true,
                expression: None,
            });
            self.send(RoomTranscriptionEffect::Write { payload });
            self.send(RoomTranscriptionEffect::CloseStream { is_final: true, expression: None });
        }
        self.send(RoomTranscriptionEffect::PublishLegacy {
            segment,
            text: self.text.clone(),
            is_final: true,
        });
        self.reset(next);
    }

    fn reset(&mut self, next: Writing) {
        self.text.clear();
        self.expression.take();
        self.writing = next;
        self.host.reset();
        self.raw_bytes = 0;
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct Subject {
    pub participant: Option<ParticipantIdentity>,
    /// Who the deprecated leg publishes as: the avatar worker's track when one publishes for them.
    pub represented_by: Option<ParticipantIdentity>,
    /// The deprecated leg's track: the subject's microphone, or the proxy's.
    pub legacy_track: Option<TrackSid>,
    /// The stream leg's track hint: the subject's own microphone, never the proxy's.
    pub stream_track: Option<TrackSid>,
}

impl Subject {
    pub fn for_participant(
        participant: Option<ParticipantIdentity>,
        local: &ParticipantIdentity,
        peers: &[ParticipantTracks],
    ) -> Self {
        let mut subject = Self {
            participant: participant.clone(),
            represented_by: participant.clone(),
            ..Self::default()
        };
        let Some(target) = participant else { return subject };
        subject.stream_track = peers
            .iter()
            .find(|peer| peer.identity == target)
            .and_then(|peer| peer.microphone.clone());
        if target == *local {
            subject.legacy_track = subject.stream_track.clone();
        }
        if subject.legacy_track.is_none()
            && let Some(peer) = peers.iter().find(|peer| {
                is_local_proxy_participant(
                    Some(&target),
                    &peer.identity,
                    publishes_on_behalf_of(&peer.attributes),
                ) && peer.microphone.is_some()
            })
        {
            subject.legacy_track = peer.microphone.clone();
            subject.represented_by = Some(peer.identity.clone());
        }
        subject
    }

    /// Who and on which track [`RoomTranscriptionEffect::PublishLegacy`] publishes: the representing
    /// participant, else the subject itself.
    pub fn legacy_target(&self) -> Option<(ParticipantIdentity, TrackSid)> {
        let identity = self.represented_by.clone().or_else(|| self.participant.clone())?;
        Some((identity, self.legacy_track.clone()?))
    }

    pub fn on_track_published(
        &mut self,
        owner: &ParticipantIdentity,
        owner_attributes: &BTreeMap<String, String>,
        source: TrackSource,
        sid: &TrackSid,
    ) {
        if source != TrackSource::Microphone {
            return;
        }
        if is_local_proxy_participant(
            self.participant.as_ref(),
            owner,
            publishes_on_behalf_of(owner_attributes),
        ) {
            self.legacy_track = Some(sid.clone());
            self.represented_by = Some(owner.clone());
        }
        if self.participant.as_ref() == Some(owner) {
            self.stream_track = Some(sid.clone());
        }
    }

    pub fn on_local_track_published(
        &mut self,
        local: &ParticipantIdentity,
        source: TrackSource,
        sid: &TrackSid,
    ) {
        if source != TrackSource::Microphone || self.participant.as_ref() != Some(local) {
            return;
        }
        self.legacy_track = Some(sid.clone());
        self.stream_track = Some(sid.clone());
    }
}

fn stream_attributes(is_final: bool, track: Option<&TrackSid>) -> BTreeMap<String, String> {
    let mut attributes = BTreeMap::new();
    attributes.insert(ATTRIBUTE_TRANSCRIPTION_FINAL.to_owned(), is_final.to_string());
    if let Some(tid) = track {
        attributes.insert(ATTRIBUTE_TRANSCRIPTION_TRACK_ID.to_owned(), tid.to_string());
    }
    attributes
}

/// The header of [`RoomTranscriptionEffect::OpenStream`]: its segment id, and the leading expression.
pub fn open_attributes(
    segment: String,
    is_final: bool,
    expression: Option<(&'static str, String)>,
    track: Option<&TrackSid>,
) -> BTreeMap<String, String> {
    let mut attributes = stream_attributes(is_final, track);
    attributes.insert(ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID.to_owned(), segment);
    if let Some((key, value)) = expression {
        attributes.insert(key.to_owned(), value);
    }
    attributes
}

/// The trailer of [`RoomTranscriptionEffect::CloseStream`], with `track` as the stream opened with it; the
/// expression never overrides the stream's own attributes.
pub fn close_attributes(
    is_final: bool,
    expression: Option<(&'static str, String)>,
    track: Option<&TrackSid>,
) -> BTreeMap<String, String> {
    let mut attributes = stream_attributes(is_final, track);
    if let Some((key, value)) = expression {
        attributes.entry(key.to_owned()).or_insert(value);
    }
    attributes
}

/// Whether the performer may abandon this effect at a failure or the cleanup deadline: a stream
/// open and its consuming close keep their SDK outcome.
pub fn cancellable(effect: &RoomTranscriptionEffect) -> bool {
    matches!(
        effect,
        RoomTranscriptionEffect::Write { .. } | RoomTranscriptionEffect::PublishLegacy { .. }
    )
}

/// Effects captured for one subject, in order, and the text bytes they hold until performed.
pub struct Batch {
    pub effects: Vec<RoomTranscriptionEffect>,
    pub subject: Subject,
    pub ticket: Ticket,
}

/// The text bytes a batch holds until [`Admission::performed`].
#[derive(Debug)]
pub struct Ticket {
    bytes: usize,
}

fn capacity(resource: &str) -> IoError {
    IoError::TranscriptionCapacity(resource.into())
}

/// The transcription performer's backlog, in capture order. Its length never refuses text: a
/// queued replacement a later one supersedes is dropped and queued deltas merge, so only a writer
/// stalled behind more than [`MAX_STREAM_BYTES`] of text fails the output. The first failure
/// latches and drops the backlog; the performer streams one segment at a time.
#[derive(Default)]
pub struct Admission {
    queue: VecDeque<Batch>,
    /// Text bytes queued and in flight.
    bytes: usize,
    closing: bool,
    failure: Option<IoError>,
}

impl Admission {
    pub fn failure(&self) -> Option<&IoError> {
        self.failure.as_ref()
    }

    /// Latches the first failure, dropping the writer's pending effects and the backlog.
    pub fn fail(&mut self, writer: &mut RoomTranscriptionWriter, error: IoError) {
        writer.clear_effects();
        writer.reset(Writing::Closed);
        for batch in self.queue.drain(..) {
            self.bytes -= batch.ticket.bytes;
        }
        self.failure.get_or_insert(error);
    }

    /// Queue the writer's pending effects for `subject`, coalesced into the last queued batch of
    /// the same subject. `Err` only when this submission fails the output.
    pub fn submit(
        &mut self,
        writer: &mut RoomTranscriptionWriter,
        subject: &Subject,
    ) -> Result<()> {
        if self.failure.is_some() {
            return Ok(());
        }
        let queued = self.try_submit(writer, subject);
        queued.inspect_err(|error| self.fail(writer, error.clone()))
    }

    fn try_submit(
        &mut self,
        writer: &mut RoomTranscriptionWriter,
        subject: &Subject,
    ) -> Result<()> {
        if writer.expression_bytes() > MAX_STREAM_BYTES {
            return Err(capacity("expression bytes"));
        }
        let mut effects = writer.take_effects();
        if effects.is_empty() {
            return Ok(());
        }
        let mut released = 0;
        if let Some(last) = self.queue.back_mut().filter(|last| last.subject == *subject) {
            effects = coalesce(std::mem::take(&mut last.effects).into_iter().chain(effects));
            released = last.ticket.bytes;
            self.queue.pop_back();
        }
        self.bytes -= released;
        let bytes = retained_bytes(&effects)
            .filter(|bytes| *bytes <= MAX_STREAM_BYTES.saturating_sub(self.bytes))
            .ok_or_else(|| capacity("queued text bytes"))?;
        self.bytes += bytes;
        self.queue.push_back(Batch { effects, subject: subject.clone(), ticket: Ticket { bytes } });
        Ok(())
    }

    /// Close: the writer's final batch, allowed twice the byte bound since it carries the open
    /// segment's closing text, ends the backlog.
    pub fn finish(
        &mut self,
        writer: &mut RoomTranscriptionWriter,
        subject: &Subject,
    ) -> Result<()> {
        if self.failure.is_some() || self.closing {
            return Ok(());
        }
        self.closing = true;
        writer.close();
        let effects = writer.take_effects();
        if retained_bytes(&effects).is_none_or(|bytes| bytes > MAX_STREAM_BYTES * 2) {
            let error = capacity("final text bytes");
            self.fail(writer, error.clone());
            return Err(error);
        }
        self.queue.push_back(Batch {
            effects,
            subject: subject.clone(),
            ticket: Ticket { bytes: 0 },
        });
        Ok(())
    }

    /// The oldest batch, to perform; its bytes stay held until [`Self::performed`].
    pub fn pop_batch(&mut self) -> Option<Batch> {
        self.queue.pop_front()
    }

    /// Nothing more will be queued and nothing is left to perform.
    pub fn drained(&self) -> bool {
        self.failure.is_some() || (self.closing && self.queue.is_empty())
    }

    pub fn performed(&mut self, ticket: Ticket) {
        self.bytes -= ticket.bytes;
    }
}

/// Drops what a later effect of the queue supersedes, a non-final replacement stream or legacy
/// publication of a segment queued again later, and merges adjacent writes.
fn coalesce(
    effects: impl Iterator<Item = RoomTranscriptionEffect>,
) -> Vec<RoomTranscriptionEffect> {
    let effects: Vec<_> = effects.collect();
    let mut superseded = vec![false; effects.len()];
    let (mut opened, mut published) = (BTreeSet::new(), BTreeSet::new());
    for (at, effect) in effects.iter().enumerate().rev() {
        match effect {
            RoomTranscriptionEffect::OpenStream { segment, is_final: false, .. }
                if opened.contains(segment)
                    && matches!(
                        effects.get(at + 1),
                        Some(RoomTranscriptionEffect::Write { .. })
                    )
                    && matches!(
                        effects.get(at + 2),
                        Some(RoomTranscriptionEffect::CloseStream { is_final: false, .. })
                    ) =>
            {
                superseded[at..at + 3].fill(true);
            }
            RoomTranscriptionEffect::PublishLegacy { segment, is_final: false, .. }
                if published.contains(segment) =>
            {
                superseded[at] = true;
            }
            _ => {}
        }
        match effect {
            RoomTranscriptionEffect::OpenStream { segment, .. } => opened.insert(segment.clone()),
            RoomTranscriptionEffect::PublishLegacy { segment, .. } => {
                published.insert(segment.clone())
            }
            _ => false,
        };
    }
    let mut kept: Vec<RoomTranscriptionEffect> = Vec::with_capacity(effects.len());
    for (effect, superseded) in effects.into_iter().zip(superseded) {
        if superseded {
            continue;
        }
        match (kept.last_mut(), effect) {
            (
                Some(RoomTranscriptionEffect::Write { payload }),
                RoomTranscriptionEffect::Write { payload: more },
            ) => {
                payload.push_str(&more);
            }
            (_, effect) => kept.push(effect),
        }
    }
    kept
}

pub fn retained_bytes(effects: &[RoomTranscriptionEffect]) -> Option<usize> {
    effects.iter().try_fold(0usize, |total, effect| {
        let bytes =
            match effect {
                RoomTranscriptionEffect::OpenStream { segment, expression, .. } => segment
                    .capacity()
                    .checked_add(expression.as_ref().map_or(0, |(_, value)| value.capacity()))?,
                RoomTranscriptionEffect::Write { payload } => payload.capacity(),
                RoomTranscriptionEffect::CloseStream { expression, .. } => {
                    expression.as_ref().map_or(0, |(_, value)| value.capacity())
                }
                RoomTranscriptionEffect::PublishLegacy { segment, text, .. } => {
                    segment.capacity().checked_add(text.capacity())?
                }
            };
        total.checked_add(bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(admission: &mut Admission) -> Vec<RoomTranscriptionEffect> {
        std::iter::from_fn(|| admission.pop_batch()).flat_map(|batch| batch.effects).collect()
    }

    #[test]
    fn text_first_released_at_segment_end_keeps_its_reserved_identity() {
        let mut writer = RoomTranscriptionWriter::new(true, false, "SG_first".into());
        // was: a trailing blank was held by the markup stripper until the segment ended; the
        // transcript arrives settled now, so it is written at once, under the reserved identity
        writer.capture(&TranscriptChunk::new(" "));
        assert_eq!(
            writer.take_effects(),
            [
                RoomTranscriptionEffect::OpenStream {
                    segment: "SG_first".into(),
                    is_final: false,
                    expression: None,
                },
                RoomTranscriptionEffect::Write { payload: " ".into() },
                RoomTranscriptionEffect::PublishLegacy {
                    segment: "SG_first".into(),
                    text: " ".into(),
                    is_final: false,
                },
            ]
        );
        writer.end_segment("SG_next".into());
        assert_eq!(
            writer.take_effects(),
            [
                RoomTranscriptionEffect::CloseStream { is_final: true, expression: None },
                RoomTranscriptionEffect::PublishLegacy {
                    segment: "SG_first".into(),
                    text: " ".into(),
                    is_final: true,
                },
            ]
        );
        writer.capture(&TranscriptChunk::new("next"));
        assert!(matches!(
            &writer.take_effects()[0],
            RoomTranscriptionEffect::OpenStream { segment, .. } if segment == "SG_next"
        ));
    }

    #[test]
    fn disconnection_retires_the_open_identity_and_closed_outputs_cannot_reopen() {
        let mut writer = RoomTranscriptionWriter::new(true, false, "SG_first".into());
        writer.capture(&TranscriptChunk::new("first"));
        writer.clear_effects();
        writer.disconnected("SG_reconnected".into());
        assert_eq!(
            writer.take_effects(),
            [RoomTranscriptionEffect::CloseStream { is_final: true, expression: None }]
        );
        writer.capture(&TranscriptChunk::new("reconnected"));
        assert!(matches!(
            &writer.take_effects()[0],
            RoomTranscriptionEffect::OpenStream { segment, .. } if segment == "SG_reconnected"
        ));
        writer.close();
        writer.clear_effects();
        writer.end_segment("SG_unused".into());
        writer.disconnected("SG_also_unused".into());
        writer.capture(&TranscriptChunk::new("late"));
        assert!(writer.closed());
        assert!(writer.take_effects().is_empty());
    }

    // Catches: a fifth queued batch latching a capacity failure while the performer was busy,
    // which ended the room's transcript (and RoomIo with it) for the rest of the call.
    #[test]
    fn a_replacement_backlog_keeps_only_the_latest_text_of_each_segment() {
        let mut admission = Admission::default();
        let mut writer = RoomTranscriptionWriter::new(false, false, "SG_1".into());
        let subject = Subject::default();
        for text in ["the", "the cat", "the cat sat"] {
            let visible = TranscriptChunk::new(text);
            writer.capture(&visible);
            admission.submit(&mut writer, &subject).unwrap();
        }
        for n in 0..8 {
            let visible = TranscriptChunk::new("next");
            writer.end_segment(format!("SG_{}", n + 2));
            writer.capture(&visible);
            admission.submit(&mut writer, &subject).unwrap();
        }
        let effects = queued(&mut admission);
        let opened: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                RoomTranscriptionEffect::OpenStream { segment, is_final, .. } => {
                    Some((segment.as_str(), *is_final))
                }
                _ => None,
            })
            .collect();
        assert_eq!(&opened[..2], [("SG_1", true), ("SG_2", true)]);
        assert_eq!(opened.last(), Some(&("SG_9", false)), "the open segment's latest interim");
        assert!(
            effects.contains(&RoomTranscriptionEffect::Write { payload: "the cat sat".into() })
        );
        assert!(!effects.contains(&RoomTranscriptionEffect::Write { payload: "the cat".into() }));
    }

    #[test]
    fn queued_deltas_merge_and_keep_every_word() {
        let mut admission = Admission::default();
        let mut writer = RoomTranscriptionWriter::new(true, false, "SG_1".into());
        let subject = Subject::default();
        for text in ["one ", "two ", "three"] {
            let visible = TranscriptChunk::new(text);
            writer.capture(&visible);
            admission.submit(&mut writer, &subject).unwrap();
        }
        assert_eq!(
            queued(&mut admission),
            [
                RoomTranscriptionEffect::OpenStream {
                    segment: "SG_1".into(),
                    is_final: false,
                    expression: None
                },
                RoomTranscriptionEffect::Write { payload: "one two three".into() },
                RoomTranscriptionEffect::PublishLegacy {
                    segment: "SG_1".into(),
                    text: "one two three".into(),
                    is_final: false
                },
            ]
        );
    }

    #[test]
    fn another_subjects_text_is_never_merged_into_the_queue() {
        let mut admission = Admission::default();
        let mut writer = RoomTranscriptionWriter::new(true, false, "SG_1".into());
        let first = Subject { participant: Some("first".into()), ..Subject::default() };
        writer.capture(&TranscriptChunk::new("hello"));
        admission.submit(&mut writer, &first).unwrap();
        writer.end_segment("SG_2".into());
        admission.submit(&mut writer, &Subject::default()).unwrap();
        assert!(admission.pop_batch().is_some_and(|batch| batch.subject == first));
        assert!(admission.pop_batch().is_some_and(|batch| batch.subject == Subject::default()));
    }

    #[test]
    fn a_backlog_past_its_bytes_fails_the_output_once() {
        let mut admission = Admission::default();
        let mut writer = RoomTranscriptionWriter::new(false, false, "SG_1".into());
        let subject = Subject::default();
        let mut failures = 0;
        for n in 0..4 {
            let visible = TranscriptChunk::new("x".repeat(MAX_STREAM_BYTES / 3));
            writer.capture(&visible);
            writer.end_segment(format!("SG_{}", n + 2));
            failures += usize::from(admission.submit(&mut writer, &subject).is_err());
        }
        assert_eq!(failures, 1, "the overflow fails the output; later text is dropped quietly");
        assert!(admission.drained());
        assert!(admission.pop_batch().is_none(), "a failed output performs nothing more");
    }
}
