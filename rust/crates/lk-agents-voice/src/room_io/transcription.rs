//! Transcript visibility, segment framing and subject routing over supplied records.

use super::ingress::{MAX_INCOMING_STREAMS, MAX_STREAM_BYTES};
use super::{is_local_proxy_participant, publishes_on_behalf_of};
use crate::transcription::{Visible, VisibleStream};
use crate::tts::markup::expression_attribute;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, TrackSid, TrackSource};
use lk_agents_abi::{ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_TRACK_ID, TimedString};
use std::collections::BTreeMap;

/// The microphone view observed by the performer when a transcript's subject changes.
pub struct ParticipantTracks {
    pub identity: ParticipantIdentity,
    pub attributes: BTreeMap<String, String>,
    pub microphone: Option<TrackSid>,
}

/// Wrap a visible chunk for the wire: protojson of `agent_pb.TimedString`, default fields omitted,
/// newline-delimited.
fn encode(clean_text: &str, json_format: bool, timing: Option<&TimedString>) -> String {
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

/// A replacement is its own short stream; everything else closes final.
#[derive(Debug, Clone, PartialEq)]
pub enum WriterEffect {
    /// Open the segment's text stream; only the replacement mode's closing stream opens final.
    OpenStream {
        segment: String,
        is_final: bool,
        expression: Option<(&'static str, String)>,
    },
    Write {
        payload: String,
    },
    CloseStream {
        is_final: bool,
        expression: Option<(&'static str, String)>,
    },
    /// Publish the segment's text so far on the deprecated API.
    PublishLegacy {
        segment: String,
        text: String,
        is_final: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writing {
    /// No segment open; the next visible text starts one.
    Idle,
    /// A segment has visible text; in delta mode its stream is open.
    Segment,
    Closed,
}

/// The writer's state: one segment's text and tags, and the performer's queue.
pub struct Writer {
    /// Deltas accumulate into one stream per segment (the agent's transcript); otherwise each text
    /// is a full replacement in its own short stream (the user's).
    is_delta_stream: bool,
    json_format: bool,
    writing: Writing,
    segment: String,
    /// The segment's visible text: accumulated for deltas, the latest for replacements.
    text: String,
    expression: Option<(&'static str, String)>,
    /// The one place a delta becomes visible text. Per segment; replacements split whole.
    visible: VisibleStream,
    pending: Vec<WriterEffect>,
    raw_bytes: usize,
}

impl Writer {
    pub fn new(is_delta_stream: bool, json_format: bool) -> Self {
        Self {
            is_delta_stream,
            json_format,
            writing: Writing::Idle,
            segment: String::new(),
            text: String::new(),
            expression: None,
            visible: VisibleStream::new(),
            pending: Vec::new(),
            raw_bytes: 0,
        }
    }

    pub fn needs_segment(&self) -> bool {
        self.writing == Writing::Idle
    }
    pub fn closed(&self) -> bool {
        self.writing == Writing::Closed
    }
    pub fn take_effects(&mut self) -> Vec<WriterEffect> {
        std::mem::take(&mut self.pending)
    }
    pub fn clear_effects(&mut self) {
        self.pending.clear();
    }
    pub fn expression_bytes(&self) -> usize {
        self.expression.as_ref().map_or(0, |(_, value)| value.capacity())
    }
    pub fn admit_text(&mut self, text: &str) -> Result<()> {
        // A closed output takes no more text; `capture` drops it.
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

    fn send(&mut self, effect: WriterEffect) {
        self.pending.push(effect);
    }

    pub fn visible_of(&mut self, raw: &str) -> Visible {
        if self.is_delta_stream {
            self.visible.push(raw)
        } else {
            // A replacement is whole text every time; don't hold a trailing space back.
            Visible::of(raw)
        }
    }

    /// Visible text: the first of a segment opens it, the rest extend (or replace) it. Blank text
    /// (a tag still held, a markup-only chunk) opens nothing.
    pub fn capture(
        &mut self,
        visible: Visible,
        timing: Option<&TimedString>,
        next_segment: Option<String>,
    ) {
        if self.writing == Writing::Closed {
            return;
        }
        if self.expression.is_none() {
            self.expression = expression_attribute(&visible.tags);
        }
        if visible.is_blank() {
            return;
        }
        let opening = self.writing != Writing::Segment;
        if opening {
            self.segment = next_segment.expect("opening text has a supplied segment id");
            self.text.clear();
        }
        let payload = encode(&visible.text, self.json_format, timing);
        let segment = self.segment.clone();
        if self.is_delta_stream {
            self.text.push_str(&visible.text);
            if opening {
                // The leading expression also goes on the opening header, so a frontend can
                // colour the turn at once.
                let expression = self.expression.clone();
                self.send(WriterEffect::OpenStream { segment, is_final: false, expression });
            }
            self.send(WriterEffect::Write { payload });
        } else {
            self.text = visible.text;
            self.send(WriterEffect::OpenStream { segment, is_final: false, expression: None });
            self.send(WriterEffect::Write { payload });
            self.send(WriterEffect::CloseStream { is_final: false, expression: None });
        }
        let (segment, text) = (self.segment.clone(), self.text.clone());
        self.send(WriterEffect::PublishLegacy { segment, text, is_final: false });
        self.writing = Writing::Segment;
    }

    /// The segment's text is complete: whatever the stripper held is visible now, then the end.
    pub fn end_segment(&mut self, next_segment: Option<String>) {
        if self.is_delta_stream {
            let held = self.visible.flush();
            if !held.is_blank() {
                self.capture(held, None, next_segment);
            }
        }
        match self.writing {
            Writing::Closed => {}
            Writing::Idle => self.reset(Writing::Idle),
            Writing::Segment => self.end(Writing::Idle),
        }
    }

    /// The room disconnected: an open segment is closed now, not leaked (a dropped host stream
    /// stays open server-side).
    pub fn disconnected(&mut self) {
        if self.writing == Writing::Segment {
            let expression = self.expression.clone();
            self.send(WriterEffect::CloseStream { is_final: true, expression });
            self.reset(Writing::Idle);
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
            self.send(WriterEffect::CloseStream { is_final: true, expression });
        } else {
            // The final replacement opens already final with the whole latest text.
            let payload = encode(&self.text, self.json_format, None);
            self.send(WriterEffect::OpenStream {
                segment: segment.clone(),
                is_final: true,
                expression: None,
            });
            self.send(WriterEffect::Write { payload });
            self.send(WriterEffect::CloseStream { is_final: true, expression: None });
        }
        self.send(WriterEffect::PublishLegacy { segment, text: self.text.clone(), is_final: true });
        self.reset(next);
    }

    pub fn reset(&mut self, next: Writing) {
        self.text.clear();
        self.expression.take();
        self.writing = next;
        self.visible = VisibleStream::new();
        self.raw_bytes = 0;
    }
}

#[derive(Clone, Default)]
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

pub fn stream_attributes(is_final: bool, track: Option<&TrackSid>) -> BTreeMap<String, String> {
    let mut attributes = BTreeMap::new();
    attributes.insert(ATTRIBUTE_TRANSCRIPTION_FINAL.to_owned(), is_final.to_string());
    if let Some(tid) = track {
        attributes.insert(ATTRIBUTE_TRANSCRIPTION_TRACK_ID.to_owned(), tid.to_string());
    }
    attributes
}

/// Whether the performer may abandon this effect at a failure or the cleanup deadline: a stream
/// open and its consuming close keep their SDK outcome.
pub fn cancellable(effect: &WriterEffect) -> bool {
    matches!(effect, WriterEffect::Write { .. } | WriterEffect::PublishLegacy { .. })
}

/// One admitted batch and the capacity it holds until performed.
pub struct Batch {
    pub effects: Vec<WriterEffect>,
    pub ticket: Ticket,
}

/// Text bytes held until the batch is performed, and stream slots reserved for its opens. The
/// close's final batch reserves its open's slot only when it opens.
#[derive(Debug)]
pub struct Ticket {
    bytes: usize,
    streams: Option<usize>,
}

fn capacity(resource: &str) -> IoError {
    IoError::TranscriptionCapacity(resource.into())
}

/// The bounded work the transcription performer holds: queued and in-flight text bytes, open and
/// reserved streams, queued batches; and the first failure, after which nothing is admitted.
#[derive(Default)]
pub struct Admission {
    bytes: usize,
    streams: usize,
    batches: usize,
    failure: Option<IoError>,
}

impl Admission {
    pub fn failure(&self) -> Option<&IoError> {
        self.failure.as_ref()
    }

    /// Latches the first failure and drops the writer's pending effects; returns the latched one.
    pub fn fail(&mut self, writer: &mut Writer, error: IoError) -> IoError {
        writer.clear_effects();
        writer.reset(Writing::Closed);
        self.failure.get_or_insert(error).clone()
    }

    /// The writer's pending effects as one batch, `None` when there are none.
    pub fn admit(&mut self, writer: &mut Writer) -> Result<Option<Batch>> {
        let admitted = self.try_admit(writer);
        admitted.map_err(|error| self.fail(writer, error))
    }

    fn try_admit(&mut self, writer: &mut Writer) -> Result<Option<Batch>> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if writer.expression_bytes() > MAX_STREAM_BYTES {
            return Err(capacity("expression bytes"));
        }
        let effects = writer.take_effects();
        if effects.is_empty() {
            return Ok(None);
        }
        let bytes = retained_bytes(&effects)
            .filter(|bytes| *bytes <= MAX_STREAM_BYTES.saturating_sub(self.bytes))
            .ok_or_else(|| capacity("queued text bytes"))?;
        let opens = effects
            .iter()
            .filter(|effect| matches!(effect, WriterEffect::OpenStream { .. }))
            .count();
        if self.batches == MAX_INCOMING_STREAMS {
            return Err(capacity("queued batches"));
        }
        if opens > MAX_INCOMING_STREAMS - self.streams {
            return Err(capacity("streams"));
        }
        self.bytes += bytes;
        self.streams += opens;
        self.batches += 1;
        Ok(Some(Batch { effects, ticket: Ticket { bytes, streams: Some(opens) } }))
    }

    /// Close's final batch, allowed twice the byte bound since it carries the open segment's
    /// closing text.
    pub fn finish(&mut self, writer: &mut Writer) -> Result<Batch> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        writer.close();
        let effects = writer.take_effects();
        if retained_bytes(&effects).is_none_or(|bytes| bytes > MAX_STREAM_BYTES * 2) {
            return Err(self.fail(writer, capacity("final text bytes")));
        }
        Ok(Batch { effects, ticket: Ticket { bytes: 0, streams: None } })
    }

    /// A stream of `ticket`'s batch opens: it holds a slot until [`Self::stream_closed`].
    pub fn open_stream(&mut self, ticket: &mut Ticket) -> Result<()> {
        match &mut ticket.streams {
            Some(reserved) => {
                *reserved -= 1;
                Ok(())
            }
            None if self.streams < MAX_INCOMING_STREAMS => {
                self.streams += 1;
                Ok(())
            }
            None => Err(capacity("streams")),
        }
    }

    pub fn stream_closed(&mut self) {
        self.streams -= 1;
    }

    /// The batch was performed or abandoned: its bytes and unopened stream slots are free.
    pub fn performed(&mut self, ticket: Ticket) {
        self.bytes -= ticket.bytes;
        if let Some(reserved) = ticket.streams {
            self.streams -= reserved;
            self.batches -= 1;
        }
    }
}

pub fn retained_bytes(effects: &[WriterEffect]) -> Option<usize> {
    effects.iter().try_fold(0usize, |total, effect| {
        let bytes = match effect {
            WriterEffect::OpenStream { segment, expression, .. } => segment
                .capacity()
                .checked_add(expression.as_ref().map_or(0, |(_, value)| value.capacity()))?,
            WriterEffect::Write { payload } => payload.capacity(),
            WriterEffect::CloseStream { expression, .. } => {
                expression.as_ref().map_or(0, |(_, value)| value.capacity())
            }
            WriterEffect::PublishLegacy { segment, text, .. } => {
                segment.capacity().checked_add(text.capacity())?
            }
        };
        total.checked_add(bytes)
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    /// A writer with one new segment's text pending: a stream open and a write.
    fn segment(writer: &mut Writer, n: usize) {
        let visible = writer.visible_of("hello");
        writer.capture(visible, None, Some(format!("SG_{n}")));
        writer.end_segment(Some(format!("SG_{}", n + 1)));
    }

    #[test]
    fn open_streams_bound_admission_until_they_close() {
        let mut admission = Admission::default();
        let mut writer = Writer::new(true, false);
        let mut tickets = Vec::new();
        for n in 0..MAX_INCOMING_STREAMS {
            segment(&mut writer, n);
            let mut batch = admission.admit(&mut writer).unwrap().unwrap();
            admission.open_stream(&mut batch.ticket).unwrap();
            tickets.push(batch.ticket);
        }
        for ticket in tickets {
            admission.performed(ticket);
        }
        segment(&mut writer, 99);
        assert!(matches!(
            admission.admit(&mut writer),
            Err(IoError::TranscriptionCapacity(resource)) if resource == "streams"
        ));
        assert!(admission.failure().is_some(), "a refusal latches the output's failure");
    }

    #[test]
    fn a_performed_batch_frees_its_bytes_and_unopened_slots() {
        let mut admission = Admission::default();
        let mut writer = Writer::new(true, false);
        segment(&mut writer, 0);
        let batch = admission.admit(&mut writer).unwrap().unwrap();
        admission.performed(batch.ticket);
        for n in 1..=MAX_INCOMING_STREAMS {
            segment(&mut writer, n);
            let batch = admission.admit(&mut writer).unwrap().unwrap();
            admission.performed(batch.ticket);
        }
    }

    #[test]
    fn the_final_batch_takes_its_stream_slot_when_it_opens() {
        let mut admission = Admission::default();
        let mut writer = Writer::new(true, false);
        let mut tickets = Vec::new();
        for n in 0..MAX_INCOMING_STREAMS {
            segment(&mut writer, n);
            let mut batch = admission.admit(&mut writer).unwrap().unwrap();
            admission.open_stream(&mut batch.ticket).unwrap();
            tickets.push(batch.ticket);
        }
        let mut last = admission.finish(&mut writer).unwrap();
        assert!(admission.open_stream(&mut last.ticket).is_err());
        admission.stream_closed();
        admission.open_stream(&mut last.ticket).unwrap();
    }
}
