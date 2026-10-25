//! `_output.py`'s `_ParticipantTranscriptionOutput`: a participant's transcript, as a text stream
//! and as a legacy transcription, and its host object.

use crate::room_io::{
    find_micro_track_id, is_local_proxy_participant, local_micro_track_id, publishes_on_behalf_of,
};
use async_trait::async_trait;
use lk_agents_abi::AbortHandle;
use lk_agents_abi::tts::markup::{ExpressiveTag, expression_attribute};
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::voice::io::{IoError, TextOutput};
use lk_agents_abi::{
    ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID, Monotone, Spawner, SpawnerExt, TOPIC_TRANSCRIPTION,
    TimedString,
};
use lk_agents_rtc::{
    ParticipantIdentity, RoomEvent, RtcRoom, StreamTextOptions, TextStreamWriter, TrackSid,
    TrackSource, Transcription, TranscriptionSegment,
};
use lk_agents_voice::transcription::{Visible, VisibleStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Deterministic, for tests and for a host that has no reason to care.
#[derive(Debug, Default)]
pub struct CountingSegmentIds(AtomicU64);

impl CountingSegmentIds {
    fn next_segment_id(&self) -> String {
        format!("SG_{}", self.0.fetch_add(1, Ordering::Relaxed))
    }
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
enum WriterEffect {
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
enum Writing {
    /// No segment open; the next visible text starts one.
    Idle,
    /// A segment has visible text; in delta mode its stream is open.
    Segment,
    Closed,
}

/// The writer's state: one segment's text and tags, and the performer's queue.
struct Writer {
    /// Deltas accumulate into one stream per segment (the agent's transcript); otherwise each text
    /// is a full replacement in its own short stream (the user's).
    is_delta_stream: bool,
    json_format: bool,
    writing: Writing,
    segment: String,
    /// The segment's visible text: accumulated for deltas, the latest for replacements.
    text: String,
    tags: Vec<ExpressiveTag>,
    /// The one place a delta becomes visible text. Per segment; replacements split whole.
    visible: VisibleStream,
    effects: Option<mpsc::UnboundedSender<WriterEffect>>,
}

impl Writer {
    fn send(&self, effect: WriterEffect) {
        if let Some(tx) = &self.effects {
            let _ = tx.send(effect);
        }
    }

    fn visible_of(&mut self, raw: &str) -> Visible {
        if self.is_delta_stream {
            self.visible.push(raw)
        } else {
            // A replacement is whole text every time; don't hold a trailing space back.
            Visible::of(raw)
        }
    }

    /// Visible text: the first of a segment opens it, the rest extend (or replace) it. Blank text
    /// (a tag still held, a markup-only chunk) opens nothing.
    fn capture(
        &mut self,
        visible: Visible,
        timing: Option<&TimedString>,
        ids: &CountingSegmentIds,
    ) {
        if self.writing == Writing::Closed {
            return;
        }
        self.tags.extend(visible.tags.iter().cloned());
        if visible.is_blank() {
            return;
        }
        let opening = self.writing != Writing::Segment;
        if opening {
            self.segment = ids.next_segment_id();
            self.text.clear();
        }
        let payload = encode(&visible.text, self.json_format, timing);
        let segment = self.segment.clone();
        if self.is_delta_stream {
            self.text.push_str(&visible.text);
            if opening {
                // The leading expression also goes on the opening header, so a frontend can
                // colour the turn at once.
                let expression = expression_attribute(&self.tags);
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
    fn end_segment(&mut self, ids: &CountingSegmentIds) {
        if self.is_delta_stream {
            let held = self.visible.flush();
            if !held.is_blank() {
                self.capture(held, None, ids);
            }
        }
        match self.writing {
            Writing::Closed => {}
            Writing::Idle => self.tags.clear(),
            Writing::Segment => self.end(Writing::Idle),
        }
    }

    /// The room disconnected: an open segment is closed now, not leaked (a dropped host stream
    /// stays open server-side).
    fn disconnected(&mut self) {
        if self.writing == Writing::Segment {
            let expression = expression_attribute(&self.tags);
            self.send(WriterEffect::CloseStream { is_final: true, expression });
            self.reset(Writing::Idle);
        }
    }

    fn close(&mut self) {
        match self.writing {
            Writing::Closed => {}
            Writing::Idle => self.writing = Writing::Closed,
            Writing::Segment => self.end(Writing::Closed),
        }
    }

    fn end(&mut self, next: Writing) {
        let expression = expression_attribute(&self.tags);
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

    fn reset(&mut self, next: Writing) {
        self.text.clear();
        self.tags.clear();
        self.writing = next;
    }
}

#[derive(Default)]
struct Subject {
    participant: Option<ParticipantIdentity>,
    /// Who the deprecated leg publishes as: the avatar worker's track when one publishes for them.
    represented_by: Option<ParticipantIdentity>,
    /// The deprecated leg's track: the subject's microphone, or the proxy's.
    legacy_track: Option<TrackSid>,
    /// The stream leg's track hint: the subject's own microphone, never the proxy's.
    stream_track: Option<TrackSid>,
}

impl Subject {
    fn set_participant(&mut self, room: &dyn RtcRoom, participant: Option<ParticipantIdentity>) {
        self.participant = participant.clone();
        self.represented_by = participant.clone();
        self.legacy_track = None;
        self.stream_track = None;
        let Some(target) = participant else { return };
        self.stream_track = find_micro_track_id(room, &target);
        if target == room.local().identity() {
            self.legacy_track = local_micro_track_id(room.local().as_ref());
        }
        if self.legacy_track.is_none() {
            for p in room.remote_participants() {
                let attrs = p.attributes();
                if !is_local_proxy_participant(
                    Some(&target),
                    p.identity(),
                    publishes_on_behalf_of(&attrs),
                ) {
                    continue;
                }
                if let Some(sid) = p
                    .track_publications()
                    .into_iter()
                    .find(|t| t.source() == TrackSource::Microphone)
                    .map(|t| t.sid().clone())
                {
                    self.legacy_track = Some(sid);
                    self.represented_by = Some(p.identity().clone());
                    break;
                }
            }
        }
    }

    fn on_track_published(
        &mut self,
        owner: &ParticipantIdentity,
        owner_attributes: &HashMap<String, String>,
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

    fn on_local_track_published(
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

/// A transcription stream's header: final or not, and the subject's microphone when known.
fn stream_attributes(is_final: bool, track: Option<&TrackSid>) -> HashMap<String, String> {
    let mut attributes = HashMap::new();
    attributes.insert(ATTRIBUTE_TRANSCRIPTION_FINAL.to_owned(), is_final.to_string());
    if let Some(tid) = track {
        attributes.insert(ATTRIBUTE_TRANSCRIPTION_TRACK_ID.to_owned(), tid.to_string());
    }
    attributes
}

struct Performer {
    room: Arc<dyn RtcRoom>,
    subject: Arc<Mutex<Subject>>,
    writer: Option<Box<dyn TextStreamWriter>>,
}

impl Performer {
    async fn perform(&mut self, effect: WriterEffect) {
        match effect {
            WriterEffect::OpenStream { segment, is_final, expression } => {
                let (participant, track) = {
                    let s = self.subject.lock().unwrap();
                    (s.participant.clone(), s.stream_track.clone())
                };
                let Some(participant) = participant else { return };
                if !self.room.connection_state().is_connected() {
                    return;
                }
                let mut attributes = stream_attributes(is_final, track.as_ref());
                attributes.insert(ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID.to_owned(), segment);
                if let Some((k, v)) = expression {
                    attributes.insert(k.to_owned(), v);
                }
                let opened = self
                    .room
                    .local()
                    .stream_text(StreamTextOptions {
                        topic: TOPIC_TRANSCRIPTION.to_owned(),
                        // On behalf of the subject, or a user transcript renders on the agent's side.
                        sender_identity: Some(participant),
                        attributes,
                    })
                    .await;
                // A failure is dropped, as in the reference; it must not stop the agent talking.
                if let Ok(w) = opened {
                    self.writer = Some(w);
                }
            }
            WriterEffect::Write { payload } => {
                if let Some(w) = &self.writer {
                    let _ = w.write(&payload).await;
                }
            }
            WriterEffect::CloseStream { is_final, expression } => {
                let Some(w) = self.writer.take() else { return };
                let track = self.subject.lock().unwrap().stream_track.clone();
                let mut attributes = stream_attributes(is_final, track.as_ref());
                if let Some((k, v)) = expression {
                    attributes.entry(k.to_owned()).or_insert(v);
                }
                let _ = w.close(attributes).await;
            }
            WriterEffect::PublishLegacy { segment, text, is_final } => {
                let (identity, track) = {
                    let s = self.subject.lock().unwrap();
                    (s.represented_by.clone().or(s.participant.clone()), s.legacy_track.clone())
                };
                let (Some(identity), Some(track)) = (identity, track) else { return };
                if !self.room.connection_state().is_connected() {
                    return;
                }
                let _ = self
                    .room
                    .local()
                    .publish_transcription(Transcription {
                        participant_identity: identity,
                        track_sid: track,
                        segments: vec![TranscriptionSegment {
                            id: segment,
                            text,
                            start_time: 0,
                            end_time: 0,
                            language: String::new(),
                            is_final,
                        }],
                    })
                    .await;
            }
        }
    }
}

/// `_ParticipantTranscriptionOutput`: the `io.TextOutput` in front of the writer; its effects are
/// performed in order by one task, off the caller.
pub struct ParticipantTranscriptionOutput {
    room: Arc<dyn RtcRoom>,
    ids: Arc<CountingSegmentIds>,
    writer: Mutex<Writer>,
    subject: Arc<Mutex<Subject>>,
    performer_done: Monotone<bool>,
    next_in_chain: Option<Arc<dyn TextOutput>>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl ParticipantTranscriptionOutput {
    /// `is_delta_stream`: the agent's transcript arrives as deltas, the user's as full replacements.
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        ids: Arc<CountingSegmentIds>,
        is_delta_stream: bool,
        json_format: bool,
        next_in_chain: Option<Arc<dyn TextOutput>>,
    ) -> Arc<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let subject = Arc::new(Mutex::new(Subject::default()));
        let this = Arc::new(ParticipantTranscriptionOutput {
            room: room.clone(),
            ids,
            writer: Mutex::new(Writer {
                is_delta_stream,
                json_format,
                writing: Writing::Idle,
                segment: String::new(),
                text: String::new(),
                tags: Vec::new(),
                visible: VisibleStream::new(),
                effects: Some(tx),
            }),
            subject: subject.clone(),
            performer_done: Monotone::new(false),
            next_in_chain,
            tasks: Mutex::new(Vec::new()),
        });
        let mut performer = Performer { room, subject, writer: None };
        let done = this.clone();
        let handle = spawner.spawn_unboxed(async move {
            while let Some(effect) = rx.recv().await {
                performer.perform(effect).await;
            }
            done.performer_done.set(true);
        });
        this.tasks.lock().unwrap().push(handle);
        this
    }

    /// The transcript's subject changed. The segment in progress ends; the tracks re-resolve.
    pub async fn set_participant(&self, participant: Option<ParticipantIdentity>) {
        self.subject.lock().unwrap().set_participant(self.room.as_ref(), participant);
        self.writer.lock().unwrap().end_segment(&self.ids);
    }

    pub async fn handle_event(&self, event: &RoomEvent) {
        match event {
            RoomEvent::TrackPublished { participant, publication } => {
                let attrs = participant.attributes();
                self.subject.lock().unwrap().on_track_published(
                    participant.identity(),
                    &attrs,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::LocalTrackPublished { publication } => {
                let local = self.room.local().identity();
                self.subject.lock().unwrap().on_local_track_published(
                    &local,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::ConnectionStateChanged { state } => {
                if !state.is_connected() {
                    self.writer.lock().unwrap().disconnected();
                }
            }
            RoomEvent::Close { reason: _ } => self.writer.lock().unwrap().disconnected(),
            RoomEvent::TrackSubscribed { participant: _, publication: _ }
            | RoomEvent::TrackUnpublished { participant: _, publication: _ }
            | RoomEvent::ParticipantConnected { participant: _ }
            | RoomEvent::ParticipantDisconnected { participant: _, reason: _ } => {}
        }
    }

    pub fn start(self: &Arc<Self>, room: &dyn RtcRoom, spawner: &dyn Spawner) {
        let mut events = room.events();
        let this = self.clone();
        let handle = spawner.spawn_unboxed(async move {
            while let Some(ev) = events.recv().await {
                this.handle_event(&ev).await;
            }
        });
        self.tasks.lock().unwrap().push(handle);
    }

    /// Waits for the close, since an open writer holds a server-side stream.
    pub async fn aclose(&self) {
        {
            let mut writer = self.writer.lock().unwrap();
            writer.close();
            writer.effects.take();
        }
        self.performer_done.wait_for(|set| *set).await;
        for h in self.tasks.lock().unwrap().drain(..) {
            h.abort();
        }
    }
}

#[async_trait]
impl TextOutput for ParticipantTranscriptionOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next_in_chain.clone()
    }

    async fn capture_text(&self, text: &str) -> Result<()> {
        self.capture_timed(&TimedString::new(text)).await
    }

    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        let shown = {
            let mut writer = self.writer.lock().unwrap();
            let visible = writer.visible_of(&text.text);
            let shown = visible.text.clone();
            writer.capture(visible, Some(text), &self.ids);
            shown
        };
        if let Some(next) = &self.next_in_chain
            && !shown.is_empty()
        {
            // The host's sink sees the same visible text the room does.
            next.capture_timed(&TimedString { text: shown, ..text.clone() }).await?;
        }
        Ok(())
    }

    fn flush(&self) {
        self.writer.lock().unwrap().end_segment(&self.ids);
        if let Some(next) = &self.next_in_chain {
            next.flush();
        }
    }
}

/// `_core.RoomTranscriptionOutput`: a participant's transcription leg as a host object (the
/// agent's, or a standalone one over a host room).
pub struct RoomTranscriptionLeg {
    inner: Arc<ParticipantTranscriptionOutput>,
    /// Built by the constructor, so closed by it; the session's own leg is its `RoomIo`'s.
    standalone: bool,
}

impl RoomTranscriptionLeg {
    pub fn of(inner: Arc<ParticipantTranscriptionOutput>) -> Self {
        RoomTranscriptionLeg { inner, standalone: false }
    }
}

#[lk_agents_abi::host(object(name = "RoomTranscriptionOutput", cell = "arc"))]
impl RoomTranscriptionLeg {
    /// A standalone leg over a host room (the reference's constructor). `next_in_chain` is a host
    /// `io.TextOutput` the text is also handed to.
    #[host(new(
        kw_only = "is_delta_stream",
        defaults(is_delta_stream = true, json_format = false)
    ))]
    pub fn standalone(
        room: Arc<dyn RtcRoom>,
        is_delta_stream: bool,
        participant: Option<ParticipantIdentity>,
        next_in_chain: Option<Arc<dyn TextOutput>>,
        json_format: bool,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let inner = ParticipantTranscriptionOutput::new(
            room.clone(),
            spawner.clone(),
            Arc::new(CountingSegmentIds::default()),
            is_delta_stream,
            json_format,
            next_in_chain,
        );
        inner.start(room.as_ref(), spawner.as_ref());
        if let Some(identity) = participant {
            let leg = inner.clone();
            spawner.spawn(Box::pin(async move { leg.set_participant(Some(identity)).await }));
        }
        RoomTranscriptionLeg { inner, standalone: true }
    }

    /// Point the leg at `identity` (`None` unlinks).
    #[host(method)]
    pub async fn set_participant(&self, #[host(required)] identity: Option<ParticipantIdentity>) {
        self.inner.set_participant(identity).await;
    }

    #[host(method)]
    pub async fn capture_text(&self, text: String) -> std::result::Result<(), IoError> {
        TextOutput::capture_text(&*self.inner, &text).await
    }

    #[host(method)]
    pub fn flush(&self) {
        TextOutput::flush(&*self.inner);
    }

    #[host(method)]
    pub async fn aclose(&self) {
        if self.standalone {
            self.inner.aclose().await;
        }
    }
}
