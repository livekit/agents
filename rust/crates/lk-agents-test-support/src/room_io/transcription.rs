//! Fake room text execution using the production transcript writer and subject-selection core.

use lk_agents_abi::voice::io::{IoError, Result, TextOutput};
use lk_agents_abi::voice::rtc::ports::{MAX_INCOMING_STREAMS, MAX_STREAM_BYTES};
use lk_agents_abi::voice::rtc::{
    ParticipantIdentity, RoomEvent, RtcRoom, StreamTextOptions, TextStreamWriter, TrackSid,
    TrackSource, Transcription, TranscriptionSegment,
};
use lk_agents_abi::{
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID, BoxFuture, Clock, Monotone, Spawner, SpawnerExt,
    TOPIC_TRANSCRIPTION, TimedString,
};
use lk_agents_voice::room_io::transcription::{
    ParticipantTracks, Subject, Writer, WriterEffect, Writing, retained_bytes, stream_attributes,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

/// Deterministic transcript identities supplied by the fake room owner.
#[derive(Default)]
pub struct SegmentIds(std::sync::atomic::AtomicU64);
impl SegmentIds {
    pub fn next_segment_id(&self) -> String {
        format!("SG_{}", self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

struct Batch {
    effects: Vec<WriterEffect>,
    subject: Subject,
    _bytes: Option<OwnedSemaphorePermit>,
    streams: Option<OwnedSemaphorePermit>,
}

struct State {
    writer: Writer,
    subject: Subject,
    sender: Option<mpsc::Sender<Batch>>,
    finish: Option<Batch>,
    error: Option<IoError>,
    deadline: Option<lk_agents_abi::Ts>,
}

pub struct RoomTranscriptionOutput {
    room: Arc<dyn RtcRoom>,
    ids: Arc<SegmentIds>,
    state: Mutex<State>,
    bytes: Arc<Semaphore>,
    streams: Arc<Semaphore>,
    done: Monotone<bool>,
    clock: Arc<dyn Clock>,
    next: Option<Arc<dyn TextOutput>>,
}

fn capacity(resource: &str) -> IoError {
    IoError::TranscriptionCapacity(resource.into())
}

impl RoomTranscriptionOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        ids: Arc<SegmentIds>,
        is_delta_stream: bool,
        json_format: bool,
        next: Option<Arc<dyn TextOutput>>,
    ) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(MAX_INCOMING_STREAMS);
        let output = Arc::new(Self {
            room,
            ids,
            clock,
            next,
            state: Mutex::new(State {
                writer: Writer::new(is_delta_stream, json_format),
                subject: Subject::default(),
                sender: Some(sender),
                finish: None,
                error: None,
                deadline: None,
            }),
            bytes: Arc::new(Semaphore::new(MAX_STREAM_BYTES)),
            streams: Arc::new(Semaphore::new(MAX_INCOMING_STREAMS)),
            done: Monotone::new(false),
        });
        let performer = output.clone();
        spawner.spawn_unboxed(async move {
            performer.run(receiver).await;
        });
        output
    }

    fn reject(&self, state: &mut State, error: IoError) -> IoError {
        let error = state.error.get_or_insert(error).clone();
        state.writer.clear_effects();
        state.writer.reset(Writing::Closed);
        state.sender.take();
        state.finish.take();
        error
    }

    fn submit(&self, state: &mut State) -> Result<()> {
        self.admit_batch(state).map_err(|error| self.reject(state, error))
    }

    fn admit_batch(&self, state: &mut State) -> Result<()> {
        if state.writer.expression_bytes() > MAX_STREAM_BYTES {
            return Err(capacity("expression bytes"));
        }
        let effects = state.writer.take_effects();
        if effects.is_empty() {
            return Ok(());
        }
        let bytes = retained_bytes(&effects)
            .filter(|bytes| *bytes <= MAX_STREAM_BYTES)
            .ok_or_else(|| capacity("queued text bytes"))?;
        let opens = effects
            .iter()
            .filter(|effect| matches!(effect, WriterEffect::OpenStream { .. }))
            .count();
        let sender = state
            .sender
            .clone()
            .ok_or_else(|| IoError::Rtc("transcription output is closed".into()))?;
        let queue = sender.try_reserve_owned().map_err(|_| capacity("queued batches"))?;
        let bytes = self
            .bytes
            .clone()
            .try_acquire_many_owned(bytes as u32)
            .map_err(|_| capacity("queued text bytes"))?;
        let streams = self
            .streams
            .clone()
            .try_acquire_many_owned(opens as u32)
            .map_err(|_| capacity("streams"))?;
        queue.send(Batch {
            effects,
            subject: state.subject.clone(),
            _bytes: Some(bytes),
            streams: Some(streams),
        });
        Ok(())
    }

    pub async fn set_participant(&self, participant: Option<ParticipantIdentity>) -> Result<()> {
        let local = self.room.local();
        let local_id = local.identity()?;
        let mut peers = vec![ParticipantTracks {
            identity: local_id.clone(),
            attributes: BTreeMap::new(),
            microphone: local
                .track_publications()?
                .into_iter()
                .find(|track| track.source() == TrackSource::Microphone)
                .map(|track| track.sid().clone()),
        }];
        for peer in self.room.remote_participants()? {
            peers.push(ParticipantTracks {
                identity: peer.identity().clone(),
                attributes: peer.attributes()?,
                microphone: peer
                    .track_publications()?
                    .into_iter()
                    .find(|track| track.source() == TrackSource::Microphone)
                    .map(|track| track.sid().clone()),
            });
        }
        let subject = Subject::for_participant(participant, &local_id, &peers);
        let mut state = self.state.lock().unwrap();
        if let Some(error) = state.error.as_ref() {
            return Err(error.clone());
        }
        let id = state.writer.needs_segment().then(|| self.ids.next_segment_id());
        state.writer.end_segment(id);
        self.submit(&mut state)?;
        state.subject = subject;
        Ok(())
    }

    pub async fn handle_event(&self, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::TrackPublished { participant, publication } => {
                let attributes = participant.attributes()?;
                self.state.lock().unwrap().subject.on_track_published(
                    participant.identity(),
                    &attributes,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::LocalTrackPublished { publication } => {
                let local = self.room.local().identity()?;
                self.state.lock().unwrap().subject.on_local_track_published(
                    &local,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::Close { .. } => {
                let mut state = self.state.lock().unwrap();
                state.writer.disconnected();
                self.submit(&mut state)?;
            }
            RoomEvent::ConnectionStateChanged { state } if !state.is_connected() => {
                let mut state = self.state.lock().unwrap();
                state.writer.disconnected();
                self.submit(&mut state)?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn run(&self, mut receiver: mpsc::Receiver<Batch>) {
        let mut open: Option<(Box<dyn TextStreamWriter>, Option<TrackSid>, OwnedSemaphorePermit)> =
            None;
        while let Some(batch) = receiver.recv().await {
            if self.state.lock().unwrap().error.is_some() {
                break;
            }
            if let Err(error) = self.perform(batch, &mut open).await {
                self.reject(&mut self.state.lock().unwrap(), error);
                break;
            }
        }
        receiver.close();
        while receiver.try_recv().is_ok() {}
        let finish = self.state.lock().unwrap().finish.take();
        if let Some(finish) = finish
            && let Err(error) = self.perform(finish, &mut open).await
        {
            self.reject(&mut self.state.lock().unwrap(), error);
        }
        if let Some((writer, track, _permit)) = open {
            if let Err(error) = writer.close(stream_attributes(true, track.as_ref())).await {
                self.reject(&mut self.state.lock().unwrap(), error.into());
            }
        }
        self.done.set(true);
    }

    async fn perform(
        &self,
        mut batch: Batch,
        open: &mut Option<(Box<dyn TextStreamWriter>, Option<TrackSid>, OwnedSemaphorePermit)>,
    ) -> Result<()> {
        for effect in batch.effects {
            match effect {
                WriterEffect::OpenStream { segment, is_final, expression } => {
                    let Some(participant) = batch.subject.participant.clone() else {
                        continue;
                    };
                    if !self.room.connection_state()?.is_connected() {
                        continue;
                    }
                    let mut attributes =
                        stream_attributes(is_final, batch.subject.stream_track.as_ref());
                    attributes.insert(ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID.to_owned(), segment);
                    if let Some((key, value)) = expression {
                        attributes.insert(key.to_owned(), value);
                    }
                    let permit = match &mut batch.streams {
                        Some(permits) => permits.split(1).expect("a batch admits each stream"),
                        None => self
                            .streams
                            .clone()
                            .try_acquire_owned()
                            .map_err(|_| capacity("streams"))?,
                    };
                    let writer = self
                        .room
                        .local()
                        .stream_text(StreamTextOptions {
                            topic: TOPIC_TRANSCRIPTION.to_owned(),
                            sender_identity: Some(participant),
                            attributes,
                        })
                        .await?;
                    *open = Some((writer, batch.subject.stream_track.clone(), permit));
                }
                WriterEffect::Write { payload } => {
                    if let Some((writer, _, _)) = open.as_ref() {
                        writer.write(&payload).await?;
                    }
                }
                WriterEffect::CloseStream { is_final, expression } => {
                    if let Some((writer, track, _permit)) = open.take() {
                        let mut attributes = stream_attributes(is_final, track.as_ref());
                        if let Some((key, value)) = expression {
                            attributes.insert(key.to_owned(), value);
                        }
                        writer.close(attributes).await?;
                    }
                }
                WriterEffect::PublishLegacy { segment, text, is_final } => {
                    let identity =
                        batch.subject.represented_by.clone().or(batch.subject.participant.clone());
                    let (Some(identity), Some(track)) =
                        (identity, batch.subject.legacy_track.clone())
                    else {
                        continue;
                    };
                    if !self.room.connection_state()?.is_connected() {
                        continue;
                    }
                    self.room
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
                        .await?;
                }
            }
        }
        Ok(())
    }

    pub async fn aclose(&self) -> Result<()> {
        let requested_deadline =
            self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let deadline = {
            let mut state = self.state.lock().unwrap();
            if state.sender.is_some() {
                state.writer.close();
                let effects = state.writer.take_effects();
                if retained_bytes(&effects).is_none_or(|bytes| bytes > MAX_STREAM_BYTES * 2) {
                    self.reject(&mut state, capacity("final text bytes"));
                } else {
                    state.finish = Some(Batch {
                        effects,
                        subject: state.subject.clone(),
                        _bytes: None,
                        streams: None,
                    });
                }
                state.sender.take();
            }
            *state.deadline.get_or_insert(requested_deadline)
        };
        if lk_agents_abi::before(
            self.clock.as_ref(),
            Some(deadline),
            self.done.wait_for(|done| *done),
        )
        .await
        .is_err()
        {
            self.reject(
                &mut self.state.lock().unwrap(),
                IoError::Rtc("transcription cleanup deadline expired".into()),
            );
        }
        self.state.lock().unwrap().error.clone().map_or(Ok(()), Err)
    }
}

impl TextOutput for RoomTranscriptionOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next.clone()
    }
    fn capture_text<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture_timed(&TimedString::new(text)).await })
    }
    fn capture_timed<'a>(&'a self, text: &'a TimedString) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let shown = {
                let mut state = self.state.lock().unwrap();
                if let Some(error) = &state.error {
                    return Err(error.clone());
                }
                if let Err(error) = state.writer.admit_text(&text.text) {
                    return Err(self.reject(&mut state, error));
                }
                let visible = state.writer.visible_of(&text.text);
                let shown = visible.text.clone();
                let id = state.writer.needs_segment().then(|| self.ids.next_segment_id());
                state.writer.capture(visible, Some(text), id);
                if let Err(error) = self.submit(&mut state) {
                    return Err(self.reject(&mut state, error));
                }
                shown
            };
            if let Some(next) = &self.next
                && !shown.is_empty()
            {
                next.capture_timed(&TimedString { text: shown, ..text.clone() }).await?;
            }
            Ok(())
        })
    }
    fn flush(&self) {
        let mut state = self.state.lock().unwrap();
        let id = state.writer.needs_segment().then(|| self.ids.next_segment_id());
        state.writer.end_segment(id);
        if let Err(error) = self.submit(&mut state) {
            self.reject(&mut state, error);
        }
        drop(state);
        if let Some(next) = &self.next {
            next.flush();
        }
    }
}
