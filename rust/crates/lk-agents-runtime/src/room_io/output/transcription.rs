//! `_output.py`'s `_ParticipantTranscriptionOutput`: a participant's transcript, as a text stream
//! and as a legacy transcription, and its host object.

use lk_agents_abi::AbortHandle;

pub use super::segment_ids::SegmentIds;
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{
    ParticipantIdentity, RoomEvent, RtcRoom, StreamTextOptions, TextStreamWriter, TrackSid,
    TrackSource, Transcription, TranscriptionSegment,
};
use lk_agents_abi::{
    Clock, HOST_TEARDOWN_TIMEOUT, Monotone, Spawner, SpawnerExt, TOPIC_TRANSCRIPTION, TimedString,
    Ts,
};
use lk_agents_voice::room_io::transcription::{
    Admission, Batch, ParticipantTracks, Subject, Writer, WriterEffect, cancellable,
    close_attributes, open_attributes,
};

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct WriterState {
    core: Writer,
    admission: Admission,
    deadline: Option<Ts>,
    subject: Subject,
}

fn subject_for_participant(
    room: &dyn RtcRoom,
    participant: Option<ParticipantIdentity>,
) -> Result<Subject> {
    if participant.is_none() {
        return Ok(Subject::default());
    }
    let local = room.local();
    let local_id = local.identity()?;
    let mut peers = vec![ParticipantTracks {
        identity: local_id.clone(),
        attributes: BTreeMap::new(),
        microphone: local
            .track_publications()?
            .into_iter()
            .find(|publication| publication.source() == TrackSource::Microphone)
            .map(|publication| publication.sid().clone()),
    }];
    for peer in room.remote_participants()? {
        peers.push(ParticipantTracks {
            identity: peer.identity().clone(),
            attributes: peer.attributes()?,
            microphone: peer
                .track_publications()?
                .into_iter()
                .find(|publication| publication.source() == TrackSource::Microphone)
                .map(|publication| publication.sid().clone()),
        });
    }
    Ok(Subject::for_participant(participant, &local_id, &peers))
}

struct OpenWriter {
    writer: Box<dyn TextStreamWriter>,
    track: Option<TrackSid>,
}

struct Performer {
    room: Arc<dyn RtcRoom>,
    writer: Option<OpenWriter>,
}

impl Performer {
    async fn perform(&mut self, effect: WriterEffect, subject: &Subject) -> Result<()> {
        match effect {
            WriterEffect::OpenStream { segment, is_final, expression } => {
                let Some(participant) = subject.participant.clone() else {
                    return Ok(());
                };
                if !self.room.connection_state()?.is_connected() {
                    return Ok(());
                }
                let attributes =
                    open_attributes(segment, is_final, expression, subject.stream_track.as_ref());
                let writer = self
                    .room
                    .local()
                    .stream_text(StreamTextOptions {
                        topic: TOPIC_TRANSCRIPTION.to_owned(),
                        sender_identity: Some(participant),
                        attributes,
                    })
                    .await?;
                self.writer = Some(OpenWriter { writer, track: subject.stream_track.clone() });
            }
            WriterEffect::Write { payload } => {
                if let Some(open) = &self.writer {
                    open.writer.write(&payload).await?;
                }
            }
            WriterEffect::CloseStream { is_final, expression } => {
                if let Some(open) = self.writer.take() {
                    let attributes = close_attributes(is_final, expression, open.track.as_ref());
                    open.writer.close(attributes).await?;
                }
            }
            WriterEffect::PublishLegacy { segment, text, is_final } => {
                let Some((identity, track)) = subject.legacy_target() else { return Ok(()) };
                if !self.room.connection_state()?.is_connected() {
                    return Ok(());
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
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(open) = self.writer.take() {
            open.writer.close(close_attributes(true, None, open.track.as_ref())).await?;
        }
        Ok(())
    }
}

/// `_ParticipantTranscriptionOutput`: the `io.TextOutput` in front of the writer; its effects are
/// performed in order by one task, off the caller.
pub struct ParticipantTranscriptionOutput {
    room: Arc<dyn RtcRoom>,
    ids: Arc<SegmentIds>,
    clock: Arc<dyn Clock>,
    writer: Mutex<WriterState>,
    changed: Notify,
    performer_done: Monotone<bool>,
    next_in_chain: Option<Arc<dyn TextOutput>>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl ParticipantTranscriptionOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        ids: Arc<SegmentIds>,
        is_delta_stream: bool,
        json_format: bool,
        next_in_chain: Option<Arc<dyn TextOutput>>,
    ) -> Arc<Self> {
        let output = Arc::new(Self {
            room: room.clone(),
            ids,
            clock,
            writer: Mutex::new(WriterState {
                core: Writer::new(is_delta_stream, json_format),
                admission: Admission::default(),
                deadline: None,
                subject: Subject::default(),
            }),
            changed: Notify::new(),
            performer_done: Monotone::new(false),
            next_in_chain,
            tasks: Mutex::new(Vec::new()),
        });
        let owner = output.clone();
        let task = spawner.spawn_unboxed(async move {
            owner.run(Performer { room, writer: None }).await;
        });
        output.tasks.lock().unwrap().push(task);
        output
    }

    /// The output's own failure stays here: it is logged once, its text is dropped from then
    /// on, and [`Self::aclose`] reports it. Neither the speech nor the room fails with it.
    fn latched(&self, error: IoError) {
        log::warn!(target: "livekit.agents", "room transcription output failed, dropping its text: {error}");
        self.changed.notify_waiters();
    }

    fn fail(&self, error: IoError) {
        let mut writer = self.writer.lock().unwrap();
        if writer.admission.failure().is_some() {
            return;
        }
        let WriterState { core, admission, .. } = &mut *writer;
        admission.fail(core, error.clone());
        drop(writer);
        self.latched(error);
    }

    fn submit(&self, writer: &mut WriterState) {
        let WriterState { core, admission, subject, .. } = writer;
        match admission.submit(core, subject) {
            Ok(()) => self.changed.notify_waiters(),
            Err(error) => self.latched(error),
        }
    }

    fn status(&self) -> (Option<IoError>, Option<Ts>) {
        let writer = self.writer.lock().unwrap();
        (writer.admission.failure().cloned(), writer.deadline)
    }

    async fn effect(
        &self,
        performer: &mut Performer,
        effect: WriterEffect,
        subject: &Subject,
    ) -> Result<()> {
        let (failure, deadline) = self.status();
        if let Some(error) = failure {
            return Err(error);
        }
        if deadline.is_some_and(|deadline| self.clock.now() >= deadline) {
            return Err(IoError::Rtc("transcription cleanup deadline expired".into()));
        }
        let cancellable = cancellable(&effect);
        let action = performer.perform(effect, subject);
        if !cancellable {
            return action.await;
        }
        tokio::pin!(action);
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let (failure, deadline) = self.status();
            if let Some(error) = failure {
                return Err(error);
            }
            let elapsed = async {
                match deadline {
                    Some(at) => self.clock.wait_until(at).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                biased;
                () = &mut changed => {},
                result = &mut action => return result,
                () = elapsed => return Err(IoError::Rtc("transcription cleanup deadline expired".into())),
            }
        }
    }

    async fn batch(&self, performer: &mut Performer, batch: Batch) -> Result<()> {
        let Batch { effects, subject, ticket } = batch;
        let mut result = Ok(());
        for effect in effects {
            result = self.effect(performer, effect, &subject).await;
            if result.is_err() {
                break;
            }
        }
        self.writer.lock().unwrap().admission.performed(ticket);
        result
    }

    async fn run(self: Arc<Self>, mut performer: Performer) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let next = {
                let mut writer = self.writer.lock().unwrap();
                if writer.admission.drained() {
                    break;
                }
                writer.admission.next()
            };
            let Some(batch) = next else {
                changed.await;
                continue;
            };
            if let Err(error) = self.batch(&mut performer, batch).await {
                self.fail(error);
                break;
            }
        }
        // The worker, not a caller, owns this consuming close even beyond a waiter's deadline.
        if let Err(error) = performer.close().await {
            self.fail(error);
        }
        self.performer_done.set(true);
    }

    pub async fn set_participant(&self, participant: Option<ParticipantIdentity>) -> Result<()> {
        let subject = subject_for_participant(self.room.as_ref(), participant)?;
        let mut writer = self.writer.lock().unwrap();
        let segment = writer.core.needs_segment().then(|| self.ids.next_segment_id());
        writer.core.end_segment(segment);
        self.submit(&mut writer);
        writer.subject = subject;
        Ok(())
    }

    pub async fn handle_event(&self, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::TrackPublished { participant, publication } => {
                let attributes = participant.attributes()?;
                self.writer.lock().unwrap().subject.on_track_published(
                    participant.identity(),
                    &attributes,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::LocalTrackPublished { publication } => {
                let local = self.room.local().identity()?;
                self.writer.lock().unwrap().subject.on_local_track_published(
                    &local,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::ConnectionStateChanged { state } if !state.is_connected() => {
                let mut writer = self.writer.lock().unwrap();
                writer.core.disconnected();
                self.submit(&mut writer);
            }
            RoomEvent::Close { .. } => {
                let mut writer = self.writer.lock().unwrap();
                writer.core.disconnected();
                self.submit(&mut writer);
            }
            _ => {}
        }
        Ok(())
    }

    pub async fn aclose(&self) -> Result<()> {
        let requested_deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        let deadline = {
            let mut writer = self.writer.lock().unwrap();
            let WriterState { core, admission, subject, .. } = &mut *writer;
            if let Err(error) = admission.finish(core, subject) {
                self.latched(error);
            }
            *writer.deadline.get_or_insert(requested_deadline)
        };
        self.changed.notify_waiters();
        let completed = lk_agents_abi::before(
            self.clock.as_ref(),
            Some(deadline),
            self.performer_done.wait_for(|done| *done),
        )
        .await;
        if completed.is_err() {
            self.fail(IoError::Rtc("transcription cleanup deadline expired".into()));
        }
        if self.performer_done.get() {
            self.tasks.lock().unwrap().clear();
        }
        self.writer.lock().unwrap().admission.failure().cloned().map_or(Ok(()), Err)
    }
}

impl TextOutput for ParticipantTranscriptionOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next_in_chain.clone()
    }
    fn capture_text<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture_timed(&TimedString::new(text)).await })
    }

    fn capture_timed<'a>(
        &'a self,
        text: &'a TimedString,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let shown = {
                let mut writer = self.writer.lock().unwrap();
                let WriterState { core, admission, .. } = &mut *writer;
                match core.admit_text(&text.text) {
                    Ok(()) => {
                        let visible = core.visible_of(&text.text);
                        let shown = visible.text.clone();
                        let segment = core.needs_segment().then(|| self.ids.next_segment_id());
                        core.capture(visible, Some(text), segment);
                        self.submit(&mut writer);
                        shown
                    }
                    Err(error) => {
                        admission.fail(core, error.clone());
                        self.latched(error);
                        String::new()
                    }
                }
            };
            if let Some(next) = &self.next_in_chain
                && !shown.is_empty()
            {
                next.capture_timed(&TimedString { text: shown, ..text.clone() }).await?;
            }
            Ok(())
        })
    }

    fn flush(&self) {
        {
            let mut writer = self.writer.lock().unwrap();
            let segment = writer.core.needs_segment().then(|| self.ids.next_segment_id());
            writer.core.end_segment(segment);
            self.submit(&mut writer);
        }
        if let Some(next) = &self.next_in_chain {
            next.flush();
        }
    }
}

#[cfg(test)]
#[path = "transcription_tests.rs"]
mod tests;
