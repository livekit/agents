//! `_output.py`'s `_ParticipantTranscriptionOutput`: a participant's transcript, as a text stream
//! and as a legacy transcription, and its host object.
//!
//! Executes RoomTranscriptionWriter's RoomTranscriptionEffect operations. Word pacing belongs
//! to TranscriptSync upstream; this adapter only publishes already-delivered transcript text.

use crate::monotone::Monotone;
use crate::telemetry;
use lk_agents_abi::OwnedTask;
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{
    ParticipantIdentity, RoomEvent, RtcRoom, StreamTextOptions, TextStreamWriter, TrackSid,
    TrackSource, Transcription, TranscriptionSegment,
};
use lk_agents_abi::{
    Clock, HOST_TEARDOWN_TIMEOUT, Spawner, SpawnerExt, TOPIC_TRANSCRIPTION, TranscriptChunk, Ts,
};
use lk_agents_voice::room_io::effects::RoomTranscriptionEffect;
use lk_agents_voice::room_io::machine::transcription::{
    Admission, Batch, ParticipantTracks, RoomTranscriptionWriter, Subject, cancellable,
    close_attributes, open_attributes,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Notify;

struct WriterState {
    core: RoomTranscriptionWriter,
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
    async fn perform(&mut self, effect: RoomTranscriptionEffect, subject: &Subject) -> Result<()> {
        match effect {
            RoomTranscriptionEffect::OpenStream { segment, is_final, expression } => {
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
                    .stream_text(
                        &telemetry::Cx::NONE,
                        StreamTextOptions {
                            topic: TOPIC_TRANSCRIPTION.to_owned(),
                            sender_identity: Some(participant),
                            attributes,
                        },
                    )
                    .await?;
                self.writer = Some(OpenWriter { writer, track: subject.stream_track.clone() });
            }
            RoomTranscriptionEffect::Write { payload } => {
                if let Some(open) = &self.writer {
                    open.writer.write(&telemetry::Cx::NONE, &payload).await?;
                }
            }
            RoomTranscriptionEffect::CloseStream { is_final, expression } => {
                if let Some(open) = self.writer.take() {
                    let attributes = close_attributes(is_final, expression, open.track.as_ref());
                    open.writer.close(&telemetry::Cx::NONE, attributes).await?;
                }
            }
            RoomTranscriptionEffect::PublishLegacy { segment, text, is_final } => {
                let Some((identity, track)) = subject.legacy_target() else { return Ok(()) };
                if !self.room.connection_state()?.is_connected() {
                    return Ok(());
                }
                self.room
                    .local()
                    .publish_transcription(
                        &telemetry::Cx::NONE,
                        Transcription {
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
                        },
                    )
                    .await?;
            }
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(open) = self.writer.take() {
            open.writer
                .close(&telemetry::Cx::NONE, close_attributes(true, None, open.track.as_ref()))
                .await?;
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
    /// The performer, which holds the output until it ends.
    performer: Mutex<Option<OwnedTask>>,
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
        let segment = ids.next_segment_id();
        let output = Arc::new(Self {
            room: room.clone(),
            ids,
            clock,
            writer: Mutex::new(WriterState {
                core: RoomTranscriptionWriter::new(is_delta_stream, json_format, segment),
                admission: Admission::default(),
                deadline: None,
                subject: Subject::default(),
            }),
            changed: Notify::new(),
            performer_done: Monotone::new(false),
            next_in_chain,
            performer: Mutex::new(None),
        });
        let owner = output.clone();
        let task = spawner.spawn(async move {
            owner.run(Performer { room, writer: None }).await;
        });
        *output.performer.lock() = Some(task);
        output
    }

    /// The output's own failure stays here: it is logged once, its text is dropped from then
    /// on, and [`Self::aclose`] reports it. Neither the speech nor the room fails with it.
    fn latched(&self, error: IoError) {
        log::warn!(target: "livekit.agents", "room transcription output failed, dropping its text: {error}");
        self.changed.notify_waiters();
    }

    fn fail(&self, error: IoError) {
        let mut writer = self.writer.lock();
        if writer.admission.failure().is_some() {
            return;
        }
        let WriterState { core, admission, .. } = &mut *writer;
        admission.fail(core, error.clone());
        drop(writer);
        self.latched(error);
    }

    /// A refusal comes back, to be latched once the guard is released: latching logs.
    fn submit(&self, writer: &mut WriterState) -> Option<IoError> {
        let WriterState { core, admission, subject, .. } = writer;
        let refused = admission.submit(core, subject).err();
        if refused.is_none() {
            self.changed.notify_waiters();
        }
        refused
    }

    /// Ends the open segment and submits; a refusal is latched after the guard.
    fn end_segment(&self, subject: Option<Subject>) {
        let refused = {
            let mut writer = self.writer.lock();
            writer.core.end_segment(self.ids.next_segment_id());
            let refused = self.submit(&mut writer);
            if let Some(subject) = subject {
                writer.subject = subject;
            }
            refused
        };
        if let Some(error) = refused {
            self.latched(error);
        }
    }

    fn disconnected(&self) {
        let refused = {
            let mut writer = self.writer.lock();
            writer.core.disconnected(self.ids.next_segment_id());
            self.submit(&mut writer)
        };
        if let Some(error) = refused {
            self.latched(error);
        }
    }

    fn status(&self) -> (Option<IoError>, Option<Ts>) {
        let writer = self.writer.lock();
        (writer.admission.failure().cloned(), writer.deadline)
    }

    async fn effect(
        &self,
        performer: &mut Performer,
        effect: RoomTranscriptionEffect,
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
        self.writer.lock().admission.performed(ticket);
        result
    }

    async fn run(self: Arc<Self>, mut performer: Performer) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let next = {
                let mut writer = self.writer.lock();
                if writer.admission.drained() {
                    break;
                }
                writer.admission.pop_batch()
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
        self.end_segment(Some(subject));
        Ok(())
    }

    pub async fn handle_event(&self, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::TrackPublished { participant, publication } => {
                let attributes = participant.attributes()?;
                self.writer.lock().subject.on_track_published(
                    participant.identity(),
                    &attributes,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::LocalTrackPublished { publication } => {
                let local = self.room.local().identity()?;
                self.writer.lock().subject.on_local_track_published(
                    &local,
                    publication.source(),
                    publication.sid(),
                );
            }
            RoomEvent::ConnectionStateChanged { state } if !state.is_connected() => {
                self.disconnected()
            }
            RoomEvent::Close { .. } => self.disconnected(),
            _ => {}
        }
        Ok(())
    }

    pub async fn aclose(&self) -> Result<()> {
        let requested_deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        let (deadline, refused) = {
            let mut writer = self.writer.lock();
            let WriterState { core, admission, subject, .. } = &mut *writer;
            let refused = admission.finish(core, subject).err();
            (*writer.deadline.get_or_insert(requested_deadline), refused)
        };
        if let Some(error) = refused {
            self.latched(error);
        }
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
            self.performer.lock().take();
        }
        self.writer.lock().admission.failure().cloned().map_or(Ok(()), Err)
    }
}

impl TextOutput for ParticipantTranscriptionOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next_in_chain.clone()
    }
    fn capture_text<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.capture_timed(&telemetry::Cx::NONE, &TranscriptChunk::new(text)).await
        })
    }

    fn capture_timed<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        text: &'a TranscriptChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let (shown, refused) = {
                let mut writer = self.writer.lock();
                let WriterState { core, admission, .. } = &mut *writer;
                match core.admit_text(&text.text) {
                    Ok(()) => {
                        let visible = core.visible_of(&text.text);
                        let shown = visible.text.clone();
                        core.capture(visible, Some(text));
                        (shown, self.submit(&mut writer))
                    }
                    Err(error) => {
                        admission.fail(core, error.clone());
                        (String::new(), Some(error))
                    }
                }
            };
            if let Some(error) = refused {
                self.latched(error);
            }
            if let Some(next) = &self.next_in_chain
                && !shown.is_empty()
            {
                next.capture_timed(
                    &telemetry::Cx::NONE,
                    &TranscriptChunk { text: shown, ..text.clone() },
                )
                .await?;
            }
            Ok(())
        })
    }

    fn flush(&self) {
        self.end_segment(None);
        if let Some(next) = &self.next_in_chain {
            next.flush();
        }
    }
}

// Transcript segment identities shared by the session and runtime output.

/// Transcript segment ids: `utils.shortuuid("SG_")`, unique across sessions in one room (a
/// frontend keys its transcript rows on them), or counted for a test that compares them.
#[derive(Debug, Default)]
pub struct SegmentIds(Option<AtomicU64>);

impl SegmentIds {
    pub fn counting() -> Self {
        SegmentIds(Some(AtomicU64::new(0)))
    }

    pub fn next_segment_id(&self) -> String {
        match &self.0 {
            Some(n) => format!("SG_{}", n.fetch_add(1, Ordering::Relaxed)),
            None => lk_agents_net::strings::shortuuid("SG_"),
        }
    }
}

#[cfg(test)]
mod segment_id_tests {
    use super::SegmentIds;

    // Catches: per-session counters in production. A second session in the same room (a warm
    // transfer, a restarted worker) would reuse `SG_0`, and a frontend keying its transcript rows
    // on the segment id overwrites the first session's lines.
    #[test]
    fn two_sessions_never_share_a_segment_id() {
        let (a, b) = (SegmentIds::default(), SegmentIds::default());
        let first: Vec<String> = (0..4).map(|_| a.next_segment_id()).collect();
        let second: Vec<String> = (0..4).map(|_| b.next_segment_id()).collect();
        assert!(first.iter().all(|id| id.starts_with("SG_") && id.len() == 15), "{first:?}");
        assert!(first.iter().all(|id| !second.contains(id)), "{first:?} {second:?}");
    }
}
