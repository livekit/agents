//! A stalled SDK writer must not allow transcript work to grow without admission or prevent cleanup.

use super::*;
use crate::test_support_rtc::{FakeRoom, FakeRtc};
use lk_agents_abi::voice::rtc::{
    AudioSource, ByteStreamSender, ConnectionState, LocalParticipant, ParticipantKind,
    RemoteParticipant, RoomEventBus, RtcError, TextStreamSender, TrackPublication,
    TrackPublishOptions,
};
use lk_agents_abi::{ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_TRACK_ID};
use lk_agents_abi::{Clock, RealClock};
use lk_agents_net::task::TokioSpawner;
use std::time::Duration;
use tokio::sync::{Notify, watch};

struct StalledWriter {
    inner: Box<dyn TextStreamWriter>,
    started: Arc<Notify>,
    flowing: watch::Receiver<bool>,
}

impl TextStreamWriter for StalledWriter {
    fn write<'a>(
        &'a self,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<(), RtcError>> {
        Box::pin(async move {
            self.started.notify_one();
            let mut flowing = self.flowing.clone();
            flowing
                .wait_for(|flowing| *flowing)
                .await
                .map_err(|_| RtcError::Closed("text writer"))?;
            self.inner.write(text).await
        })
    }

    fn close(
        self: Box<Self>,
        attributes: BTreeMap<String, String>,
    ) -> lk_agents_abi::BoxFuture<'static, std::result::Result<(), RtcError>> {
        Box::pin(async move { self.inner.close(attributes).await })
    }
}

struct StalledLocal {
    inner: Arc<dyn LocalParticipant>,
    started: Arc<Notify>,
    flowing: watch::Receiver<bool>,
}

impl LocalParticipant for StalledLocal {
    fn identity(&self) -> std::result::Result<ParticipantIdentity, RtcError> {
        self.inner.identity()
    }
    fn kind(&self) -> std::result::Result<ParticipantKind, RtcError> {
        self.inner.kind()
    }
    fn track_publications(&self) -> std::result::Result<Vec<Arc<dyn TrackPublication>>, RtcError> {
        self.inner.track_publications()
    }
    fn publish_audio_track<'a>(
        &'a self,
        options: TrackPublishOptions,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        std::result::Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>), RtcError>,
    > {
        Box::pin(async move { self.inner.publish_audio_track(options).await })
    }
    fn set_attributes<'a>(
        &'a self,
        attributes: BTreeMap<String, String>,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<(), RtcError>> {
        Box::pin(async move { self.inner.set_attributes(attributes).await })
    }
    fn publish_transcription<'a>(
        &'a self,
        transcription: Transcription,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<(), RtcError>> {
        Box::pin(async move { self.inner.publish_transcription(transcription).await })
    }
    fn stream_text<'a>(
        &'a self,
        options: StreamTextOptions,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Box<dyn TextStreamWriter>, RtcError>>
    {
        Box::pin(async move {
            Ok(Box::new(StalledWriter {
                inner: self.inner.stream_text(options).await?,
                started: self.started.clone(),
                flowing: self.flowing.clone(),
            }) as Box<dyn TextStreamWriter>)
        })
    }
}

struct StalledRoom {
    inner: Arc<FakeRoom>,
    local: Arc<dyn LocalParticipant>,
    clock: Arc<dyn Clock>,
}

impl lk_agents_abi::utils::clock::HostClock for StalledRoom {
    fn host_clock(&self) -> Arc<dyn Clock> {
        self.clock.clone()
    }
}

impl RtcRoom for StalledRoom {
    fn name(&self) -> std::result::Result<String, RtcError> {
        self.inner.name()
    }
    fn connection_state(&self) -> std::result::Result<ConnectionState, RtcError> {
        self.inner.connection_state()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local
    }
    fn participant(
        &self,
        identity: &ParticipantIdentity,
    ) -> std::result::Result<Option<Arc<dyn RemoteParticipant>>, RtcError> {
        self.inner.participant(identity)
    }
    fn remote_participants(
        &self,
    ) -> std::result::Result<Vec<Arc<dyn RemoteParticipant>>, RtcError> {
        self.inner.remote_participants()
    }
    fn event_bus(&self) -> &RoomEventBus {
        self.inner.event_bus()
    }
    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: ByteStreamSender,
    ) -> std::result::Result<(), RtcError> {
        self.inner.register_byte_stream_handler(topic, streams)
    }
    fn unregister_byte_stream_handler(&self, topic: &str) -> std::result::Result<(), RtcError> {
        self.inner.unregister_byte_stream_handler(topic)
    }
    fn register_text_stream_handler(
        &self,
        topic: &str,
        streams: TextStreamSender,
    ) -> std::result::Result<(), RtcError> {
        self.inner.register_text_stream_handler(topic, streams)
    }
    fn unregister_text_stream_handler(&self, topic: &str) -> std::result::Result<(), RtcError> {
        self.inner.unregister_text_stream_handler(topic)
    }
}

fn stalled_output(
    name: &str,
    is_delta_stream: bool,
) -> (Arc<ParticipantTranscriptionOutput>, Arc<FakeRoom>, Arc<Notify>, watch::Sender<bool>) {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room(name);
    room.add_participant("person", ParticipantKind::Standard).publish_microphone("microphone");
    let started = Arc::new(Notify::new());
    let (flowing, flow) = watch::channel(false);
    let local = Arc::new(StalledLocal {
        inner: room.local().clone(),
        started: started.clone(),
        flowing: flow,
    });
    let sdk = Arc::new(StalledRoom { inner: room.clone(), local, clock: clock.clone() });
    let output = ParticipantTranscriptionOutput::new(
        sdk,
        Arc::new(TokioSpawner::current()),
        clock,
        Arc::new(SegmentIds::counting()),
        is_delta_stream,
        false,
        None,
    );
    (output, room, started, flowing)
}

#[tokio::test(start_paused = true)]
async fn a_writer_stalled_past_the_text_bound_drops_the_transcript_not_the_caller() {
    use lk_agents_voice::room_io::ingress::MAX_STREAM_BYTES;

    let (output, room, started, _flowing) = stalled_output("transcription-bytes", true);
    output.set_participant(Some("person".into())).await.unwrap();
    output.capture_text("accepted prefix").await.unwrap();
    started.notified().await;
    for _ in 0..3 {
        output
            .capture_text(&"x".repeat(MAX_STREAM_BYTES / 2))
            .await
            .expect("a transcription failure is not the speech's error");
    }
    output.capture_text("dropped").await.unwrap();
    let closed = lk_agents_abi::timeout(&RealClock::new(), Duration::from_secs(1), output.aclose())
        .await
        .expect("the reserved cleanup path must bypass saturated transcript data");
    assert!(
        closed.is_err_and(|error| error.to_string().contains("capacity")),
        "the overload stays observable at cleanup"
    );
    let streams = room.local_participant().text_streams();
    assert_eq!(streams.len(), 1, "no stream opens after the failure");
    assert!(streams[0].closed, "the stalled SDK writer must be closed before cleanup returns");
    assert!(streams[0].text().is_empty(), "the refused text must not partially reach the SDK");
}

// Catches: a fifth batch queued behind the writer latching "queued batches", which failed the
// transcript (and RoomIo with it) for the rest of the call.
#[tokio::test(start_paused = true)]
async fn a_transcript_backlog_behind_a_stalled_writer_is_delivered_coalesced() {
    let (output, room, started, flowing) = stalled_output("transcription-backlog", false);
    output.set_participant(Some("person".into())).await.unwrap();
    output.capture_text("first").await.unwrap();
    started.notified().await;
    for turn in 0..6 {
        for words in ["a", "a b", "a b c"] {
            output.capture_text(&format!("{turn}: {words}")).await.unwrap();
        }
        output.flush();
    }
    flowing.send_replace(true);
    lk_agents_abi::timeout(&RealClock::new(), Duration::from_secs(1), output.aclose())
        .await
        .unwrap()
        .expect("a backlog is delivered, not failed");
    let shown: Vec<_> = room
        .local_participant()
        .text_streams()
        .into_iter()
        .filter(|stream| {
            stream.close_attributes.get(ATTRIBUTE_TRANSCRIPTION_FINAL).is_some_and(|f| f == "true")
        })
        .map(|stream| stream.text())
        .collect();
    assert_eq!(shown, (0..6).map(|turn| format!("{turn}: a b c")).collect::<Vec<_>>());
    assert!(
        room.local_participant().text_streams().len() < 6 * 4,
        "superseded interim text is coalesced, not replayed"
    );
}

#[tokio::test(start_paused = true)]
async fn queued_transcript_keeps_its_subject_and_track_through_replacement() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("transcription-subject");
    room.add_participant("first", ParticipantKind::Standard).publish_microphone("first-mic");
    room.add_participant("second", ParticipantKind::Standard).publish_microphone("second-mic");
    let started = Arc::new(Notify::new());
    let (flowing, flow) = watch::channel(false);
    let local = Arc::new(StalledLocal {
        inner: room.local().clone(),
        started: started.clone(),
        flowing: flow,
    });
    let sdk = Arc::new(StalledRoom { inner: room.clone(), local, clock: clock.clone() });
    let output = ParticipantTranscriptionOutput::new(
        sdk,
        Arc::new(TokioSpawner::current()),
        clock,
        Arc::new(SegmentIds::counting()),
        true,
        false,
        None,
    );
    output.set_participant(Some("first".into())).await.unwrap();
    output.capture_text("first").await.unwrap();
    started.notified().await;
    output.set_participant(Some("second".into())).await.unwrap();
    output.capture_text("second").await.unwrap();
    output.flush();
    flowing.send_replace(true);
    lk_agents_abi::timeout(&RealClock::new(), Duration::from_secs(1), output.aclose())
        .await
        .unwrap()
        .unwrap();
    let streams = room.local_participant().text_streams();
    assert_eq!(streams.iter().map(|stream| stream.text()).collect::<Vec<_>>(), ["first", "second"]);
    assert_eq!(
        streams[0].close_attributes.get(ATTRIBUTE_TRANSCRIPTION_TRACK_ID).map(String::as_str),
        Some("first-mic"),
        "retiring the old transcript must retain its original track"
    );
    assert_eq!(
        streams[1].close_attributes.get(ATTRIBUTE_TRANSCRIPTION_TRACK_ID).map(String::as_str),
        Some("second-mic")
    );
    let legacy = room.local_participant().transcriptions();
    assert_eq!(
        legacy[0].participant_identity.as_str(),
        "first",
        "queued legacy text must retain the subject that admitted it"
    );
    assert!(streams.iter().all(|stream| stream.closed));
}

// Catches: a speech still producing text when the room closed failing with "transcription
// output is closed": a closed output takes no more text, and that is not the speech's error.
#[tokio::test(start_paused = true)]
async fn text_captured_after_close_is_dropped_not_an_error() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("transcription-after-close");
    room.add_participant("user", ParticipantKind::Standard);
    let output = ParticipantTranscriptionOutput::new(
        room.clone(),
        Arc::new(TokioSpawner::current()),
        clock,
        Arc::new(SegmentIds::counting()),
        true,
        false,
        None,
    );
    output.set_participant(Some("user".into())).await.unwrap();
    output.aclose().await.unwrap();
    output.capture_text("late").await.expect("a closed output drops late text");
    output.flush();
    assert!(room.local_participant().text_streams().iter().all(|stream| stream.text().is_empty()));
}
