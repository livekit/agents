//! A stalled SDK writer must not allow transcript work to grow without admission or prevent cleanup.

use super::*;
use crate::test_support_rtc::{FakeRoom, FakeRtc};
use lk_agents_abi::ATTRIBUTE_TRANSCRIPTION_TRACK_ID;
use lk_agents_abi::voice::rtc::{
    AudioSource, ByteStreamSender, ConnectionState, LocalParticipant, ParticipantKind,
    RemoteParticipant, RoomEventBus, RtcError, TextStreamSender, TrackPublication,
    TrackPublishOptions,
};
use lk_agents_abi::{Clock, RealClock, TokioSpawner};
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

#[derive(Clone, Copy, Debug)]
enum Saturation {
    TextBytes,
    Streams,
    Batches,
}

#[tokio::test(start_paused = true)]
async fn saturated_transcription_refuses_the_whole_capture_and_closes_the_stalled_sdk_writer() {
    use lk_agents_voice::room_io::ingress::{MAX_INCOMING_STREAMS, MAX_STREAM_BYTES};

    for saturation in [Saturation::TextBytes, Saturation::Streams, Saturation::Batches] {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("transcription-admission");
        let person = room.add_participant("person", ParticipantKind::Standard);
        person.publish_microphone("microphone");
        let started = Arc::new(Notify::new());
        let (_flowing, flow) = watch::channel(false);
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
            !matches!(saturation, Saturation::Streams),
            false,
            None,
        );
        output.set_participant(Some("person".into())).await.unwrap();
        output.capture_text("accepted prefix").await.unwrap();
        started.notified().await;
        let mut failure = None;
        for _ in 0..=MAX_INCOMING_STREAMS + 1 {
            let text = match saturation {
                Saturation::TextBytes => "x".repeat(MAX_STREAM_BYTES / 2),
                _ => "queued text".into(),
            };
            if let Err(error) = output.capture_text(&text).await {
                failure = Some(error);
                break;
            }
        }
        let failure =
            failure.expect("a stalled SDK writer must have a finite transcript admission bound");
        assert!(failure.to_string().contains("capacity"), "{saturation:?}: {failure}");
        assert!(output.capture_text("must be refused too").await.is_err());
        let closed =
            lk_agents_abi::timeout(&RealClock::new(), Duration::from_secs(1), output.aclose())
                .await
                .expect("the reserved cleanup path must bypass saturated transcript data");
        assert!(closed.is_err(), "the overload remains observable after cleanup");
        let streams = room.local_participant().text_streams();
        assert_eq!(
            streams.len(),
            1,
            "queued or rejected captures must not open extra streams after failure"
        );
        assert!(streams[0].closed, "the stalled SDK writer must be closed before cleanup returns");
        assert!(
            streams[0].text().is_empty(),
            "the rejected capture must not partially reach the SDK"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn queued_transcript_keeps_its_subject_and_track_through_replacement() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
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
