//! Participant transcription stalls, backlog, replacement and close contracts.

use super::*;

// --------------------------------------------------------------------- the agent's transcript

/// A text writer that waits until the test lets it write.
struct StalledWriter {
    inner: Box<dyn TextStreamWriter>,
    started: Arc<Notify>,
    flowing: watch::Receiver<bool>,
}

impl TextStreamWriter for StalledWriter {
    fn write<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<(), RtcError>> {
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
    ) -> lk_agents_abi::BoxFuture<'static, Result<(), RtcError>> {
        Box::pin(async move { self.inner.close(attributes).await })
    }
}

struct StalledLocal {
    inner: Arc<dyn LocalParticipant>,
    started: Arc<Notify>,
    flowing: watch::Receiver<bool>,
}

impl LocalParticipant for StalledLocal {
    fn identity(&self) -> Result<ParticipantIdentity, RtcError> {
        self.inner.identity()
    }
    fn kind(&self) -> Result<ParticipantKind, RtcError> {
        self.inner.kind()
    }
    fn track_publications(&self) -> Result<Vec<Arc<dyn TrackPublication>>, RtcError> {
        self.inner.track_publications()
    }
    fn publish_audio_track<'a>(
        &'a self,
        options: TrackPublishOptions,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>), RtcError>,
    > {
        Box::pin(async move { self.inner.publish_audio_track(options).await })
    }
    fn set_attributes<'a>(
        &'a self,
        attributes: BTreeMap<String, String>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { self.inner.set_attributes(attributes).await })
    }
    fn publish_transcription<'a>(
        &'a self,
        transcription: Transcription,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { self.inner.publish_transcription(transcription).await })
    }
    fn stream_text<'a>(
        &'a self,
        options: StreamTextOptions,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Box<dyn TextStreamWriter>, RtcError>> {
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
}

impl RtcRoom for StalledRoom {
    fn name(&self) -> Result<String, RtcError> {
        self.inner.name()
    }
    fn connection_state(&self) -> Result<ConnectionState, RtcError> {
        self.inner.connection_state()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local
    }
    fn participant(
        &self,
        identity: &ParticipantIdentity,
    ) -> Result<Option<Arc<dyn RemoteParticipant>>, RtcError> {
        self.inner.participant(identity)
    }
    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>, RtcError> {
        self.inner.remote_participants()
    }
    fn event_bus(&self) -> &Arc<dyn RoomEventSink> {
        self.inner.event_bus()
    }
    fn rpc_router(&self) -> &Arc<dyn lk_agents_abi::voice::rtc::ports::RpcRoutes> {
        self.inner.rpc_router()
    }
    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: Arc<dyn ByteStreamIngress>,
    ) -> Result<(), RtcError> {
        self.inner.register_byte_stream_handler(topic, streams)
    }
    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<(), RtcError> {
        self.inner.unregister_byte_stream_handler(topic)
    }
    fn register_text_stream_handler(
        &self,
        topic: &str,
        streams: Arc<dyn TextStreamIngress>,
    ) -> Result<(), RtcError> {
        self.inner.register_text_stream_handler(topic, streams)
    }
    fn unregister_text_stream_handler(&self, topic: &str) -> Result<(), RtcError> {
        self.inner.unregister_text_stream_handler(topic)
    }
}

/// A transcription output whose SDK writer stalls until the returned sender lets it flow.
fn stalled_output(
    room: &Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    is_delta_stream: bool,
) -> (Arc<ParticipantTranscriptionOutput>, Arc<Notify>, watch::Sender<bool>) {
    let started = Arc::new(Notify::new());
    let (flowing, flow) = watch::channel(false);
    let local = Arc::new(StalledLocal {
        inner: room.local().clone(),
        started: started.clone(),
        flowing: flow,
    });
    let sdk = Arc::new(StalledRoom { inner: room.clone(), local });
    let output = ParticipantTranscriptionOutput::new(
        sdk,
        Arc::new(TokioSpawner::current()),
        clock,
        Arc::new(SegmentIds::counting()),
        is_delta_stream,
        false,
        None,
    );
    (output, started, flowing)
}

fn room_with_person(name: &str) -> (Arc<FakeRoom>, Arc<VirtualClock>) {
    let (room, clock) = fake_room(name);
    room.add_participant("person", ParticipantKind::Standard).publish_microphone("microphone");
    (room, clock)
}

#[tokio::test(start_paused = true)]
async fn a_writer_stalled_past_the_text_bound_drops_the_transcript_not_the_caller() {
    use lk_agents_voice::room_io::MAX_STREAM_BYTES;

    let (room, clock) = room_with_person("transcription-bytes");
    let (output, started, _flowing) = stalled_output(&room, clock, true);
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
    let closed = within_a_second(output.aclose())
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
    let (room, clock) = room_with_person("transcription-backlog");
    let (output, started, flowing) = stalled_output(&room, clock, false);
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
    within_a_second(output.aclose()).await.unwrap().expect("a backlog is delivered, not failed");
    let streams = room.local_participant().text_streams();
    let shown: Vec<_> = streams
        .iter()
        .filter(|stream| {
            stream.close_attributes.get(ATTRIBUTE_TRANSCRIPTION_FINAL).is_some_and(|f| f == "true")
        })
        .map(|stream| stream.text())
        .collect();
    assert_eq!(shown, (0..6).map(|turn| format!("{turn}: a b c")).collect::<Vec<_>>());
    assert!(streams.len() < 6 * 4, "superseded interim text is coalesced, not replayed");
}

#[tokio::test(start_paused = true)]
async fn queued_transcript_keeps_its_subject_and_track_through_replacement() {
    let (room, clock) = fake_room("transcription-subject");
    room.add_participant("first", ParticipantKind::Standard).publish_microphone("first-mic");
    room.add_participant("second", ParticipantKind::Standard).publish_microphone("second-mic");
    let (output, started, flowing) = stalled_output(&room, clock, true);
    output.set_participant(Some("first".into())).await.unwrap();
    output.capture_text("first").await.unwrap();
    started.notified().await;
    output.set_participant(Some("second".into())).await.unwrap();
    output.capture_text("second").await.unwrap();
    output.flush();
    flowing.send_replace(true);
    within_a_second(output.aclose()).await.unwrap().unwrap();
    let streams = room.local_participant().text_streams();
    assert_eq!(streams.iter().map(|stream| stream.text()).collect::<Vec<_>>(), ["first", "second"]);
    let track = |n: usize| {
        streams[n].close_attributes.get(ATTRIBUTE_TRANSCRIPTION_TRACK_ID).map(String::as_str)
    };
    assert_eq!(track(0), Some("first-mic"), "retiring the old transcript must retain its track");
    assert_eq!(track(1), Some("second-mic"));
    let legacy = room.local_participant().transcriptions();
    assert_eq!(
        legacy[0].participant_identity.as_str(),
        "first",
        "queued legacy text must retain the subject that admitted it"
    );
    assert!(streams.iter().all(|stream| stream.closed));
}

// Closing text output drops late text without failing the producing speech.
#[tokio::test(start_paused = true)]
async fn text_captured_after_close_is_dropped_not_an_error() {
    let (room, clock) = fake_room("transcription-after-close");
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
