//! The runtime's room performers over FakeRoom, in what the DST's room legs do not reach: SDK
//! failures and stalls, cancelled and racing waiters, reattachment races, overload bounds and the
//! pre-connect buffer's edges.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputExt, PlaybackFinished, TextOutput,
};
use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextInput, TextOutputConfig,
};
use lk_agents_abi::voice::room_owner::{RoomIo as _, RoomIoEvent};
use lk_agents_abi::voice::rtc::{
    AudioInputStream, AudioSource, AudioStreamItem, AudioStreamOptions, AudioTrackFeature,
    ByteStreamIngress, ConnectionState, DisconnectReason, LocalParticipant, ParticipantIdentity,
    ParticipantKind, ProcessorFailure, RemoteParticipant, RoomEvent, RoomEventSink, RtcError,
    RtcRoom, StreamInfo, StreamTextOptions, TextStreamIngress, TextStreamWriter, TrackPublication,
    TrackPublishOptions, TrackSid, TrackSource, Transcription, VideoFrameRef, silence_frame,
};
use lk_agents_abi::{
    ATTRIBUTE_TRANSCRIPTION_FINAL, ATTRIBUTE_TRANSCRIPTION_TRACK_ID, AudioFrame, Closed,
    HOST_TEARDOWN_TIMEOUT, StreamOutput, TOPIC_CHAT, Ts, VirtualClock,
};
use lk_agents_fuzz::fakes::room::{
    FailingProcessors, FakeRemoteParticipant, FakeRoom, FakeRtc, FakeTrackPublication, Faults,
    SdkRx, SdkTx, sdk_queue,
};
use lk_agents_runtime::net::clock::RealClock;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::RoomIo;
use lk_agents_runtime::room_io::input::ParticipantAudioInput;
use lk_agents_runtime::room_io::output::ParticipantAudioOutput;
use lk_agents_runtime::room_io::output::transcription::{
    ParticipantTranscriptionOutput, SegmentIds,
};
use lk_agents_runtime::room_io::pre_connect_audio::PreConnectAudioHandler;
use lk_agents_runtime::rtc::events::byte_streams;
use lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use std::collections::BTreeMap;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;
use tokio::sync::{Notify, mpsc, watch};

/// Polls `future` once, asserting it is still pending.
async fn pending<F: Future>(mut future: Pin<&mut F>) {
    poll_fn(|context| {
        assert!(future.as_mut().poll(context).is_pending(), "operation should remain pending");
        Poll::Ready(())
    })
    .await;
}

/// Awaits `future` for at most a second of real time.
async fn within_a_second<F: Future>(future: F) -> Option<F::Output> {
    lk_agents_abi::timeout(&RealClock::new(), Duration::from_secs(1), future).await.ok()
}

async fn yields(n: usize) {
    for _ in 0..n {
        tokio::task::yield_now().await;
    }
}

fn fake_room(name: &str) -> (Arc<FakeRoom>, Arc<VirtualClock>) {
    let (rtc, clock) =
        FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room(name);
    std::mem::forget(rtc);
    (room, clock)
}

// ------------------------------------------------------------------------------ RoomIo

fn config() -> RoomIoConfig {
    RoomIoConfig {
        participant_identity: None,
        participant_kinds: Vec::new(),
        close_on_disconnect: true,
        delete_room_on_close: false,
        text_input: false,
        video_input: false,
        audio_input: None,
        audio_output: Some(AudioOutputConfig::default()),
        text_output: None,
    }
}

fn microphone_only() -> RoomIoConfig {
    RoomIoConfig {
        audio_output: None,
        audio_input: Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() }),
        ..config()
    }
}

fn room_io(room: Arc<dyn RtcRoom>, clock: Arc<VirtualClock>, config: RoomIoConfig) -> Arc<RoomIo> {
    let ids = Arc::new(SegmentIds::counting());
    let tracer = lk_agents_runtime::net::telemetry::Tracer::disabled(clock.clone());
    RoomIo::new(room, clock, Arc::new(TokioSpawner::current()), ids, config, None, None, tracer)
}

#[tokio::test(start_paused = true)]
async fn publication_failure_settles_every_room_readiness_waiter() {
    let (room, clock) = fake_room("failed-ready");
    room.add_participant("person", ParticipantKind::Standard);
    room.local_participant().fail_with(RtcError::host("publication refused"));
    let io = room_io(room, clock, config());
    let mut first = Box::pin(io.wait_for_ready());
    let mut second = Box::pin(io.wait_for_ready());
    pending(first.as_mut()).await;
    pending(second.as_mut()).await;
    io.clone().start(Default::default());
    let completed = within_a_second(async {
        let first = first.await.expect_err("publication failure must fail readiness");
        let second = second.await.expect_err("every waiter must see the same failure");
        assert_eq!(first, second);
        assert!(first.to_string().contains("publication refused"));
    })
    .await;
    io.clone().aclose().await;
    assert!(completed.is_some(), "publication failure must release every readiness waiter");
}

#[tokio::test(start_paused = true)]
async fn closing_before_linking_settles_room_readiness() {
    let (room, clock) = fake_room("closed-ready");
    let io = room_io(room, clock, config());
    io.clone().start(Default::default());
    io.clone().aclose().await;
    let ready = within_a_second(io.wait_for_ready())
        .await
        .expect("close must not leave readiness waiting for a participant");
    assert!(ready.is_err(), "close must never fabricate successful readiness");
}

#[tokio::test(start_paused = true)]
async fn room_disconnection_settles_readiness_before_a_participant_links() {
    let (room, clock) = fake_room("disconnected-ready");
    let io = room_io(room.clone(), clock, config());
    io.clone().start(Default::default());
    room.close(DisconnectReason::ClientInitiated);
    let ready = within_a_second(io.wait_for_ready()).await;
    io.clone().aclose().await;
    let ready = ready.expect("a disconnected room must settle readiness without a participant");
    assert!(ready.is_err(), "a disconnected room cannot become ready");
}

/// FakeRoom with its chat handler captured, so the test opens chat streams by hand.
struct StreamingRoom {
    room: Arc<FakeRoom>,
    local: Arc<dyn LocalParticipant>,
    text: watch::Sender<Option<Arc<dyn TextStreamIngress>>>,
}

struct TextChunks(mpsc::Receiver<String>);
impl StreamOutput<String> for TextChunks {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<String> for TextChunks {}

impl RtcRoom for StreamingRoom {
    fn name(&self) -> Result<String, RtcError> {
        self.room.name()
    }
    fn connection_state(&self) -> Result<ConnectionState, RtcError> {
        self.room.connection_state()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local
    }
    fn participant(
        &self,
        id: &ParticipantIdentity,
    ) -> Result<Option<Arc<dyn RemoteParticipant>>, RtcError> {
        self.room.participant(id)
    }
    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>, RtcError> {
        self.room.remote_participants()
    }
    fn event_bus(&self) -> &Arc<dyn RoomEventSink> {
        self.room.event_bus()
    }
    fn rpc_router(&self) -> &Arc<dyn lk_agents_abi::voice::rtc::ports::RpcRoutes> {
        self.room.rpc_router()
    }
    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: Arc<dyn ByteStreamIngress>,
    ) -> Result<(), RtcError> {
        self.room.register_byte_stream_handler(topic, streams)
    }
    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<(), RtcError> {
        self.room.unregister_byte_stream_handler(topic)
    }
    fn register_text_stream_handler(
        &self,
        _: &str,
        streams: Arc<dyn TextStreamIngress>,
    ) -> Result<(), RtcError> {
        self.text.send_replace(Some(streams));
        Ok(())
    }
    fn unregister_text_stream_handler(&self, _: &str) -> Result<(), RtcError> {
        self.text.send_replace(None);
        Ok(())
    }
}

impl StreamingRoom {
    fn open_chat(&self) -> mpsc::Sender<String> {
        let (chunks, receiver) = mpsc::channel(4);
        assert!(self.text.borrow().as_ref().unwrap().open(
            StreamInfo { topic: TOPIC_CHAT.into(), ..Default::default() },
            "person".into(),
            Box::new(TextChunks(receiver)),
        ));
        chunks
    }
}

#[tokio::test]
async fn text_from_an_earlier_attachment_cannot_drive_the_new_attachment() {
    let (room, clock) = fake_room("text-attachment");
    room.add_participant("person", ParticipantKind::Standard);
    let room =
        Arc::new(StreamingRoom { local: room.local().clone(), room, text: watch::channel(None).0 });
    let config = RoomIoConfig { audio_output: None, text_input: true, ..config() };
    let io = room_io(room.clone(), clock, config);
    let (mut events, _) = io.events().unwrap();
    io.clone().start(Default::default());
    let _ = io.wait_for_ready().await;
    while events.try_recv().is_ok() {}
    let chunks = room.open_chat();
    chunks.send("old attachment".into()).await.unwrap();
    // The text reader has accepted this stream and is waiting for its trailer.
    yields(4).await;
    io.set_participant(Some("person".into())).await.unwrap();
    drop(chunks);
    yields(4).await;
    let mut delivered_stale = false;
    while let Ok(event) = events.try_recv() {
        delivered_stale |= matches!(event, RoomIoEvent::TextInput(_));
    }
    assert!(!delivered_stale, "a completed old chat stream must retain its attachment identity");
    let chunks = room.open_chat();
    chunks.send("current attachment".into()).await.unwrap();
    drop(chunks);
    let delivered = events.recv().await.expect("the current attachment can send chat");
    assert!(
        matches!(delivered, RoomIoEvent::TextInput(TextInput { text, .. }) if text == "current attachment")
    );
    io.clone().aclose().await;
}

#[tokio::test(start_paused = true)]
async fn room_close_deadline_survives_a_cancelled_waiter_and_a_blocked_sdk_source() {
    let (room, clock) = fake_room("close-deadline");
    room.add_participant("person", ParticipantKind::Standard);
    let io = room_io(room.clone(), clock.clone(), config());
    io.clone().start(Default::default());
    io.wait_for_ready().await.unwrap();
    let source = room.local_participant().sources()[0].clone();
    source.flow_closes(false);
    let output = io.audio_output().unwrap();
    output.capture_frame(&AudioFrame::new(vec![0; 960], 24_000, 1, 480)).await.unwrap();
    let mut first = Box::pin(io.clone().aclose());
    pending(first.as_mut()).await;
    source.wait_for_close_started().await;
    let playback = within_a_second(output.tracker().wait_for_playout()).await;
    drop(first);
    let mut second = Box::pin(io.clone().aclose());
    pending(second.as_mut()).await;
    clock.advance(HOST_TEARDOWN_TIMEOUT);
    let outcome = within_a_second(second).await;
    let unpublished_while_source_blocked = room.local_participant().publications().is_empty();
    source.flow_closes(true);
    within_a_second(async {
        while !source.is_closed() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the SDK cleanup owner survives its waiters");
    io.clone().aclose().await;
    let playback =
        playback.expect("closing a segment must settle playback before SDK cleanup completes");
    assert!(playback.interrupted);
    assert_eq!(playback.playback_position, Duration::ZERO, "cleanup cannot fabricate heard audio");
    assert!(
        unpublished_while_source_blocked,
        "source cleanup must not block retiring the publication SID"
    );
    assert_eq!(
        outcome.expect("a blocked SDK source must not hold room close beyond its deadline"),
        Closed::TimedOut
    );
    assert_eq!(
        io.clone().aclose().await,
        Closed::TimedOut,
        "repeated close must retain the original timeout outcome"
    );
}

// Catches: a second `start()` subscribing to the room's events again, so every event is handled
// twice.
#[tokio::test]
async fn repeated_room_start_keeps_one_event_subscription() {
    let (room, clock) = fake_room("one-start");
    let io = room_io(room.clone(), clock, config());
    io.clone().start(Default::default());
    io.clone().start(Default::default());
    assert_eq!(room.event_subscriber_count(), 1);
    io.clone().aclose().await;
}

// Catches: the input staying on a departed participant's track when another person replaces
// them, with no explicit selection (the DST's caller always rejoins under the same identity).
#[tokio::test]
async fn room_relinks_input_after_the_linked_participant_is_replaced() {
    let (room, clock) = fake_room("relinked-input");
    room.add_participant("first", ParticipantKind::Standard).publish_microphone("TR_first");
    let io = room_io(room.clone(), clock, microphone_only());
    io.clone().start(Default::default());
    io.wait_for_ready().await.unwrap();
    assert_eq!(io.audio_input_leg().unwrap().reading(), Some("TR_first".into()));
    room.remove_participant(&"first".into(), DisconnectReason::ClientInitiated);
    room.add_participant("second", ParticipantKind::Standard).publish_microphone("TR_second");
    yields(8).await;
    let linked = io.linked_participant();
    let reading = io.audio_input_leg().unwrap().reading();
    io.clone().aclose().await;
    assert_eq!(linked, Some("second".into()), "the replacement participant must become linked");
    assert_eq!(reading, Some("TR_second".into()), "the microphone must follow the new link");
}

#[tokio::test]
async fn selecting_the_same_participant_retires_buffered_input_from_the_old_attachment() {
    let (room, clock) = fake_room("reselected-input");
    let person = room.add_participant("person", ParticipantKind::Standard);
    let microphone = person.publish_microphone("TR_mic");
    let config =
        RoomIoConfig { audio_output: Some(AudioOutputConfig::default()), ..microphone_only() };
    let io = room_io(room, clock, config);
    io.clone().start(Default::default());
    io.wait_for_ready().await.unwrap();
    let input = io.audio_input().unwrap();
    assert!(microphone.push_frame(AudioFrame::new(vec![1; 960], 24_000, 1, 480)) > 0);
    yields(4).await;
    io.set_participant(Some("person".into())).await.unwrap();
    assert!(microphone.push_frame(AudioFrame::new(vec![2; 960], 24_000, 1, 480)) > 0);
    let received = input.next_frame(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap().unwrap();
    io.clone().aclose().await;
    assert_eq!(received.data.as_ref(), &[2; 960], "old queued audio must not cross reattachment");
}

#[tokio::test(start_paused = true)]
async fn delayed_departure_cannot_detach_a_rejoined_participant_with_the_same_identity() {
    let (room, clock) = fake_room("participant-lifetime");
    let old = room.add_participant("person", ParticipantKind::Standard);
    let old_publication = old.publish_microphone("microphone");
    let io = room_io(room.clone(), clock, microphone_only());
    let (mut events, _) = io.events().unwrap();
    io.clone().start(Default::default());
    io.wait_for_ready().await.unwrap();
    room.remove_participant(&"person".into(), DisconnectReason::ClientInitiated);
    let current = room.add_participant("person", ParticipantKind::Standard);
    current.publish_microphone("microphone");
    io.set_participant(Some("person".into())).await.unwrap();
    room.event_bus().emit(RoomEvent::TrackUnpublished {
        participant: old.clone(),
        publication: old_publication,
    });
    room.event_bus().emit(RoomEvent::ParticipantDisconnected {
        participant: old,
        reason: DisconnectReason::ClientInitiated,
    });
    room.press_dtmf(&"person".into(), "1");
    within_a_second(async {
        while !matches!(events.recv().await, Some(RoomIoEvent::KeyPressed(_))) {}
    })
    .await
    .expect("the current participant must remain linked after the delayed old departure");
    assert_eq!(io.audio_input_leg().unwrap().reading(), Some("microphone".into()));
    io.clone().aclose().await;
}

#[tokio::test]
async fn saturated_room_events_are_reported_to_the_selection_caller() {
    let (room, clock) = fake_room("event-capacity");
    room.add_participant("first", ParticipantKind::Standard);
    room.add_participant("second", ParticipantKind::Standard);
    let io = room_io(room, clock, RoomIoConfig { audio_output: None, ..config() });
    // Nobody reads the events: selections fill the queue until one is refused.
    let mut refused = None;
    for n in 0..1_000 {
        let identity = if n % 2 == 0 { "first" } else { "second" };
        if let Err(error) = io.set_participant(Some(identity.into())).await {
            refused = Some(error);
            break;
        }
    }
    assert!(
        refused.is_some_and(|error| error.to_string().contains("capacity exceeded")),
        "an overflowing selection must be refused, not dropped"
    );
    io.clone().aclose().await;
}

// Catches: a burst of transcripts the performer had not drained yet latched a capacity failure,
// and the next room event carried it into RoomIo::fail, closing the room's audio output: the
// agent was mute for the rest of the call.
#[tokio::test(start_paused = true)]
async fn a_transcript_burst_leaves_the_room_audio_output_open() {
    let (room, clock) = fake_room("transcript-burst");
    room.add_participant("person", ParticipantKind::Standard);
    let config = RoomIoConfig { text_output: Some(TextOutputConfig::default()), ..config() };
    let io = room_io(room.clone(), clock, config);
    io.clone().start(Default::default());
    io.wait_for_ready().await.unwrap();
    for n in 0..8 {
        io.on_user_input_transcribed(&format!("word {n}"), n % 2 == 1)
            .await
            .expect("a transcript backlog is not the caller's error");
    }
    room.remove_participant(&"person".into(), DisconnectReason::ClientInitiated);
    yields(8).await;
    let output = io.audio_output().unwrap();
    let captured = output.capture_frame(&silence_frame(480, 24_000, 1)).await;
    io.clone().aclose().await;
    assert!(captured.is_ok(), "a transcription backlog must not close the room's audio output");
    let streams = room.local_participant().text_streams();
    assert_eq!(
        streams.last().map(|stream| stream.text()),
        Some("word 7".into()),
        "the backlog delivers the latest transcript"
    );
}

// ------------------------------------------------------------------------- the microphone

/// A microphone whose SDK processing the test controls.
struct Processed {
    track: Arc<FakeTrackPublication>,
    processor: Arc<Processor>,
}

impl TrackPublication for Processed {
    fn sid(&self) -> &TrackSid {
        self.track.sid()
    }
    fn name(&self) -> &str {
        self.track.name()
    }
    fn source(&self) -> TrackSource {
        self.track.source()
    }
    fn kind(&self) -> i32 {
        self.track.kind()
    }
    fn is_subscribed(&self) -> Result<bool, RtcError> {
        Ok(true)
    }
    fn audio_features(&self) -> Vec<AudioTrackFeature> {
        vec![AudioTrackFeature::PreconnectBuffer]
    }
    fn open_audio_stream(
        &self,
        _: AudioStreamOptions,
    ) -> Result<Arc<dyn AudioInputStream>, RtcError> {
        Ok(self.processor.clone())
    }
    fn open_video_stream(&self) -> Result<Box<dyn StreamOutput<VideoFrameRef>>, RtcError> {
        self.track.open_video_stream()
    }
}

struct Frames(SdkRx<AudioFrame>);

impl StreamOutput<AudioStreamItem> for Frames {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<AudioStreamItem>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await.map(AudioStreamItem::Frame)) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<AudioStreamItem> for Frames {}

struct Processor {
    fail: bool,
    live: Mutex<Option<SdkRx<AudioFrame>>>,
    formats: Mutex<Vec<(u32, u16)>>,
}

impl Processor {
    fn new(fail: bool) -> (Arc<Self>, SdkTx<AudioFrame>) {
        let (tx, rx) = sdk_queue(50);
        let processor = Self { fail, live: Mutex::new(Some(rx)), formats: Mutex::new(Vec::new()) };
        (Arc::new(processor), tx)
    }
}

impl AudioInputStream for Processor {
    fn frames(&self) -> Result<Box<dyn StreamOutput<AudioStreamItem>>, RtcError> {
        Ok(Box::new(Frames(self.live.lock().unwrap().take().expect("one live stream"))))
    }
    fn process_audio<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        mut frames: Vec<AudioFrame>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<AudioStreamItem>, RtcError>> {
        Box::pin(async move {
            self.formats
                .lock()
                .unwrap()
                .extend(frames.iter().map(|frame| (frame.sample_rate, frame.num_channels)));
            if self.fail {
                return Err(RtcError::host("processor failed"));
            }
            for frame in &mut frames {
                frame.data = [11, 0].repeat(frame.samples_per_channel as usize).into();
            }
            Ok(frames.into_iter().map(AudioStreamItem::Frame).collect())
        })
    }
}

fn subscribed(
    participant: &Arc<FakeRemoteParticipant>,
    publication: Arc<dyn TrackPublication>,
) -> RoomEvent {
    RoomEvent::TrackSubscribed { participant: participant.clone(), publication }
}

fn pre_connect(room: &Arc<FakeRoom>, clock: Arc<VirtualClock>) -> Arc<PreConnectAudioHandler> {
    let spawner = Arc::new(TokioSpawner::current());
    PreConnectAudioHandler::new(
        room.clone(),
        spawner,
        clock,
        Duration::from_secs(3),
        (24_000, 1),
        None,
    )
}

fn input(
    room: &Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    handler: Option<Arc<PreConnectAudioHandler>>,
) -> Arc<ParticipantAudioInput> {
    session_input(room, clock, AudioInputConfig::default(), handler).0
}

/// One session's microphone reader, and the processor failures it reports to that session, one
/// at a time: the session takes each report before the next fits.
fn session_input(
    room: &Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    config: AudioInputConfig,
    handler: Option<Arc<PreConnectAudioHandler>>,
) -> (Arc<ParticipantAudioInput>, tokio::sync::mpsc::Receiver<ProcessorFailure>) {
    let spawner = Arc::new(TokioSpawner::current());
    let (failures, reported) = tokio::sync::mpsc::channel(1);
    (ParticipantAudioInput::new(room.clone(), spawner, clock, config, handler, failures), reported)
}

// Catches: a processor failure reported to every session in the room or once per failed frame,
// a report dropped because the session had yet to take the previous one, and a recovered
// processor never reported again.
#[tokio::test(start_paused = true)]
async fn a_failing_processor_span_is_one_error_on_the_session_whose_input_failed() {
    let (room, clock) = fake_room("processor-spans");
    let user = room.add_participant("user", ParticipantKind::Standard);
    let mic = user.publish_microphone("TR_mic");
    // The denoiser rejects frames 0, 1, 3 and 4: two spans, the second after a clean frame.
    mic.fail_processors(FailingProcessors {
        denoiser: Faults::Flapping { period: 3, fails: 2 },
        gain_control: Faults::None,
        first_ref: 1 << 40,
        refs: Arc::new(NoRefs),
        spans: Arc::new(|_, _| {}),
    });
    let denoised = AudioInputConfig {
        noise_cancellation: Some(lk_agents_abi::HostRef::new(1, &(Arc::new(NoRefs) as _))),
        auto_gain_control: false,
        pre_connect_audio: false,
        ..Default::default()
    };
    let plain = AudioInputConfig { noise_cancellation: None, ..denoised.clone() };
    let (failing, mut failing_reports) = session_input(&room, clock.clone(), denoised, None);
    let (healthy, mut healthy_reports) = session_input(&room, clock, plain, None);
    for input in [&failing, &healthy] {
        input.set_participant(Some("user".into())).unwrap();
        input.handle_event(&subscribed(&user, mic.clone())).unwrap();
    }
    yields(4).await;
    // The frames arrive together: the second span opens before the session took the first report.
    for chunk in 0..5i16 {
        mic.push_frame(AudioFrame::new(chunk.to_le_bytes().repeat(480), 24000, 1, 480));
    }
    // A backlog may arrive merged: each input hears the five frames' samples.
    let hearing = async {
        for input in [&failing, &healthy] {
            let mut heard = 0;
            while heard < 5 * 480 {
                let frame = input.next_frame(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap();
                heard += frame.expect("the input stopped hearing").samples_per_channel;
            }
        }
    };
    let reports = async { [failing_reports.recv().await, failing_reports.recv().await] };
    let ((), reports) =
        tokio::time::timeout(Duration::from_secs(1), async { tokio::join!(hearing, reports) })
            .await
            .expect("every failing span is reported");
    assert_eq!(reports.map(|failure| failure.unwrap().label), ["fake.Denoiser"; 2]);
    yields(4).await;
    assert!(failing_reports.try_recv().is_err(), "one report per failing span");
    assert!(healthy_reports.try_recv().is_err(), "another session's input never failed");
    failing.aclose().await;
    healthy.aclose().await;
}

struct NoRefs;

impl lk_agents_abi::errors::RefRegistry for NoRefs {
    fn release(&self, _: u64) {}
}

#[tokio::test]
async fn track_replacement_releases_pending_preconnect_input() {
    let (room, clock) = fake_room("replace-pending-preconnect");
    let handler = pre_connect(&room, clock.clone());
    let input = input(&room, clock, Some(handler.clone()));
    let user = room.add_participant("user", ParticipantKind::Standard);
    input.set_participant(Some("user".into())).unwrap();
    let (processor, live) = Processor::new(false);
    let old = Processed { track: user.publish_microphone("old"), processor };
    input.handle_event(&subscribed(&user, Arc::new(old))).unwrap();
    tokio::task::yield_now().await;
    assert_eq!(input.reading(), Some("old".into()), "the head is awaited, the track is read");
    let new = user.publish_microphone("new");
    input.handle_event(&subscribed(&user, new)).unwrap();
    yields(4).await;
    assert!(
        !live.push(silence_frame(480, 24000, 1)),
        "replacing a track must release its stream without waiting for pre-connect data"
    );
    input.aclose().await;
    handler.aclose().await;
}

#[tokio::test]
async fn preconnect_uses_session_format_and_never_replays_failed_processing() {
    for fail in [false, true] {
        let (room, clock) = fake_room("process");
        let handler = pre_connect(&room, clock.clone());
        assert!(handler.register());
        let input = input(&room, clock, Some(handler.clone()));
        let user = room.add_participant("user", ParticipantKind::Standard);
        input.set_participant(Some("user".into())).unwrap();
        let (rate, channels, pcm) = if fail {
            ("24000", "1", [100, 0].repeat(480))
        } else {
            ("48000", "2", [100, 0, 200, 0].repeat(960))
        };
        let info = pcm_info("TR_buffer", rate, channels);
        room.send_byte_stream("user", info, [pcm]).unwrap();
        let (processor, _live) = Processor::new(fail);
        let track = Processed { track: user.publish_microphone("TR_buffer"), processor };
        let processor = track.processor.clone();
        input.handle_event(&subscribed(&user, Arc::new(track))).unwrap();
        if fail {
            assert!(
                matches!(input.next_frame(&lk_agents_abi::telemetry::Cx::NONE).await, Err(error) if error.to_string().contains("processor failed")),
                "processing failure must reach the session input"
            );
        } else {
            let frame =
                input.next_frame(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap().unwrap();
            assert_eq!((frame.sample_rate, frame.num_channels), (24000, 1));
            assert!(
                frame.data.as_chunks::<2>().0.iter().all(|b| i16::from_le_bytes(*b) == 11),
                "only processed normalized PCM may reach the session"
            );
        }
        assert!(
            processor.formats.lock().unwrap().iter().all(|format| *format == (24000, 1)),
            "processing sees normalized session audio"
        );
        input.aclose().await;
        handler.aclose().await;
    }
}

// ------------------------------------------------------------------------ the pre-connect buffer

fn pcm_info(track: &str, rate: &str, channels: &str) -> StreamInfo {
    StreamInfo {
        id: track.into(),
        topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
        attributes: [
            ("trackId".into(), track.into()),
            ("sampleRate".into(), rate.into()),
            ("channels".into(), channels.into()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

#[tokio::test]
async fn failed_registration_does_not_remove_another_stream_owner() {
    let (room, clock) = fake_room("registration-owner");
    let mut original = byte_streams(room.as_ref(), PRE_CONNECT_AUDIO_BUFFER_STREAM).unwrap();
    let handler = pre_connect(&room, clock);
    assert!(!handler.register());
    handler.aclose().await;
    let info = StreamInfo {
        id: "original-owner".into(),
        topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
        ..Default::default()
    };
    room.send_byte_stream("user", info, [vec![1, 2]]).unwrap();
    let mut stream = original.next().await.unwrap().unwrap();
    assert_eq!(stream.chunks.next().await.unwrap().unwrap(), vec![1, 2]);
    let deadline = Ts::from_nanos(u64::MAX);
    assert_eq!(stream.close(deadline).await, Closed::Clean);
    assert_eq!(original.close(deadline).await, Closed::Clean);
}

#[tokio::test]
async fn closed_preconnect_input_cannot_register_again() {
    let (room, clock) = fake_room("closed-registration");
    let handler = pre_connect(&room, clock);
    handler.aclose().await;
    assert!(!handler.register());
    assert!(byte_streams(room.as_ref(), PRE_CONNECT_AUDIO_BUFFER_STREAM).is_ok());
}

#[tokio::test]
async fn preconnect_overload_preserves_already_admitted_tracks() {
    let (room, clock) = fake_room("track-capacity");
    let handler = pre_connect(&room, clock);
    assert!(handler.register());
    let tracks = (0..lk_agents_voice::room_io::machine::pre_connect_state::MAX_TRACKS)
        .map(|n| TrackSid::from(format!("TR_{n}")))
        .collect::<Vec<_>>();
    let mut waiters =
        tracks.iter().map(|track| Box::pin(handler.wait_for_data(track))).collect::<Vec<_>>();
    for waiter in &mut waiters {
        pending(waiter.as_mut()).await;
    }
    let overload = handler.wait_for_data(&"TR_overload".into()).await;
    assert!(overload.is_err_and(|error| error.to_string().contains("track capacity")));
    room.send_byte_stream("user", pcm_info("TR_0", "24000", "1"), [vec![1; 960]]).unwrap();
    let frames = waiters.remove(0).await.unwrap();
    assert_eq!(frames.iter().map(|frame| frame.data.len()).sum::<usize>(), 960);
    handler.aclose().await;
    for waiter in waiters {
        assert!(waiter.await.is_err());
    }
}

#[tokio::test]
async fn preconnect_receive_deadline_discards_partial_pcm() {
    let (room, clock) = fake_room("receive-deadline");
    let handler = pre_connect(&room, clock.clone());
    assert!(handler.register());
    let chunks = room.open_byte_stream("user", pcm_info("TR_expired", "24000", "1")).unwrap();
    chunks.send(vec![1; 960]).await.unwrap();
    let track = TrackSid::from("TR_expired");
    let mut result = Box::pin(handler.wait_for_data(&track));
    pending(result.as_mut()).await;
    tokio::task::yield_now().await;
    clock.advance(Duration::from_secs(3));
    assert!(result.await.is_err());
    tokio::task::yield_now().await;
    assert!(chunks.send(vec![2; 960]).await.is_err(), "expired input must be closed");
    handler.aclose().await;
}

#[tokio::test]
async fn preconnect_duration_is_bounded_before_transfer() {
    let (room, clock) = fake_room("duration");
    let handler = pre_connect(&room, clock);
    assert!(handler.register());
    let info = pcm_info("TR_buffer", "24000", "1");
    room.send_byte_stream("user", info, (0..501).map(|_| vec![1; 960])).unwrap();
    assert!(
        handler.wait_for_data(&"TR_buffer".into()).await.is_err(),
        "more than ten seconds of pre-connect PCM must be refused"
    );
    handler.aclose().await;
}

// ------------------------------------------------------------------------- the agent's track

fn audio_output(room: Arc<FakeRoom>, clock: Arc<VirtualClock>) -> Arc<ParticipantAudioOutput> {
    let spawner = Arc::new(TokioSpawner::current());
    ParticipantAudioOutput::new(room, clock, spawner, AudioOutputConfig::default())
}

// Catches: closing an output that holds a captured segment it never played (paused) leaving that
// segment's playout waiting for ever.
#[tokio::test]
async fn close_finishes_a_captured_segment_as_interrupted() {
    let (room, clock) = fake_room("close-segment");
    let output = audio_output(room, clock);
    output.start().await.unwrap();
    output.pause();
    output.capture_frame(&silence_frame(480, 24000, 1)).await.unwrap();
    output.aclose().await;
    let finished = within_a_second(output.tracker().wait_for_playout())
        .await
        .expect("close must settle the captured segment");
    assert!(finished.interrupted);
    assert_eq!(finished.playback_position, Duration::ZERO);
}

#[tokio::test]
async fn publication_arriving_after_close_is_retired() {
    let (room, clock) = fake_room("late-publish");
    let local = room.local_participant();
    local.flow_publications(false);
    let output = audio_output(room, clock);
    let mut start = Box::pin(output.start());
    pending(start.as_mut()).await;
    let mut close = Box::pin(output.aclose());
    pending(close.as_mut()).await;
    local.flow_publications(true);
    close.await;
    let result = start.await;
    assert!(local.publications().is_empty(), "the late SID must be unpublished");
    assert!(result.is_err(), "closed output must refuse adoption");
}

#[tokio::test]
async fn publication_failure_releases_capture_waiters() {
    let (room, clock) = fake_room("failed-publish");
    room.local_participant().fail_with(RtcError::host("publish failed"));
    let output = audio_output(room, clock);
    assert!(output.start().await.is_err());
    let result = within_a_second(
        output.capture(&lk_agents_abi::telemetry::Cx::NONE, &silence_frame(480, 24000, 1)),
    )
    .await;
    assert!(matches!(result, Some(Err(_))), "capture must receive the publication failure");
    output.aclose().await;
}

#[tokio::test]
async fn paused_room_output_backpressures_capture() {
    let (room, clock) = fake_room("backpressure");
    let output = audio_output(room, clock);
    output.start().await.unwrap();
    output.pause();
    let result = lk_agents_abi::timeout(
        &RealClock::new(),
        Duration::from_millis(50),
        output.capture(&lk_agents_abi::telemetry::Cx::NONE, &silence_frame(24000, 24000, 1)),
    )
    .await;
    assert!(
        result.is_err(),
        "a paused sink must bound accepted audio instead of buffering a second"
    );
    output.aclose().await;
}

#[tokio::test]
async fn concurrent_starts_share_one_publication() {
    let (room, clock) = fake_room("joined-start");
    let local = room.local_participant();
    let output = audio_output(room, clock);
    local.flow_publications(false);
    let mut first = Box::pin(output.start());
    let mut second = Box::pin(output.start());
    pending(first.as_mut()).await;
    pending(second.as_mut()).await;
    local.flow_publications(true);
    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(local.publications().len(), 1, "start joins its existing creation");
    output.aclose().await;
    assert!(local.publications().is_empty());
}

#[tokio::test]
async fn cancelled_start_keeps_ownership_of_the_created_track() {
    let (room, clock) = fake_room("cancel-start");
    let local = room.local_participant();
    local.flow_publications(false);
    let output = audio_output(room, clock);
    let mut start = Box::pin(output.start());
    pending(start.as_mut()).await;
    tokio::task::yield_now().await;
    assert_eq!(
        local.publications().len(),
        1,
        "the server has created the track before acknowledging it"
    );
    drop(start);
    let mut close = Box::pin(output.aclose());
    pending(close.as_mut()).await;
    local.flow_publications(true);
    close.await;
    assert!(local.publications().is_empty(), "cancellation must not orphan a created SID");
}

#[tokio::test]
async fn source_capture_failure_releases_playout_and_refuses_further_audio() {
    async fn finished(output: &ParticipantAudioOutput) -> PlaybackFinished {
        within_a_second(output.tracker().wait_for_playout())
            .await
            .expect("source failure must settle the captured segment")
    }

    let mut outcomes = Vec::new();
    for mode in ["fresh", "queued", "heard", "blocked"] {
        let (room, clock) = fake_room(mode);
        let local = room.local_participant();
        let output = audio_output(room, clock.clone());
        output.start().await.unwrap();
        let frame = silence_frame(480, 24_000, 1);
        let (refused, playback) = if mode == "blocked" {
            output.pause();
            for _ in 0..10 {
                output.capture_frame(&frame).await.unwrap();
            }
            let mut waiting = Box::pin(output.capture_frame(&frame));
            pending(waiting.as_mut()).await;
            local.sources()[0].close(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap();
            output.resume();
            let refused = waiting.await;
            (refused, finished(&output).await)
        } else {
            if mode == "queued" || mode == "heard" {
                output.capture_frame(&frame).await.unwrap();
                yields(8).await;
                assert_eq!(local.sources()[0].captured().len(), 1);
                if mode == "heard" {
                    clock.advance(Duration::from_millis(10));
                }
            }
            local.sources()[0].close(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap();
            output.capture_frame(&frame).await.unwrap();
            output.flush();
            yields(8).await;
            let playback = finished(&output).await;
            (output.capture_frame(&frame).await, playback)
        };
        output.flush_segment();
        output.aclose().await;
        let settled = within_a_second(output.tracker().wait_for_playout()).await.is_some();
        outcomes.push((
            mode,
            playback.interrupted,
            playback.playback_position,
            refused.is_err(),
            settled,
        ));
    }
    assert_eq!(
        outcomes,
        [
            ("fresh", true, Duration::ZERO, true, true),
            ("queued", true, Duration::ZERO, true, true),
            ("heard", true, Duration::from_millis(10), true, true),
            ("blocked", true, Duration::ZERO, true, true),
        ],
        "source failure must preserve heard media, refuse parked and later captures, and settle every waiter"
    );
}

#[path = "room_transcription_tests.rs"]
mod transcription;

#[derive(Default)]
struct Spans(Mutex<Vec<lk_agents_abi::telemetry::record::SpanRecord>>);

impl lk_agents_abi::telemetry::record::RecordSink for Spans {
    fn record(&self, record: lk_agents_abi::telemetry::record::Record) {
        if let lk_agents_abi::telemetry::record::Record::Span(span) = record {
            self.0.lock().unwrap().push(span);
        }
    }
}

// Catches: the room's startup spans missing or opened outside `session_start`, the participant
// wait not ended by the link, the audio wait missing its track or first frame, or the agent's
// publication without its track.
#[tokio::test]
async fn the_rooms_startup_runs_under_session_start() {
    use lk_agents_abi::telemetry::record::{SpanId, SpanRef, TraceContext, TraceId, Value};
    let (room, clock) = fake_room("startup-spans");
    let spans = Arc::new(Spans::default());
    let tracer = lk_agents_runtime::net::telemetry::Tracer::new(spans.clone(), clock.clone(), 1);
    let config =
        RoomIoConfig { audio_output: Some(AudioOutputConfig::default()), ..microphone_only() };
    let spawner = Arc::new(TokioSpawner::current());
    let ids = Arc::new(SegmentIds::counting());
    let io = RoomIo::new(room.clone(), clock, spawner, ids, config, None, None, tracer);
    let start = SpanRef { trace_id: TraceId(5), span_id: SpanId(6), sampled: true };
    io.clone().start(TraceContext { span: Some(start), ..Default::default() });
    let microphone =
        room.add_participant("person", ParticipantKind::Standard).publish_microphone("TR_mic");
    io.wait_for_ready().await.unwrap();
    let input = io.audio_input().unwrap();
    assert!(microphone.push_frame(AudioFrame::new(vec![1; 960], 24_000, 1, 480)) > 0);
    input.next_frame(&lk_agents_abi::telemetry::Cx::NONE).await.unwrap().unwrap();
    io.clone().aclose().await;

    let spans = spans.0.lock().unwrap().clone();
    let named = |name: &str| spans.iter().find(|s| s.name == name).expect(name).clone();
    let attr = |s: &lk_agents_abi::telemetry::record::SpanRecord, key: &str| {
        s.attributes.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())
    };
    let (participant, audio, publish) = (
        named("wait_for_participant"),
        named("wait_for_audio_track"),
        named("publish_audio_output"),
    );
    for span in [&participant, &audio, &publish] {
        assert_eq!(span.parent, Some(SpanId(6)), "{} under session_start", span.name);
    }
    assert_eq!(
        attr(&participant, "lk.pii.participant_identity"),
        Some(Value::String("person".into()))
    );
    assert!(participant.end_unix_nanos <= audio.start_unix_nanos, "the link starts the audio wait");
    let events: Vec<_> = audio.events.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(events, ["track_subscribed", "first_frame"]);
    assert!(attr(&audio, "lk.first_frame_delay").is_some());
    assert!(matches!(attr(&publish, "lk.track_id"), Some(Value::String(sid)) if !sid.is_empty()));
}
