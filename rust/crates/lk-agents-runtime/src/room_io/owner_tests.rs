use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::task::Poll;

pub(crate) async fn pending<F: Future>(mut future: Pin<&mut F>) {
    poll_fn(|context| {
        assert!(future.as_mut().poll(context).is_pending(), "operation should remain pending");
        Poll::Ready(())
    })
    .await;
}

use super::*;
use crate::test_support_rtc::FakeRtc;
use lk_agents_abi::voice::room_io::{AudioInputConfig, AudioOutputConfig};
use lk_agents_abi::voice::rtc::{LocalParticipant, RtcError, StreamInfo};
use lk_agents_abi::{StreamSource, TokioSpawner, VirtualClock};
use std::time::Duration;

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

fn room_io(room: Arc<dyn RtcRoom>, clock: Arc<VirtualClock>, config: RoomIoConfig) -> Arc<RoomIo> {
    RoomIo::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        Arc::new(SegmentIds::counting()),
        config,
        None,
        None,
    )
}

#[tokio::test(start_paused = true)]
async fn publication_failure_settles_every_room_readiness_waiter() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("failed-ready");
    room.add_participant("person", ParticipantKind::Standard);
    room.local_participant().fail_with(RtcError::host("publication refused"));
    let io = room_io(room, clock, config());
    let mut first = Box::pin(io.wait_for_ready());
    let mut second = Box::pin(io.wait_for_ready());
    pending(first.as_mut()).await;
    pending(second.as_mut()).await;
    io.start();
    let completed =
        lk_agents_abi::timeout(&lk_agents_abi::RealClock::new(), Duration::from_secs(1), async {
            let first = first.await.expect_err("publication failure must fail readiness");
            let second = second.await.expect_err("every waiter must see the same failure");
            assert_eq!(first, second);
            assert!(first.to_string().contains("publication refused"));
        })
        .await;
    io.aclose().await;
    assert!(completed.is_ok(), "publication failure must release every readiness waiter");
}

#[tokio::test(start_paused = true)]
async fn closing_before_linking_settles_room_readiness() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("closed-ready");
    let io = room_io(room, clock, config());
    io.start();
    io.aclose().await;
    let ready = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        io.wait_for_ready(),
    )
    .await
    .expect("close must not leave readiness waiting for a participant");
    assert!(ready.is_err(), "close must never fabricate successful readiness");
}

#[tokio::test]
async fn repeated_room_start_keeps_one_event_subscription() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("one-start");
    let io = room_io(room.clone(), clock, config());
    io.start();
    io.start();
    assert_eq!(room.event_subscriber_count(), 1);
    io.aclose().await;
}

struct StreamingRoom {
    room: Arc<crate::test_support_rtc::FakeRoom>,
    local: Arc<dyn LocalParticipant>,
    text: watch::Sender<Option<lk_agents_abi::voice::rtc::TextStreamSender>>,
}

impl lk_agents_abi::utils::clock::HostClock for StreamingRoom {
    fn host_clock(&self) -> Arc<dyn Clock> {
        self.room.host_clock()
    }
}

struct TextChunks(mpsc::Receiver<String>);
impl StreamSource<String> for TextChunks {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        std::result::Result<Option<String>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl RtcRoom for StreamingRoom {
    fn name(&self) -> std::result::Result<String, RtcError> {
        self.room.name()
    }
    fn connection_state(&self) -> std::result::Result<ConnectionState, RtcError> {
        self.room.connection_state()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local
    }
    fn participant(
        &self,
        id: &ParticipantIdentity,
    ) -> std::result::Result<Option<Arc<dyn lk_agents_abi::voice::rtc::RemoteParticipant>>, RtcError>
    {
        self.room.participant(id)
    }
    fn remote_participants(
        &self,
    ) -> std::result::Result<Vec<Arc<dyn lk_agents_abi::voice::rtc::RemoteParticipant>>, RtcError>
    {
        self.room.remote_participants()
    }
    fn event_bus(&self) -> &lk_agents_abi::voice::rtc::RoomEventBus {
        self.room.event_bus()
    }
    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: lk_agents_abi::voice::rtc::ByteStreamSender,
    ) -> std::result::Result<(), RtcError> {
        self.room.register_byte_stream_handler(topic, streams)
    }
    fn unregister_byte_stream_handler(&self, topic: &str) -> std::result::Result<(), RtcError> {
        self.room.unregister_byte_stream_handler(topic)
    }
    fn register_text_stream_handler(
        &self,
        _topic: &str,
        streams: lk_agents_abi::voice::rtc::TextStreamSender,
    ) -> std::result::Result<(), RtcError> {
        self.text.send_replace(Some(streams));
        Ok(())
    }
    fn unregister_text_stream_handler(&self, _topic: &str) -> std::result::Result<(), RtcError> {
        self.text.send_replace(None);
        Ok(())
    }
}

#[tokio::test]
async fn text_from_an_earlier_attachment_cannot_drive_the_new_attachment() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("text-attachment");
    room.add_participant("person", ParticipantKind::Standard);
    let room =
        Arc::new(StreamingRoom { local: room.local().clone(), room, text: watch::channel(None).0 });
    let mut config = config();
    config.audio_output = None;
    config.text_input = true;
    let io = room_io(room.clone(), clock, config);
    let mut events = io.events().unwrap();
    io.start();
    let _ = io.wait_for_ready().await;
    while events.try_recv().is_ok() {}
    let (chunks, receiver) = mpsc::channel(4);
    assert!(room.text.borrow().as_ref().unwrap().open(
        StreamInfo { topic: TOPIC_CHAT.into(), ..Default::default() },
        "person".into(),
        Box::new(TextChunks(receiver)),
    ));
    chunks.send("old attachment".into()).await.unwrap();
    // The text reader has accepted this stream and is waiting for its trailer.
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    io.set_participant(Some("person".into())).await.unwrap();
    drop(chunks);
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    let mut delivered_stale = false;
    while let Ok(event) = events.try_recv() {
        delivered_stale |= matches!(event, RoomIoEvent::TextInput(_));
    }
    assert!(!delivered_stale, "a completed old chat stream must retain its attachment identity");
    let (chunks, receiver) = mpsc::channel(4);
    assert!(room.text.borrow().as_ref().unwrap().open(
        StreamInfo { topic: TOPIC_CHAT.into(), ..Default::default() },
        "person".into(),
        Box::new(TextChunks(receiver)),
    ));
    chunks.send("current attachment".into()).await.unwrap();
    drop(chunks);
    let delivered = events.recv().await.expect("the current attachment can send chat");
    assert!(
        matches!(delivered, RoomIoEvent::TextInput(TextInput { text, .. }) if text == "current attachment")
    );
    io.aclose().await;
}

#[tokio::test(start_paused = true)]
async fn room_close_deadline_survives_a_cancelled_waiter_and_a_blocked_sdk_source() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("close-deadline");
    let native_room = room.clone();
    room.add_participant("person", ParticipantKind::Standard);
    let io = room_io(room.clone(), clock.clone(), config());
    io.start();
    io.wait_for_ready().await.unwrap();
    let source = room.local_participant().sources()[0].clone();
    source.flow_closes(false);
    let output = io.audio_output().unwrap();
    let frame = lk_agents_abi::AudioFrame::new(vec![0; 960], 24_000, 1, 480);
    lk_agents_abi::voice::io::AudioOutputExt::capture_frame(output.as_ref(), &frame).await.unwrap();
    let mut first = Box::pin(io.aclose());
    pending(first.as_mut()).await;
    source.wait_for_close_started().await;
    let playback = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        output.tracker().wait_for_playout(),
    )
    .await;
    drop(first);
    let mut second = Box::pin(io.aclose());
    pending(second.as_mut()).await;
    clock.advance(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    let outcome =
        lk_agents_abi::timeout(&lk_agents_abi::RealClock::new(), Duration::from_secs(1), second)
            .await;
    let unpublished_while_source_blocked =
        native_room.local_participant().publications().is_empty();
    source.flow_closes(true);
    lk_agents_abi::timeout(&lk_agents_abi::RealClock::new(), Duration::from_secs(1), async {
        while !source.is_closed() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the SDK cleanup owner survives its waiters");
    io.aclose().await;
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
        lk_agents_abi::Closed::TimedOut
    );
    assert_eq!(
        io.aclose().await,
        lk_agents_abi::Closed::TimedOut,
        "repeated close must retain the original timeout outcome"
    );
}

#[tokio::test(start_paused = true)]
async fn room_disconnection_settles_readiness_before_a_participant_links() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("disconnected-ready");
    let io = room_io(room.clone(), clock, config());
    io.start();
    room.close(DisconnectReason::ClientInitiated);
    let ready = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        io.wait_for_ready(),
    )
    .await;
    io.aclose().await;
    assert!(ready.is_ok(), "a disconnected room must settle readiness without a participant");
    assert!(ready.unwrap().is_err(), "a disconnected room cannot become ready");
}

#[tokio::test]
async fn room_relinks_input_after_the_linked_participant_is_replaced() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("relinked-input");
    let original = room.add_participant("first", ParticipantKind::Standard);
    original.publish_microphone("TR_first");
    let mut config = config();
    config.audio_input = Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() });
    let io = room_io(room.clone(), clock, config);
    io.start();
    io.wait_for_ready().await.unwrap();
    assert_eq!(io.audio_input_leg().unwrap().reading(), Some("TR_first".into()));
    room.remove_participant(&"first".into(), DisconnectReason::ClientInitiated);
    let replacement = room.add_participant("second", ParticipantKind::Standard);
    replacement.publish_microphone("TR_second");
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    let linked = io.linked_participant();
    let reading = io.audio_input_leg().unwrap().reading();
    io.aclose().await;
    assert_eq!(linked, Some("second".into()), "the replacement participant must become linked");
    assert_eq!(reading, Some("TR_second".into()), "the microphone must follow the new link");
}

#[tokio::test]
async fn selecting_the_same_participant_retires_buffered_input_from_the_old_attachment() {
    use lk_agents_abi::voice::io::AudioInput;

    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("reselected-input");
    let person = room.add_participant("person", ParticipantKind::Standard);
    let microphone = person.publish_microphone("TR_mic");
    let mut config = config();
    config.audio_input = Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() });
    let io = room_io(room, clock, config);
    io.start();
    io.wait_for_ready().await.unwrap();
    let input = io.audio_input_leg().unwrap();
    assert!(
        microphone.push_frame(lk_agents_abi::AudioFrame::new(vec![1; 960], 24_000, 1, 480)) > 0
    );
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    io.set_participant(Some("person".into())).await.unwrap();
    assert!(
        microphone.push_frame(lk_agents_abi::AudioFrame::new(vec![2; 960], 24_000, 1, 480)) > 0
    );
    let received = input.next_frame().await.unwrap().unwrap();
    io.aclose().await;
    assert_eq!(received.data.as_ref(), &[2; 960], "old queued audio must not cross reattachment");
}

#[tokio::test(start_paused = true)]
async fn delayed_departure_cannot_detach_a_rejoined_participant_with_the_same_identity() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("participant-lifetime");
    let old = room.add_participant("person", ParticipantKind::Standard);
    let old_publication = old.publish_microphone("microphone");
    let mut config = config();
    config.audio_output = None;
    config.audio_input = Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() });
    let io = room_io(room.clone(), clock, config);
    let mut events = io.events().unwrap();
    io.start();
    io.wait_for_ready().await.unwrap();
    room.remove_participant(&"person".into(), DisconnectReason::ClientInitiated);
    let current = room.add_participant("person", ParticipantKind::Standard);
    current.publish_microphone("microphone");
    io.set_participant(Some("person".into())).await.unwrap();
    room.event_bus().track_unpublished(old.clone(), old_publication);
    room.event_bus().participant_disconnected(old, DisconnectReason::ClientInitiated);
    room.press_dtmf("1");
    lk_agents_abi::timeout(&lk_agents_abi::RealClock::new(), Duration::from_secs(1), async {
        loop {
            if matches!(events.recv().await, Some(RoomIoEvent::KeyPressed)) {
                break;
            }
        }
    })
    .await
    .expect("the current participant must remain linked after the delayed old departure");
    assert_eq!(io.audio_input_leg().unwrap().reading(), Some("microphone".into()));
    io.aclose().await;
}

#[tokio::test]
async fn saturated_room_events_are_reported_to_the_selection_caller() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("event-capacity");
    room.add_participant("first", ParticipantKind::Standard);
    room.add_participant("second", ParticipantKind::Standard);
    let mut config = config();
    config.audio_output = None;
    let io = room_io(room, clock, config);
    for n in 0..ROOM_EVENT_CAPACITY {
        io.set_participant(Some(if n % 2 == 0 { "first" } else { "second" }.into())).await.unwrap();
    }
    assert!(
        matches!(io.set_participant(Some("first".into())).await, Err(IoError::Rtc(error)) if error.contains("capacity exceeded"))
    );
    io.aclose().await;
}

#[tokio::test(start_paused = true)]
async fn a_rejoined_identity_gets_a_new_attachment_without_an_explicit_selection() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("automatic-rejoin");
    let old = room.add_participant("person", ParticipantKind::Standard);
    old.publish_microphone("old-track");
    let mut config = config();
    config.audio_output = None;
    config.audio_input = Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() });
    let io = room_io(room.clone(), clock, config);
    let mut events = io.events().unwrap();
    io.start();
    io.wait_for_ready().await.unwrap();
    while events.try_recv().is_ok() {}
    room.remove_participant(&"person".into(), DisconnectReason::ClientInitiated);
    let current = room.add_participant("person", ParticipantKind::Standard);
    current.publish_microphone("new-track");
    let event = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        events.recv(),
    )
    .await
    .expect("a new participant lifetime must be linked without an explicit application selection");
    assert!(
        matches!(event, Some(RoomIoEvent::ParticipantLinked(ParticipantLinked { identity })) if identity.as_str()=="person")
    );
    assert_eq!(io.audio_input_leg().unwrap().reading(), Some("new-track".into()));
    io.aclose().await;
}

// Catches: a burst of transcripts the performer had not drained yet latched a capacity failure,
// and the next room event carried it into RoomIo::fail, closing the room's audio output: the
// agent was mute for the rest of the call.
#[tokio::test(start_paused = true)]
async fn a_transcript_burst_leaves_the_room_audio_output_open() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("transcript-burst");
    room.add_participant("person", ParticipantKind::Standard);
    let mut config = config();
    config.text_output = Some(lk_agents_abi::voice::room_io::TextOutputConfig::default());
    let io = room_io(room.clone(), clock, config);
    io.start();
    io.wait_for_ready().await.unwrap();
    for n in 0..8 {
        io.on_user_input_transcribed(&format!("word {n}"), n % 2 == 1)
            .await
            .expect("a transcript backlog is not the caller's error");
    }
    room.remove_participant(&"person".into(), DisconnectReason::ClientInitiated);
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    let closing = io.is_closing();
    io.aclose().await;
    assert!(!closing, "a transcription backlog must not close the room's audio output");
    let streams = room.local_participant().text_streams();
    assert_eq!(
        streams.last().map(|stream| stream.text()),
        Some("word 7".into()),
        "the backlog delivers the latest transcript"
    );
}
