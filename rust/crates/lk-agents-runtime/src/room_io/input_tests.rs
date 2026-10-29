use super::*;
use crate::test_support_rtc::{FakeRtc, FakeTrackPublication};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::rtc::{
    ParticipantKind, RtcError, StreamInfo, TrackSource, silence_frame,
};
use lk_agents_net::task::TokioSpawner;
use std::time::Duration;

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
    fn source(&self) -> lk_agents_abi::voice::rtc::TrackSource {
        self.track.source()
    }
    fn kind(&self) -> i32 {
        self.track.kind()
    }
    fn is_subscribed(&self) -> std::result::Result<bool, RtcError> {
        Ok(true)
    }
    fn audio_features(&self) -> Vec<AudioTrackFeature> {
        vec![AudioTrackFeature::PreconnectBuffer]
    }
    fn open_audio_stream(
        &self,
        _: AudioStreamOptions,
    ) -> std::result::Result<Arc<dyn AudioInputStream>, RtcError> {
        Ok(self.processor.clone())
    }
    fn open_video_stream(
        &self,
    ) -> std::result::Result<Box<dyn StreamSource<VideoFrameRef>>, RtcError> {
        self.track.open_video_stream()
    }
}

struct Frames(BoundedLossy<AudioFrame>);

impl StreamSource<AudioFrame> for Frames {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Option<AudioFrame>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

struct Processor {
    fail: bool,
    live: Mutex<Option<BoundedLossy<AudioFrame>>>,
    formats: Mutex<Vec<(u32, u16)>>,
}

impl Processor {
    fn new(fail: bool) -> (Arc<Self>, LossyTx<AudioFrame>) {
        let (tx, rx) = BoundedLossy::audio();
        let processor = Self { fail, live: Mutex::new(Some(rx)), formats: Mutex::new(Vec::new()) };
        (Arc::new(processor), tx)
    }
}

impl AudioInputStream for Processor {
    fn frames(&self) -> std::result::Result<Box<dyn StreamSource<AudioFrame>>, RtcError> {
        Ok(Box::new(Frames(self.live.lock().unwrap().take().expect("one live stream"))))
    }
    fn process_audio<'a>(
        &'a self,
        mut frames: Vec<AudioFrame>,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Vec<AudioFrame>, RtcError>> {
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
            Ok(frames)
        })
    }
}

fn subscribed(
    participant: &Arc<crate::test_support_rtc::FakeRemoteParticipant>,
    publication: Arc<dyn TrackPublication>,
) -> RoomEvent {
    RoomEvent::TrackSubscribed { participant: participant.clone(), publication }
}

#[tokio::test]
async fn track_replacement_releases_pending_preconnect_input() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("replace-pending-preconnect");
    let spawner = Arc::new(TokioSpawner::current());
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        spawner.clone(),
        clock.clone(),
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    let input = ParticipantAudioInput::new(
        room.clone(),
        spawner,
        clock,
        AudioInputConfig::default(),
        Some(handler.clone()),
    );
    let user = room.add_participant("user", ParticipantKind::Standard);
    input.set_participant(Some("user".into())).unwrap();
    let (processor, live) = Processor::new(false);
    let old = Processed { track: user.publish_microphone("old"), processor };
    input.handle_event(&subscribed(&user, Arc::new(old))).unwrap();
    tokio::task::yield_now().await;
    assert_eq!(input.reading(), Some("old".into()), "the head is awaited, the track is read");
    let new = user.publish_microphone("new");
    input.handle_event(&subscribed(&user, new)).unwrap();
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        live.push(silence_frame(480, 24000, 1)),
        Pushed::Closed,
        "replacing a track must release its stream without waiting for pre-connect data"
    );
    input.aclose().await;
    handler.aclose().await;
}

#[tokio::test]
async fn linking_an_arrived_participant_reads_their_existing_track() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("participant-arrival");
    let input = ParticipantAudioInput::new(
        room.clone(),
        Arc::new(TokioSpawner::current()),
        clock,
        AudioInputConfig::default(),
        None,
    );
    input.set_participant(Some("user".into())).unwrap();
    let participant = room.add_participant("user", ParticipantKind::Standard);
    let microphone = participant.publish_track("microphone", TrackSource::Microphone);
    input.set_participant(Some("user".into())).unwrap();
    assert_eq!(input.reading(), Some("microphone".into()));
    input.set_participant(Some("user".into())).unwrap();
    assert_eq!(microphone.stream_requests().len(), 1, "an active track must remain attached");
    assert_eq!(microphone.push_frame(silence_frame(480, 24000, 1)), 1);
    assert_eq!(input.next_frame().await.unwrap().unwrap().samples_per_channel, 480);
    input.aclose().await;
}

#[tokio::test]
async fn preconnect_uses_session_format_and_never_replays_failed_processing() {
    for fail in [false, true] {
        let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
        let room = rtc.room("process");
        let spawner = Arc::new(TokioSpawner::current());
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            spawner.clone(),
            clock.clone(),
            Duration::from_secs(3),
            (24_000, 1),
            None,
        );
        assert!(handler.register());
        let input = ParticipantAudioInput::new(
            room.clone(),
            spawner,
            clock,
            AudioInputConfig::default(),
            Some(handler.clone()),
        );
        let user = room.add_participant("user", ParticipantKind::Standard);
        input.set_participant(Some("user".into())).unwrap();
        room.send_byte_stream(
            "user",
            StreamInfo {
                id: "buffer".into(),
                topic: lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
                attributes: [
                    ("trackId".into(), "TR_buffer".into()),
                    ("sampleRate".into(), if fail { "24000" } else { "48000" }.into()),
                    ("channels".into(), if fail { "1" } else { "2" }.into()),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
            [if fail { [100, 0].repeat(480) } else { [100, 0, 200, 0].repeat(960) }],
        )
        .unwrap();
        let (processor, _live) = Processor::new(fail);
        let track = Processed { track: user.publish_microphone("TR_buffer"), processor };
        let processor = track.processor.clone();
        input.handle_event(&subscribed(&user, Arc::new(track))).unwrap();
        if fail {
            assert!(
                matches!(input.next_frame().await, Err(error) if error.to_string().contains("processor failed")),
                "processing failure must reach the session input"
            );
        } else {
            let frame = input.next_frame().await.unwrap().unwrap();
            assert_eq!((frame.sample_rate, frame.num_channels), (24000, 1));
            assert!(
                frame
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|b| i16::from_le_bytes([b[0], b[1]]) == 11),
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
