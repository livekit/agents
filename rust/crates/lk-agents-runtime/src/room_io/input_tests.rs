use super::*;
use crate::room_io::owner::tests::pending;
use crate::test_support_rtc::FakeRtc;
use lk_agents_abi::{TokioSpawner, errors::ProviderError};
use std::time::Duration;

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
    formats: Mutex<Vec<(u32, u16)>>,
}

impl AudioInputStream for Processor {
    fn frames(
        &self,
    ) -> std::result::Result<Box<dyn StreamSource<AudioFrame>>, lk_agents_abi::voice::rtc::RtcError>
    {
        let (tx, rx) = BoundedLossy::audio();
        drop(tx);
        Ok(Box::new(Frames(rx)))
    }
    fn process_audio<'a>(
        &'a self,
        mut frames: Vec<AudioFrame>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        std::result::Result<Vec<AudioFrame>, lk_agents_abi::voice::rtc::RtcError>,
    > {
        Box::pin(async move {
            self.formats
                .lock()
                .unwrap()
                .extend(frames.iter().map(|frame| (frame.sample_rate, frame.num_channels)));
            if self.fail {
                return Err(lk_agents_abi::voice::rtc::RtcError::host("processor failed"));
            }
            for frame in &mut frames {
                frame.data = [11, 0].repeat(frame.samples_per_channel as usize).into();
            }
            Ok(frames)
        })
    }
}

#[tokio::test]
async fn track_replacement_releases_pending_preconnect_input() {
    use std::future::Future;
    use std::task::Poll;

    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("replace-pending-preconnect");
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(TokioSpawner::current()),
        clock.clone(),
        Duration::from_secs(3),
        (24_000, 1),
    );
    let input = ParticipantAudioInput::new(
        room,
        Arc::new(TokioSpawner::current()),
        clock,
        AudioInputConfig::default(),
        Some(handler.clone()),
    );
    input.reader.link(Some("user".into()));
    let old = input
        .reader
        .claim(&"user".into(), TrackSource::Microphone, &"old".into(), AUDIO_SOURCES)
        .unwrap();
    let processor = Arc::new(Processor { fail: false, formats: Mutex::new(Vec::new()) });
    let (frames, stream) = BoundedLossy::audio();
    let mut pump =
        Box::pin(input.clone().pump(old, "old".into(), Box::new(Frames(stream)), Some(processor)));
    pending(pump.as_mut()).await;
    let new = input
        .reader
        .claim(&"user".into(), TrackSource::Microphone, &"new".into(), AUDIO_SOURCES)
        .unwrap();
    input.reader.opening_failed(&new);
    std::future::poll_fn(|cx| {
        assert!(
            pump.as_mut().poll(cx).is_ready(),
            "replacing a track must release its stream without waiting for pre-connect data"
        );
        Poll::Ready(())
    })
    .await;
    assert_eq!(
        frames.push(silence_frame(480, 24000, 1)),
        Pushed::Closed,
        "the obsolete subscription must be closed"
    );
    input.aclose().await;
    handler.aclose().await;
}

#[tokio::test]
async fn linking_an_arrived_participant_reads_their_existing_track() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("participant-arrival");
    let input = ParticipantAudioInput::new(
        room.clone(),
        Arc::new(TokioSpawner::current()),
        clock,
        AudioInputConfig::default(),
        None,
    );
    input.set_participant(Some("user".into())).unwrap();
    let participant =
        room.add_participant("user", lk_agents_abi::voice::rtc::ParticipantKind::Standard);
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
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("process");
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(TokioSpawner::current()),
            clock.clone(),
            Duration::from_secs(3),
            (24_000, 1),
        );
        assert!(handler.register());
        let input = ParticipantAudioInput::new(
            room.clone(),
            Arc::new(TokioSpawner::current()),
            clock,
            AudioInputConfig::default(),
            Some(handler.clone()),
        );
        input.reader.link(Some("user".into()));
        let lease = input
            .reader
            .claim(&"user".into(), TrackSource::Microphone, &"TR_buffer".into(), AUDIO_SOURCES)
            .unwrap();
        room.send_byte_stream(
            "user",
            lk_agents_abi::voice::rtc::StreamInfo {
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
        let processor = Arc::new(Processor { fail, formats: Mutex::new(Vec::new()) });
        let (tx, stream) = BoundedLossy::audio();
        drop(tx);
        input
            .clone()
            .pump(lease, "TR_buffer".into(), Box::new(Frames(stream)), Some(processor.clone()))
            .await;
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
