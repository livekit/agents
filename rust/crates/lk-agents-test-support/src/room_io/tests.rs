use super::input::ParticipantAudioInput;
use super::owner::RoomIo;
use super::pre_connect_audio::PreConnectAudioHandler;
use crate::rtc::{AudioTrackFeature, FakeRtc, ParticipantKind, StreamInfo};
use lk_agents_abi::voice::io::AudioInput;
use lk_agents_abi::voice::room_io::{AudioInputConfig, RoomIoConfig};
use lk_agents_abi::voice::room_owner::{RoomAudioInputPort, RoomIo as _};
use lk_agents_abi::{AudioFrame, Closed, Spawner, TokioSpawner};
use std::sync::Arc;
use std::time::Duration;

fn spawner() -> Arc<dyn Spawner> {
    Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()))
}

fn sample(value: u8) -> AudioFrame {
    AudioFrame::new(vec![value; 320], 16_000, 1, 160)
}

#[tokio::test(start_paused = true)]
async fn closing_input_settles_late_pulls_and_releases_the_subscription() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("close-input");
    let caller = room.add_participant("caller", ParticipantKind::Standard);
    let microphone = caller.publish_microphone("mic");
    let input =
        ParticipantAudioInput::new(room, clock, spawner(), AudioInputConfig::default(), None);
    input.set_participant(Some("caller".into())).unwrap();
    assert_eq!(input.aclose().await, Closed::Clean);
    assert!(
        tokio::time::timeout(Duration::from_millis(10), input.next_frame())
            .await
            .expect("closed input left a late pull parked")
            .unwrap()
            .is_none()
    );
    assert_eq!(microphone.push_frame(sample(1)), 0, "the closed subscription leaked");
    assert_eq!(input.aclose().await, Closed::Clean);
}

#[tokio::test(start_paused = true)]
async fn reattachment_discards_queued_frames_from_the_previous_attachment() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("reattach-input");
    let caller = room.add_participant("caller", ParticipantKind::Standard);
    let microphone = caller.publish_microphone("mic");
    let input = ParticipantAudioInput::new(
        room,
        clock,
        spawner(),
        AudioInputConfig { sample_rate: 16_000, ..Default::default() },
        None,
    );
    input.set_participant(Some("caller".into())).unwrap();
    assert_eq!(microphone.push_frame(sample(1)), 1);
    tokio::task::yield_now().await;
    input.on_detached();
    input.on_attached();
    assert_eq!(microphone.push_frame(sample(2)), 1);
    let received = tokio::time::timeout(Duration::from_millis(10), input.next_frame())
        .await
        .expect("new attachment lost its first frame")
        .unwrap()
        .unwrap();
    assert_eq!(received.data, sample(2).data);
    assert_eq!(input.aclose().await, Closed::Clean);
}

#[tokio::test(start_paused = true)]
async fn closing_input_cancels_a_pending_preconnect_head_before_its_deadline() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("cancel-head");
    let caller = room.add_participant("caller", ParticipantKind::Standard);
    let microphone = caller.publish_microphone("mic");
    microphone.set_audio_features(vec![AudioTrackFeature::PreconnectBuffer]);
    let head = PreConnectAudioHandler::new(
        room.clone(),
        spawner(),
        clock.clone(),
        Duration::from_secs(30),
        (16_000, 1),
    )
    .unwrap();
    let input = ParticipantAudioInput::new(
        room,
        clock,
        spawner(),
        AudioInputConfig::default(),
        Some(head.clone()),
    );
    input.set_participant(Some("caller".into())).unwrap();
    tokio::task::yield_now().await;
    assert_eq!(
        tokio::time::timeout(Duration::from_millis(10), input.aclose())
            .await
            .expect("close waited for the receive deadline"),
        Closed::Clean
    );
    assert_eq!(microphone.push_frame(sample(1)), 0);
    assert_eq!(head.aclose().await, Closed::Clean);
}

fn media_off() -> RoomIoConfig {
    RoomIoConfig {
        audio_input: None,
        audio_output: None,
        text_output: None,
        text_input: false,
        video_input: false,
        ..Default::default()
    }
}

#[tokio::test(start_paused = true)]
async fn closing_before_a_participant_joins_keeps_unowned_chat_handlers() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("topic-owner");
    let mut application_chat = room.text_streams(lk_agents_abi::TOPIC_CHAT).unwrap();
    let io = RoomIo::new(room.clone(), clock, spawner(), media_off());
    io.clone().start();
    assert_eq!(io.clone().aclose().await, Closed::Clean);
    assert!(io.wait_for_ready().await.is_err(), "closed startup fabricated readiness");
    room.send_text_stream(
        "caller",
        StreamInfo { topic: lk_agents_abi::TOPIC_CHAT.into(), ..Default::default() },
        ["application message"],
    )
    .unwrap();
    let mut message = application_chat.next().await.unwrap().unwrap();
    assert_eq!(message.read_all().await.unwrap(), "application message");
    message.close(lk_agents_abi::Ts::ZERO).await;
    application_chat.close(lk_agents_abi::Ts::ZERO).await;
}

#[tokio::test(start_paused = true)]
async fn chat_registration_failure_settles_readiness_without_stealing_the_topic() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("topic-refusal");
    let mut application_chat = room.text_streams(lk_agents_abi::TOPIC_CHAT).unwrap();
    let io = RoomIo::new(
        room.clone(),
        clock,
        spawner(),
        RoomIoConfig { text_input: true, ..media_off() },
    );
    io.clone().start();
    assert!(io.wait_for_ready().await.is_err());
    assert_eq!(io.clone().aclose().await, Closed::Failed);
    room.send_text_stream(
        "caller",
        StreamInfo { topic: lk_agents_abi::TOPIC_CHAT.into(), ..Default::default() },
        ["still owned"],
    )
    .unwrap();
    assert_eq!(
        application_chat.next().await.unwrap().unwrap().read_all().await.unwrap(),
        "still owned"
    );
    application_chat.close(lk_agents_abi::Ts::ZERO).await;
}

#[tokio::test(start_paused = true)]
async fn closing_preconnect_releases_a_stream_that_never_sent_its_trailer() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("head-without-trailer");
    let head = PreConnectAudioHandler::new(
        room.clone(),
        spawner(),
        clock,
        Duration::from_secs(30),
        (16_000, 1),
    )
    .unwrap();
    let source = room
        .open_byte_stream(
            "caller",
            StreamInfo {
                id: "head".into(),
                topic: lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
                mime_type: "audio/pcm".into(),
                attributes: std::collections::BTreeMap::from([
                    ("trackId".into(), "mic".into()),
                    ("sampleRate".into(), "16000".into()),
                    ("channels".into(), "1".into()),
                ]),
                ..Default::default()
            },
        )
        .unwrap();
    tokio::task::yield_now().await;
    assert_eq!(
        tokio::time::timeout(Duration::from_millis(10), head.aclose())
            .await
            .expect("close waited for the absent trailer"),
        Closed::Clean
    );
    assert!(source.is_closed(), "the admitted byte source was dropped without explicit cleanup");
}
