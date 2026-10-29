use super::*;
use crate::room_io::owner::tests::pending;
use lk_agents_abi::Ts;

#[tokio::test]
async fn failed_registration_does_not_remove_another_stream_owner() {
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("registration-owner");
    let mut original =
        crate::rtc::events::byte_streams(room.as_ref(), PRE_CONNECT_AUDIO_BUFFER_STREAM).unwrap();
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        clock,
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    assert!(!handler.register());
    handler.aclose().await;
    let info = lk_agents_abi::voice::rtc::StreamInfo {
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
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("closed-registration");
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        clock,
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    handler.aclose().await;
    assert!(!handler.register());
    assert!(
        crate::rtc::events::byte_streams(room.as_ref(), PRE_CONNECT_AUDIO_BUFFER_STREAM).is_ok()
    );
}

fn pcm_info(track: &str) -> lk_agents_abi::voice::rtc::StreamInfo {
    lk_agents_abi::voice::rtc::StreamInfo {
        id: track.into(),
        topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
        attributes: [
            ("trackId".into(), track.into()),
            ("sampleRate".into(), "24000".into()),
            ("channels".into(), "1".into()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}

#[tokio::test]
async fn preconnect_overload_preserves_already_admitted_tracks() {
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("track-capacity");
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        clock,
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    assert!(handler.register());
    let tracks = (0..lk_agents_voice::room_io::pre_connect_state::MAX_TRACKS)
        .map(|n| TrackSid::from(format!("TR_{n}")))
        .collect::<Vec<_>>();
    let mut waiters =
        tracks.iter().map(|track| Box::pin(handler.wait_for_data(track))).collect::<Vec<_>>();
    for waiter in &mut waiters {
        pending(waiter.as_mut()).await;
    }
    assert!(matches!(
        handler.wait_for_data(&"TR_overload".into()).await,
        Err(IoError::PreConnectMalformed(error)) if error.contains("track capacity")
    ));
    room.send_byte_stream("user", pcm_info("TR_0"), [vec![1; 960]]).unwrap();
    let frames = waiters.remove(0).await.unwrap();
    assert_eq!(frames.iter().map(|frame| frame.data.len()).sum::<usize>(), 960);
    handler.aclose().await;
    for waiter in waiters {
        assert!(waiter.await.is_err());
    }
}

#[tokio::test]
async fn preconnect_receive_deadline_discards_partial_pcm() {
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("receive-deadline");
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        clock.clone(),
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    assert!(handler.register());
    let chunks = room.open_byte_stream("user", pcm_info("TR_expired")).unwrap();
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
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("duration");
    let handler = PreConnectAudioHandler::new(
        room.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        clock,
        Duration::from_secs(3),
        (24_000, 1),
        None,
    );
    let info = lk_agents_abi::voice::rtc::StreamInfo {
        id: "buffer".into(),
        topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
        attributes: [
            ("trackId".into(), "TR_buffer".into()),
            ("sampleRate".into(), "24000".into()),
            ("channels".into(), "1".into()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    assert!(handler.register());
    room.send_byte_stream("user", info, (0..501).map(|_| vec![1; 960])).unwrap();
    assert!(
        handler.wait_for_data(&"TR_buffer".into()).await.is_err(),
        "more than ten seconds of pre-connect PCM must be refused"
    );
    handler.aclose().await;
}

#[cfg(feature = "media-ffmpeg")]
#[test]
fn compressed_heads_stop_at_their_decoded_duration_bound() {
    use lk_agents_voice::room_io::pre_connect_state::{HeadBytes, HeadError, MAX_DURATION};
    let seconds = MAX_DURATION.as_secs() as usize + 1;
    let mut encoder = lk_agents_media::ffmpeg::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
    let mut data = encoder.push(&vec![100; 48_000 * seconds]).unwrap();
    data.extend(encoder.finish().unwrap());
    let info = lk_agents_abi::voice::rtc::StreamInfo {
        mime_type: "audio/ogg;codecs=opus".into(),
        attributes: [("sampleRate".into(), "48000".into()), ("channels".into(), "1".into())]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let mut head = HeadBytes::new(&info).unwrap();
    head.push(&data).unwrap();
    assert_eq!(head.decode((24_000, 1)), Err(HeadError::Capacity("decoded duration")));
}
