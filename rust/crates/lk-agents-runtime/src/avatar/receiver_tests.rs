//! Receiver conformance against supplied-clock RTC ports.
use super::*;
use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use lk_agents_abi::voice::rtc::ParticipantIdentity;
use lk_agents_abi::{AudioFrame, Closed, SpawnerExt, TokioSpawner, VirtualClock};
use lk_agents_test_support::rtc::{FakeRoom, FakeRtc, ParticipantKind, TrackSource};
use lk_agents_voice::avatar::{
    RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED, playback_method, stream_name, stream_segment,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
fn receipt_segment(id: &str) -> AvatarPlaybackSegment {
    AvatarPlaybackSegment {
        playback: PlaybackSegment {
            attachment: 1,
            segment: id.strip_prefix('S').and_then(|id| id.parse().ok()).unwrap_or(1),
        },
        stream_id: Some(id.to_owned()),
    }
}

async fn settle() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

// ------------------------------------------------------------------------- the avatar's side

fn audio_info(id: &str) -> lk_agents_test_support::rtc::StreamInfo {
    lk_agents_test_support::rtc::StreamInfo {
        id: id.to_owned(),
        name: Some(stream_name(receipt_segment(id).playback)),
        topic: AUDIO_STREAM_TOPIC.to_owned(),
        attributes: BTreeMap::from([
            ("sample_rate".to_owned(), "10000".to_owned()),
            ("num_channels".to_owned(), "1".to_owned()),
        ]),
        ..Default::default()
    }
}

struct RxFixture {
    room: Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    rx: DataStreamAudioReceiver,
    cleared: Arc<std::sync::atomic::AtomicUsize>,
}

async fn receiver(name: &str) -> RxFixture {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room(name);
    std::mem::forget(rtc);
    room.add_participant("agent-1", ParticipantKind::Agent);
    let cleared = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let on_clear = {
        let cleared = cleared.clone();
        Arc::new(move || {
            cleared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }) as Arc<dyn Fn() + Send + Sync>
    };
    let rx = DataStreamAudioReceiver::new(
        crate::rtc::room::Room::wrap(room.clone()),
        None,
        std::num::NonZeroU32::new(100).unwrap(),
        3,
        Some(on_clear),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    );
    rx.start().await.unwrap();
    RxFixture { room, clock, rx, cleared }
}

fn kinds(items: &[Received]) -> Vec<String> {
    items
        .iter()
        .map(|i| match i {
            Received::Audio(f) => format!("frame {}", f.samples_per_channel),
            Received::SegmentEnd => "end".to_owned(),
        })
        .collect()
}

async fn take(rx: &DataStreamAudioReceiver, n: usize) -> Vec<Received> {
    let mut out = Vec::new();
    for _ in 0..n {
        out.push(
            tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .3
                .unwrap(),
        );
    }
    out
}

// Catches: frames cut at the wrong size (the reference frames at `frame_size_ms`), the tail lost
// at the trailer, or no segment end for the avatar to report playback on.
#[tokio::test]
async fn a_segment_comes_out_in_frame_size_frames_then_its_end() {
    let f = receiver("rx1").await;
    // 250 ms at 10 kHz mono: two 100 ms frames and a 50 ms tail
    f.room
        .send_byte_stream("agent-1", audio_info("S1"), [vec![0u8; 3000], vec![0u8; 2000]])
        .unwrap();
    let items = take(&f.rx, 4).await;
    assert_eq!(kinds(&items), ["frame 1000", "frame 1000", "frame 500", "end"]);
}

// Catches: a lost trailer wedging the avatar (it stops receiving audio): a new stream must end
// the open one.
#[tokio::test]
async fn a_new_stream_ends_one_whose_trailer_never_came() {
    let f = receiver("rx2").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 2000]).await.unwrap();
    let first = take(&f.rx, 1).await;
    assert_eq!(kinds(&first), ["frame 1000"]);
    f.room.send_byte_stream("agent-1", audio_info("S2"), [vec![0u8; 2000]]).unwrap();
    let items = take(&f.rx, 3).await;
    assert_eq!(kinds(&items), ["end", "frame 1000", "end"]);
    drop(open);
}

// Catches: clear_buffer leaving buffered audio to play, accepted from a stranger, or not raised
// as the runner's clear_buffer event.
#[tokio::test]
async fn clear_buffer_from_the_agent_drops_the_segment_and_is_raised() {
    let f = receiver("rx3").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 4000]).await.unwrap();
    take(&f.rx, 1).await;
    let local = f.room.local_participant();
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "mallory", "").as_deref(), Some("reject"));
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "agent-1", "").as_deref(), Some("ok"));
    assert_eq!(f.cleared.load(std::sync::atomic::Ordering::Relaxed), 1);
    let items = take(&f.rx, 1).await;
    assert_eq!(kinds(&items), ["end"], "the buffered frame is gone and the segment ends");
    drop(open);
}

// Catches: playback reports with the wrong method or payload, or a failing report retried forever
// (the reference gives up after `rpc_max_retries` and fails the receiver).
#[tokio::test]
async fn playback_reports_go_to_the_agent_and_a_failing_one_ends_the_receiver() {
    let f = receiver("rx4").await;
    f.rx.notify_playback_started(receipt_segment("S1")).unwrap();
    f.rx.notify_playback_finished(receipt_segment("S1"), 1.5, true).unwrap();
    settle().await;
    let calls = f.room.local_participant().rpc_calls();
    assert_eq!(
        calls[0],
        (
            ParticipantIdentity("agent-1".into()),
            playback_method(RPC_PLAYBACK_STARTED, &receipt_segment("S1")).unwrap(),
            String::new()
        )
    );
    assert_eq!(calls[1].1, playback_method(RPC_PLAYBACK_FINISHED, &receipt_segment("S1")).unwrap());
    let payload: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(payload["playback_position"], 1.5);
    assert_eq!(payload["interrupted"], true);

    f.room.local_participant().set_rpc_reply(1500, "boom");
    f.rx.notify_playback_finished(receipt_segment("S2"), 2.0, false).unwrap();
    for _ in 0..4 {
        settle().await;
        f.clock.advance(Duration::from_millis(100));
    }
    settle().await;
    assert_eq!(f.room.local_participant().rpc_calls().len(), 5, "three attempts");
    match tokio::time::timeout(Duration::from_secs(5), f.rx.recv()).await.unwrap() {
        Err(ReceiverEnd::Failed(e)) => assert!(e.contains("1500"), "{e}"),
        other => panic!("{other:?}"),
    }
}

// Catches: a stream with no format read as some default, which would play garbage.
#[tokio::test]
async fn a_stream_without_its_format_fails_the_receiver() {
    let f = receiver("rx5").await;
    let mut info = audio_info("S1");
    info.attributes.clear();
    f.room.send_byte_stream("agent-1", info, [vec![0u8; 10]]).unwrap();
    match tokio::time::timeout(Duration::from_secs(5), f.rx.recv()).await.unwrap() {
        Err(ReceiverEnd::BadStream(_)) => {}
        other => panic!("{other:?}"),
    }
}

// Catches: a join wait that starts before the room connects (the reference arms it on the
// connection), or that ends at the participant without its video track.
#[tokio::test]
async fn the_join_wait_starts_at_the_connection_and_ends_at_the_avatar_video() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("join");
    room.set_connection_state(lk_agents_test_support::rtc::ConnectionState::Disconnected);
    let wait = tokio::spawn(wait_avatar_join_room(
        crate::rtc::room::Room::wrap(room.clone()),
        "avatar".into(),
        clock.clone(),
    ));
    clock.advance(Duration::from_secs(1));
    tokio::task::yield_now().await;
    let connected_at = lk_agents_abi::unix_secs(clock.wall());
    room.set_connection_state(lk_agents_test_support::rtc::ConnectionState::Connected);
    let avatar = room.add_participant("avatar", ParticipantKind::Agent);
    tokio::task::yield_now().await;
    clock.advance(Duration::from_secs(2));
    assert!(!wait.is_finished());
    avatar.publish_track("TR_cam", TrackSource::Camera);
    let (started, joined) =
        tokio::time::timeout(Duration::from_secs(5), wait).await.unwrap().unwrap().unwrap();
    assert!((started - connected_at).abs() < 0.01, "{started} {connected_at}");
    assert!((joined - connected_at - 2.0).abs() < 0.01, "{joined} {connected_at}");
}

#[tokio::test]
async fn a_failed_byte_stream_does_not_flush_a_partial_avatar_segment() {
    let f = receiver("rx-failed-chunk").await;
    f.room
        .send_byte_stream(
            "agent-1",
            audio_info("failed"),
            [vec![0; 1500], vec![0; lk_agents_voice::room_io::ingress::MAX_STREAM_CHUNK_BYTES + 1]],
        )
        .unwrap();
    let outcome = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        f.rx.recv(),
    )
    .await;
    f.rx.aclose();
    let outcome = outcome.expect("the failed stream must settle its receiver");
    assert!(
        matches!(outcome, Err(ReceiverEnd::BadStream(_))),
        "a failed stream must not produce its buffered tail or a successful segment end"
    );
}
