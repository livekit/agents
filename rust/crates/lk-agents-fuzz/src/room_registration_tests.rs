//! Room handler registrations: owned by one room I/O or receiver for its lifetime, released at
//! close, and scoped to their own room.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::TOPIC_CHAT;
use lk_agents_abi::voice::room_io::{RoomIoConfig, TextInput};
use lk_agents_abi::voice::room_owner::{RoomIo as _, RoomIoEvent};
use lk_agents_abi::voice::rtc::capabilities::RtcRoom as _;
use lk_agents_abi::voice::rtc::{ParticipantKind, StreamInfo};
use lk_agents_fuzz::fakes::room::FakeRtc;
use lk_agents_runtime::avatar::receiver::DataStreamAudioReceiver;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::RoomIo;
use lk_agents_runtime::room_io::output::transcription::SegmentIds;
use lk_agents_runtime::rtc::events::RoomEventBus;
use lk_agents_voice::avatar::RPC_CLEAR_BUFFER;

/// A room I/O created after an earlier one on the same room closed receives chat: the closed one
/// released the chat topic, and its repeated start did not register it twice. https://github.com/livekit/agents/issues/2034
#[tokio::test]
async fn a_room_io_after_a_restarted_and_closed_one_receives_chat() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("chat-lifecycle");
    room.add_participant("person", ParticipantKind::Standard);
    let config = RoomIoConfig {
        text_input: true,
        audio_input: None,
        audio_output: None,
        transcription_output: None,
        close_on_disconnect: false,
        ..Default::default()
    };
    let spawner = Arc::new(TokioSpawner::current());
    let first = RoomIo::new(
        room.clone(),
        clock.clone(),
        spawner.clone(),
        Arc::new(SegmentIds::counting()),
        config.clone(),
        None,
        None,
    );
    first.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    first.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    first.wait_for_ready().await.expect("first room io ready");
    first.clone().aclose().await;

    let second = RoomIo::new(
        room.clone(),
        clock,
        spawner,
        Arc::new(SegmentIds::counting()),
        config,
        None,
        None,
    );
    let (mut events, _) = second.events().expect("the room io's events");
    second.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    second.wait_for_ready().await.expect("second room io ready");
    let chat = StreamInfo { topic: TOPIC_CHAT.into(), ..Default::default() };
    room.send_text_stream("person", chat, ["hello"])
        .expect("the closed room io kept the chat topic registered");
    let delivered = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if let Some(RoomIoEvent::TextInput(TextInput { text, .. })) = events.recv().await {
                return text;
            }
        }
    })
    .await
    .expect("the chat reached the new room io");
    assert_eq!(delivered, "hello");
    second.clone().aclose().await;
}

/// An avatar's clear-buffer request reaches only the receiver in its own room, even when both
/// rooms' avatars share an identity. https://github.com/livekit/agents/issues/8288
#[tokio::test(start_paused = true)]
async fn clear_buffer_reaches_only_the_receiver_in_its_room() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let mut rooms = Vec::new();
    for name in ["room-a", "room-b"] {
        let room = rtc.room(name);
        room.add_participant("agent-1", ParticipantKind::Agent);
        let cleared = Arc::new(AtomicUsize::new(0));
        let counter = cleared.clone();
        let receiver = DataStreamAudioReceiver::new(
            room.clone(),
            None,
            std::num::NonZeroU32::new(100).unwrap(),
            3,
            Some(Arc::new(move || {
                counter.fetch_add(1, Ordering::Relaxed);
            })),
            clock.clone(),
            Arc::new(TokioSpawner::current()),
        );
        receiver.start().await.unwrap();
        rooms.push((room, receiver, cleared));
    }
    assert_eq!(rooms[0].0.rpc_router().call(RPC_CLEAR_BUFFER, "agent-1", ""), "ok");
    assert_eq!(rooms[0].2.load(Ordering::Relaxed), 1, "room A's receiver was cleared");
    assert_eq!(rooms[1].2.load(Ordering::Relaxed), 0, "room A's clear reached room B");
    for (_, receiver, _) in &rooms {
        receiver.aclose();
    }
}

/// A second room I/O on the same room leaves the first one's chat in place, also after the second
/// closes. https://github.com/livekit/agents/issues/5226 https://github.com/livekit/agents/issues/4535
#[tokio::test]
async fn a_second_room_io_closing_leaves_the_first_ones_chat() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("two-sessions");
    room.add_participant("person", ParticipantKind::Standard);
    let config = RoomIoConfig {
        text_input: true,
        audio_input: None,
        audio_output: None,
        transcription_output: None,
        close_on_disconnect: false,
        ..Default::default()
    };
    let spawner = Arc::new(TokioSpawner::current());
    let first = RoomIo::new(
        room.clone(),
        clock.clone(),
        spawner.clone(),
        Arc::new(SegmentIds::counting()),
        config.clone(),
        None,
        None,
    );
    let (mut events, _) = first.events().expect("the first room io's events");
    first.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    first.wait_for_ready().await.expect("first room io ready");

    let second = RoomIo::new(
        room.clone(),
        clock,
        spawner,
        Arc::new(SegmentIds::counting()),
        config,
        None,
        None,
    );
    second.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    second.wait_for_ready().await.expect("second room io ready");
    second.clone().aclose().await;

    let chat = StreamInfo { topic: TOPIC_CHAT.into(), ..Default::default() };
    room.send_text_stream("person", chat, ["still here"])
        .expect("closing the second room io unregistered the first one's chat");
    let delivered = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if let Some(RoomIoEvent::TextInput(TextInput { text, .. })) = events.recv().await {
                return text;
            }
        }
    })
    .await
    .expect("the chat reached the first room io");
    assert_eq!(delivered, "still here");
    first.clone().aclose().await;
}
