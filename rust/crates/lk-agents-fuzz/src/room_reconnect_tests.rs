//! A resumed room connection keeps the agent's published audio as it was.

use super::*;

/// Resuming after a dropped connection publishes no second track and keeps playing into the same
/// source. https://github.com/livekit/agents/pull/5698 https://github.com/livekit/agents/issues/5706
#[tokio::test(start_paused = true)]
async fn a_resumed_connection_keeps_the_one_audio_publication() {
    let (room, clock) = fake_room("resumed");
    room.add_participant("person", ParticipantKind::Standard);
    let io = room_io(room.clone(), clock.clone(), config());
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    io.wait_for_ready().await.unwrap();
    let local = room.local_participant();
    let before: Vec<_> = local.publications().iter().map(|p| p.sid().clone()).collect();
    assert_eq!(before.len(), 1);
    let output = io.audio_output().unwrap();
    output.capture_frame(&AudioFrame::new(vec![0; 960], 24_000, 1, 480)).await.unwrap();
    yields(8).await;
    clock.advance(Duration::from_millis(100));
    yields(8).await;
    room.set_connection_state(ConnectionState::Reconnecting);
    yields(8).await;
    assert!(room.reconnected());
    yields(8).await;
    output.capture_frame(&AudioFrame::new(vec![0; 960], 24_000, 1, 480)).await.unwrap();
    yields(8).await;
    clock.advance(Duration::from_millis(100));
    yields(8).await;
    output.flush();
    yields(8).await;
    clock.advance(Duration::from_millis(100));
    yields(8).await;
    let after: Vec<_> = local.publications().iter().map(|p| p.sid().clone()).collect();
    assert_eq!(after, before, "the reconnect republished the agent's audio");
    assert_eq!(local.sources().len(), 1);
    let samples: usize =
        local.sources()[0].captured().iter().map(|f| f.samples_per_channel as usize).sum();
    assert_eq!(samples, 960, "audio after the resume was lost");
    io.clone().aclose().await;
}

/// A room I/O started before the room connected publishes its audio once, on the connect, and
/// plays into it. https://github.com/livekit/agents/pull/2167 https://github.com/livekit/agents/pull/3270
#[tokio::test(start_paused = true)]
async fn a_room_io_started_before_the_connect_publishes_its_audio_once_connected() {
    let (room, clock) = fake_room("started-early");
    room.set_connection_state(ConnectionState::Disconnected);
    room.add_participant("person", ParticipantKind::Standard);
    let io = room_io(room.clone(), clock.clone(), config());
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    yields(8).await;
    room.set_connection_state(ConnectionState::Connected);
    within_a_second(io.wait_for_ready()).await.unwrap().unwrap();

    let output = io.audio_output().unwrap();
    output.capture_frame(&AudioFrame::new(vec![0; 960], 24_000, 1, 480)).await.unwrap();
    output.flush();
    yields(8).await;
    clock.advance(Duration::from_millis(100));
    yields(8).await;
    let local = room.local_participant();
    assert_eq!(local.publications().len(), 1, "the agent's audio was not published exactly once");
    let samples: usize =
        local.sources()[0].captured().iter().map(|f| f.samples_per_channel as usize).sum();
    assert_eq!(samples, 480, "audio played after the connect was lost");
    io.clone().aclose().await;
}
