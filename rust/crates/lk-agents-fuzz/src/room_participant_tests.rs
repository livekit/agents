//! Participant selection: which participant and microphone track the room input links.

use super::*;

/// Explicit caller selection ignores earlier agents and unrelated participants. https://github.com/livekit/agents/issues/1596 https://github.com/livekit/agents/issues/2839
#[tokio::test(start_paused = true)]
async fn explicit_sip_identity_selects_only_the_intended_caller() {
    let (room, clock) = fake_room("selected-caller");
    let config = RoomIoConfig {
        participant_identity: Some("callee".into()),
        participant_kinds: vec![ParticipantKind::Sip],
        ..microphone_only()
    };
    let io = room_io(room.clone(), clock, config);
    room.add_participant("other-agent", ParticipantKind::Agent).publish_microphone("wrong-agent");
    room.add_participant("other-sip", ParticipantKind::Sip).publish_microphone("wrong-sip");
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    yields(16).await;
    assert!(io.linked_participant().is_none());

    let selected = room.add_participant("callee", ParticipantKind::Sip).publish_microphone("mic");
    within_a_second(io.wait_for_ready()).await.unwrap().unwrap();
    assert_eq!(io.linked_participant(), Some("callee".into()));
    let input = io.audio_input().unwrap();
    assert!(selected.push_frame(AudioFrame::new(vec![7; 960], 24_000, 1, 480)) > 0);
    let heard = within_a_second(input.next_frame(&lk_agents_abi::telemetry::Cx::NONE)).await;
    assert_eq!(heard.unwrap().unwrap().unwrap().data.as_ref(), &[7; 960]);
    io.clone().aclose().await;
}

/// Replacing a microphone under the same SID delivers only new-track audio. https://github.com/livekit/agents/pull/6744 https://github.com/livekit/agents/pull/1568
#[tokio::test(start_paused = true)]
#[ignore = "https://github.com/livekit/agents/pull/6744 reproduces: same-SID replacement keeps the old microphone reader"]
async fn same_sid_microphone_replacement_retires_the_old_reader() {
    let (room, clock) = fake_room("same-sid-track");
    let io = room_io(room.clone(), clock, microphone_only());
    let person = room.add_participant("person", ParticipantKind::Standard);
    // A video publication must never be selected as audio input.
    person.publish_track("camera", TrackSource::Camera);
    let old = person.publish_microphone("microphone");
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    within_a_second(io.wait_for_ready()).await.unwrap().unwrap();
    let input = io.audio_input().unwrap();
    old.push_frame(AudioFrame::new(vec![1; 960], 24_000, 1, 480));
    yields(16).await;

    let replacement = person.publish_microphone("microphone");
    yields(16).await;
    old.push_frame(AudioFrame::new(vec![2; 960], 24_000, 1, 480));
    replacement.push_frame(AudioFrame::new(vec![3; 960], 24_000, 1, 480));
    let heard = within_a_second(input.next_frame(&lk_agents_abi::telemetry::Cx::NONE)).await;
    let heard = heard.unwrap().unwrap().unwrap();
    assert_eq!(heard.data.as_ref(), &[3; 960], "queued old audio crossed track replacement");
    io.clone().aclose().await;
}
