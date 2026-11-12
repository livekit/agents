//! The deprecated `rtc.Transcription` packet: published only while some client in the room still
//! renders it, never alongside the `lk.transcription` stream for clients that rebuild it.

use std::sync::Arc;

use lk_agents_abi::voice::io::TranscriptionOutput as _;
use lk_agents_abi::voice::rtc::ParticipantKind;
use lk_agents_fuzz::fakes::room::FakeRtc;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::output::transcription::{
    ParticipantTranscriptionOutput, SegmentIds,
};
use lk_agents_runtime::rtc::events::RoomEventBus;

/// A room whose only remote participant is a SIP caller, which renders no transcripts, gets the
/// transcript once, as a stream, and no legacy packet. https://github.com/livekit/agents/pull/7240
#[tokio::test(start_paused = true)]
#[ignore = "https://github.com/livekit/agents/pull/7240 reproduces: the legacy packet is published regardless of who is in the room (CLIENT_PROTOCOL_TRANSCRIPTION_STREAMS is unused and the port exposes no client_protocol)"]
async fn a_room_without_legacy_clients_gets_no_legacy_transcription() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("sip-only-transcription");
    room.add_participant("caller", ParticipantKind::Sip).publish_microphone("caller-mic");
    let output = ParticipantTranscriptionOutput::new(
        room.clone(),
        Arc::new(TokioSpawner::current()),
        clock,
        Arc::new(SegmentIds::counting()),
        true,
        false,
        None,
    );
    output.set_participant(Some("caller".into())).await.unwrap();
    output.capture_text(&lk_agents_abi::telemetry::Cx::NONE, "hello").await.unwrap();
    output.flush();
    output.aclose().await.unwrap();
    let streams = room.local_participant().text_streams();
    assert_eq!(streams.iter().map(|stream| stream.text()).collect::<Vec<_>>(), ["hello"]);
    assert!(
        room.local_participant().transcriptions().is_empty(),
        "the legacy packet duplicated the stream for a room with no legacy client"
    );
}
