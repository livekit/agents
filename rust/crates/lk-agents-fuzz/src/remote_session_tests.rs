//! The remote session protocol over the room: who may drive the agent's session.

use std::time::Duration;

use lk_agents_abi::voice::console::SessionTransport as _;
use lk_agents_abi::voice::rtc::{ParticipantKind, StreamInfo};
use lk_agents_fuzz::fakes::room::FakeRtc;
use lk_agents_runtime::voice::remote_session::room::RoomTransport;
use lk_agents_runtime::voice::remote_session::{TOPIC_SESSION_MESSAGES, proto};
use prost::Message as _;

/// A participant without the CanManageAgentSession grant cannot send the agent's session
/// requests. https://github.com/livekit/agents/pull/5487
#[tokio::test(start_paused = true)]
#[ignore = "https://github.com/livekit/agents/pull/5487 reproduces: the room transport accepts session requests from any participant; the RTC port carries no grants"]
async fn a_participant_without_the_session_grant_cannot_send_requests() {
    let (rtc, clock) =
        FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room("remote-session");
    room.add_participant("guest", ParticipantKind::Standard);
    let transport = RoomTransport::new(room.clone(), clock);
    transport.start().await.unwrap();

    let history = proto::AgentSessionMessage {
        message: Some(proto::agent_session_message::Message::Request(proto::SessionRequest {
            request_id: "from-guest".into(),
            request: Some(proto::session_request::Request::GetChatHistory(Default::default())),
        })),
    };
    let info = StreamInfo { topic: TOPIC_SESSION_MESSAGES.into(), ..Default::default() };
    room.send_byte_stream("guest", info, [history.encode_to_vec()]).unwrap();

    let received = tokio::time::timeout(Duration::from_secs(5), transport.recv()).await;
    assert!(received.is_err(), "an ungranted participant's request reached the session");
}
