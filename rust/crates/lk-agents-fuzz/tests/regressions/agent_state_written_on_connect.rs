//! The agent state taken before the room connected reaches the room once it does.
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::ATTRIBUTE_AGENT_STATE;
use lk_agents_abi::voice::room_owner::RoomIo as _;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::RoomIo;
use lk_agents_test_support::rtc::{ConnectionState, FakeRtc, RtcRoom};
use lk_agents_voice::room_io::RoomIoConfig;

// Catches: `lk.agent.state` written only while the room is connected, and the session started
// alongside the job's `connect()`: the `listening` it takes at start was dropped, so until the
// agent first spoke a frontend (or a caller waiting for the agent to be ready) saw no state.
#[tokio::test]
async fn a_state_taken_before_the_connect_is_written_on_it() {
    let (rtc, clock) =
        FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room("agent-state-on-connect");
    room.set_connection_state(ConnectionState::Disconnected);
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let ids = Arc::new(lk_agents_runtime::room_io::output::transcription::SegmentIds::default());
    let config = RoomIoConfig::default();
    let io = RoomIo::new(room.clone() as Arc<dyn RtcRoom>, clock, spawner, ids, config, None, None);
    io.clone().start();
    io.clone().on_agent_state_changed("listening");
    tokio::time::sleep(Duration::from_millis(50)).await;
    room.set_connection_state(ConnectionState::Connected);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let state = room.local_participant().attributes().get(ATTRIBUTE_AGENT_STATE).cloned();
    assert_eq!(state.as_deref(), Some("listening"));
    assert_eq!(io.aclose().await, lk_agents_abi::Closed::Clean);
}
