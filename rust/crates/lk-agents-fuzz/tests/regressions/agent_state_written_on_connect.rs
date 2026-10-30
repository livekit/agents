//! The agent state reaches the room: written by the session as it changes, and a state taken
//! before the room connected written once it does.
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

// Catches: `RoomIo::on_agent_state_changed` with no caller, so the agent never writes
// `lk.agent.state` and a frontend keyed on it shows the agent stuck.
#[tokio::test(start_paused = true)]
async fn the_session_writes_the_agent_state_to_the_room_as_it_changes() {
    use lk_agents_abi::voice::session::{VoiceSession as _, VoiceSessionArgs};
    use lk_agents_runtime::room_io::FixedRoom;
    use lk_agents_runtime::voice::session::{SessionRunner, Timebase};
    use lk_agents_runtime::voice::{Executor, VoiceSession};
    use lk_agents_voice::agent_session::{Command, HookOutcome, HostEvent};

    let (rtc, clock) =
        FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room("agent-state");
    let fixed = FixedRoom { rtc: Some(room.clone()), time: Timebase::from_virtual(clock), pool: None };
    let args = VoiceSessionArgs {
        agent: "agent".into(),
        room_io_config: Some(Default::default()),
        ..Default::default()
    };
    let env = lk_agents_runtime::net::env::ProcessEnv::current_runtime();
    let session = VoiceSession::new(&env, Arc::new(fixed), args, None).expect("a valid room");
    let opening = session
        .open(
            Box::new(Executor::new(session.time().clone(), session.spawner())),
            None,
            move |mut runner: SessionRunner| {
                Box::pin(async move {
                    while let Some(event) = runner.next_host_event().await {
                        if let HostEvent::Hook(call) = event {
                            let outcome = HookOutcome::Returned;
                            let _ = runner.api().send(Command::HookDone { id: call.id(), outcome });
                        }
                    }
                })
            },
        )
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    let local = room.local_participant();
    let written: Vec<String> = local
        .attribute_calls()
        .iter()
        .filter_map(|attributes| attributes.get(ATTRIBUTE_AGENT_STATE).cloned())
        .collect();
    assert!(!written.is_empty(), "no `lk.agent.state` write reached the room");
    assert_eq!(
        local.attributes().get(ATTRIBUTE_AGENT_STATE).map(String::as_str),
        Some("listening"),
        "the room's attribute is not the agent's current state (writes: {written:?})"
    );
}
