//! Room effects observed through the public session API and real runtime room performer.

use lk_agents_abi::Maybe;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_voice::agent_session::commands::{Command, HookOutcome};
use lk_agents_voice::agent_session::{HostEvent, SessionApi, SessionRunner};
use std::sync::Arc;

#[tokio::test(start_paused = true)]
async fn the_agent_state_is_written_to_the_room_as_it_changes() {
    let (rtc, clock) = crate::test_support_rtc::FakeRtc::with_virtual_clock();
    let room = rtc.room("agent-state");
    let spawner = Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current()));
    let api = SessionApi::host_new(
        Arc::new(super::FixedRoom {
            rtc: room.clone(),
            time: lk_agents_voice::agent_session::Timebase::from_virtual(clock),
        }),
        None,
        None,
        "agent".into(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        true,
        24_000,
        24_000,
        None,
        None,
        None,
        None,
        None,
        None,
        true,
        Some(lk_agents_abi::voice::room_io::RoomIoConfig {
            audio_input: None,
            audio_output: None,
            text_output: None,
            text_input: false,
            ..Default::default()
        }),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        None,
        false,
        None,
        3,
        None,
        None,
        None,
        Maybe::NotGiven,
        None,
        None,
        None,
        None,
        spawner,
    );
    let api = api.unwrap();
    api.open(
        false,
        Box::new(crate::voice::Executor::default()),
        move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(event) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = event {
                        let _ = runner.api().send(Command::HookDone {
                            id: call.id(),
                            outcome: HookOutcome::Returned,
                        });
                    }
                }
            })
        },
    )
    .expect("a ready session opens")
    .await
    .expect("the loop starts");
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    let calls = room.local_participant().attribute_calls();
    assert!(!calls.is_empty(), "no agent state write reached the room");
    assert_eq!(
        room.local_participant()
            .attributes()
            .get(lk_agents_abi::ATTRIBUTE_AGENT_STATE)
            .map(String::as_str),
        Some("listening")
    );
    api.close(CloseReason::UserInitiated, false, None).await;
}
