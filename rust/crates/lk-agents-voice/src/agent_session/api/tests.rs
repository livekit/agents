//! `SessionApi` over a fake room: the host's `start()` / `aclose()` pair without a host.

use super::*;
use crate::agent_session::commands::{FrameCommand, HookId, HookOutcome, TurnCommand};
use crate::agent_session::host_events::{HookCall, HookKind};
use lk_agents_abi::voice::events::CloseReason;
use std::sync::atomic::{AtomicBool, Ordering};

/// A fake room on a virtual clock, and the same room as a session's.
fn fake_room(name: &str) -> (Arc<lk_agents_rtc::FakeRoom>, FixedRoom) {
    let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
    let room = rtc.room(name);
    // The `FakeRtc` is the room's owner; the room keeps it alive through its own `Arc`.
    std::mem::forget(rtc);
    (room.clone(), FixedRoom { rtc: room, time: Timebase::from_virtual(clock) })
}

/// A room-only session, every provider left out, through the host's constructor.
fn room_only(room: FixedRoom, hook_bound: Option<f64>) -> SessionApi {
    let spawner = Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current()));
    #[rustfmt::skip]
    let api = SessionApi::host_new(
        Arc::new(room), None, "agent".into(), None, None, None, None, None, None, None, None,
        None, None, true, 24_000, 24_000, None, None, None, false, None, true,
        Some(crate::room_io::types::RoomIoConfig::default()), None, None, None, None, None, None,
        None, None, None, None, false, None, false, hook_bound, 3, None, None, None, spawner,
    );
    api.expect("a room-only session assembles")
}

// Catches: a `close` that waits for the host's drain while still holding a `PeerHandle` sender on
// the host stream, so the drain can never end.
#[tokio::test(start_paused = true)]
async fn close_on_a_virtual_clock_ends_the_drain_and_returns() {
    let api = room_only(fake_room("session-api-test").1, None);
    let saw_close = Arc::new(AtomicBool::new(false));
    let drain_ended = Arc::new(AtomicBool::new(false));
    let (saw, ended) = (Arc::clone(&saw_close), Arc::clone(&drain_ended));
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    match ev {
                        HostEvent::Hook(call) => {
                            let _ = runner.api().send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            });
                        }
                        HostEvent::Session(SessionEvent::Close { .. }) => {
                            saw.store(true, Ordering::SeqCst);
                        }
                        HostEvent::Session(_) | HostEvent::Room(_) => {}
                    }
                }
                ended.store(true, Ordering::SeqCst);
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");

    // Bounded on the paused runtime clock: a close parked on the virtual clock fails here.
    tokio::time::timeout(
        Duration::from_secs(60),
        api.close(CloseReason::UserInitiated, false, None),
    )
    .await
    .expect("close hung: the drain was still waiting on a sender the stage held");

    assert!(saw_close.load(Ordering::SeqCst), "the drain never saw the `close` event");
    assert!(
        drain_ended.load(Ordering::SeqCst),
        "close returned before the drain ended: the host's `close` handlers were not queued"
    );
    assert!(matches!(api.send(TurnCommand::Clear), Err(SessionRefused::Closed)));
}

// Catches: `RoomIo::on_agent_state_changed` with no caller, so the agent never writes
// `lk.agent.state` and a frontend keyed on it shows the agent stuck.
#[tokio::test(start_paused = true)]
async fn the_agent_state_is_written_to_the_room_as_it_changes() {
    let (room, fixed) = fake_room("agent-state");
    let api = room_only(fixed, None);
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        let _ = runner.api().send(Command::HookDone {
                            id: call.id(),
                            outcome: HookOutcome::Returned,
                        });
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    settle().await;

    let written: Vec<String> = room
        .local_participant()
        .attribute_calls()
        .iter()
        .filter_map(|a| a.get(lk_agents_abi::ATTRIBUTE_AGENT_STATE).cloned())
        .collect();
    assert!(!written.is_empty(), "no `lk.agent.state` write reached the room");
    assert_eq!(
        room.local_participant()
            .attributes()
            .get(lk_agents_abi::ATTRIBUTE_AGENT_STATE)
            .map(String::as_str),
        Some("listening"),
        "the room's attribute is not the agent's current state (writes: {written:?})"
    );
}

/// A room-only session whose host answers every hook but `on_exit`, which the test answers.
async fn session_holding_exits() -> (SessionApi, tokio::sync::mpsc::UnboundedReceiver<HookId>) {
    // A host that runs hooks: without a bound the driver treats `on_exit` as over at once.
    let api = room_only(fake_room("session-api-test").1, Some(10.0));
    let (exits_tx, exits) = tokio::sync::mpsc::unbounded_channel();
    let opening = api
        .open(false, move |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(ev) = runner.next_host_event().await {
                    if let HostEvent::Hook(call) = ev {
                        if matches!(call, HookCall::OnExit { .. }) {
                            let _ = exits_tx.send(call.id());
                        } else {
                            let _ = runner.api().send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            });
                        }
                    }
                }
            })
        })
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    (api, exits)
}

fn answer_exit(api: &SessionApi, id: HookId) {
    api.send(Command::HookDone { id, outcome: HookOutcome::Returned }).expect("the loop runs");
}

/// The `on_exit`s issued, of `agent`'s frame or of any.
fn on_exits(api: &SessionApi, agent: Option<&str>) -> usize {
    api.steps()
        .iter()
        .filter(|s| {
            matches!(
                s,
                Step::Handoff(crate::trace::HandoffStep::HookIssued {
                    kind: HookKind::OnExit, agent: a, ..
                }) if agent.is_none() || a.as_deref() == agent
            )
        })
        .count()
}

async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}

// Catches: a `drain()` refused while a frame change is in flight; it must wait for the swap,
// without running the arriving frame's `on_exit`.
#[tokio::test(start_paused = true)]
async fn a_drain_during_a_handoff_joins_it_and_is_answered_at_the_swap() {
    let (api, mut exits) = session_holding_exits().await;
    let (call, swap) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = AgentUpdate::frame("next", Instructions::plain("You are next."));
    update.call = Some(call);
    api.send(FrameCommand::Update(Box::new(update))).expect("the loop runs");
    let exit = exits.recv().await.expect("the handoff issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send(FrameCommand::Drain { call: Some(call) }).expect("the loop runs");
    settle().await;
    assert!(
        matches!(drain.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the drain was answered while the handoff's on_exit still ran"
    );

    answer_exit(&api, exit);
    assert_eq!(swap.await.expect("answered"), Ok(()), "the handoff lands");
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined, not refused");
    assert_eq!(on_exits(&api, None), 1, "the arriving frame's on_exit ran: the drain moved frames");
    api.close(CloseReason::UserInitiated, false, None).await;
}

// Catches: a drain during an inline task's entry handed to the *task's* frame; the parent's
// `on_exit` must run once, and not again when it is later closed.
#[tokio::test(start_paused = true)]
async fn a_drain_during_an_inline_tasks_entry_exits_the_paused_parent_not_the_task() {
    let (api, mut exits) = session_holding_exits().await;
    // A speech still being said holds the entry's pause, so the drain arrives inside it.
    let (words, rx) = tokio::sync::mpsc::channel::<lk_agents_abi::AudioFrame>(4);
    api.send(SpeechCommand::Say {
        text: SayText::Audio { text: "One moment.".into(), rx },
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: None,
    })
    .expect("the loop runs");
    settle().await;
    let (call, mut entry) = api.call(Answer::handoff).expect("the loop runs");
    let mut update = AgentUpdate::frame("task", Instructions::plain("You are the task."));
    update.previous = PreviousActivity::Pause;
    update.call = Some(call);
    api.send(FrameCommand::Update(Box::new(update))).expect("the loop runs");
    settle().await;
    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send(FrameCommand::Drain { call: Some(call) }).expect("the loop runs");
    settle().await;
    assert!(
        matches!(entry.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the entry landed before the drain arrived: {:#?}",
        api.steps()
    );
    drop(words);
    assert_eq!(entry.await.expect("answered"), Ok(()), "the task's frame lands");

    let exit = exits.recv().await.expect("the drain runs the paused parent's on_exit");
    assert_eq!(on_exits(&api, Some("agent")), 1, "{:#?}", api.steps());
    assert_eq!(on_exits(&api, Some("task")), 0, "the drain moved to the task's frame");
    let steps = api.steps();
    let issued = steps.iter().position(|s| {
        matches!(
            s,
            Step::Handoff(crate::trace::HandoffStep::HookIssued { kind: HookKind::OnExit, .. })
        )
    });
    let installed = steps
        .iter()
        .position(|s| matches!(s, Step::Handoff(crate::trace::HandoffStep::AgentUpdated { .. })));
    assert!(issued < installed, "the parent's on_exit is its own frame's, before the task's");
    settle().await;
    assert!(
        matches!(drain.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the drain was answered while the parent's on_exit still ran"
    );
    answer_exit(&api, exit);
    assert_eq!(drain.await.expect("answered"), Ok(()), "bound to the parent, answered");

    // The task hands back; the parent resumes live and is closed: its `on_exit` ran already.
    let (call, resume) = api.call(Answer::handoff).expect("the loop runs");
    let mut back = AgentUpdate::frame("agent", Instructions::plain(String::new()));
    back.next = NewActivity::Resume;
    back.call = Some(call);
    api.send(FrameCommand::Update(Box::new(back))).expect("the loop runs");
    let task_exit = exits.recv().await.expect("leaving the task runs its own on_exit");
    answer_exit(&api, task_exit);
    assert_eq!(resume.await.expect("answered"), Ok(()), "the parent resumes");
    assert_eq!(on_exits(&api, Some("agent")), 1);
    assert_eq!(on_exits(&api, Some("task")), 1);
    let closing = api.close(CloseReason::UserInitiated, false, None);
    tokio::pin!(closing);
    let mut at_close = 0;
    loop {
        tokio::select! {
            _ = &mut closing => break,
            Some(id) = exits.recv() => {
                at_close += 1;
                answer_exit(&api, id);
            }
        }
    }
    assert_eq!(at_close, 0, "the close ran the resumed parent's on_exit a second time");
}

// Catches: a `drain()` refused during the close; it must wait for the close's drain.
#[tokio::test(start_paused = true)]
async fn a_drain_during_the_close_is_answered_by_the_closes_drain() {
    let (api, mut exits) = session_holding_exits().await;
    api.send(Command::Close(Close::default())).expect("the loop runs");
    let exit = exits.recv().await.expect("the close issues the frame's on_exit");

    let (call, mut drain) = api.call(Answer::drain).expect("the loop runs");
    api.send(FrameCommand::Drain { call: Some(call) }).expect("the loop runs");
    settle().await;
    assert!(
        matches!(drain.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)),
        "the drain was answered while the close's on_exit still ran"
    );

    answer_exit(&api, exit);
    assert_eq!(drain.await.expect("answered"), Ok(()), "joined the close, not refused");
    assert_eq!(on_exits(&api, None), 1, "one on_exit for the frame, the close's");
    api.close(CloseReason::UserInitiated, false, None).await;
}
