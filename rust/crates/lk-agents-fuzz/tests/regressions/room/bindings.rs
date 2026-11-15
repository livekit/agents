use super::*;
use lk_agents_abi::voice::room_owner::SessionRoom as _;
use lk_agents_abi::voice::room_owner::{RoomAttachments as _, RoomAttachmentsPort as _};
use lk_agents_runtime::room_io::FixedRoom;
use lk_agents_runtime::voice::session::Timebase;
use std::task::{Context, Waker};

#[tokio::test(start_paused = true)]
async fn closing_the_shared_room_again_joins_a_cancelled_close_waiters_cleanup() {
    let (room, clock) = fake_room("shared-close-waiters");
    room.add_participant("person", ParticipantKind::Standard);
    let registry = lk_agents_runtime::room_io::AttachmentRegistry::default();
    let attached = FixedRoom::new(Some(room.clone()), Timebase::from_virtual(clock.clone()), None);
    let spawner = Arc::new(TokioSpawner::current());
    let performer = spawner.clone();
    let io = registry
        .attach(clock, spawner, Box::new(move || attached.create_io(performer, config())))
        .unwrap();
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    io.wait_for_ready().await.unwrap();
    io.session_started().unwrap();
    let local = room.local_participant();
    let source = local.sources()[0].clone();
    source.flow_closes(false);
    let mut first = Box::pin(registry.aclose());
    assert!(first.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
    drop(first);
    let mut second = Box::pin(registry.aclose());
    assert!(second.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
    source.flow_closes(true);
    within_a_second(second).await.unwrap().unwrap();
    assert!(source.is_closed());
    assert!(local.publications().is_empty());
    assert_eq!(io.aclose().await, Closed::Clean);
}

#[tokio::test(start_paused = true)]
async fn closing_a_binding_retires_only_its_real_room_publication() {
    let (room, clock) = fake_room("shared-binding-ownership");
    room.add_participant("person", ParticipantKind::Standard);
    let attached = FixedRoom::new(Some(room.clone()), Timebase::from_virtual(clock.clone()), None);
    let first = attached.room_io(Arc::new(TokioSpawner::current()), config()).unwrap();
    first.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    first.wait_for_ready().await.unwrap();
    first.session_started().unwrap();
    // A separately constructed view of the same SDK room resolves to the same owner.
    let another_view = FixedRoom::new(Some(room.clone()), Timebase::from_virtual(clock), None);
    let second = another_view.room_io(Arc::new(TokioSpawner::current()), config()).unwrap();
    second.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    second.wait_for_ready().await.unwrap();
    second.session_started().unwrap();
    let local = room.local_participant();
    assert_eq!(local.publications().len(), 2);
    let second_sid = local.publications()[1].sid().clone();

    assert_eq!(first.clone().aclose().await, Closed::Clean);
    assert!(first.session_started().is_err(), "closed bindings cannot be reactivated");
    assert_eq!(local.publications().len(), 1);
    assert_eq!(local.publications()[0].sid(), &second_sid);
    // The surviving binding's actual output executor still accepts audio.
    second.audio_output().unwrap().capture_frame(&silence_frame(480, 24_000, 1)).await.unwrap();
    assert_eq!(second.aclose().await, Closed::Clean);
    assert!(local.publications().is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_queued_binding_cleanup_does_not_restart_the_room_close_deadline() {
    let (room, clock) = fake_room("binding-close-deadline");
    room.add_participant("person", ParticipantKind::Standard);
    let attached = FixedRoom::new(Some(room.clone()), Timebase::from_virtual(clock.clone()), None);
    let io = attached.room_io(Arc::new(TokioSpawner::current()), config()).unwrap();
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    io.wait_for_ready().await.unwrap();
    io.session_started().unwrap();
    let local = room.local_participant();
    let source = local.sources()[0].clone();
    source.flow_closes(false);

    // Close is admitted now. Delay the cleanup task's first poll past its deadline.
    let closing = io.aclose();
    clock.advance(HOST_TEARDOWN_TIMEOUT);
    assert_eq!(
        within_a_second(closing).await.expect("cleanup must use its original deadline"),
        Closed::TimedOut
    );
    source.flow_closes(true);
    within_a_second(async {
        while !source.is_closed() || !local.publications().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the admitted resource cleanup must finish after its waiter timed out");
}
