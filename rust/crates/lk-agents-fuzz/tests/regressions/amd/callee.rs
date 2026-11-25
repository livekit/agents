//! AMD's callee leg over a fake room: whom it listens to, and how it lets go of the room's
//! events. Expected values come from `voice/amd/detector.py` `_setup_listening` on `main`.

use lk_agents_abi::voice::room_io::{AudioInputConfig, RoomIoConfig};
use lk_agents_abi::voice::room_owner::RoomIo as _;
use lk_agents_abi::voice::rtc::{
    DisconnectReason, ParticipantKind, RtcRoom, TrackPublication as _,
};
use lk_agents_fuzz::fakes::room::{EventStreamEnd, FakeRoom, FakeRtc};
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::RoomIo;
use lk_agents_runtime::voice::amd::callee::{self, CalleeSetup, LinkedWait};
use lk_agents_runtime::voice::session::Timebase;
use lk_agents_voice::amd::machine::Callee;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

fn fake_room(name: &str) -> (Arc<FakeRoom>, Arc<lk_agents_abi::VirtualClock>) {
    let (rtc, clock) =
        FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room(name);
    std::mem::forget(rtc);
    (room, clock)
}

/// `identity` joins and publishes a microphone the agent is subscribed to, as an auto-subscribing
/// room leaves it.
async fn joins_with_a_microphone(room: &FakeRoom, identity: &str) {
    let mic = room
        .add_participant(identity, ParticipantKind::Standard)
        .publish_microphone(format!("TR_{identity}"));
    mic.set_subscribed(&lk_agents_abi::telemetry::Cx::NONE, true).await.expect("subscribed");
}

async fn yields(n: usize) {
    for _ in 0..n {
        tokio::task::yield_now().await;
    }
}

/// What the leg reported, in order.
fn reporter() -> (callee::Report, Arc<Mutex<Vec<Callee>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let sink = heard.clone();
    let report: callee::Report = Box::new(move |callee| {
        sink.lock().unwrap().push(callee);
        true
    });
    (report, heard)
}

// Catches: a run without an identity that listens to whichever participant publishes audio
// first, not to the one the session hears. Python: `_wait_for_audio` waits for
// `linked_io._wait_for_linked_participant()` before the track
// (`test_without_an_identity_amd_listens_to_the_linked_participant`).
#[tokio::test(start_paused = true)]
async fn without_an_identity_the_leg_listens_to_the_linked_participant() {
    let (room, clock) = fake_room("amd-linked");
    // The room I/O links only the caller.
    let config = RoomIoConfig {
        participant_identity: Some("caller".into()),
        audio_input: Some(AudioInputConfig { pre_connect_audio: false, ..Default::default() }),
        ..RoomIoConfig::default()
    };
    let ids = Arc::new(lk_agents_runtime::room_io::output::transcription::SegmentIds::counting());
    let io = RoomIo::new(
        room.clone(),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
        ids,
        config,
        None,
        None,
    );
    io.clone().start(lk_agents_abi::telemetry::Cx::NONE);
    // Someone else publishes audio first.
    joins_with_a_microphone(&room, "guest").await;
    let linked: LinkedWait = {
        let io = io.clone();
        Arc::new(move || {
            let io = io.clone();
            Box::pin(async move { io.wait_linked_participant().await.map(|id| id.0) })
        })
    };
    let setup = CalleeSetup {
        room: Some(room.clone() as Arc<dyn RtcRoom>),
        participant: None,
        linked: Some(linked),
        wait_until_answered: true,
    };
    let (report, heard) = reporter();
    let stop = CancellationToken::new();
    let leg =
        tokio::spawn(callee::watch(setup, Timebase::from_virtual(clock), stop.clone(), report));
    yields(20).await;
    assert!(heard.lock().unwrap().is_empty(), "the guest's audio is not the callee's");

    joins_with_a_microphone(&room, "caller").await;
    yields(20).await;
    assert_eq!(*heard.lock().unwrap(), [Callee::Answered]);
    room.remove_participant(&"guest".into(), DisconnectReason::ClientInitiated);
    yields(20).await;
    assert_eq!(*heard.lock().unwrap(), [Callee::Answered], "the guest leaving ends nothing");
    room.remove_participant(&"caller".into(), DisconnectReason::ClientInitiated);
    yields(20).await;
    assert_eq!(*heard.lock().unwrap(), [Callee::Answered, Callee::Disconnected]);
    leg.await.expect("the leg ends");
    io.clone().aclose().await;
}

// Catches: a run that completes while its participant is still connected cancelling the leg's
// room events instead of closing them within their bound (runtime.md "Stream lifecycle": the
// runtime closes streams it owns). The executor joins the leg before its outcome, so the stream
// is closed before `execute()` returns.
#[tokio::test(start_paused = true)]
async fn a_stopped_leg_closes_its_room_events_with_the_participant_still_there() {
    let (room, clock) = fake_room("amd-stop");
    joins_with_a_microphone(&room, "caller").await;
    let setup = CalleeSetup {
        room: Some(room.clone() as Arc<dyn RtcRoom>),
        participant: Some("caller".into()),
        linked: None,
        wait_until_answered: true,
    };
    let (report, heard) = reporter();
    let stop = CancellationToken::new();
    let leg =
        tokio::spawn(callee::watch(setup, Timebase::from_virtual(clock), stop.clone(), report));
    yields(20).await;
    assert_eq!(*heard.lock().unwrap(), [Callee::Answered]);
    // The leg's own subscription comes first; the track wait's follows and ends with its wait.
    assert_eq!(room.event_streams()[0], EventStreamEnd::Open, "watched while running");

    stop.cancel();
    leg.await.expect("the leg ends once stopped");
    assert_eq!(room.event_streams()[0], EventStreamEnd::Closed);
    assert_eq!(*heard.lock().unwrap(), [Callee::Answered], "a stop is no disconnect");
    assert!(room.fake_participant(&"caller".into()).is_some(), "the caller is still there");
}

// Catches: a leg stopped while it still waits for the callee leaving its events open or
// cancelled.
#[tokio::test(start_paused = true)]
async fn a_leg_stopped_during_its_setup_closes_its_room_events() {
    let (room, clock) = fake_room("amd-stop-setup");
    let setup = CalleeSetup {
        room: Some(room.clone() as Arc<dyn RtcRoom>),
        participant: Some("caller".into()),
        linked: None,
        wait_until_answered: true,
    };
    let (report, heard) = reporter();
    let stop = CancellationToken::new();
    let leg =
        tokio::spawn(callee::watch(setup, Timebase::from_virtual(clock), stop.clone(), report));
    yields(20).await;
    stop.cancel();
    leg.await.expect("the leg ends once stopped");
    assert_eq!(room.event_streams()[0], EventStreamEnd::Closed);
    assert!(heard.lock().unwrap().is_empty(), "a stop reports nothing");
}
