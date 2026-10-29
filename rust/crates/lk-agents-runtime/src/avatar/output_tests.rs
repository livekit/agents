use super::*;
use lk_agents_abi::voice::io::{AudioOutputExt, PlaybackEvent, PlaybackObserver};
use lk_agents_abi::{SpawnerExt, VirtualClock};
use lk_agents_net::task::TokioSpawner;
use lk_agents_test_support::rtc::{FakeRoom, FakeRtc, ParticipantKind, TrackSource};
use lk_agents_voice::avatar::{RPC_PLAYBACK_STARTED, stream_segment};

#[derive(Default)]
struct Events(Mutex<Vec<String>>);

impl PlaybackObserver for Events {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let label = match ev {
            PlaybackEvent::Started(_) => "started".to_owned(),
            PlaybackEvent::Progressed(_) => "progressed".to_owned(),
            PlaybackEvent::Finished { event: f, .. } => {
                format!("finished {:.2} {}", f.playback_position.as_secs_f64(), f.interrupted)
            }
        };
        self.0.lock().unwrap().push(label);
    }
}

fn frame() -> AudioFrame {
    // 10 ms at 24 kHz mono
    AudioFrame::new(vec![1u8; 480], 24_000, 1, 240)
}

struct Fixture {
    room: Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    out: Arc<DataStreamAudioOutput>,
    events: Arc<Events>,
}

fn fixture(name: &str, wait_remote_track: Option<i32>, wait_playback_start: bool) -> Fixture {
    let (rtc, clock) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room(name);
    let out = Arc::new(DataStreamAudioOutput::new(
        room.clone(),
        "avatar".to_owned(),
        None,
        wait_remote_track,
        Some(Duration::from_secs(2)),
        wait_playback_start,
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    ));
    let events = Arc::new(Events::default());
    out.tracker().observe(events.clone());
    std::mem::forget(rtc);
    Fixture { room, clock, out, events }
}

fn output_method(f: &Fixture, base: &str, index: usize) -> String {
    let stream = &f.room.local_participant().byte_streams()[index];
    playback_method(
        base,
        &AvatarPlaybackSegment {
            playback: stream_segment(&stream.opts.name).unwrap(),
            stream_id: Some(stream.id.clone()),
        },
    )
    .unwrap()
}

async fn settle() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

// Catches: audio streamed before the avatar is in the room, a stream per frame instead of per
// segment, the wrong topic/destination/attributes, or a segment left open by flush.
#[tokio::test]
async fn a_segment_is_one_byte_stream_to_the_avatar_once_it_has_joined() {
    let f = fixture("r1", None, false);
    let capture = {
        let out = f.out.clone();
        tokio::spawn(async move {
            out.capture_frame(&frame()).await.unwrap();
            out.capture_frame(&frame()).await.unwrap();
        })
    };
    settle().await;
    assert!(
        f.room.local_participant().byte_streams().is_empty(),
        "nothing before the avatar joins"
    );
    f.room.add_participant("avatar", ParticipantKind::Agent);
    capture.await.unwrap();
    f.out.flush();
    settle().await;

    let streams = f.room.local_participant().byte_streams();
    assert_eq!(streams.len(), 1);
    let s = &streams[0];
    assert_eq!(s.opts.topic, AUDIO_STREAM_TOPIC);
    assert_eq!(s.opts.destination_identities, vec![ParticipantIdentity("avatar".into())]);
    assert_eq!(s.opts.attributes.get("sample_rate").map(String::as_str), Some("24000"));
    assert_eq!(s.opts.attributes.get("num_channels").map(String::as_str), Some("1"));
    assert!(s.opts.name.starts_with("AUDIO_"));
    assert_eq!(s.chunks.len(), 2);
    assert!(s.closed, "flush closes the segment's stream");
    assert_eq!(
        *f.events.0.lock().unwrap(),
        ["started"],
        "no lk.playback_started: the first frame stands for it"
    );
}

// Catches: a capture dropped mid-write (the playout declines a push on interruption) taking the
// segment's writer with it: the host stream stays open, and the flush finds no writer, so the
// avatar never sees the segment end.
#[tokio::test]
async fn a_capture_dropped_mid_write_leaves_the_segment_to_its_flush() {
    let f = fixture("r_drop", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    settle().await;
    f.room.local_participant().flow_byte_writes(false);
    let capture = {
        let out = f.out.clone();
        tokio::spawn(async move { out.capture_frame(&frame()).await })
    };
    settle().await;
    capture.abort();
    let _ = capture.await;
    f.out.flush();
    f.room.local_participant().flow_byte_writes(true);
    settle().await;

    let streams = f.room.local_participant().byte_streams();
    assert_eq!(streams.len(), 1);
    assert!(streams[0].closed, "the dropped capture's stream was never closed");
}

// Catches: a playback report from a stranger accepted, or the avatar's report not reaching the
// session's playback bookkeeping.
#[tokio::test]
async fn the_avatars_playback_finished_rpc_ends_the_segment_and_a_stranger_is_rejected() {
    let f = fixture("r2", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let local = f.room.local_participant();
    assert_eq!(
        local.invoke_rpc(&output_method(&f, RPC_PLAYBACK_FINISHED, 0), "mallory", "{}").as_deref(),
        Some("reject: unexpected participant")
    );
    let reply = local.invoke_rpc(
        &output_method(&f, RPC_PLAYBACK_FINISHED, 0),
        "avatar",
        r#"{"playback_position": 0.01, "interrupted": false}"#,
    );
    assert_eq!(reply.as_deref(), Some("ok"));
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.01 false"]);
}

// Catches: the avatar's idle timeout (no chunk for 10 s: a paused `say(audio=)` stream, a slow
// voice) finishing a segment the agent was still feeding. The output retired it, the next capture
// failed with "the avatar segment retired", and the playout read that as the speech's normal end:
// the rest of its audio was dropped. The segment continues on a new stream, and its one finish
// covers both.
#[tokio::test]
async fn an_idle_finish_of_a_segment_still_capturing_continues_it_on_a_new_stream() {
    let f = fixture("idle_finish", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    let local = f.room.local_participant();
    let reply = local.invoke_rpc(
        &output_method(&f, RPC_PLAYBACK_FINISHED, 0),
        "avatar",
        r#"{"playback_position": 0.01, "interrupted": false}"#,
    );
    assert_eq!(reply.as_deref(), Some("ok"));
    f.out.capture_frame(&frame()).await.expect("the agent is still feeding this segment");
    f.out.flush();
    settle().await;
    let streams = local.byte_streams();
    assert_eq!(streams.len(), 2, "the rest of the segment goes on a new stream");
    assert!(streams.iter().all(|stream| stream.closed));
    assert_eq!(*f.events.0.lock().unwrap(), ["started"], "the segment is not done yet");
    let reply = local.invoke_rpc(
        &output_method(&f, RPC_PLAYBACK_FINISHED, 1),
        "avatar",
        r#"{"playback_position": 0.01, "interrupted": false}"#,
    );
    assert_eq!(reply.as_deref(), Some("ok"));
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.02 false"]);
}

// Catches: an idle finish that covered all a segment's audio leaving its playout waiting on a
// report that will never come once the agent flushes it.
#[tokio::test]
async fn an_idle_finish_covering_the_whole_segment_finishes_it_at_its_flush() {
    let f = fixture("idle_finish_flushed", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    let reply = f.room.local_participant().invoke_rpc(
        &output_method(&f, RPC_PLAYBACK_FINISHED, 0),
        "avatar",
        r#"{"playback_position": 0.01, "interrupted": false}"#,
    );
    assert_eq!(reply.as_deref(), Some("ok"));
    f.out.flush();
    settle().await;
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.01 false"]);
    assert_eq!(f.room.local_participant().byte_streams().len(), 1);
}

// Catches: an interruption that never asks the avatar to clear, or one whose segment never
// finishes when the avatar does not answer (the session would wait forever).
#[tokio::test]
async fn an_unanswered_clear_buffer_finishes_the_segment_after_its_timeout() {
    let f = fixture("r3", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.clear_buffer();
    settle().await;
    let calls = f.room.local_participant().rpc_calls();
    assert_eq!(
        calls,
        vec![(ParticipantIdentity("avatar".into()), RPC_CLEAR_BUFFER.to_owned(), String::new())]
    );
    assert_eq!(*f.events.0.lock().unwrap(), ["started"]);
    f.clock.advance(Duration::from_secs(2));
    settle().await;
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.00 true"]);
}

// Catches: audio sent before the avatar's video track is up, when the plugin asks to wait for it.
#[tokio::test]
async fn wait_remote_track_holds_audio_until_the_avatar_publishes_video() {
    let f = fixture("r4", Some(lk_agents_test_support::rtc::TRACK_KIND_VIDEO), true);
    let avatar = f.room.add_participant("avatar", ParticipantKind::Agent);
    let capture = {
        let out = f.out.clone();
        tokio::spawn(async move { out.capture_frame(&frame()).await.unwrap() })
    };
    settle().await;
    avatar.publish_track("TR_mic", TrackSource::Microphone);
    settle().await;
    assert!(!capture.is_finished(), "an audio track is not the video the plugin waits for");
    avatar.publish_track("TR_cam", TrackSource::Camera);
    capture.await.unwrap();
    assert!(f.events.0.lock().unwrap().is_empty(), "the avatar reports lk.playback_started itself");
    assert_eq!(
        f.room
            .local_participant()
            .invoke_rpc(&output_method(&f, RPC_PLAYBACK_STARTED, 0), "avatar", "")
            .as_deref(),
        Some("ok")
    );
    assert_eq!(*f.events.0.lock().unwrap(), ["started"]);
}

#[tokio::test]
async fn closing_output_releases_startup_and_refuses_later_audio() {
    let f = fixture("close_waiting_output", None, false);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    let capture = TokioSpawner::current().spawn(async move {
        let _ = sent.send(output.capture_frame(&frame()).await);
    });
    settle().await;
    assert!(!capture.is_finished());
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(received.await.unwrap().is_err());
    f.room.add_participant("avatar", ParticipantKind::Agent);
    assert!(f.out.capture_frame(&frame()).await.is_err());
    assert!(f.room.local_participant().byte_streams().is_empty());
}

#[tokio::test]
async fn closing_output_cancels_a_blocked_write_and_retires_its_playback_route() {
    let f = fixture("close_writing_output", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.room.local_participant().flow_byte_writes(false);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    let capture = TokioSpawner::current().spawn(async move {
        let _ = sent.send(output.capture_frame(&frame()).await);
    });
    settle().await;
    assert!(!capture.is_finished());
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(received.await.unwrap().is_err());
    assert!(f.room.local_participant().byte_streams()[0].closed);
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(f.out.wait_for_playout().await.interrupted);
    assert_eq!(
        f.room.local_participant().invoke_rpc(RPC_PLAYBACK_FINISHED, "avatar", "{}").as_deref(),
        None
    );
    assert_eq!(*f.events.0.lock().unwrap(), ["started"]);
}

#[tokio::test]
async fn cancelling_close_does_not_leave_the_avatar_playback_route_live() {
    let f = fixture("cancel_output_close", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.room.local_participant().flow_byte_writes(false);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    TokioSpawner::current().spawn_detached(async move {
        let _ = sent.send(output.capture_frame(&frame()).await);
    });
    settle().await;
    let method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    let mut close = Box::pin(f.out.aclose());
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(close.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    drop(close);
    assert_eq!(f.room.local_participant().invoke_rpc(&method, "avatar", "{}").as_deref(), None);
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(received.await.unwrap().is_err());
}

#[tokio::test]
async fn close_deadline_retains_a_blocked_writer_until_its_acknowledgement() {
    let f = fixture("close_deadline_output", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.room.local_participant().flow_byte_closes(false);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    TokioSpawner::current().spawn_detached(async move {
        let _ = sent.send(output.aclose().await);
    });
    settle().await;
    assert!(!f.room.local_participant().byte_streams()[0].closed);
    f.clock.advance(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    assert_eq!(received.await.unwrap(), Closed::TimedOut);
    assert_eq!(f.out.inner.closing.names(), ["avatar_stream_close"]);
    f.room.local_participant().flow_byte_closes(true);
    settle().await;
    assert!(f.room.local_participant().byte_streams()[0].closed);
    assert!(f.out.inner.closing.is_empty());
    assert_eq!(f.out.aclose().await, Closed::Clean);
}

#[tokio::test]
async fn a_retired_segment_cannot_be_reopened_by_an_admitted_write() {
    let f = fixture("close_before_write", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    let segment = Arc::new(Segment {
        key: PlaybackSegment { attachment: 1, segment: 1 },
        writer: tokio::sync::Mutex::new(SegmentWriter::Unopened),
    });
    f.out.inner.close_segment(Arc::clone(&segment));
    settle().await;
    assert!(f.out.inner.write(&segment, &frame()).await.is_err());
    assert!(f.room.local_participant().byte_streams().is_empty());
    f.out.aclose().await;
}

#[tokio::test]
async fn dropping_a_native_owner_cancels_startup_and_releases_its_inner() {
    let f = fixture("drop_waiting_output", None, false);
    let inner = Arc::downgrade(&f.out.inner);
    drop(f.out);
    settle().await;
    assert!(inner.upgrade().is_none(), "the start task retained a dropped native output");
}

#[tokio::test]
async fn a_delayed_finish_from_a_reused_stream_id_cannot_finish_a_new_output_generation() {
    let f = fixture("reused-stream-playback", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.room.local_participant().set_byte_stream_id(Some("REUSED".into()));
    let old = PlaybackSegment { attachment: 1, segment: 1 };
    let new = PlaybackSegment { attachment: 2, segment: 2 };
    let payload = r#"{"playback_position":0.01,"interrupted":false}"#;
    f.out.begin_segment(old).unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let old_method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    assert_eq!(
        f.room.local_participant().invoke_rpc(&old_method, "avatar", payload).as_deref(),
        Some("ok")
    );
    assert!(f.room.local_participant().invoke_rpc(&old_method, "avatar", payload).is_none());

    f.out.begin_segment(new).unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let new_method = output_method(&f, RPC_PLAYBACK_FINISHED, 1);
    assert_ne!(old_method, new_method);
    assert!(f.room.local_participant().invoke_rpc(&old_method, "avatar", payload).is_none());
    // A callback already dispatched before unregistration retains the old key as well.
    assert_eq!(f.out.inner.on_playback_finished(old, payload), Reply::Stale);
    assert!(f.out.tracker().has_pending_segment(new));
    assert_eq!(
        f.room.local_participant().invoke_rpc(&new_method, "avatar", payload).as_deref(),
        Some("ok")
    );
    assert!(!f.out.tracker().has_pending_segment(new));
    assert_eq!(f.out.aclose().await, Closed::Clean);
}

#[tokio::test]
async fn a_dropped_output_still_closes_its_streams_on_its_executor() {
    let f = fixture("drop_closes_streams", None, false);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.room.local_participant().flow_byte_closes(false);
    drop(f.out);
    settle().await;
    assert!(!f.room.local_participant().byte_streams()[0].closed);
    f.room.local_participant().flow_byte_closes(true);
    settle().await;
    assert!(f.room.local_participant().byte_streams()[0].closed);
}

#[test]
fn drop_cleanup_runs_after_the_callers_async_runtime_has_closed() {
    let host_loop = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let native = crate::process::handle().unwrap();
    let (rtc, _) = FakeRtc::with_virtual_clock(crate::rtc::events::RoomEventBus::port);
    let room = rtc.room("drop_after_host_loop");
    room.add_participant("avatar", ParticipantKind::Agent);
    let clock = Arc::new(lk_agents_abi::RealClock::new());
    let output = DataStreamAudioOutput::new(
        room.clone(),
        "avatar".into(),
        None,
        None,
        None,
        false,
        clock.clone(),
        Arc::new(TokioSpawner::new(native.clone())),
    );
    host_loop.block_on(output.capture_frame(&frame())).unwrap();
    drop(host_loop);
    drop(output);
    native.block_on(async {
        lk_agents_abi::timeout(&*clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, async {
            while !room.local_participant().byte_streams()[0].closed {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    });
}
