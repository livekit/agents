//! The avatar's two ends over FakeRoom, in what the DST's avatar seeds do not reach: the agent's
//! `DataStreamAudioOutput` (joins, idle finishes, give-up timer, close and drop) and the worker's
//! `DataStreamAudioReceiver` (lost trailers, strangers, failing reports, bad streams, the join wait).
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, AvatarPlaybackSegment, PlaybackEvent, PlaybackObserver,
    PlaybackSegment,
};
use lk_agents_abi::voice::rtc::{
    ConnectionState, ParticipantIdentity, ParticipantKind, StreamInfo, TRACK_KIND_VIDEO,
    TrackSource,
};
use lk_agents_abi::{AudioFrame, Clock as _, Closed, SpawnerExt, VirtualClock};
use lk_agents_fuzz::fakes::room::{FakeRoom, FakeRtc};
use lk_agents_runtime::avatar::output::DataStreamAudioOutput;
use lk_agents_runtime::avatar::receiver::{DataStreamAudioReceiver, wait_avatar_join_room};
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::rtc::events::RoomEventBus;
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, RPC_CLEAR_BUFFER, RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED, Received,
    ReceiverEnd, playback_method, stream_name, stream_segment,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

async fn settle() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

// ------------------------------------------------------------------------- the agent's side

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

/// 10 ms at 24 kHz mono.
fn frame() -> AudioFrame {
    AudioFrame::new(vec![1u8; 480], 24_000, 1, 240)
}

struct Fixture {
    room: Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    out: Arc<DataStreamAudioOutput>,
    events: Arc<Events>,
}

impl Fixture {
    fn events(&self) -> Vec<String> {
        self.events.0.lock().unwrap().clone()
    }

    fn rpc(&self, method: &str, caller: &str, payload: &str) -> Option<String> {
        self.room.local_participant().invoke_rpc(method, caller, payload)
    }
}

fn fixture(name: &str, wait_remote_track: Option<i32>) -> Fixture {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room(name);
    let out = Arc::new(DataStreamAudioOutput::new(
        room.clone(),
        "avatar".to_owned(),
        None,
        wait_remote_track,
        Some(Duration::from_secs(2)),
        wait_remote_track.is_some(),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    ));
    let events = Arc::new(Events::default());
    out.tracker().observe(events.clone());
    std::mem::forget(rtc);
    Fixture { room, clock, out, events }
}

/// The RPC method the avatar reports the `index`th byte stream's playback on.
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

const PLAYED_10MS: &str = r#"{"playback_position": 0.01, "interrupted": false}"#;

// Catches: audio streamed before the avatar is in the room, a stream per frame instead of per
// segment, the wrong topic/destination/attributes, or a segment left open by flush.
#[tokio::test]
async fn a_segment_is_one_byte_stream_to_the_avatar_once_it_has_joined() {
    let f = fixture("r1", None);
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
    assert_eq!(f.events(), ["started"], "no lk.playback_started: the first frame stands for it");
}

// Catches: a capture dropped mid-write (the playout declines a push on interruption) taking the
// segment's writer with it: the host stream stays open, and the flush finds no writer, so the
// avatar never sees the segment end.
#[tokio::test]
async fn a_capture_dropped_mid_write_leaves_the_segment_to_its_flush() {
    let f = fixture("r_drop", None);
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
    let f = fixture("r2", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    assert_eq!(f.rpc(&method, "mallory", "{}").as_deref(), Some("reject: unexpected participant"));
    assert_eq!(f.rpc(&method, "avatar", PLAYED_10MS).as_deref(), Some("ok"));
    assert_eq!(f.events(), ["started", "finished 0.01 false"]);
}

// Catches: the avatar's idle timeout (no chunk for 10 s: a paused `say(audio=)` stream, a slow
// voice) finishing a segment the agent was still feeding. The output retired it, the next capture
// failed with "the avatar segment retired", and the playout read that as the speech's normal end:
// the rest of its audio was dropped. The segment continues on a new stream, and its one finish
// covers both.
#[tokio::test]
async fn an_idle_finish_of_a_segment_still_capturing_continues_it_on_a_new_stream() {
    let f = fixture("idle_finish", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    let method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    assert_eq!(f.rpc(&method, "avatar", PLAYED_10MS).as_deref(), Some("ok"));
    f.out.capture_frame(&frame()).await.expect("the agent is still feeding this segment");
    f.out.flush();
    settle().await;
    let streams = f.room.local_participant().byte_streams();
    assert_eq!(streams.len(), 2, "the rest of the segment goes on a new stream");
    assert!(streams.iter().all(|stream| stream.closed));
    assert_eq!(f.events(), ["started"], "the segment is not done yet");
    let method = output_method(&f, RPC_PLAYBACK_FINISHED, 1);
    assert_eq!(f.rpc(&method, "avatar", PLAYED_10MS).as_deref(), Some("ok"));
    assert_eq!(f.events(), ["started", "finished 0.02 false"]);
}

// Catches: an idle finish that covered all a segment's audio leaving its playout waiting on a
// report that will never come once the agent flushes it.
#[tokio::test]
async fn an_idle_finish_covering_the_whole_segment_finishes_it_at_its_flush() {
    let f = fixture("idle_finish_flushed", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    let method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    assert_eq!(f.rpc(&method, "avatar", PLAYED_10MS).as_deref(), Some("ok"));
    f.out.flush();
    settle().await;
    assert_eq!(f.events(), ["started", "finished 0.01 false"]);
    assert_eq!(f.room.local_participant().byte_streams().len(), 1);
}

// Catches: an interruption that never asks the avatar to clear, or one whose segment never
// finishes when the avatar does not answer (the session would wait forever). The DST's avatar
// seeds run without this give-up timer.
#[tokio::test]
async fn an_unanswered_clear_buffer_finishes_the_segment_after_its_timeout() {
    let f = fixture("r3", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.clear_buffer();
    settle().await;
    assert_eq!(
        f.room.local_participant().rpc_calls(),
        vec![(ParticipantIdentity("avatar".into()), RPC_CLEAR_BUFFER.to_owned(), String::new())]
    );
    assert_eq!(f.events(), ["started"]);
    f.clock.advance(Duration::from_secs(2));
    settle().await;
    assert_eq!(f.events(), ["started", "finished 0.00 true"]);
}

// Catches: audio sent before the avatar's video track is up, when the plugin asks to wait for it.
#[tokio::test]
async fn wait_remote_track_holds_audio_until_the_avatar_publishes_video() {
    let f = fixture("r4", Some(TRACK_KIND_VIDEO));
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
    assert!(f.events().is_empty(), "the avatar reports lk.playback_started itself");
    let method = output_method(&f, RPC_PLAYBACK_STARTED, 0);
    assert_eq!(f.rpc(&method, "avatar", "").as_deref(), Some("ok"));
    assert_eq!(f.events(), ["started"]);
}

#[tokio::test]
async fn closing_output_releases_startup_and_refuses_later_audio() {
    let f = fixture("close_waiting_output", None);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    let capture = tokio::spawn(async move {
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
    let f = fixture("close_writing_output", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.room.local_participant().flow_byte_writes(false);
    let (sent, received) = tokio::sync::oneshot::channel();
    let output = Arc::clone(&f.out);
    let capture = tokio::spawn(async move {
        let _ = sent.send(output.capture_frame(&frame()).await);
    });
    settle().await;
    assert!(!capture.is_finished());
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(received.await.unwrap().is_err());
    assert!(f.room.local_participant().byte_streams()[0].closed);
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(f.out.wait_for_playout().await.interrupted);
    assert_eq!(f.rpc(RPC_PLAYBACK_FINISHED, "avatar", "{}"), None);
    assert_eq!(f.events(), ["started"]);
}

#[tokio::test]
async fn cancelling_close_does_not_leave_the_avatar_playback_route_live() {
    let f = fixture("cancel_output_close", None);
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
    assert_eq!(f.rpc(&method, "avatar", "{}"), None);
    assert_eq!(f.out.aclose().await, Closed::Clean);
    assert!(received.await.unwrap().is_err());
}

#[tokio::test]
async fn close_deadline_retains_a_blocked_writer_until_its_acknowledgement() {
    let f = fixture("close_deadline_output", None);
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
    f.room.local_participant().flow_byte_closes(true);
    settle().await;
    assert!(f.room.local_participant().byte_streams()[0].closed);
    assert_eq!(f.out.aclose().await, Closed::Clean);
}

#[tokio::test]
async fn a_delayed_finish_from_a_reused_stream_id_cannot_finish_a_new_output_generation() {
    let f = fixture("reused-stream-playback", None);
    f.room.add_participant("avatar", ParticipantKind::Agent);
    f.room.local_participant().set_byte_stream_id(Some("REUSED".into()));
    let old = PlaybackSegment { attachment: 1, segment: 1 };
    let new = PlaybackSegment { attachment: 2, segment: 2 };
    let payload = r#"{"playback_position":0.01,"interrupted":false}"#;
    f.out.begin_segment(old).unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let old_method = output_method(&f, RPC_PLAYBACK_FINISHED, 0);
    assert_eq!(f.rpc(&old_method, "avatar", payload).as_deref(), Some("ok"));
    assert!(f.rpc(&old_method, "avatar", payload).is_none());

    f.out.begin_segment(new).unwrap();
    f.out.capture_frame(&frame()).await.unwrap();
    f.out.flush();
    let new_method = output_method(&f, RPC_PLAYBACK_FINISHED, 1);
    assert_ne!(old_method, new_method);
    assert!(f.rpc(&old_method, "avatar", payload).is_none());
    assert!(f.out.tracker().has_pending_segment(new));
    assert_eq!(f.rpc(&new_method, "avatar", payload).as_deref(), Some("ok"));
    assert!(!f.out.tracker().has_pending_segment(new));
    assert_eq!(f.out.aclose().await, Closed::Clean);
}

#[tokio::test]
async fn a_dropped_output_still_closes_its_streams_on_its_executor() {
    let f = fixture("drop_closes_streams", None);
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
    let native = lk_agents_runtime::process::handle().unwrap();
    let (rtc, _) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("drop_after_host_loop");
    room.add_participant("avatar", ParticipantKind::Agent);
    let clock = Arc::new(lk_agents_runtime::net::clock::RealClock::new());
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

// ------------------------------------------------------------------------- the avatar's side

fn receipt_segment(id: &str) -> AvatarPlaybackSegment {
    AvatarPlaybackSegment {
        playback: PlaybackSegment {
            attachment: 1,
            segment: id.strip_prefix('S').and_then(|id| id.parse().ok()).unwrap_or(1),
        },
        stream_id: Some(id.to_owned()),
    }
}

fn audio_info(id: &str) -> StreamInfo {
    StreamInfo {
        id: id.to_owned(),
        name: Some(stream_name(receipt_segment(id).playback)),
        topic: AUDIO_STREAM_TOPIC.to_owned(),
        attributes: BTreeMap::from([
            ("sample_rate".to_owned(), "10000".to_owned()),
            ("num_channels".to_owned(), "1".to_owned()),
        ]),
        ..Default::default()
    }
}

struct RxFixture {
    room: Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    rx: DataStreamAudioReceiver,
    cleared: Arc<AtomicUsize>,
}

async fn receiver(name: &str) -> RxFixture {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room(name);
    std::mem::forget(rtc);
    room.add_participant("agent-1", ParticipantKind::Agent);
    let cleared = Arc::new(AtomicUsize::new(0));
    let on_clear = {
        let cleared = cleared.clone();
        Arc::new(move || {
            cleared.fetch_add(1, Ordering::Relaxed);
        }) as Arc<dyn Fn() + Send + Sync>
    };
    let rx = DataStreamAudioReceiver::new(
        room.clone(),
        None,
        std::num::NonZeroU32::new(100).unwrap(),
        3,
        Some(on_clear),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    );
    rx.start().await.unwrap();
    RxFixture { room, clock, rx, cleared }
}

/// The next `n` items the receiver yields, as `frame <samples>` or `end`.
async fn take(rx: &DataStreamAudioReceiver, n: usize) -> Vec<String> {
    let mut out = Vec::new();
    for _ in 0..n {
        let item = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap();
        out.push(match item.unwrap().3.unwrap() {
            Received::Audio(f) => format!("frame {}", f.samples_per_channel),
            Received::SegmentEnd => "end".to_owned(),
        });
    }
    out
}

// Catches: byte-stream chunks handed on as they arrived instead of frames of the receiver's frame
// size, or a segment's short tail dropped before its end.
#[tokio::test]
async fn a_segment_comes_out_in_frame_size_frames_then_its_end() {
    let f = receiver("rx1").await;
    // 250 ms at 10 kHz mono: two 100 ms frames and a 50 ms tail
    f.room
        .send_byte_stream("agent-1", audio_info("S1"), [vec![0u8; 3000], vec![0u8; 2000]])
        .unwrap();
    assert_eq!(take(&f.rx, 4).await, ["frame 1000", "frame 1000", "frame 500", "end"]);
}

// Catches: a lost trailer wedging the avatar (it stops receiving audio): a new stream must end
// the open one.
#[tokio::test]
async fn a_new_stream_ends_one_whose_trailer_never_came() {
    let f = receiver("rx2").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 2000]).await.unwrap();
    assert_eq!(take(&f.rx, 1).await, ["frame 1000"]);
    f.room.send_byte_stream("agent-1", audio_info("S2"), [vec![0u8; 2000]]).unwrap();
    assert_eq!(take(&f.rx, 3).await, ["end", "frame 1000", "end"]);
    drop(open);
}

// Catches: clear_buffer leaving buffered audio to play, accepted from a stranger, or not raised
// as the runner's clear_buffer event.
#[tokio::test]
async fn clear_buffer_from_the_agent_drops_the_segment_and_is_raised() {
    let f = receiver("rx3").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 4000]).await.unwrap();
    take(&f.rx, 1).await;
    let local = f.room.local_participant();
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "mallory", "").as_deref(), Some("reject"));
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "agent-1", "").as_deref(), Some("ok"));
    assert_eq!(f.cleared.load(Ordering::Relaxed), 1);
    assert_eq!(take(&f.rx, 1).await, ["end"], "the buffered frame is gone and the segment ends");
    drop(open);
}

// Catches: playback reports with the wrong method or payload, or a failing report retried forever
// (the reference gives up after `rpc_max_retries` and fails the receiver).
#[tokio::test]
async fn playback_reports_go_to_the_agent_and_a_failing_one_ends_the_receiver() {
    let f = receiver("rx4").await;
    f.rx.notify_playback_started(receipt_segment("S1")).unwrap();
    f.rx.notify_playback_finished(receipt_segment("S1"), 1.5, true).unwrap();
    settle().await;
    let calls = f.room.local_participant().rpc_calls();
    assert_eq!(
        calls[0],
        (
            ParticipantIdentity("agent-1".into()),
            playback_method(RPC_PLAYBACK_STARTED, &receipt_segment("S1")).unwrap(),
            String::new()
        )
    );
    assert_eq!(calls[1].1, playback_method(RPC_PLAYBACK_FINISHED, &receipt_segment("S1")).unwrap());
    let payload: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(payload["playback_position"], 1.5);
    assert_eq!(payload["interrupted"], true);

    f.room.local_participant().set_rpc_reply(1500, "boom");
    f.rx.notify_playback_finished(receipt_segment("S2"), 2.0, false).unwrap();
    for _ in 0..4 {
        settle().await;
        f.clock.advance(Duration::from_millis(100));
    }
    settle().await;
    assert_eq!(f.room.local_participant().rpc_calls().len(), 5, "three attempts");
    match tokio::time::timeout(Duration::from_secs(5), f.rx.recv()).await.unwrap() {
        Err(ReceiverEnd::Failed(e)) => assert!(e.contains("1500"), "{e}"),
        other => panic!("{other:?}"),
    }
}

// Catches: a stream with no format read as some default, which would play garbage.
#[tokio::test]
async fn a_stream_without_its_format_fails_the_receiver() {
    let f = receiver("rx5").await;
    let mut info = audio_info("S1");
    info.attributes.clear();
    f.room.send_byte_stream("agent-1", info, [vec![0u8; 10]]).unwrap();
    match tokio::time::timeout(Duration::from_secs(5), f.rx.recv()).await.unwrap() {
        Err(ReceiverEnd::BadStream(_)) => {}
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_failed_byte_stream_does_not_flush_a_partial_avatar_segment() {
    let f = receiver("rx-failed-chunk").await;
    let oversized = lk_agents_voice::room_io::ingress::MAX_STREAM_CHUNK_BYTES + 1;
    f.room
        .send_byte_stream("agent-1", audio_info("failed"), [vec![0; 1500], vec![0; oversized]])
        .unwrap();
    let outcome = lk_agents_abi::timeout(
        &lk_agents_runtime::net::clock::RealClock::new(),
        Duration::from_secs(1),
        f.rx.recv(),
    )
    .await;
    f.rx.aclose();
    let outcome = outcome.expect("the failed stream must settle its receiver");
    assert!(
        matches!(outcome, Err(ReceiverEnd::BadStream(_))),
        "a failed stream must not produce its buffered tail or a successful segment end"
    );
}

// Catches: a join wait that starts before the room connects (the reference arms it on the
// connection), or that ends at the participant without its video track.
#[tokio::test]
async fn the_join_wait_starts_at_the_connection_and_ends_at_the_avatar_video() {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("join");
    room.set_connection_state(ConnectionState::Disconnected);
    let wait = tokio::spawn(wait_avatar_join_room(room.clone(), "avatar".into(), clock.clone()));
    clock.advance(Duration::from_secs(1));
    tokio::task::yield_now().await;
    let connected_at = lk_agents_abi::unix_secs(clock.wall());
    room.set_connection_state(ConnectionState::Connected);
    let avatar = room.add_participant("avatar", ParticipantKind::Agent);
    tokio::task::yield_now().await;
    clock.advance(Duration::from_secs(2));
    assert!(!wait.is_finished());
    avatar.publish_track("TR_cam", TrackSource::Camera);
    let (started, joined) =
        tokio::time::timeout(Duration::from_secs(5), wait).await.unwrap().unwrap().unwrap();
    assert!((started - connected_at).abs() < 0.01, "{started} {connected_at}");
    assert!((joined - connected_at - 2.0).abs() < 0.01, "{joined} {connected_at}");
}

#[tokio::test(start_paused = true)]
async fn a_closed_receiver_cannot_restart_or_steal_a_replacement_route() {
    let first = receiver("closed-receiver-route").await;
    first.rx.aclose();
    settle().await;
    let replacement = DataStreamAudioReceiver::new(
        first.room.clone(),
        None,
        std::num::NonZeroU32::new(100).unwrap(),
        3,
        None,
        first.clock.clone(),
        Arc::new(TokioSpawner::current()),
    );
    replacement.start().await.unwrap();
    assert_eq!(first.rx.start().await, Err(ReceiverEnd::Closed));
    first.rx.aclose();
    use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
    assert_eq!(first.room.rpc_router().call("agent-1", ""), "ok");
    assert_eq!(first.cleared.load(Ordering::Relaxed), 0);
    replacement.aclose();
}

#[tokio::test(start_paused = true)]
async fn closing_a_receiver_settles_startup_waiting_for_its_sender() {
    use std::future::Future as _;
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room("receiver-start-close");
    let rx = DataStreamAudioReceiver::new(
        room,
        None,
        std::num::NonZeroU32::new(100).unwrap(),
        3,
        None,
        clock,
        Arc::new(TokioSpawner::current()),
    );
    let mut starting = Box::pin(rx.start());
    let pending =
        std::future::poll_fn(|cx| std::task::Poll::Ready(starting.as_mut().poll(cx).is_pending()))
            .await;
    assert!(pending, "startup must wait for the absent sender");
    rx.aclose();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), starting).await.unwrap(),
        Err(ReceiverEnd::Closed)
    );
}

#[tokio::test(start_paused = true)]
async fn failed_receiver_startup_does_not_retire_the_running_receivers_route() {
    let first = receiver("receiver-failed-start").await;
    let competing = DataStreamAudioReceiver::new(
        first.room.clone(),
        None,
        std::num::NonZeroU32::new(100).unwrap(),
        3,
        None,
        first.clock.clone(),
        Arc::new(TokioSpawner::current()),
    );
    assert!(competing.start().await.is_err(), "one SDK topic admits one receiver");
    competing.aclose();
    use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
    assert_eq!(first.room.rpc_router().call("agent-1", ""), "ok");
    assert_eq!(first.cleared.load(Ordering::Relaxed), 1);
    first.rx.aclose();
}

#[tokio::test(start_paused = true)]
async fn duplicate_receiver_start_does_not_close_its_existing_registration() {
    let first = receiver("receiver-duplicate-start").await;
    assert!(first.rx.start().await.is_err());
    use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
    assert_eq!(first.room.rpc_router().call("agent-1", ""), "ok");
    assert_eq!(first.cleared.load(Ordering::Relaxed), 1);
    first.rx.aclose();
}
