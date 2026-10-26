use super::*;
use lk_agents_abi::voice::io::{AudioOutputExt, PlaybackEvent, PlaybackObserver};
use lk_agents_abi::{TokioSpawner, VirtualClock};
use lk_agents_rtc::{FakeRoom, FakeRtc, ParticipantKind, TrackSource};

#[derive(Default)]
struct Events(Mutex<Vec<String>>);

impl PlaybackObserver for Events {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        let label = match ev {
            PlaybackEvent::Started(_) => "started".to_owned(),
            PlaybackEvent::Progressed(_) => "progressed".to_owned(),
            PlaybackEvent::Finished(f) => {
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
    let (rtc, clock) = FakeRtc::with_virtual_clock();
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
    assert!(f.room.local_participant().byte_streams().is_empty(), "nothing before the avatar joins");
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
    assert_eq!(*f.events.0.lock().unwrap(), ["started"], "no lk.playback_started: the first frame stands for it");
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
    assert_eq!(local.invoke_rpc(RPC_PLAYBACK_FINISHED, "mallory", "{}").as_deref(), Some("reject"));
    let reply = local.invoke_rpc(RPC_PLAYBACK_FINISHED, "avatar", r#"{"playback_position": 0.01, "interrupted": false}"#);
    assert_eq!(reply.as_deref(), Some("ok"));
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.01 false"]);
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
    assert_eq!(calls, vec![(ParticipantIdentity("avatar".into()), RPC_CLEAR_BUFFER.to_owned(), String::new())]);
    assert_eq!(*f.events.0.lock().unwrap(), ["started"]);
    f.clock.advance(Duration::from_secs(2));
    settle().await;
    assert_eq!(*f.events.0.lock().unwrap(), ["started", "finished 0.02 true"]);
}

// Catches: audio sent before the avatar's video track is up, when the plugin asks to wait for it.
#[tokio::test]
async fn wait_remote_track_holds_audio_until_the_avatar_publishes_video() {
    let f = fixture("r4", Some(lk_agents_rtc::TRACK_KIND_VIDEO), true);
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
    assert_eq!(f.room.local_participant().invoke_rpc(RPC_PLAYBACK_STARTED, "avatar", "").as_deref(), Some("ok"));
    assert_eq!(*f.events.0.lock().unwrap(), ["started"]);
}

// ------------------------------------------------------------------------- the avatar's side

fn audio_info(id: &str) -> lk_agents_rtc::StreamInfo {
    lk_agents_rtc::StreamInfo {
        id: id.to_owned(),
        topic: AUDIO_STREAM_TOPIC.to_owned(),
        attributes: HashMap::from([
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
    cleared: Arc<std::sync::atomic::AtomicUsize>,
}

async fn receiver(name: &str) -> RxFixture {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room(name);
    std::mem::forget(rtc);
    room.add_participant("agent-1", ParticipantKind::Agent);
    let cleared = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let on_clear = {
        let cleared = cleared.clone();
        Arc::new(move || {
            cleared.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }) as Arc<dyn Fn() + Send + Sync>
    };
    let rx = DataStreamAudioReceiver::new(
        room.clone(),
        None,
        100,
        3,
        Some(on_clear),
        clock.clone(),
        Arc::new(TokioSpawner::current()),
    );
    rx.start().await.unwrap();
    RxFixture { room, clock, rx, cleared }
}

fn kinds(items: &[Received]) -> Vec<String> {
    items
        .iter()
        .map(|i| match i {
            Received::Frame(f) => format!("frame {}", f.samples_per_channel),
            Received::SegmentEnd => "end".to_owned(),
        })
        .collect()
}

async fn take(rx: &DataStreamAudioReceiver, n: usize) -> Vec<Received> {
    let mut out = Vec::new();
    for _ in 0..n {
        out.push(tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap());
    }
    out
}

// Catches: frames cut at the wrong size (the reference frames at `frame_size_ms`), the tail lost
// at the trailer, or no segment end for the avatar to report playback on.
#[tokio::test]
async fn a_segment_comes_out_in_frame_size_frames_then_its_end() {
    let f = receiver("rx1").await;
    // 250 ms at 10 kHz mono: two 100 ms frames and a 50 ms tail
    f.room.send_byte_stream("agent-1", audio_info("S1"), [vec![0u8; 3000], vec![0u8; 2000]]).unwrap();
    let items = take(&f.rx, 4).await;
    assert_eq!(kinds(&items), ["frame 1000", "frame 1000", "frame 500", "end"]);
}

// Catches: a lost trailer wedging the avatar (it stops receiving audio): a new stream must end
// the open one.
#[tokio::test]
async fn a_new_stream_ends_one_whose_trailer_never_came() {
    let f = receiver("rx2").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 2000]).unwrap();
    let first = take(&f.rx, 1).await;
    assert_eq!(kinds(&first), ["frame 1000"]);
    f.room.send_byte_stream("agent-1", audio_info("S2"), [vec![0u8; 2000]]).unwrap();
    let items = take(&f.rx, 3).await;
    assert_eq!(kinds(&items), ["end", "frame 1000", "end"]);
    drop(open);
}

// Catches: clear_buffer leaving buffered audio to play, accepted from a stranger, or not raised
// as the runner's clear_buffer event.
#[tokio::test]
async fn clear_buffer_from_the_agent_drops_the_segment_and_is_raised() {
    let f = receiver("rx3").await;
    let open = f.room.open_byte_stream("agent-1", audio_info("S1")).unwrap();
    open.send(vec![0u8; 4000]).unwrap();
    take(&f.rx, 1).await;
    let local = f.room.local_participant();
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "mallory", "").as_deref(), Some("reject"));
    assert_eq!(local.invoke_rpc(RPC_CLEAR_BUFFER, "agent-1", "").as_deref(), Some("ok"));
    assert_eq!(f.cleared.load(std::sync::atomic::Ordering::Relaxed), 1);
    let items = take(&f.rx, 1).await;
    assert_eq!(kinds(&items), ["end"], "the buffered frame is gone and the segment ends");
    drop(open);
}

// Catches: playback reports with the wrong method or payload, or a failing report retried forever
// (the reference gives up after `rpc_max_retries` and fails the receiver).
#[tokio::test]
async fn playback_reports_go_to_the_agent_and_a_failing_one_ends_the_receiver() {
    let f = receiver("rx4").await;
    f.rx.notify_playback_started();
    f.rx.notify_playback_finished(1.5, true);
    settle().await;
    let calls = f.room.local_participant().rpc_calls();
    assert_eq!(calls[0], (ParticipantIdentity("agent-1".into()), RPC_PLAYBACK_STARTED.to_owned(), String::new()));
    assert_eq!(calls[1].1, RPC_PLAYBACK_FINISHED);
    let payload: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(payload["playback_position"], 1.5);
    assert_eq!(payload["interrupted"], true);

    f.room.local_participant().set_rpc_reply(1500, "boom");
    f.rx.notify_playback_finished(2.0, false);
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
