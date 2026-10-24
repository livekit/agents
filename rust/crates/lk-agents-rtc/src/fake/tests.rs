//! The fake's own tests: playout, stream conversion, room and participant lifecycle, events.

use super::*;
use crate::{AudioStreamOptions, PlaybackInfo, TranscriptionSegment, VideoFrameRef, silence_frame};
use tokio::time::timeout;

/// Long enough that a future which is merely slow would still finish. Under
/// `start_paused = true` tokio auto-advances to it the moment every task is parked, so this
/// costs no wall-clock time and the outcome is deterministic: expiring means *nothing can
/// make progress*, which is exactly the "is it blocked?" question.
const NEVER: Duration = Duration::from_secs(3600);

/// Await `f`, failing the test rather than hanging when the fake cannot make progress.
///
/// Every async test here is `start_paused`, so this is free. It exists because most of the
/// ways this double can break -- a `clear_queue` that does not wake, an unpublish that does not
/// close the stream, an event that is never emitted -- present as a hang, and a hang is a
/// timeout in CI rather than a named failure.
async fn eventually<F: std::future::Future>(f: F) -> F::Output {
    timeout(NEVER, f).await.expect("the fake made no progress: this future never completed")
}

/// 20 ms of mono 24 kHz, tagged in its first byte so assertions can tell frames apart.
fn frame(tag: u8) -> AudioFrame {
    let mut pcm = vec![0u8; 480 * 2];
    pcm[0] = tag;
    AudioFrame::new(pcm, 24_000, 1, 480)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn tags(frames: &[AudioFrame]) -> Vec<u8> {
    frames.iter().map(|f| f.data[0]).collect()
}

/// A room with the agent's track already published.
async fn published() -> (Arc<FakeRtc>, Arc<FakeRoom>, Arc<FakeAudioSource>) {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let room = rtc.room("session");
    room.local().publish_audio_track(TrackPublishOptions::default()).await.unwrap();
    let source = room.local_participant().sources()[0].clone();
    (rtc, room, source)
}

// ---- faithful about playout --------------------------------------------------------------

// Catches the bug that would make this whole double worthless: a `clear_queue` that resets a
// counter without dropping queued audio. Every interruption test in the pipeline would then
// pass while interruption shipped broken -- the agent talks over the user and nothing fails.
#[tokio::test(start_paused = true)]
async fn clear_queue_really_discards_the_audio_that_had_not_played() {
    let (rtc, _room, source) = published().await;
    for tag in 1..=5 {
        eventually(source.capture(&frame(tag))).await.unwrap(); // 5 x 20 ms = 100 ms
    }
    assert_eq!(source.queued_duration(), ms(100));

    rtc.advance(ms(30)); // frame 1 has played; frame 2 is halfway through
    source.clear_queue();

    let info = eventually(source.wait_for_playout()).await;
    assert_eq!(info.played, ms(30), "exactly the audio the clock let through");
    assert_eq!(info.discarded, ms(70));
    assert!(info.interrupted());
    assert_eq!(source.queued_duration(), Duration::ZERO);

    assert_eq!(tags(&source.captured()), vec![1, 2, 3, 4, 5]);
    assert_eq!(tags(&source.played()), vec![1], "only whole frames reach the speaker");
    assert_eq!(tags(&source.discarded()), vec![2, 3, 4, 5]);
}

// Catches: a `wait_for_playout` that resolves as soon as it is called, or counters that never
// reset. The reference races it
// against the interruption event, so a double that never waits makes every playout look
// finished-at-zero and hides the ordering bugs the race exists to expose.
#[tokio::test(start_paused = true)]
async fn wait_for_playout_waits_for_the_clock() {
    let (rtc, _room, source) = published().await;
    eventually(source.capture(&frame(1))).await.unwrap();
    eventually(source.capture(&frame(2))).await.unwrap(); // 40 ms queued

    assert!(timeout(NEVER, source.wait_for_playout()).await.is_err(), "returned with 40 ms left");
    rtc.advance(ms(20));
    assert!(timeout(NEVER, source.wait_for_playout()).await.is_err(), "returned with 20 ms left");
    rtc.advance(ms(20));

    let info = timeout(NEVER, source.wait_for_playout()).await.expect("the queue has drained");
    assert_eq!(info, PlaybackInfo { played: ms(40), discarded: Duration::ZERO });
    assert!(!info.interrupted());

    // The counters cover one wait, not the session: the reference zeroes them at the end of
    // every `_wait_for_playout`, or the next utterance reports this one's duration too.
    eventually(source.capture(&frame(3))).await.unwrap();
    rtc.advance(ms(20));
    assert_eq!(eventually(source.wait_for_playout()).await.played, ms(20));
}

// Catches: `clear_queue` not waking a parked `wait_for_playout`. That is the interruption path,
// and the symptom is an agent that goes silent but never reports the interruption -- the turn
// never ends and the session wedges.
#[tokio::test(start_paused = true)]
async fn clear_queue_releases_a_parked_wait_for_playout() {
    let (_rtc, _room, source) = published().await;
    for tag in 1..=4 {
        eventually(source.capture(&frame(tag))).await.unwrap();
    }

    let waiting = source.wait_for_playout();
    let interrupting = async { source.clear_queue() };
    let (info, ()) = tokio::join!(eventually(waiting), interrupting);

    assert_eq!(info, PlaybackInfo { played: Duration::ZERO, discarded: ms(80) });
}

// Catches: an unbounded `capture`. `rtc.AudioSource.capture_frame` blocks past `queue_size_ms`
// and that block is the output path's only backpressure -- without it the forwarding loop runs
// a whole utterance into the source in one tick, after which `clear_queue` has nothing left to
// cut and interruption silently stops working.
#[tokio::test(start_paused = true)]
async fn capture_blocks_once_the_queue_is_full() {
    let (rtc, room, _s) = published().await;
    let opts = TrackPublishOptions { queue_size_ms: 60, ..TrackPublishOptions::default() };
    room.local().publish_audio_track(opts).await.unwrap();
    let source = room.local_participant().sources()[1].clone();

    for tag in 1..=3 {
        eventually(source.capture(&frame(tag))).await.unwrap(); // fills 60 ms exactly
    }
    assert!(
        timeout(NEVER, source.capture(&frame(4))).await.is_err(),
        "the 4th frame must wait for the queue to drain"
    );
    assert_eq!(tags(&source.captured()), vec![1, 2, 3], "a blocked frame must not be queued");

    rtc.advance(ms(60));
    eventually(source.capture(&frame(4))).await.unwrap();
    assert_eq!(tags(&source.captured()), vec![1, 2, 3, 4]);
}

// Catches: a frame longer than `queue_size_ms` deadlocking. `rtc` unblocks such a frame once
// the queue drains; here nothing else drains it, so applying the bound unconditionally would
// hang the very first capture -- and 100 ms is `AudioByteStream`'s default frame.
#[tokio::test(start_paused = true)]
async fn a_frame_longer_than_the_queue_still_captures() {
    let (_rtc, room, _s) = published().await;
    let opts = TrackPublishOptions { queue_size_ms: 20, ..TrackPublishOptions::default() };
    let (source, _pub_) = room.local().publish_audio_track(opts).await.unwrap();

    let big = silence_frame(2_400, 24_000, 1); // 100 ms
    timeout(NEVER, source.capture(&big)).await.expect("an empty queue always accepts").unwrap();
    assert_eq!(source.queued_duration(), ms(100));
}

// Catches: `close` leaving queued audio unaccounted, so `played + discarded` stops equalling
// what was captured and a session torn down mid-utterance reports playing audio nobody heard.
#[tokio::test(start_paused = true)]
async fn closing_the_source_accounts_the_queue_as_discarded() {
    let (_rtc, _room, source) = published().await;
    eventually(source.capture(&frame(1))).await.unwrap();
    eventually(source.capture(&frame(2))).await.unwrap();
    assert!(!source.is_closed());

    source.close().await.unwrap();

    assert!(source.is_closed());
    assert_eq!(
        eventually(source.wait_for_playout()).await,
        PlaybackInfo { played: Duration::ZERO, discarded: ms(40) }
    );
    assert_eq!(eventually(source.capture(&frame(3))).await, Err(RtcError::Closed("audio source")));
}

// ---- the stream honours what it was opened with ------------------------------------------

/// `n` samples of a mono ramp at `rate`, so a resample is visible in the data and not only in
/// the header.
fn ramp(rate: u32, channels: u16, samples: u32) -> AudioFrame {
    let mut pcm = Vec::with_capacity(samples as usize * usize::from(channels) * 2);
    for i in 0..samples {
        for c in 0..channels {
            let v = (i as i16).wrapping_mul(10).wrapping_add(c as i16 * 1000);
            pcm.extend_from_slice(&v.to_le_bytes());
        }
    }
    AudioFrame::new(pcm, rate, channels, samples)
}

// Catches: `audio_stream` ignoring its `AudioStreamOptions`. Against a real room
// `rtc.AudioStream` converts the track natively before the frame crosses, so a fake that hands
// back whatever `push_frame` was given makes `AudioInputConfig::sample_rate` unobservable --
// every test built on it passes whether the input path forwards the requested rate or drops
// it, and the difference in production is the STT leg reading 48 kHz where it asked for 24.
#[tokio::test(start_paused = true)]
async fn a_stream_reads_the_rate_it_asked_for_whatever_the_track_pushes() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");

    let mut asked = publication.audio_stream(AudioStreamOptions {
        sample_rate: Some(24_000),
        ..AudioStreamOptions::default()
    });
    // A second reader on the same track that asked for nothing: it must still see the track.
    let mut untouched = publication.audio_stream(AudioStreamOptions::default());

    assert_eq!(publication.push_frame(ramp(48_000, 1, 480)), 2);

    let got = eventually(asked.recv()).await.expect("the stream delivered nothing");
    assert_eq!(got.sample_rate, 24_000, "the requested rate was ignored");
    assert_eq!(got.samples_per_channel, 240, "half the rate is half the samples");
    assert_eq!(got.data.len(), 240 * 2);
    assert_eq!(got.duration(), 0.01, "and the frame still describes 10 ms of audio");

    let raw = eventually(untouched.recv()).await.expect("the stream delivered nothing");
    assert_eq!(raw.sample_rate, 48_000, "`None` must leave the track's own rate alone");
    assert_eq!(raw.samples_per_channel, 480);
}

// Catches: a channel count that is forwarded in the header but not applied to the samples --
// a frame that claims mono and carries interleaved stereo is twice as long as it says it is,
// and every consumer downstream reads the wrong half of it.
#[tokio::test(start_paused = true)]
async fn a_downmix_averages_the_channels_and_shortens_the_frame() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");
    let mut stream = publication.audio_stream(AudioStreamOptions {
        num_channels: Some(1),
        ..AudioStreamOptions::default()
    });

    // Channel 0 is `i * 10`, channel 1 is `i * 10 + 1000`, so mono must be `i * 10 + 500`.
    assert_eq!(publication.push_frame(ramp(24_000, 2, 4)), 1);
    let got = eventually(stream.recv()).await.expect("the stream delivered nothing");

    assert_eq!(got.num_channels, 1);
    assert_eq!(got.samples_per_channel, 4);
    assert_eq!(got.data.len(), 4 * 2, "a mono frame is half the bytes of the stereo one");
    let pcm: Vec<i16> =
        got.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    assert_eq!(pcm, vec![500, 510, 520, 530]);
}

// Catches: a zero rate or a truncated buffer taking the conversion path. Both can come off a
// host object, and §1.6 #6 says the answer is "pass it through", never a panic or a divide by
// zero in the middle of the audio path.
#[tokio::test(start_paused = true)]
async fn a_degenerate_frame_is_passed_through_rather_than_converted() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");
    let mut stream = publication.audio_stream(AudioStreamOptions {
        sample_rate: Some(24_000),
        ..AudioStreamOptions::default()
    });

    for bad in [
        AudioFrame::new(vec![0u8; 8], 0, 1, 4),
        AudioFrame::new(vec![0u8; 8], 48_000, 0, 4),
        AudioFrame::new(Vec::new(), 48_000, 1, 4),
    ] {
        assert_eq!(publication.push_frame(bad.clone()), 1);
        assert_eq!(
            eventually(stream.recv()).await,
            Some(bad),
            "a frame this fake cannot convert must arrive unchanged"
        );
    }
}

// ---- faithful about lifetimes ------------------------------------------------------------

// Catches: an audio stream that never ends. §1.6 #1's symptom exactly -- the reader waits
// forever on a track that is gone, and the turn hangs with no error anywhere.
#[tokio::test(start_paused = true)]
async fn unpublishing_ends_the_frame_stream() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");
    let mut stream = publication.audio_stream(AudioStreamOptions::default());
    let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRegistry);
    let camera = user.publish_track("TR_cam", TrackSource::Camera);
    let mut video = camera.video_stream();

    assert_eq!(publication.push_frame(frame(7)), 1);
    assert_eq!(eventually(stream.recv()).await.map(|f| f.data[0]), Some(7));
    assert_eq!(camera.push_video_frame(video_frame(&registry, 1)), 1);
    assert_eq!(eventually(video.recv()).await.map(|f| f.media.id()), Some(1));

    assert!(user.unpublish_track(&TrackSid::from("TR_mic")));
    assert_eq!(eventually(stream.recv()).await, None, "the stream must end, not hang");
    assert!(!publication.is_subscribed());
    // The video list is cleared beside the audio one and is easy to forget: a video pump that
    // never ends holds decoded images the host cannot free.
    assert!(user.unpublish_track(&TrackSid::from("TR_cam")));
    assert_eq!(eventually(video.recv()).await, None, "the video stream must end too");
}

// Catches: `close` on the room leaving subscribed tracks live. A disconnect that does not end
// the input stream leaves the STT leg reading from a room nobody is in.
#[tokio::test(start_paused = true)]
async fn closing_the_room_ends_every_stream_and_reports_disconnected() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let mut stream = user.publish_microphone("TR_mic").audio_stream(AudioStreamOptions::default());
    let mut events = room.events();

    room.close(DisconnectReason::RoomDeleted);

    assert_eq!(eventually(stream.recv()).await, None);
    assert_eq!(room.connection_state(), ConnectionState::Disconnected);
    assert!(!room.connection_state().is_connected());
    assert!(
        !room.local_participant().publications()[0].is_subscribed(),
        "the agent's own publication survived the disconnect as subscribed"
    );
    assert!(matches!(
        eventually(events.recv()).await,
        Some(RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected })
    ));
    assert!(matches!(
        eventually(events.recv()).await,
        Some(RoomEvent::Close { reason: DisconnectReason::RoomDeleted })
    ));
    assert!(room.remote_participants().is_empty());
}

// Catches: a closed room coming back. `disconnected` is the last thing an `rtc.Room` emits; a
// fake that took a `Connected` afterwards would let a harness talk into streams the close ended
// and call it a reconnect (the room verbs draw connection states after a `RoomClose`).
#[tokio::test(start_paused = true)]
async fn a_closed_room_takes_no_connection_state_and_emits_nothing_more() {
    let (_rtc, room, _s) = published().await;
    room.close(DisconnectReason::RoomDeleted);
    let mut events = room.events();
    room.set_connection_state(ConnectionState::Connected);
    room.set_connection_state(ConnectionState::Reconnecting);
    assert_eq!(room.connection_state(), ConnectionState::Disconnected);
    assert!(events.try_recv().is_err(), "a closed room emitted an event");
}

// Catches: a disconnect emitted twice for one participant. `RoomIO` resets its linked-
// participant future on each one, so a duplicate drops a participant that has just relinked.
#[tokio::test(start_paused = true)]
async fn a_participant_only_disconnects_once() {
    let (_rtc, room, _s) = published().await;
    room.add_participant("user", ParticipantKind::Sip);
    let id = ParticipantIdentity::from("user");

    assert!(room.remove_participant(&id, DisconnectReason::ClientInitiated));
    assert!(!room.remove_participant(&id, DisconnectReason::ClientInitiated));
    assert!(room.participant(&id).is_none());
}

// Catches: a departure that leaves the departed participant's tracks published, or reports
// the departure before the tracks are gone. `_input.py` learns a track is dead from
// `track_unpublished` and nothing else; a fake that skipped it would leave the input's
// "reading" pointed at a stream that ended, and a reconnect would then be judged against a
// track the reader never released.
#[tokio::test(start_paused = true)]
async fn leaving_unpublishes_every_track_before_the_departure_is_reported() {
    let (_rtc, room, _s) = published().await;
    let user = room.add_participant("user", ParticipantKind::Standard);
    let mic = user.publish_microphone("TR_mic");
    user.publish_track("TR_cam", TrackSource::Camera);
    let mut stream = mic.audio_stream(AudioStreamOptions::default());
    let mut events = room.events();

    assert!(room.remove_participant(&"user".into(), DisconnectReason::Migration));
    assert_eq!(eventually(stream.recv()).await, None, "the microphone's stream must end");
    assert!(!mic.is_subscribed());
    let mut seen = Vec::new();
    while let Ok(ev) = events.try_recv() {
        seen.push(match ev {
            RoomEvent::TrackUnpublished { participant, publication } => {
                assert_eq!(participant.identity().as_str(), "user");
                format!("unpublished {}", publication.sid().as_str())
            }
            RoomEvent::ParticipantDisconnected { participant, reason } => {
                assert_eq!(participant.identity().as_str(), "user");
                assert!(
                    participant.track_publications().is_empty(),
                    "the departed participant still lists tracks"
                );
                format!("disconnected {reason:?}")
            }
            other => panic!("unexpected {other:?}"),
        });
    }
    assert_eq!(seen, ["unpublished TR_mic", "unpublished TR_cam", "disconnected Migration"]);
}

// Catches: two participants with one identity. The server refuses that: the older session is
// dropped with `DUPLICATE_IDENTITY` before the new one is admitted, and `RoomIO` relies on
// that order (the departure clears the link, the arrival re-links). A fake that admitted both
// would let a reconnect test pass against a room no server can produce.
#[tokio::test(start_paused = true)]
async fn a_duplicate_identity_drops_the_older_participant_first() {
    let (_rtc, room, _s) = published().await;
    let first = room.add_participant("user", ParticipantKind::Standard);
    first.publish_microphone("TR_one");
    let mut events = room.events();

    let second = room.add_participant("user", ParticipantKind::Standard);
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(room.remote_participants().len(), 1, "one participant per identity");
    assert!(Arc::ptr_eq(&room.fake_participant(&"user".into()).expect("the newcomer"), &second));
    let mut seen = Vec::new();
    while let Ok(ev) = events.try_recv() {
        seen.push(match ev {
            RoomEvent::TrackUnpublished { .. } => "track_unpublished".to_owned(),
            RoomEvent::ParticipantDisconnected { reason, .. } => {
                format!("participant_disconnected {reason:?}")
            }
            RoomEvent::ParticipantConnected { participant } => {
                assert_eq!(participant.identity().as_str(), "user");
                "participant_connected".to_owned()
            }
            other => panic!("unexpected {other:?}"),
        });
    }
    assert_eq!(
        seen,
        [
            "track_unpublished",
            "participant_disconnected DuplicateIdentity",
            "participant_connected"
        ]
    );
}

// ---- faithful about subscription ---------------------------------------------------------

// Catches: `wait_for_subscription` resolving before anyone subscribed. The reference gates the
// first frame on it so the greeting is not published into a room with nobody listening; a
// double that returns immediately lets that regression through untouched.
#[tokio::test(start_paused = true)]
async fn wait_for_subscription_gates_the_first_frame() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let room = rtc.room("session");
    room.set_auto_subscribe(false);
    let (_source, publication) =
        room.local().publish_audio_track(TrackPublishOptions::default()).await.unwrap();
    assert!(!publication.is_subscribed());

    assert!(timeout(NEVER, publication.wait_for_subscription()).await.is_err());

    subscribe_local(&room.local_participant().publications()[0]);
    timeout(NEVER, publication.wait_for_subscription())
        .await
        .expect("subscription must release the wait")
        .unwrap();
    assert!(publication.is_subscribed());
}

// Catches: `set_subscribed(true)` on an already-subscribed publication emitting a second
// `track_subscribed`. `_input.py` tears down and rebuilds its stream on each one, so a
// spurious repeat drops audio mid-turn.
#[tokio::test(start_paused = true)]
async fn track_subscribed_fires_only_on_the_rising_edge() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let room = rtc.room("session");
    room.set_auto_subscribe(false);
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");
    let mut events = room.events();

    publication.set_subscribed(true).await.unwrap();
    publication.set_subscribed(true).await.unwrap();

    assert!(matches!(eventually(events.recv()).await, Some(RoomEvent::TrackSubscribed { .. })));
    assert!(events.try_recv().is_err(), "a second rising edge was invented");
}

// Catches: the local/remote split answering a question that has no meaning on its side.
// `set_subscribed` on the agent's own track and `wait_for_subscription` on someone else's are
// both nonsense; a double that returned `Ok` for them would let the confusion ship.
#[tokio::test(start_paused = true)]
async fn local_and_remote_publication_verbs_do_not_cross() {
    let (_rtc, room, _s) = published().await;
    let local = room.local_participant().track_publications();
    let local = local.first().expect("the agent's own track");
    let remote =
        room.add_participant("user", ParticipantKind::Standard).publish_microphone("TR_mic");

    assert_eq!(local.set_subscribed(true).await, Err(RtcError::Unsupported("set_subscribed")));
    assert_eq!(
        eventually(remote.wait_for_subscription()).await,
        Err(RtcError::Unsupported("wait_for_subscription"))
    );
}

// ---- faithful about events ---------------------------------------------------------------

// Catches: an event the double cannot produce. §1.5 fixes the set at eight and every one has a
// handler in `room_io/`; a missing one makes that handler untestable, which is how it rots.
#[tokio::test(start_paused = true)]
async fn all_eight_room_events_can_be_produced() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let room = rtc.room("session");
    room.set_auto_subscribe(false);
    let mut events = room.events();

    room.set_connection_state(ConnectionState::Reconnecting);
    assert_eq!(room.connection_state(), ConnectionState::Reconnecting);
    let user = room.add_participant("user", ParticipantKind::Standard);
    let mic = user.publish_microphone("TR_mic");
    mic.set_subscribed(true).await.unwrap();
    user.unpublish_track(&TrackSid::from("TR_mic"));
    room.local().publish_audio_track(TrackPublishOptions::default()).await.unwrap();
    room.remove_participant(&"user".into(), DisconnectReason::UserRejected);
    room.close(DisconnectReason::ClientInitiated);

    let mut seen = Vec::new();
    while let Ok(ev) = events.try_recv() {
        seen.push(match ev {
            RoomEvent::ConnectionStateChanged { .. } => "connection_state_changed",
            RoomEvent::ParticipantConnected { .. } => "participant_connected",
            RoomEvent::TrackPublished { .. } => "track_published",
            RoomEvent::TrackSubscribed { .. } => "track_subscribed",
            RoomEvent::TrackUnpublished { .. } => "track_unpublished",
            RoomEvent::LocalTrackPublished { .. } => "local_track_published",
            RoomEvent::ParticipantDisconnected { .. } => "participant_disconnected",
            RoomEvent::Close { .. } => "close",
        });
    }
    for expected in [
        "connection_state_changed",
        "participant_connected",
        "track_published",
        "track_subscribed",
        "track_unpublished",
        "local_track_published",
        "participant_disconnected",
        "close",
    ] {
        assert!(seen.contains(&expected), "{expected} never arrived; got {seen:?}");
    }
}

// Catches: `events()` handing out one shared receiver. The reference registers four
// independent `room.on(...)` handlers, so if a second subscriber steals from the first,
// whichever component subscribes second silently stops seeing participants.
#[tokio::test(start_paused = true)]
async fn every_subscriber_sees_every_event() {
    let (_rtc, room, _s) = published().await;
    let mut a = room.events();
    let mut b = room.events();

    room.add_participant("user", ParticipantKind::Standard);

    assert!(matches!(eventually(a.recv()).await, Some(RoomEvent::ParticipantConnected { .. })));
    assert!(matches!(eventually(b.recv()).await, Some(RoomEvent::ParticipantConnected { .. })));
}

// ---- faithful about the control plane ----------------------------------------------------

// Catches: a handler map that silently replaces on a duplicate topic. The SDK raises and both
// callers catch and warn; replacing instead would let a late registration steal chat input
// from `RoomIO` with no warning at all.
#[tokio::test(start_paused = true)]
async fn registering_a_topic_twice_is_refused_not_replaced() {
    let (_rtc, room, _s) = published().await;
    let mut first = room.text_streams("lk.chat").unwrap();

    assert_eq!(
        room.text_streams("lk.chat").unwrap_err(),
        RtcError::TopicInUse { topic: "lk.chat".into() }
    );

    let info = StreamInfo { topic: "lk.chat".into(), ..StreamInfo::default() };
    room.send_text_stream("user", info, ["hello ", "world"]).unwrap();
    let mut stream = eventually(first.recv()).await.expect("delivered to the original handler");
    assert_eq!(eventually(stream.read_all()).await, "hello world");

    room.unregister_text_stream_handler("lk.chat");
    room.unregister_text_stream_handler("lk.chat"); // idempotent: aclose runs unconditionally
    room.text_streams("lk.chat").expect("the topic is free again");
}

// Catches: a byte-stream double that drops the header. `_pre_connect_audio.py` reads
// `trackId`, `sampleRate` and `channels` out of `reader.info.attributes` and treats a missing
// key as a hard error, so a fake carrying only the payload cannot test that path at all.
#[tokio::test(start_paused = true)]
async fn byte_streams_carry_their_header() {
    let (_rtc, room, _s) = published().await;
    let topic = "lk.agent.pre-connect-audio-buffer";
    let mut handler = room.byte_streams(topic).unwrap();
    assert_eq!(room.byte_streams(topic).unwrap_err(), RtcError::TopicInUse { topic: topic.into() });

    let info = StreamInfo {
        topic: topic.into(),
        mime_type: "audio/opus".into(),
        attributes: HashMap::from([
            ("trackId".to_owned(), "TR_mic".to_owned()),
            ("sampleRate".to_owned(), "24000".to_owned()),
        ]),
        ..StreamInfo::default()
    };
    room.send_byte_stream("user", info, [vec![1u8, 2], vec![3u8]]).unwrap();

    let mut stream = eventually(handler.recv()).await.unwrap();
    assert_eq!(stream.info.attributes.get("trackId").map(String::as_str), Some("TR_mic"));
    assert_eq!(stream.info.mime_type, "audio/opus");
    assert_eq!(stream.sender, "user".into());
    let mut chunks = Vec::new();
    while let Some(c) = eventually(stream.chunks.recv()).await {
        chunks.push(c);
    }
    assert_eq!(chunks, vec![vec![1u8, 2], vec![3u8]]);

    room.unregister_byte_stream_handler(topic);
    room.byte_streams(topic).expect("the topic is free again");
}

// Catches: `stream_text` modelled as one call per delta. The reference opens **one** writer per
// transcript segment and closes it with `lk.transcription_final=true`; a double that lost the
// writer's identity would hide a shim opening a stream per token, which frontends render as
// one message per word.
#[tokio::test(start_paused = true)]
async fn one_text_stream_per_segment_carries_its_deltas_and_final_header() {
    let (_rtc, room, _s) = published().await;

    let opts = StreamTextOptions {
        topic: "lk.transcription".into(),
        sender_identity: Some("agent".into()),
        ..StreamTextOptions::default()
    };
    let writer = room.local().stream_text(opts).await.unwrap();
    writer.write("Hello").await.unwrap();
    writer.write(" there").await.unwrap();
    let final_attrs = HashMap::from([("lk.transcription_final".to_owned(), "true".to_owned())]);
    writer.close(final_attrs.clone()).await.unwrap();

    let recorded = room.local_participant().text_streams();
    assert_eq!(recorded.len(), 1, "one stream per segment, not one per delta");
    assert_eq!(recorded[0].chunks, vec!["Hello", " there"]);
    assert_eq!(recorded[0].text(), "Hello there");
    assert!(recorded[0].closed);
    assert_eq!(recorded[0].close_attributes, final_attrs);
    assert_eq!(recorded[0].opts.sender_identity, Some("agent".into()));

    // A second writer is recorded apart from the first, and an open one is not closed.
    let second = room.local().stream_text(StreamTextOptions::default()).await.unwrap();
    second.write("turn two").await.unwrap();
    let recorded = room.local_participant().text_streams();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[1].text(), "turn two");
    assert!(!recorded[1].closed, "an open stream must not report itself closed");
}

// Catches: a double that cannot fail. The reference wraps every publish in `try/except` and
// logs; with no injectable failure those `except` arms are dead code in every test.
#[tokio::test(start_paused = true)]
async fn control_calls_can_be_made_to_fail() {
    let (_rtc, room, _s) = published().await;
    room.local()
        .publish_transcription(Transcription {
            participant_identity: "user".into(),
            track_sid: "TR_mic".into(),
            segments: vec![TranscriptionSegment {
                text: "hi".into(),
                is_final: true,
                ..TranscriptionSegment::default()
            }],
        })
        .await
        .unwrap();

    room.local_participant().fail_with(RtcError::NotConnected);
    assert_eq!(room.local().set_attributes(HashMap::new()).await, Err(RtcError::NotConnected));
    assert_eq!(room.local_participant().transcriptions().len(), 1, "the good call stands");
    assert!(
        room.local_participant().attribute_calls().is_empty(),
        "the failed call recorded nothing"
    );
}

// Catches: `set_attributes` replacing the whole map instead of merging, which would drop every
// attribute the agent set earlier each time `lk.agent.state` changes.
#[tokio::test(start_paused = true)]
async fn set_attributes_merges_and_every_call_is_visible() {
    let (_rtc, room, _s) = published().await;
    let local = room.local();
    let attr = |k: &str, v: &str| HashMap::from([(k.to_owned(), v.to_owned())]);

    local.set_attributes(attr("lk.agent.state", "listening")).await.unwrap();
    local.set_attributes(attr("custom", "x")).await.unwrap();
    local.set_attributes(attr("lk.agent.state", "speaking")).await.unwrap();

    let attrs = room.local_participant().attributes();
    assert_eq!(attrs.get("lk.agent.state").map(String::as_str), Some("speaking"));
    assert_eq!(attrs.get("custom").map(String::as_str), Some("x"));
    assert_eq!(room.local_participant().attribute_calls().len(), 3);
}

// ---- the host: a second room, with no network --------------------------------------------

// Catches: `connect` handing back the room the session is already in. `warm_transfer` dials the
// human agent into a *separate* room and then moves them across; a fake that aliased the two
// would make the move a no-op and the whole workflow untestable.
#[tokio::test(start_paused = true)]
async fn connect_opens_a_second_independent_room() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let caller = rtc.room("caller");
    caller.add_participant("caller-1", ParticipantKind::Sip);

    let human = rtc.connect("wss://example.livekit.cloud", "jwt-token").await.unwrap();

    assert_ne!(human.name(), caller.name());
    assert!(human.remote_participants().is_empty(), "the second room starts empty");
    assert_eq!(caller.remote_participants().len(), 1);
    assert_eq!(
        rtc.connect_calls(),
        vec![ConnectCall {
            url: "wss://example.livekit.cloud".into(),
            token: "jwt-token".into(),
            room: human.name().to_owned(),
        }]
    );
    assert_eq!(rtc.rooms().len(), 2);
}

// Catches: an injected connect failure that sticks, or one that opens a room anyway.
// `warm_transfer` surfaces a dial failure as a `ToolError` and must stay usable afterwards.
#[tokio::test(start_paused = true)]
async fn a_failed_connect_is_reported_once_and_opens_nothing() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    rtc.fail_next_connect(RtcError::host("connection refused"));

    assert_eq!(rtc.connect("wss://x", "t").await.err(), Some(RtcError::host("connection refused")));
    assert!(rtc.rooms().is_empty(), "a failed dial must not leave a room behind");
    assert!(rtc.connect("wss://x", "t").await.is_ok(), "the failure must not stick");
    assert_eq!(rtc.connect_calls().len(), 1);
}

// Catches: `room()` minting a new room per call. `RoomIO` and the avatar path both reach for
// the session's room; two different objects put the agent's track in one and the user's audio
// in the other.
#[tokio::test(start_paused = true)]
async fn room_is_looked_up_by_name() {
    let (rtc, _) = FakeRtc::with_virtual_clock();
    let a = rtc.room("session");
    a.add_participant("user", ParticipantKind::Standard);
    let b = rtc.room("session");

    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(b.remote_participants().len(), 1);
    assert_eq!(rtc.rooms().len(), 1);
}

// Catches: `advance` silently doing nothing on a fake that shares someone else's clock. The
// playout model would never move and every `wait_for_playout` would hang with no explanation.
#[test]
#[should_panic(expected = "with_virtual_clock")]
fn advance_refuses_a_clock_it_does_not_own() {
    FakeRtc::new(Arc::new(VirtualClock::new())).advance(ms(10));
}

// ---- the input path ----------------------------------------------------------------------

// Catches: publishing a track without announcing it, or announcing it without the handle a
// consumer needs. `_input.py` reads the publication straight off the `track_subscribed` event
// and opens its stream from there; anything less and that path cannot be exercised.
#[tokio::test(start_paused = true)]
async fn a_subscribed_track_is_readable_from_the_event_it_arrived_on() {
    let (_rtc, room, _s) = published().await;
    let mut events = room.events();
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_microphone("TR_mic");

    assert!(matches!(
        eventually(events.recv()).await,
        Some(RoomEvent::ParticipantConnected { .. })
    ));
    assert!(matches!(eventually(events.recv()).await, Some(RoomEvent::TrackPublished { .. })));
    let mut stream = match eventually(events.recv()).await {
        Some(RoomEvent::TrackSubscribed { participant, publication }) => {
            assert_eq!(participant.identity().as_str(), "user");
            assert_eq!(publication.source(), TrackSource::Microphone);
            assert!(publication.is_subscribed());
            publication.audio_stream(AudioStreamOptions::default())
        }
        other => panic!("expected track_subscribed, got {other:?}"),
    };

    publication.push_frame(frame(9));
    assert_eq!(eventually(stream.recv()).await.map(|f| f.data[0]), Some(9));
}

// Catches: synthetic speech that a silence check cannot tell from silence, or two speech
// frames that are the same frame -- either lets a leg that dropped or duplicated audio pass
// the gate. Also pins `push_frames`' count: frames pushed with no reader are reported as
// undelivered rather than silently gone.
#[test]
fn synthetic_speech_is_non_silent_and_distinct_per_frame_and_undelivered_frames_count_zero() {
    let synth = FrameSynth::new(48_000, 20);
    let a = synth.frame(false);
    let b = synth.frame(false);
    let quiet = synth.frame(true);
    assert_eq!(a.samples_per_channel, 960, "20 ms at 48 kHz");
    assert_eq!(a.data.len(), 960 * 2);
    assert!(a.data.iter().any(|&x| x != 0) && a.data != b.data, "speech is distinct per frame");
    assert!(quiet.data.iter().all(|&x| x == 0));

    let (rtc, _) = FakeRtc::with_virtual_clock();
    let mic =
        rtc.room("s").add_participant("user", ParticipantKind::Standard).publish_microphone("TR");
    assert_eq!(mic.push_frames([synth.frame(false)]), 0, "nobody is reading");
    let _stream = mic.audio_stream(AudioStreamOptions::default());
    assert_eq!(mic.push_frames((0..3).map(|_| synth.frame(false))), 3);
}

// Catches: `offline` building a room on a clock of its own. The console session's timers and
// its playout model must share one timeline, so the room takes the caller's clock.
#[tokio::test(start_paused = true)]
async fn an_offline_room_is_connected_and_runs_on_the_callers_clock() {
    let clock = Arc::new(VirtualClock::new());
    let (rtc, room) = FakeRtc::offline("console", clock.clone() as Arc<dyn Clock>);
    assert_eq!(room.name(), "console");
    assert!(room.connection_state().is_connected());
    assert!(room.remote_participants().is_empty());
    let (source, _) =
        room.local().publish_audio_track(TrackPublishOptions::default()).await.unwrap();
    eventually(source.capture(&frame(1))).await.unwrap();
    clock.advance(ms(20));
    rtc.settle();
    assert_eq!(eventually(source.wait_for_playout()).await.played, ms(20));
    assert_eq!(room.local_participant().captured_bytes(), 480 * 2);
    assert_eq!(room.local_participant().played_bytes(), 480 * 2);
    assert_eq!(room.local_participant().discarded_bytes(), 0);
}

// Catches: a stream list shared across publications. Two participants talking at once is the
// ordinary case in a SIP conference, and crossed streams are invisible to any test that only
// ever has one.
#[tokio::test(start_paused = true)]
async fn frames_reach_only_their_own_track() {
    let (_rtc, room, _s) = published().await;
    let a = room.add_participant("a", ParticipantKind::Standard).publish_microphone("TR_a");
    let b = room.add_participant("b", ParticipantKind::Standard).publish_microphone("TR_b");
    let mut sa = a.audio_stream(AudioStreamOptions::default());
    let mut sb = b.audio_stream(AudioStreamOptions::default());

    a.push_frame(frame(1));
    b.push_frame(frame(2));

    assert_eq!(eventually(sa.recv()).await.map(|f| f.data[0]), Some(1));
    assert_eq!(eventually(sb.recv()).await.map(|f| f.data[0]), Some(2));
    assert!(sa.is_empty(), "b's frame leaked into a's stream");
}

/// Nothing pure can resolve a `MediaRef` (the pure tier is given `release` and never
/// `resolve`), so a test asserts on the *id* -- which is exactly what the core may see.
struct NoRegistry;

impl lk_agents_abi::RefRegistry for NoRegistry {
    fn release(&self, _id: u64) {}
}

fn video_frame(registry: &Arc<dyn lk_agents_abi::RefRegistry>, id: u64) -> VideoFrameRef {
    VideoFrameRef {
        media: lk_agents_abi::MediaRef::new(id, registry),
        width: 1280,
        height: 720,
        timestamp_us: id as i64 * 33_000,
    }
}

// Catches: a video queue sized like the audio one. `VIDEO_QUEUE_FRAMES` frames fit; the frame
// after that has to evict the *oldest*, because the newest frame is the one worth looking at
// and because 50 pinned 1080p frames is 400 MB of host memory a stalled reader would hold.
#[tokio::test(start_paused = true)]
async fn the_video_queue_is_short_and_drops_the_oldest() {
    let (_rtc, room, _s) = published().await;
    let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRegistry);
    let user = room.add_participant("user", ParticipantKind::Standard);
    let publication = user.publish_track("TR_cam", TrackSource::Camera);
    let mut stream = publication.video_stream();

    let overflow = crate::VIDEO_QUEUE_FRAMES as u64 + 1;
    for id in 1..=overflow {
        publication.push_video_frame(video_frame(&registry, id));
    }
    assert_eq!(
        eventually(stream.recv()).await.map(|f| f.media.id()),
        Some(2),
        "frame 1 was evicted to make room for the newest one"
    );
}
