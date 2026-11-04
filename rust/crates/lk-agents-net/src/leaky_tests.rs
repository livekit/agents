use super::*;
use std::pin::pin;

/// Real-time guard: a lost wakeup fails instead of hanging.
const GUARD: Duration = Duration::from_secs(2);

const CATCH_UP: Policy = Policy::CatchUp { max: Duration::from_millis(100) };

/// One poll: every wait below is on a channel that already holds its answer.
fn ready<F: Future>(f: F) -> F::Output {
    match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("the channel hung instead of answering"),
    }
}

/// Polls `f` on this thread, parked between polls; a wait no wake ends fails.
fn woken<F: Future>(f: F) -> F::Output {
    struct Unpark(std::thread::Thread, std::sync::OnceLock<()>);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            let _ = self.1.set(());
            self.0.unpark();
        }
    }
    let mut f = pin!(f);
    loop {
        let unpark = Arc::new(Unpark(std::thread::current(), std::sync::OnceLock::new()));
        let waker = Waker::from(unpark.clone());
        if let Poll::Ready(v) = f.as_mut().poll(&mut Context::from_waker(&waker)) {
            return v;
        }
        std::thread::park_timeout(GUARD);
        assert!(unpark.1.get().is_some(), "the receiver slept through a push (lost wakeup)");
    }
}

/// `ms` of 16 kHz mono, every sample `tag`.
fn frame(ms: u32, tag: i16) -> AudioFrame {
    let samples = 16 * ms;
    AudioFrame::new(tag.to_le_bytes().repeat(samples as usize), 16_000, 1, samples)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Each item as `(kind, ms, first sample)`, so assertions read as the timeline.
fn timeline(items: &[Item<&'static str>]) -> Vec<(&'static str, u64, i16)> {
    items
        .iter()
        .map(|item| match item {
            Item::Audio(f) => (
                "audio",
                frame_duration(f).as_millis() as u64,
                i16::from_le_bytes([f.data[0], f.data[1]]),
            ),
            Item::Gap(d) => ("gap", d.as_millis() as u64, 0),
            Item::Control(_) => ("control", 0, 0),
        })
        .collect()
}

fn samples(items: &[Item<&'static str>]) -> Vec<i16> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Audio(f) => Some(f.data.chunks(2).map(|s| i16::from_le_bytes([s[0], s[1]]))),
            _ => None,
        })
        .flatten()
        .collect()
}

// Catches: merging (or copying) frames for a consumer that keeps up, which would delay or change
// what it hears.
#[test]
fn a_consumer_that_keeps_up_receives_the_pushed_frames_themselves() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    for tag in 0..5 {
        let pushed = frame(10, tag);
        let data = pushed.data.as_ptr();
        tx.push(pushed);
        let batch = ready(rx.recv()).unwrap();
        let [Item::Audio(got)] = batch.as_slice() else { panic!("one frame: {batch:?}") };
        assert_eq!(got.data.as_ptr(), data, "the same buffer, not a copy");
        assert_eq!(got.samples_per_channel, 160);
    }
}

// Catches: a backlog delivered one 10 ms item at a time, or merged past 50 ms, or reordered.
#[test]
fn a_backlog_comes_out_as_frames_of_at_most_50ms_in_order() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::CatchUp { max: ms(10_000) });
    for tag in 0..23 {
        tx.push(frame(10, tag));
    }
    let batch = ready(rx.recv()).unwrap();
    assert_eq!(
        timeline(&batch),
        [
            ("audio", 50, 0),
            ("audio", 50, 5),
            ("audio", 50, 10),
            ("audio", 50, 15),
            ("audio", 30, 20)
        ]
    );
    let expected: Vec<i16> = (0..23).flat_map(|tag| [tag; 160]).collect();
    assert_eq!(samples(&batch), expected, "every sample, in order");
}

// Catches: merging that waits for a frame to fill, delivering a sample later than unmerged.
#[test]
fn a_merged_batch_is_never_later_than_the_frames_unmerged() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::CatchUp { max: ms(10_000) });
    let mut receiving = pin!(rx.recv());
    let noop = &mut Context::from_waker(Waker::noop());
    assert!(receiving.as_mut().poll(noop).is_pending());
    tx.push(frame(10, 1));
    tx.push(frame(10, 2));
    let Poll::Ready(Some(batch)) = receiving.as_mut().poll(noop) else {
        panic!("pending audio is delivered at once");
    };
    assert_eq!(timeline(&batch), [("audio", 20, 1)]);
}

// Catches: a merged frame over 50 ms for any mix of frame sizes.
#[test]
fn no_merged_frame_exceeds_50ms() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::CatchUp { max: ms(10_000) });
    for size in [10, 20, 30, 40, 50, 10, 10, 25, 5, 45, 15, 20] {
        tx.push(frame(size, 0));
    }
    tx.push(frame(80, 0));
    let batch = ready(rx.recv()).unwrap();
    let durations: Vec<u64> = timeline(&batch).iter().map(|(_, ms, _)| *ms).collect();
    assert_eq!(durations.iter().sum::<u64>(), 360);
    assert!(durations[..durations.len() - 1].iter().all(|ms| *ms <= 50), "{durations:?}");
    assert_eq!(durations.last(), Some(&80), "a long frame passes through unmerged");
}

// Catches: merging across a format change, a control or a gap.
#[test]
fn merging_stops_at_a_format_change_a_control_and_a_gap() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::CatchUp { max: ms(30) });
    tx.push(frame(10, 1));
    tx.push(AudioFrame::new(vec![0; 640], 32_000, 1, 320));
    tx.push(frame(10, 2));
    tx.control("commit").unwrap();
    for tag in 3..8 {
        tx.push(frame(10, tag));
    }
    let batch = ready(rx.recv()).unwrap();
    let kinds: Vec<_> = timeline(&batch).iter().map(|(kind, ms, _)| (*kind, *ms)).collect();
    assert_eq!(kinds, [("audio", 10), ("audio", 10), ("audio", 10), ("control", 0)]);
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("gap", 20, 0), ("audio", 30, 5)]);
}

// Catches: dropping the newest audio instead of the oldest, a gap per dropped frame, or drops
// that are not counted.
#[test]
fn catch_up_drops_the_oldest_frames_into_one_gap() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    for tag in 0..15 {
        tx.push(frame(10, tag));
    }
    assert_eq!(rx.dropped(), ms(50));
    let batch = ready(rx.recv()).unwrap();
    assert_eq!(timeline(&batch), [("gap", 50, 0), ("audio", 50, 5), ("audio", 50, 10)]);
    tx.push(frame(10, 15));
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("audio", 10, 15)]);
    assert_eq!(rx.dropped(), ms(50), "monotone, and nothing new dropped");
}

// Catches: a lagging `Now` consumer handed a bounded backlog of old audio instead of the present.
#[test]
fn now_drops_to_the_newest_frame_once_the_consumer_lags() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::Now { max_lag: ms(50) });
    for tag in 0..5 {
        tx.push(frame(10, tag));
    }
    assert_eq!(rx.dropped(), Duration::ZERO, "within its lag");
    tx.push(frame(10, 5));
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("gap", 50, 0), ("audio", 10, 5)]);
}

// Catches: the pre-connect head dropped by `Now`, losing the start of the first sentence.
#[test]
fn the_head_is_kept_in_full_under_now() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::Now { max_lag: ms(20) });
    for tag in 0..10 {
        tx.head(frame(10, tag));
    }
    for tag in 10..14 {
        tx.push(frame(10, tag));
    }
    let batch = ready(rx.recv()).unwrap();
    assert_eq!(&samples(&batch)[..1600], (0..10).flat_map(|tag| [tag; 160]).collect::<Vec<_>>());
    assert_eq!(timeline(&batch)[2..], [("gap", 20, 0), ("audio", 20, 12)]);
}

// Catches: the policy dropping audio a control was ordered after (a committed buffer).
#[test]
fn audio_before_a_control_is_never_dropped_by_the_policy() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    for tag in 0..20 {
        tx.push(frame(10, tag));
    }
    tx.control("commit").unwrap();
    for tag in 20..40 {
        tx.push(frame(10, tag));
    }
    let first = ready(rx.recv()).unwrap();
    assert_eq!(timeline(&first)[0], ("gap", 100, 0), "dropped before the control was sent");
    assert_eq!(timeline(&first).last(), Some(&("control", 0, 0)));
    let second = ready(rx.recv()).unwrap();
    assert_eq!(timeline(&second)[0], ("gap", 100, 0), "the policy applies after it");
    assert_eq!(samples(&second)[0], 30);
}

// Catches: `clear()` reaching past a control, or reporting the cleared audio as a gap.
#[test]
fn clear_drops_back_to_the_last_control_without_a_gap() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    tx.push(frame(10, 1));
    tx.control("commit").unwrap();
    for tag in 2..20 {
        tx.push(frame(10, tag));
    }
    tx.clear();
    tx.control("cleared").unwrap();
    tx.push(frame(10, 30));
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("audio", 10, 1), ("control", 0, 0)]);
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("control", 0, 0)]);
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("audio", 10, 30)]);
    assert_eq!(rx.dropped(), ms(80), "only the policy's drops before the clear count");
}

// Catches: controls piling up without bound behind a consumer that stopped draining.
#[test]
fn a_consumer_that_stops_draining_controls_is_stuck() {
    let (tx, mut rx) = audio_channel::<u32>(CATCH_UP);
    for n in 0..MAX_CONTROLS as u32 {
        tx.control(n).unwrap();
    }
    assert_eq!(tx.control(99), Err(Stuck));
    assert_eq!(ready(rx.recv()).unwrap(), [Item::Control(0)]);
    assert_eq!(tx.control(64), Ok(()), "draining one makes room for one");
}

// Catches: `recv` never ending after the sender hangs up, or hang-up discarding what is queued.
#[test]
fn recv_drains_then_ends_when_the_sender_hangs_up() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    tx.push(frame(10, 1));
    tx.control("end").unwrap();
    drop(tx);
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("audio", 10, 1), ("control", 0, 0)]);
    assert_eq!(ready(rx.recv()), None);
}

// Catches: a torn-down consumer still collecting audio.
#[test]
fn a_dropped_receiver_closes_the_channel() {
    let (tx, rx) = audio_channel::<&str>(CATCH_UP);
    assert!(!tx.is_closed());
    drop(rx);
    assert!(tx.is_closed());
    tx.push(frame(10, 1));
    assert_eq!(tx.control("ignored"), Ok(()));
}

// Catches: a `select!` arm dropping the in-flight `recv` and taking audio with it.
#[test]
fn an_abandoned_recv_does_not_consume_audio() {
    let (tx, mut rx) = audio_channel::<&str>(CATCH_UP);
    {
        let waiting = pin!(rx.recv());
        assert!(waiting.poll(&mut Context::from_waker(Waker::noop())).is_pending());
        tx.push(frame(10, 7));
    }
    assert_eq!(timeline(&ready(rx.recv()).unwrap()), [("audio", 10, 7)]);
}

// Catches a lost wakeup: checking before registering. A real OS thread: the push side is
// synchronous host code.
#[test]
fn a_push_racing_the_receiver_still_wakes_it() {
    let (tx, mut rx) = audio_channel::<&str>(Policy::CatchUp { max: ms(10_000) });
    let start = Arc::new(std::sync::Barrier::new(2));
    let producer = {
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            for tag in 0..200 {
                tx.push(frame(10, tag));
            }
        })
    };
    start.wait();
    let mut heard = Vec::new();
    while let Some(batch) = woken(rx.recv()) {
        heard.extend(samples(&batch));
    }
    producer.join().unwrap();
    assert_eq!(heard, (0..200).flat_map(|tag| [tag; 160]).collect::<Vec<_>>());
}

// Catches: drops that never reach the `audio_dropped` metric, or a measurement per frame pushed.
#[test]
fn drops_are_measured_as_seconds_of_audio_for_the_observed_consumer() {
    use lk_agents_abi::telemetry::record::{Record, RecordSink};
    #[derive(Default)]
    struct Collect(Mutex<Vec<Record>>);
    impl RecordSink for Collect {
        fn record(&self, record: Record) {
            self.0.lock().push(record);
        }
    }
    let sink = Arc::new(Collect::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let tracer = telemetry::Tracer::new(sink.clone(), clock, 7);
    let (tx, _rx) = audio_channel::<&str>(CATCH_UP);
    tx.observe(tracer.root(), telemetry::AudioConsumer::RoomInput);
    for tag in 0..12 {
        tx.push(frame(10, tag));
    }
    let measured: Vec<_> = sink
        .0
        .lock()
        .iter()
        .filter_map(|r| match r {
            Record::Measurement(m) => Some((m.instrument, format!("{:?}", m.value))),
            _ => None,
        })
        .collect();
    assert_eq!(measured.len(), 2, "one measurement per push that dropped: {measured:?}");
    assert!(measured.iter().all(|(name, _)| *name == "lk.agents.audio.dropped"));
}

// Catches `mpsc::try_send`'s behaviour on video: rejecting the newest frame when full.
#[test]
fn a_full_video_channel_drops_the_oldest_frame() {
    let (tx, mut rx) = video_channel(3);
    for n in 1..=5 {
        tx.push(n);
    }
    assert_eq!(rx.dropped(), 2);
    assert_eq!([ready(rx.recv()), ready(rx.recv()), ready(rx.recv())], [Some(3), Some(4), Some(5)]);
    drop(tx);
    assert_eq!(ready(rx.recv()), None, "drained, then ended by the sender's drop");
}

// Catches: an abandoned video `recv` taking a frame, and a torn-down consumer still collecting.
#[test]
fn video_recv_is_cancel_safe_and_a_dropped_receiver_closes() {
    let (tx, mut rx) = video_channel(4);
    {
        let waiting = pin!(rx.recv());
        assert!(waiting.poll(&mut Context::from_waker(Waker::noop())).is_pending());
        tx.push(7);
    }
    assert_eq!(woken(rx.recv()), Some(7));
    assert!(!tx.is_closed());
    drop(rx);
    assert!(tx.is_closed());
    tx.push(8);
}
