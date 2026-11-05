use super::*;
use crate::media_queue::MAX_EVENTS;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

const CATCH_UP: Policy = Policy::CatchUp(Duration::from_millis(100));

/// `ms` of 16 kHz mono, every sample `tag`.
fn frame(ms: u32, tag: i16) -> AudioFrame {
    let samples = 16 * ms;
    AudioFrame::new(tag.to_le_bytes().repeat(samples as usize), 16_000, 1, samples)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// What is queued now, without waiting.
fn now<F: Future>(f: F) -> Option<F::Output> {
    match pin!(f).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

fn next<E>(rx: &mut AudioReceiver<E>) -> Option<Item<AudioFrame, E>> {
    now(rx.recv()).flatten()
}

fn drain<E>(rx: &mut AudioReceiver<E>) -> Vec<Item<AudioFrame, E>> {
    std::iter::from_fn(|| next(rx)).collect()
}

/// Each item as `(kind, ms, first sample)`, so assertions read as the timeline.
fn timeline(items: &[Item<AudioFrame, &'static str>]) -> Vec<(&'static str, u64, i16)> {
    items
        .iter()
        .map(|item| match item {
            Item::Media(f) => (
                "audio",
                frame_duration(f).as_millis() as u64,
                i16::from_le_bytes([f.data[0], f.data[1]]),
            ),
            Item::Gap(d) => ("gap", d.as_millis() as u64, 0),
            Item::Event(_) => ("event", 0, 0),
        })
        .collect()
}

fn samples(items: &[Item<AudioFrame, &'static str>]) -> Vec<i16> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Media(f) => Some(f.data.chunks(2).map(|s| i16::from_le_bytes([s[0], s[1]]))),
            _ => None,
        })
        .flatten()
        .collect()
}

// Catches: merging (or copying) frames for a consumer that keeps up, which would delay or change
// what it hears.
#[test]
fn a_consumer_that_keeps_up_receives_the_pushed_frames_themselves() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    for tag in 0..5 {
        let pushed = frame(10, tag);
        let data = pushed.data.as_ptr();
        tx.push(pushed);
        let Some(Item::Media(got)) = next(&mut rx) else { panic!("one frame") };
        assert_eq!(got.data.as_ptr(), data, "the same buffer, not a copy");
        assert_eq!(got.samples_per_channel, 160);
        assert_eq!(next(&mut rx), None);
    }
}

// Catches: a backlog delivered one 10 ms item at a time, or merged past 50 ms, or reordered.
#[test]
fn a_backlog_comes_out_as_frames_of_at_most_50ms_in_order() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(ms(10_000)));
    for tag in 0..23 {
        tx.push(frame(10, tag));
    }
    let items = drain(&mut rx);
    assert_eq!(
        timeline(&items),
        [
            ("audio", 50, 0),
            ("audio", 50, 5),
            ("audio", 50, 10),
            ("audio", 50, 15),
            ("audio", 30, 20)
        ]
    );
    let expected: Vec<i16> = (0..23).flat_map(|tag| [tag; 160]).collect();
    assert_eq!(samples(&items), expected, "every sample, in order");
}

// Catches: merging that waits for a frame to fill, delivering a sample later than unmerged.
#[test]
fn a_merged_frame_is_never_later_than_the_frames_unmerged() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(ms(10_000)));
    tx.push(frame(10, 1));
    tx.push(frame(10, 2));
    assert_eq!(timeline(&drain(&mut rx)), [("audio", 20, 1)], "pending audio, at once");
}

// Catches: a merged frame over 50 ms for any mix of frame sizes.
#[test]
fn no_merged_frame_exceeds_50ms() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(ms(10_000)));
    for size in [10, 20, 30, 40, 50, 10, 10, 25, 5, 45, 15, 20] {
        tx.push(frame(size, 0));
    }
    tx.push(frame(80, 0));
    let durations: Vec<u64> = timeline(&drain(&mut rx)).iter().map(|(_, ms, _)| *ms).collect();
    assert_eq!(durations.iter().sum::<u64>(), 360);
    assert!(durations[..durations.len() - 1].iter().all(|ms| *ms <= 50), "{durations:?}");
    assert_eq!(durations.last(), Some(&80), "a long frame passes through unmerged");
}

// Catches: merging across a format change, an event or a gap.
#[test]
fn merging_stops_at_a_format_change_an_event_and_a_gap() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(ms(30)));
    tx.push(frame(10, 1));
    tx.push(AudioFrame::new(vec![0; 640], 32_000, 1, 320));
    tx.push(frame(10, 2));
    tx.event("commit").unwrap();
    for tag in 3..8 {
        tx.push(frame(10, tag));
    }
    let kinds: Vec<_> =
        timeline(&drain(&mut rx)).iter().map(|(kind, ms, _)| (*kind, *ms)).collect();
    assert_eq!(
        kinds,
        [("audio", 10), ("audio", 10), ("audio", 10), ("event", 0), ("gap", 20), ("audio", 30)]
    );
}

// Catches: dropping the newest audio instead of the oldest, a gap per dropped frame, or drops
// that are not counted.
#[test]
fn catch_up_drops_the_oldest_frames_into_one_gap() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    for tag in 0..15 {
        assert_eq!(tx.push(frame(10, tag)), if tag > 9 { ms(10) } else { Duration::ZERO });
    }
    assert_eq!(rx.dropped(), 5);
    assert_eq!(timeline(&drain(&mut rx)), [("gap", 50, 0), ("audio", 50, 5), ("audio", 50, 10)]);
    tx.push(frame(10, 15));
    assert_eq!(timeline(&drain(&mut rx)), [("audio", 10, 15)]);
    assert_eq!(rx.dropped(), 5, "monotone, and nothing new dropped");
}

// Catches: the bound measuring audio the consumer already took, so a consumer that keeps up
// still loses audio.
#[test]
fn received_audio_no_longer_counts_against_the_bound() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    for tag in 0..100 {
        tx.push(frame(10, tag));
        next(&mut rx);
    }
    assert_eq!(rx.dropped(), 0);
}

// Catches: a lagging `Now` consumer handed a bounded backlog of old audio instead of the present.
#[test]
fn now_drops_to_the_newest_frame_once_the_consumer_lags() {
    let (tx, mut rx) = audio::<&str>(Policy::Now(ms(50)));
    for tag in 0..5 {
        tx.push(frame(10, tag));
    }
    assert_eq!(rx.dropped(), 0, "within its lag");
    tx.push(frame(10, 5));
    assert_eq!(timeline(&drain(&mut rx)), [("gap", 50, 0), ("audio", 10, 5)]);
}

// Catches: the pre-connect head dropped by `Now`, losing the start of the first sentence.
#[test]
fn the_head_is_kept_in_full_under_now() {
    let (tx, mut rx) = audio::<&str>(Policy::Now(ms(20)));
    for tag in 0..10 {
        tx.head(frame(10, tag));
    }
    for tag in 10..14 {
        tx.push(frame(10, tag));
    }
    let items = drain(&mut rx);
    assert_eq!(&samples(&items)[..1600], (0..10).flat_map(|tag| [tag; 160]).collect::<Vec<_>>());
    assert_eq!(timeline(&items)[2..], [("gap", 20, 0), ("audio", 20, 12)]);
}

// Catches: a consumer that cannot tell the head from live audio, so it cannot measure its
// catch-up, or a merged frame that straddles the end of the head.
#[test]
fn head_progress_counts_the_head_received_and_never_merges_it_with_live_audio() {
    let (tx, mut rx) = audio::<&str>(Policy::Now(ms(500)));
    for tag in 0..8 {
        tx.head(frame(10, tag));
    }
    tx.push(frame(10, 8));
    assert_eq!(rx.head(), Head { received: ms(0), pending: true });
    assert_eq!(timeline(&[next(&mut rx).unwrap()]), [("audio", 50, 0)]);
    assert_eq!(rx.head(), Head { received: ms(50), pending: true });
    assert_eq!(timeline(&[next(&mut rx).unwrap()]), [("audio", 30, 5)], "the head's rest alone");
    assert_eq!(rx.head(), Head { received: ms(80), pending: false });
    assert_eq!(timeline(&[next(&mut rx).unwrap()]), [("audio", 10, 8)]);
    assert_eq!(rx.head().received, ms(80), "live audio is not head");
    tx.head(frame(10, 9));
    tx.flush("source changed").unwrap();
    assert_eq!(rx.head(), Head { received: ms(80), pending: false }, "flushed with the rest");
}

// Catches: the bound dropping audio an event was ordered after (a committed buffer).
#[test]
fn audio_before_an_event_is_never_dropped_by_the_bound() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    for tag in 0..20 {
        tx.push(frame(10, tag));
    }
    tx.event("commit").unwrap();
    for tag in 20..40 {
        tx.push(frame(10, tag));
    }
    let items = timeline(&drain(&mut rx));
    assert_eq!(items[0], ("gap", 100, 0), "dropped before the event was sent");
    let event = items.iter().position(|item| item.0 == "event").unwrap();
    assert_eq!(items[1..event].iter().map(|item| item.1).sum::<u64>(), 100);
    assert_eq!(items[event + 1], ("gap", 100, 0), "the bound applies after it");
    assert_eq!(items[event + 2].2, 30);
}

// Catches: `flush` keeping media, dropping an event, or reporting the flushed audio as a gap.
#[test]
fn flush_drops_the_queued_media_and_keeps_the_events() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    tx.push(frame(10, 1));
    tx.event("commit").unwrap();
    for tag in 2..20 {
        tx.push(frame(10, tag));
    }
    tx.flush("cleared").unwrap();
    tx.push(frame(10, 30));
    assert_eq!(timeline(&drain(&mut rx)), [("event", 0, 0), ("event", 0, 0), ("audio", 10, 30)]);
    assert_eq!(rx.dropped(), 8, "only the bound's drops before the flush count");
}

// Catches: events piling up without bound behind a consumer that stopped draining, or a stuck
// flush keeping the media it was asked to drop.
#[test]
fn a_consumer_that_stops_draining_events_is_stuck() {
    let (tx, mut rx) = audio::<u32>(CATCH_UP);
    for n in 0..MAX_EVENTS as u32 {
        tx.event(n).unwrap();
    }
    assert_eq!(tx.event(99), Err(Stuck));
    tx.push(frame(10, 1));
    assert_eq!(tx.flush(99), Err(Stuck));
    assert_eq!(next(&mut rx), Some(Item::Event(0)));
    assert_eq!(tx.event(64), Ok(()), "draining one makes room for one");
    assert_eq!(drain(&mut rx).len(), MAX_EVENTS, "the flushed frame is gone");
}

// ------------------------------------------------------------------- waking and closing

/// Real-time guard: a lost wakeup fails instead of hanging.
const GUARD: Duration = Duration::from_secs(2);

/// One poll: every wait below is on a queue that already holds its answer.
fn ready<F: Future>(f: F) -> F::Output {
    now(f).expect("the queue hung instead of answering")
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

fn first_sample(item: &Item<AudioFrame, &str>) -> i16 {
    let Item::Media(frame) = item else { panic!("audio expected: {item:?}") };
    i16::from_le_bytes([frame.data[0], frame.data[1]])
}

// Catches: the handle copying a frame for a consumer that keeps up, or merging outside the core's
// rules.
#[test]
fn recv_hands_over_the_pushed_frame_and_merges_a_backlog() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(Duration::from_secs(10)));
    let pushed = frame(10, 1);
    let data = pushed.data.as_ptr();
    tx.push(pushed);
    let Some(Item::Media(got)) = ready(rx.recv()) else { panic!("one frame") };
    assert_eq!(got.data.as_ptr(), data, "the same buffer, not a copy");
    for tag in 2..9 {
        tx.push(frame(10, tag));
    }
    let Some(Item::Media(merged)) = ready(rx.recv()) else { panic!("a merged frame") };
    assert_eq!(merged.samples_per_channel, 800);
    assert_eq!(first_sample(&ready(rx.recv()).unwrap()), 7);
}

// Catches: `recv` never ending after the sender hangs up, or hang-up discarding what is queued.
#[test]
fn recv_drains_then_ends_when_the_sender_hangs_up() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    tx.push(frame(10, 1));
    tx.event("end").unwrap();
    drop(tx);
    assert_eq!(first_sample(&ready(rx.recv()).unwrap()), 1);
    assert_eq!(ready(rx.recv()), Some(Item::Event("end")));
    assert_eq!(ready(rx.recv()), None);
}

// Catches: a torn-down consumer still collecting media, or a closed queue failing its producer.
#[test]
fn a_dropped_receiver_closes_the_queue() {
    let (tx, rx) = audio::<&str>(CATCH_UP);
    assert!(!tx.is_closed());
    drop(rx);
    assert!(tx.is_closed());
    tx.push(frame(10, 1));
    tx.head(frame(10, 1));
    assert_eq!(tx.event("ignored"), Ok(()));
    assert_eq!(tx.flush("ignored"), Ok(()));
}

// Catches: a `select!` arm dropping the in-flight `recv` and taking media with it.
#[test]
fn an_abandoned_recv_does_not_consume_media() {
    let (tx, mut rx) = audio::<&str>(CATCH_UP);
    {
        let waiting = pin!(rx.recv());
        assert!(waiting.poll(&mut Context::from_waker(Waker::noop())).is_pending());
        tx.push(frame(10, 7));
    }
    assert_eq!(first_sample(&ready(rx.recv()).unwrap()), 7);
}

// Catches a lost wakeup: checking before registering. A real OS thread: the push side is
// synchronous host code.
#[test]
fn a_push_racing_the_receiver_still_wakes_it() {
    let (tx, mut rx) = audio::<&str>(Policy::CatchUp(Duration::from_secs(10)));
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
    while let Some(item) = woken(rx.recv()) {
        let Item::Media(frame) = item else { panic!("only audio was pushed") };
        heard.extend(frame.data.chunks(2).map(|s| i16::from_le_bytes([s[0], s[1]])));
    }
    producer.join().unwrap();
    assert_eq!(heard, (0..200).flat_map(|tag| [tag; 160]).collect::<Vec<_>>());
}
