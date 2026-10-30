//! `_core.PlaybackTracker`: voice's [`PlaybackSegments`] deciding, this executing (the lock,
//! the waits and the fan-out to observers).

use lk_agents_abi::voice::io::{
    Playback, PlaybackEvent, PlaybackFinished, PlaybackObserver, PlaybackReports, PlaybackSegment,
    PlaybackTracker,
};
use lk_agents_abi::{BoxFuture, TimedString, Ts, WallAnchor};
use lk_agents_voice::playback::PlaybackSegments;
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::watch;

/// The tracker of one output node, also the observer a child registers with, so a terminal's
/// event counts at every level.
pub struct Tracker {
    /// Notifies waits only when a finish or reset may settle them.
    segments: watch::Sender<PlaybackSegments>,
    observers: Mutex<Vec<Arc<dyn PlaybackObserver>>>,
    /// The session's clock as wall seconds, for a host sink's `time.time()` stamps. Set once.
    anchor: OnceLock<WallAnchor>,
}

impl Tracker {
    pub fn new() -> Self {
        Tracker {
            segments: watch::Sender::new(PlaybackSegments::default()),
            observers: Mutex::new(Vec::new()),
            anchor: OnceLock::new(),
        }
    }

    /// A change no wait observes.
    fn update<T>(&self, change: impl FnOnce(&mut PlaybackSegments) -> T) -> T {
        let mut out = None;
        self.segments.send_if_modified(|s| {
            out = Some(change(s));
            false
        });
        out.expect("the change ran")
    }

    /// Cloned out: an observer is often a parent tracker that re-enters.
    fn emit(&self, notify: impl Fn(&dyn PlaybackObserver)) {
        let observers = self.observers.lock().clone();
        for o in observers {
            notify(&*o);
        }
        // Whoever stopped listening is forgotten here, not at `aclose`.
        self.observers.lock().retain(|o| !o.is_closed());
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Tracker::new()
    }
}

impl PlaybackTracker for Tracker {
    fn anchor_to(&self, anchor: WallAnchor) {
        let _ = self.anchor.set(anchor);
    }

    fn anchor(&self) -> WallAnchor {
        *self
            .anchor
            .get_or_init(|| WallAnchor { wall: lk_agents_net::clock::unix_now(), at: Ts::ZERO })
    }

    fn begin_segment(&self, segment: PlaybackSegment) -> bool {
        self.update(|s| s.begin(segment))
    }

    fn has_pending_segment(&self, segment: PlaybackSegment) -> bool {
        self.segments.borrow().is_pending(segment)
    }

    fn accepts_report(&self, segment: PlaybackSegment) -> bool {
        self.segments.borrow().accepts(segment)
    }

    fn current_segment(&self) -> Option<PlaybackSegment> {
        self.segments.borrow().current()
    }

    fn note_captured(&self, duration: Duration) -> PlaybackSegment {
        self.update(|s| s.captured(duration))
    }

    fn take_uncleared(&self) -> Duration {
        self.update(PlaybackSegments::take_uncleared)
    }

    fn note_flush(&self) {
        self.update(PlaybackSegments::flush)
    }

    fn reset(&self) {
        self.segments.send_modify(PlaybackSegments::reset);
        self.on_playback_reset();
    }

    fn segments(&self) -> u64 {
        self.segments.borrow().segments()
    }

    fn position(&self) -> Duration {
        self.segments.borrow().position()
    }

    fn playback_started(&self, segment: PlaybackSegment, created_at: Ts) -> bool {
        if !self.accepts_report(segment) {
            return false;
        }
        let ev = PlaybackEvent::Started(lk_agents_abi::voice::io::PlaybackStarted {
            segment,
            created_at,
        });
        self.emit(|o| o.on_playback_event(&ev));
        true
    }

    fn playback_progressed(
        &self,
        segment: PlaybackSegment,
        started_at: Ts,
        offset: Duration,
        duration: Duration,
    ) -> bool {
        if !self.update(|s| s.progressed(segment, duration)) {
            return false;
        }
        let ev = PlaybackEvent::Progressed(lk_agents_abi::voice::io::PlaybackProgressed {
            segment,
            started_at,
            offset,
            duration,
        });
        self.emit(|o| o.on_playback_event(&ev));
        true
    }

    fn playback_finished(&self, segment: PlaybackSegment, ev: PlaybackFinished) -> bool {
        if !self.segments.send_if_modified(|s| s.finished(segment, ev.clone())) {
            return false;
        }
        self.emit(|o| o.on_segment_finished(segment, &ev));
        true
    }

    fn transcript_shown(&self, segment: Option<PlaybackSegment>, word: &TimedString) {
        let observers = self.observers.lock().clone();
        for o in observers {
            o.on_transcript_shown(segment, word);
        }
    }

    fn observe(&self, observer: Arc<dyn PlaybackObserver>) {
        self.observers.lock().push(observer);
    }

    fn unobserve(&self, observer: &Arc<dyn PlaybackObserver>) {
        self.observers.lock().retain(|o| !Arc::ptr_eq(o, observer));
    }

    fn wait_for_playout(&self) -> BoxFuture<'_, PlaybackFinished> {
        Box::pin(async move {
            let mut settled = self.segments.subscribe();
            let wait = settled.borrow().wait();
            let s = settled
                .wait_for(|s| s.settled(wait).is_some())
                .await
                .expect("the tracker owns its sender for the whole wait");
            s.settled(wait).expect("checked").clone()
        })
    }
}

impl Playback for Tracker {
    fn new() -> Self {
        Tracker::new()
    }
}

impl PlaybackReports for Tracker {}

impl PlaybackObserver for Tracker {
    /// Forward a child's event as this node's own (`_forward_next_playback_finished`).
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                self.playback_started(e.segment, e.created_at);
            }
            PlaybackEvent::Progressed(e) => {
                self.playback_progressed(e.segment, e.started_at, e.offset, e.duration);
            }
            PlaybackEvent::Finished { segment, event } => {
                self.playback_finished(*segment, event.clone());
            }
        }
    }

    fn on_transcript_shown(&self, segment: Option<PlaybackSegment>, word: &TimedString) {
        self.transcript_shown(segment, word);
    }

    /// A reset below is told above, not repeated: each node's count is its own.
    fn on_playback_reset(&self) {
        self.update(PlaybackSegments::reset_below);
        let observers = self.observers.lock().clone();
        for o in observers {
            o.on_playback_reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Poll, Wake, Waker};

    struct Woken(OnceLock<()>);

    impl Wake for Woken {
        fn wake(self: Arc<Self>) {
            let _ = self.0.set(());
        }
    }

    #[test]
    fn cancellation_releases_a_playback_waiter_and_finish_releases_every_remaining_waiter() {
        let tracker = Tracker::new();
        let segment = tracker.note_captured(Duration::ZERO);
        let task = Arc::new(Woken(OnceLock::new()));
        let waker = Waker::from(task.clone());
        let mut cancelled = tracker.wait_for_playout();
        assert!(cancelled.as_mut().poll(&mut Context::from_waker(&waker)).is_pending());
        drop(cancelled);
        drop(waker);
        assert_eq!(Arc::strong_count(&task), 1, "a cancelled wait must release its task");

        let mut first = tracker.wait_for_playout();
        let mut second = tracker.wait_for_playout();
        let first_woken = Arc::new(Woken(OnceLock::new()));
        let second_woken = Arc::new(Woken(OnceLock::new()));
        let first_waker = Waker::from(first_woken.clone());
        let second_waker = Waker::from(second_woken.clone());
        assert!(first.as_mut().poll(&mut Context::from_waker(&first_waker)).is_pending());
        assert!(second.as_mut().poll(&mut Context::from_waker(&second_waker)).is_pending());
        let finished =
            PlaybackFinished { playback_position: Duration::from_millis(20), ..Default::default() };
        assert!(tracker.playback_finished(segment, finished.clone()));
        assert!(first_woken.0.get().is_some());
        assert!(second_woken.0.get().is_some());
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(first.as_mut().poll(&mut cx), Poll::Ready(finished.clone()));
        assert_eq!(second.as_mut().poll(&mut cx), Poll::Ready(finished));
    }

    #[test]
    fn reset_releases_a_parked_waiter() {
        let tracker = Tracker::new();
        tracker.note_captured(Duration::ZERO);
        let mut waiting = tracker.wait_for_playout();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(waiting.as_mut().poll(&mut cx).is_pending());
        PlaybackTracker::reset(&tracker);
        let Poll::Ready(finished) = waiting.as_mut().poll(&mut cx) else {
            panic!("reset must release a parked waiter");
        };
        assert!(finished.interrupted);
    }
}
