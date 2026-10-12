//! [`PlayoutLedger`]: the two counters `rtc.AudioSource` does not keep, so played/discarded per
//! `wait_for_playout` window is computed once instead of reconstructed from a race. Atomics, not a
//! `Mutex`: `clear_queue` is the barge-in path and must not wait behind `capture`. Subtractions
//! saturate because `queued` is host-supplied.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::PlaybackInfo;

#[derive(Debug, Default)]
pub struct PlayoutLedger {
    pushed_ns: AtomicU64,
    discarded_ns: AtomicU64,
    reported_played_ns: AtomicU64,
    reported_discarded_ns: AtomicU64,
}

/// A host audio source keeps one (`_core_rtc.AudioSource`): it counts what `rtc.AudioSource` does
/// not, and hands back the windows the core reads.
#[lk_agents_abi::host(object(cell = "arc"))]
impl PlayoutLedger {
    #[host(new)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_pushed(&self, d: Duration) {
        self.pushed_ns.fetch_add(nanos(d), Ordering::AcqRel);
    }

    /// [`add_pushed`](Self::add_pushed) of a frame's exact length ([`frame_duration`](crate::frame_duration)'s
    /// integer nanoseconds), as a host that holds only the frame's counts reports it.
    #[host(method)]
    pub fn add_pushed_samples(&self, samples_per_channel: u32, sample_rate: u32) {
        if sample_rate > 0 {
            self.add_pushed(Duration::from_nanos(
                u64::from(samples_per_channel).saturating_mul(1_000_000_000) / u64::from(sample_rate),
            ));
        }
    }

    /// Queued audio was thrown away (`clear_queue`, or a close that drops the queue).
    #[host(method)]
    pub fn add_discarded(&self, d: Duration) {
        self.discarded_ns.fetch_add(nanos(d), Ordering::AcqRel);
    }

    /// The window the next [`settle`](Self::settle) will report, read now and left open.
    ///
    /// `queued` is what the source still holds behind the speaker: it has neither played nor been
    /// discarded, so it is in neither figure. Once the queue drains (or is cleared) the two add up
    /// to what was pushed, exactly as `settle` will then report.
    #[host(method)]
    pub fn snapshot(&self, queued: Duration) -> PlaybackInfo {
        let pushed = self.pushed_ns.load(Ordering::Acquire);
        let discarded = self.discarded_ns.load(Ordering::Acquire);
        let reported_played = self.reported_played_ns.load(Ordering::Acquire);
        let reported_discarded = self.reported_discarded_ns.load(Ordering::Acquire);
        PlaybackInfo {
            played: Duration::from_nanos(
                pushed
                    .saturating_sub(discarded)
                    .saturating_sub(nanos(queued))
                    .saturating_sub(reported_played),
            ),
            discarded: Duration::from_nanos(discarded.saturating_sub(reported_discarded)),
        }
    }

    /// Close the window: everything pushed and not discarded has now played. Returns the window
    /// and starts the next one where this one ended.
    ///
    /// The caller has already waited for the queue to drain (or cleared it), which is why nothing
    /// queued is subtracted here.
    #[host(method)]
    pub fn settle(&self) -> PlaybackInfo {
        let pushed = self.pushed_ns.load(Ordering::Acquire);
        let discarded = self.discarded_ns.load(Ordering::Acquire);
        let prev_played =
            self.reported_played_ns.swap(pushed.saturating_sub(discarded), Ordering::AcqRel);
        let prev_discarded = self.reported_discarded_ns.swap(discarded, Ordering::AcqRel);
        PlaybackInfo {
            played: Duration::from_nanos(
                pushed.saturating_sub(discarded).saturating_sub(prev_played),
            ),
            discarded: Duration::from_nanos(discarded.saturating_sub(prev_discarded)),
        }
    }
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    // Catches: a `settle` that reports cumulative totals rather than the window since the last
    // one. `_ParticipantAudioOutput` waits once per flush and reads per-utterance figures off the
    // result; a cumulative answer would report every earlier utterance's audio as this one's.
    #[test]
    fn each_settle_reports_only_the_window_since_the_previous_one() {
        let ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(1000));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(1000), discarded: ms(0) });
        ledger.add_pushed(ms(500));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(500), discarded: ms(0) });
        assert_eq!(ledger.settle(), PlaybackInfo::default(), "an empty window is empty");
    }

    // Catches: discarded audio counted as played, or a window's discard leaking into the next.
    // `discarded` is the barge-in signal `on_playback_finished(interrupted=True)` derives from, so
    // both errors change what the user is told about an interruption.
    #[test]
    fn a_clear_moves_the_unplayed_part_from_played_to_discarded() {
        let ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(1000));
        ledger.add_discarded(ms(400));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(600), discarded: ms(400) });
        ledger.add_pushed(ms(300));
        assert_eq!(
            ledger.settle(),
            PlaybackInfo { played: ms(300), discarded: ms(0) },
            "the earlier discard is not reported twice"
        );
    }

    // Catches: `snapshot` consuming the window, or forgetting to subtract what is still queued.
    // The output pump reads it mid-segment at a pause; audio behind the speaker has not played.
    #[test]
    fn a_snapshot_subtracts_the_queue_and_leaves_the_window_open() {
        let ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(1000));
        assert_eq!(ledger.snapshot(ms(300)), PlaybackInfo { played: ms(700), discarded: ms(0) });
        ledger.add_discarded(ms(300));
        assert_eq!(ledger.snapshot(ms(0)), PlaybackInfo { played: ms(700), discarded: ms(300) });
        assert_eq!(
            ledger.settle(),
            PlaybackInfo { played: ms(700), discarded: ms(300) },
            "the snapshot did not close the window"
        );
        assert_eq!(ledger.snapshot(ms(0)), PlaybackInfo::default());
    }

    // Catches: an underflow panic on a host-supplied figure. The SDK's `queued_duration` is read
    // off a host object, and a source that reports more queued than it was handed must read as
    // "nothing played", not take the audio path down.
    #[test]
    fn host_figures_larger_than_what_was_pushed_saturate_at_zero() {
        let ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(100));
        assert_eq!(ledger.snapshot(ms(500)).played, ms(0));
        ledger.add_discarded(ms(500));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(0), discarded: ms(500) });
        assert_eq!(ledger.settle(), PlaybackInfo::default());
    }
}
