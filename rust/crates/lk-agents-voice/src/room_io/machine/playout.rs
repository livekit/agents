//! Playout window accounting shared by SDK source adapters and simulated sources.

use lk_agents_abi::voice::rtc::PlaybackInfo;
use std::time::Duration;

#[derive(Debug, Default)]
pub struct PlayoutLedger {
    pushed_ns: u64,
    discarded_ns: u64,
    reported_played_ns: u64,
    reported_discarded_ns: u64,
}

impl PlayoutLedger {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_pushed(&mut self, duration: Duration) {
        self.pushed_ns = self.pushed_ns.wrapping_add(nanos(duration));
    }
    pub fn add_pushed_samples(&mut self, samples_per_channel: u32, sample_rate: u32) {
        if sample_rate > 0 {
            self.add_pushed(Duration::from_nanos(
                u64::from(samples_per_channel).saturating_mul(1_000_000_000)
                    / u64::from(sample_rate),
            ));
        }
    }
    pub fn add_discarded(&mut self, duration: Duration) {
        self.discarded_ns = self.discarded_ns.wrapping_add(nanos(duration));
    }
    /// Read the current window without consuming it; source-queued samples remain unheard.
    pub fn snapshot(&self, queued: Duration) -> PlaybackInfo {
        PlaybackInfo {
            played: Duration::from_nanos(
                self.pushed_ns
                    .saturating_sub(self.discarded_ns)
                    .saturating_sub(nanos(queued))
                    .saturating_sub(self.reported_played_ns),
            ),
            discarded: Duration::from_nanos(
                self.discarded_ns.saturating_sub(self.reported_discarded_ns),
            ),
        }
    }
    /// Close exactly one source window after the queue has drained or been cleared.
    pub fn settle(&mut self) -> PlaybackInfo {
        let played = self.pushed_ns.saturating_sub(self.discarded_ns);
        let prev_played = std::mem::replace(&mut self.reported_played_ns, played);
        let prev_discarded = std::mem::replace(&mut self.reported_discarded_ns, self.discarded_ns);
        PlaybackInfo {
            played: Duration::from_nanos(played.saturating_sub(prev_played)),
            discarded: Duration::from_nanos(self.discarded_ns.saturating_sub(prev_discarded)),
        }
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
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
        let mut ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(1000));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(1000), discarded: ms(0) });
        ledger.add_pushed(ms(500));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(500), discarded: ms(0) });
        assert_eq!(ledger.settle(), PlaybackInfo::default(), "an empty window is empty");
    }

    // Catches: discarded audio counted as played, or a window's discard leaking into the next.
    // `discarded` is the interruption signal `on_playback_finished(interrupted=True)` derives from, so
    // both errors change what the user is told about an interruption.
    #[test]
    fn a_clear_moves_the_unplayed_part_from_played_to_discarded() {
        let mut ledger = PlayoutLedger::new();
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
        let mut ledger = PlayoutLedger::new();
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
        let mut ledger = PlayoutLedger::new();
        ledger.add_pushed(ms(100));
        assert_eq!(ledger.snapshot(ms(500)).played, ms(0));
        ledger.add_discarded(ms(500));
        assert_eq!(ledger.settle(), PlaybackInfo { played: ms(0), discarded: ms(500) });
        assert_eq!(ledger.settle(), PlaybackInfo::default());
    }
}
