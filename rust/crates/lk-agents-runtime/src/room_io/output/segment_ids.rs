//! Transcript segment identities shared by the session and runtime output.

use std::sync::atomic::{AtomicU64, Ordering};

/// Transcript segment ids: `utils.shortuuid("SG_")`, unique across sessions in one room (a
/// frontend keys its transcript rows on them), or counted for a test that compares them.
#[derive(Debug, Default)]
pub struct SegmentIds(Option<AtomicU64>);

impl SegmentIds {
    pub fn counting() -> Self {
        SegmentIds(Some(AtomicU64::new(0)))
    }

    pub fn next_segment_id(&self) -> String {
        match &self.0 {
            Some(n) => format!("SG_{}", n.fetch_add(1, Ordering::Relaxed)),
            None => lk_agents_net::strings::shortuuid("SG_"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentIds;

    // Catches: per-session counters in production. A second session in the same room (a warm
    // transfer, a restarted worker) would reuse `SG_0`, and a frontend keying its transcript rows
    // on the segment id overwrites the first session's lines.
    #[test]
    fn two_sessions_never_share_a_segment_id() {
        let (a, b) = (SegmentIds::default(), SegmentIds::default());
        let first: Vec<String> = (0..4).map(|_| a.next_segment_id()).collect();
        let second: Vec<String> = (0..4).map(|_| b.next_segment_id()).collect();
        assert!(first.iter().all(|id| id.starts_with("SG_") && id.len() == 15), "{first:?}");
        assert!(first.iter().all(|id| !second.contains(id)), "{first:?} {second:?}");
    }
}
