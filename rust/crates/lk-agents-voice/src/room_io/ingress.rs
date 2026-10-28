//! Bounds on what remote participants may stream into the session.

use lk_agents_abi::voice::rtc::RtcError;

/// Streams of one topic admitted and not yet closed.
pub const MAX_INCOMING_STREAMS: usize = 4;
pub const MAX_STREAM_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// What one incoming stream has delivered so far.
#[derive(Default)]
pub struct StreamBudget {
    bytes: usize,
}

impl StreamBudget {
    /// Refuses a chunk past either bound before its payload is retained.
    pub fn admit(&mut self, chunk: usize) -> Result<(), RtcError> {
        if chunk > MAX_STREAM_CHUNK_BYTES {
            return Err(RtcError::StreamLimit {
                resource: "chunk bytes",
                limit: MAX_STREAM_CHUNK_BYTES,
            });
        }
        if chunk > MAX_STREAM_BYTES.saturating_sub(self.bytes) {
            return Err(RtcError::StreamLimit { resource: "bytes", limit: MAX_STREAM_BYTES });
        }
        self.bytes += chunk;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_is_refused_past_its_total_bytes() {
        let mut budget = StreamBudget::default();
        for _ in 0..MAX_STREAM_BYTES / MAX_STREAM_CHUNK_BYTES {
            budget.admit(MAX_STREAM_CHUNK_BYTES).unwrap();
        }
        assert!(budget.admit(1).is_err());
        assert!(StreamBudget::default().admit(MAX_STREAM_CHUNK_BYTES + 1).is_err());
    }
}
