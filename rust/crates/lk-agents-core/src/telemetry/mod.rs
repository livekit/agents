//! Telemetry on supplied data: GenAI provider names ([`gen_ai`]), destination content policy and
//! retention charges ([`policy`]), the session report's privacy projection, records and tags
//! ([`report`]), the spans sans-I/O machines own ([`span`]) and the W3C parents transports carry
//! ([`transport`]).

pub use lk_agents_abi::telemetry::spans::common::log_keys;

pub mod gen_ai;
pub mod policy;
pub mod report;
pub mod span;

/// A span context as the W3C `traceparent` the transports carry.
pub mod transport {
    /// A validated W3C version-00 parent.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TraceParent {
        pub trace_id: [u8; 16],
        pub span_id: [u8; 8],
        pub sampled: bool,
    }

    impl TraceParent {
        pub fn format(trace_id: u128, span_id: u64, sampled: bool) -> Option<String> {
            if trace_id == 0 || span_id == 0 {
                return None;
            }
            Some(format!("00-{trace_id:032x}-{span_id:016x}-{:02x}", u8::from(sampled)))
        }

        pub fn parse(value: &str) -> Option<Self> {
            let bytes = value.as_bytes();
            if bytes.len() != 55 || &bytes[..3] != b"00-" || bytes[35] != b'-' || bytes[52] != b'-'
            {
                return None;
            }
            fn decode<const N: usize>(text: &[u8]) -> Option<[u8; N]> {
                let mut out = [0; N];
                fn hex(b: u8) -> Option<u8> {
                    match b {
                        b'0'..=b'9' => Some(b - b'0'),
                        b'a'..=b'f' => Some(b - b'a' + 10),
                        _ => None,
                    }
                }
                for (value, pair) in out.iter_mut().zip(text.as_chunks::<2>().0) {
                    *value = hex(pair[0])? * 16 + hex(pair[1])?;
                }
                Some(out)
            }
            let trace_id = decode(&bytes[3..35])?;
            let span_id = decode(&bytes[36..52])?;
            let flags: [u8; 1] = decode(&bytes[53..])?;
            if trace_id == [0; 16] || span_id == [0; 8] {
                return None;
            }
            Some(Self { trace_id, span_id, sampled: flags[0] & 1 != 0 })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn trace_parent_refuses_malformed_oversized_and_zero_identifiers() {
            let valid = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
            let parent = TraceParent::parse(valid).unwrap();
            assert!(parent.sampled);
            assert_eq!(
                TraceParent::format(
                    u128::from_be_bytes(parent.trace_id),
                    u64::from_be_bytes(parent.span_id),
                    parent.sampled,
                )
                .as_deref(),
                Some(valid),
            );
            assert!(TraceParent::format(0, 1, true).is_none());
            assert!(TraceParent::format(1, 0, false).is_none());
            for invalid in [
                "",
                "00-00000000000000000000000000000000-0123456789abcdef-01",
                "00-0123456789abcdef0123456789abcdef-0000000000000000-01",
                "00-0123456789ABCDEF0123456789abcdef-0123456789abcdef-01",
            ] {
                assert!(TraceParent::parse(invalid).is_none());
            }
            assert!(TraceParent::parse(&format!("{valid}extra")).is_none());
        }
    }
}
