//! A host SDK's span context as the W3C `traceparent` the transports carry.

use lk_agents_abi::telemetry::observation::TraceParent;

pub struct TraceParents;

impl lk_agents_abi::telemetry::transport::TraceParents for TraceParents {
    fn _core_format_trace_parent(trace_id: u128, span_id: u64, sampled: bool) -> Option<String> {
        TraceParent::format(trace_id, span_id, sampled)
    }

    fn _core_trace_parent(traceparent: Option<String>) -> Option<(u128, u64, bool)> {
        let parent = TraceParent::parse(traceparent.as_deref()?)?;
        Some((
            u128::from_be_bytes(parent.trace_id),
            u64::from_be_bytes(parent.span_id),
            parent.sampled,
        ))
    }
}
