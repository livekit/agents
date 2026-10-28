//! Transport parents are plain wire data; validation precedes host SDK activation.

use super::observation::TraceParent;

/// Wire values from the SDK's integer identifiers. Invalid parents are omitted.
#[crate::host(export)]
pub fn _core_format_trace_parent(trace_id: u128, span_id: u64, sampled: bool) -> Option<String> {
    TraceParent::format(trace_id, span_id, sampled)
}

/// SDK construction values from a validated parent. Invalid context never refuses the request.
#[crate::host(export)]
pub fn _core_trace_parent(traceparent: Option<String>) -> Option<(u128, u64, bool)> {
    let parent = TraceParent::parse(traceparent.as_deref()?)?;
    Some((u128::from_be_bytes(parent.trace_id), u64::from_be_bytes(parent.span_id), parent.sampled))
}
