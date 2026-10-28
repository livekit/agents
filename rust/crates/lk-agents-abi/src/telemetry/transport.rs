//! Transport parents are plain wire data; validation precedes host SDK activation.

use super::observation::TraceParent;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "telemetry_transport", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraceCarrier {
    #[host(default = None)]
    pub traceparent: Option<String>,
}

/// SDK construction values from a validated parent. Invalid context never refuses the request.
#[crate::host(export)]
pub fn _core_trace_parent(traceparent: Option<String>) -> Option<(String, String, bool)> {
    let parent = TraceParent::parse(traceparent.as_deref()?)?;
    Some((
        format!("{:032x}", u128::from_be_bytes(parent.trace_id)),
        format!("{:016x}", u64::from_be_bytes(parent.span_id)),
        parent.sampled,
    ))
}
