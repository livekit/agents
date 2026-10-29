//! Transport parents are plain wire data; validation precedes host SDK activation.

/// The W3C `traceparent` a host SDK's span context is carried as.
#[crate::host(functions)]
pub trait TraceParents {
    /// Wire values from the SDK's integer identifiers. Invalid parents are omitted.
    #[host(function)]
    fn _core_format_trace_parent(trace_id: u128, span_id: u64, sampled: bool) -> Option<String>;

    /// SDK construction values from a validated parent. Invalid context never refuses the
    /// request.
    #[host(function)]
    fn _core_trace_parent(
        #[host(default = None)] traceparent: Option<String>,
    ) -> Option<(u128, u64, bool)>;
}
