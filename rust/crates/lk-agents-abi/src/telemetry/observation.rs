//! A session report's host event retention, as the host declares it.

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[crate::host(class(name = "_core_EventRetention"))]
pub trait EventRetention: Send + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env, capture_content: bool) -> Result<Self, super::upload::UploadError>;

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[host(call)]
    fn claim(&mut self, job_id: String);

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[host(call)]
    fn allows_read(&self, job_id: String) -> bool;

    /// Immutable consent captured when this session's report retention was created.
    #[host(getter)]
    fn capture_content(&self) -> bool;

    /// Reject content without creation-time consent, unknown schemas and live-object events.
    #[host(call)]
    fn accepts(&self, kind: String, redaction: bool) -> bool;

    /// Expire old entries, then admit only within the item and byte budgets. The adapter removes
    /// exactly the returned prefix before appending the newly admitted snapshot.
    #[host(call)]
    fn admit(&mut self, retained_bytes: usize, now_ns: u64) -> (usize, bool);

    #[host(call)]
    fn expire(&mut self, now_ns: u64) -> usize;

    #[host(getter)]
    fn diagnostics(&self) -> Vec<(String, u64)>;
}
