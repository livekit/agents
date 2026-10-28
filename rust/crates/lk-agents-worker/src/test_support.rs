//! Native runtime fixtures own their service; saturation never changes another test's capacity.

pub(super) fn native_workers() -> crate::io::WorkerIo {
    crate::io::WorkerIo::new(tokio::runtime::Handle::current(), 4).unwrap()
}
