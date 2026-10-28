//! Native runtime fixtures own their service; saturation never changes another test's capacity.

pub(super) fn native_workers() -> crate::media_workers::MediaWorkers {
    crate::media_workers::MediaWorkers::new(
        tokio::runtime::Handle::current(),
        crate::media_workers::MediaWorkerLimits::default(),
    )
    .unwrap()
}
