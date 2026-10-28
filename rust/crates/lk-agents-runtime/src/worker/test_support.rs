//! Native runtime fixtures own their service; saturation never changes another test's capacity.

pub(super) fn native_workers() -> lk_agents_media::workers::MediaWorkers {
    lk_agents_media::workers::MediaWorkers::new(
        lk_agents_media::workers::MediaWorkerLimits::default(),
    )
    .unwrap()
}
