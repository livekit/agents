//! Native runtime fixtures own their service; saturation never changes another test's capacity.

pub(super) fn native_workers() -> lk_agents_net::worker_io::WorkerIo {
    lk_agents_net::worker_io::WorkerIo::new(tokio::runtime::Handle::current(), 4).unwrap()
}
