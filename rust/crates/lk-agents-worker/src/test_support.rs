//! Native runtime fixtures own their service; saturation never changes another test's capacity.

pub(super) fn blocking_io() -> lk_agents_net::blocking_io::BlockingIo {
    lk_agents_net::blocking_io::BlockingIo::with_limits("lk-test-io", 4, &[4]).unwrap()
}
