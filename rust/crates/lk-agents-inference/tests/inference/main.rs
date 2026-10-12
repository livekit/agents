//! The crate's one integration binary (TESTING.md §9, DESIGN.md §5 M2 step 9). `live_gateway`'s
//! tests stay `#[ignore]`d: run them with `--ignored` and credentials in the environment.

mod support;

mod interruption_transport;
mod interruption_websocket;
mod live_gateway;
mod fake_socket;
mod live_realtime;
