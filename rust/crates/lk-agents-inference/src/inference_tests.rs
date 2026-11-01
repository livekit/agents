//! The crate's one integration binary (TESTING.md §9, DESIGN.md §5 M2 step 9). `live_gateway`'s
//! tests stay `#[ignore]`d: run them with `--ignored` and credentials in the environment.

mod support {
    //! Test-only helpers. Shared by the integration harness; not a separate Cargo target.
    //!
    //! # Why every await here is bounded
    //!
    //! A test that can hang is not just slow: under load (a workspace-wide `cargo mutants` sweep, say)
    //! an unbounded `.await` turns a mutant that the test *would* have caught into a timeout, and the
    //! survivor list becomes noise. Every await in this crate's integration tests goes through
    //! [`within`] or [`until`], including the ones inside the fake gateway's own task.

    #![allow(dead_code)]

    use std::future::Future;
    use std::time::Duration;

    /// Generous enough that a loaded machine does not flake, short enough that a genuine hang is
    /// reported as a failure rather than as a stalled run.
    pub const LIMIT: Duration = Duration::from_secs(10);

    /// Await `fut`, failing the test with `what` if it does not finish inside [`LIMIT`].
    pub async fn within<F: Future>(what: &str, fut: F) -> F::Output {
        match tokio::time::timeout(LIMIT, fut).await {
            Ok(v) => v,
            Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
        }
    }

    /// Poll `cond` until it holds, failing the test with `what` if it never does.
    ///
    /// A bounded replacement for `while !cond() { yield_now().await }`, which is the other way a test
    /// hangs forever on a loaded machine.
    pub async fn until(what: &str, mut cond: impl FnMut() -> bool) {
        within(what, async {
            while !cond() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await;
    }
}

#[path = "avatar_tests.rs"]
mod avatar_peer;
#[path = "interruption/transport_tests.rs"]
mod interruption_transport;
#[path = "interruption/websocket_tests.rs"]
mod interruption_websocket;
#[path = "gateway_tests.rs"]
mod live_gateway;
#[path = "llm/live_tests.rs"]
mod llm_live;
#[path = "llm_tests.rs"]
mod llm_peer;
#[path = "realtime_tests.rs"]
mod realtime;
#[path = "tts/connection_tests.rs"]
mod tts_connection_reused;
#[path = "vad_tests.rs"]
mod vad;
