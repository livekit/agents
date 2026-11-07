//! Against a live interpreter and a running loop.
use super::*;
use std::time::Duration;

use lk_agents_runtime::net::clock::RealClock;

pub(crate) fn setup(py: Python<'_>) -> (Py<PyAny>, HostRef) {
    let g = crate::py_test_support::fixture(py, "streams");
    let pair = g.getattr("capture").unwrap().call1(("caller",)).unwrap();
    g.setattr("pair", &pair).unwrap();
    (g.unbind(), crate::py_objects::retain(pair.unbind()))
}

fn open(g: &Py<PyAny>, context: &HostRef, generator: &str) -> HostStream<String> {
    Python::attach(|py| {
        let iterable = g.bind(py).getattr(generator).unwrap().call0().unwrap();
        let convert: Convert<String> = |o| Ok(o.str()?.to_string());
        HostStream::open(py, &iterable, context, convert, Arc::new(RealClock::new())).unwrap()
    })
}

fn deadline() -> Ts {
    RealClock::new().now().saturating_add(Duration::from_secs(2))
}

// Catches: a fresh context per pull (a scope opened before a yield lost after it), a dropped
// pull advancing the iterator twice, or close not settling the blocked pull before `aclose`.
#[tokio::test]
async fn one_context_spans_pulls_and_close_and_a_dropped_pull_is_resumed() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let mut stream = open(&g, &context, "scoped");
    assert_eq!(stream.next().await.unwrap().as_deref(), Some("inside"));
    // The second pull blocks on the gate; its caller gives up, then a new pull resumes it.
    let blocked = tokio::time::timeout(Duration::from_millis(30), stream.next()).await;
    assert!(blocked.is_err());
    Python::attach(|py| {
        let g = g.bind(py);
        let set = g.getattr("gate").unwrap().getattr("set").unwrap();
        g.getattr("loop").unwrap().call_method1("call_soon_threadsafe", (set,)).unwrap();
    });
    assert_eq!(stream.next().await.unwrap().as_deref(), Some("inside"));
    assert_eq!(Box::new(stream).close(deadline()).await, Closed::Clean);
    Python::attach(|py| {
        let log = g.bind(py).getattr("log").unwrap();
        assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
        // The captured context is as the caller left it.
        let g = g.bind(py);
        let var = g.getattr("var").unwrap();
        let ctx = g.getattr("pair").unwrap().get_item(1).unwrap();
        assert_eq!(ctx.get_item(var).unwrap().str().unwrap().to_string(), "caller");
    });
}

// Catches: closing a stream whose pull is blocked waiting behind that pull instead of
// cancelling it.
#[tokio::test]
async fn close_cancels_a_blocked_pull() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let mut stream = open(&g, &context, "scoped");
    stream.next().await.unwrap();
    assert!(tokio::time::timeout(Duration::from_millis(30), stream.next()).await.is_err());
    assert_eq!(Box::new(stream).close(deadline()).await, Closed::Clean);
    Python::attach(|py| {
        let log = g.bind(py).getattr("log").unwrap();
        assert_eq!(log.str().unwrap().to_string(), "[('closed in', 'inside')]");
    });
}

fn duplex(g: &Py<PyAny>, context: &HostRef) -> (HostSink<String>, HostStream<String>) {
    Python::attach(|py| {
        let obj = g.bind(py).getattr("Duplex").unwrap().call0().unwrap();
        let to_host: ToHost<String> = |py, s| pyo3::IntoPyObjectExt::into_bound_py_any(s, py);
        let convert: Convert<String> = |o| Ok(o.str()?.to_string());
        HostStream::duplex(py, &obj, context, to_host, convert, Arc::new(RealClock::new()))
            .unwrap()
            .split()
    })
}

// Catches: input delivered out of order, outside the stream's context, flush or end of input
// lost among the items, or ending the input closing the output instead of letting it drain.
#[tokio::test]
async fn duplex_input_reaches_the_host_in_order_in_the_streams_context() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let (sink, mut source) = duplex(&g, &context);
    sink.push("a".into()).unwrap();
    sink.flush().unwrap();
    sink.push("b".into()).unwrap();
    sink.end_input();
    assert!(sink.push("late".into()).is_err());
    assert!(sink.flush().is_err());
    assert_eq!(
        source.next().await.unwrap().as_deref(),
        Some("[('a', 'stream'), ('flush', 'stream'), ('b', 'stream'), ('end', 'stream')]")
    );
}

#[tokio::test]
async fn closing_output_refuses_input_without_delivering_queued_items() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let (sink, source) = duplex(&g, &context);
    assert_eq!(Box::new(source).close(deadline()).await, Closed::Clean);
    assert!(sink.push("late".into()).is_err());
    assert!(sink.flush().is_err());
}

// Catches: a delivery the host refused dropped silently, input still accepted after it.
#[tokio::test]
async fn a_refused_delivery_wakes_the_output_and_refuses_the_next_push() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let (sink, mut source) = duplex(&g, &context);
    sink.push("bad".into()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(2), source.next())
        .await
        .expect("input failure must wake output")
        .unwrap_err();
    assert!(error.to_string().contains("refused"));
    assert!(sink.push("next".into()).is_err());
    assert_eq!(Box::new(source).close(deadline()).await, Closed::Clean);
}

// Catches: the host's CancelledError read as the end of the stream.
#[tokio::test]
async fn a_host_cancellation_is_a_failure_not_the_end() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let mut stream = open(&g, &context, "cancelled");
    assert_eq!(stream.next().await.unwrap().as_deref(), Some("1"));
    assert!(stream.next().await.is_err());
}

// Catches: a host stream whose `aclose()` raised reported as closed cleanly (B-10).
#[tokio::test]
async fn a_failed_aclose_is_a_failed_close() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let mut stream = open(&g, &context, "unclosable");
    assert_eq!(stream.next().await.unwrap().as_deref(), Some("x"));
    assert_eq!(Box::new(stream).close(deadline()).await, Closed::Failed);
}

#[tokio::test]
async fn a_failed_cleanup_lookup_is_a_failed_close() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let stream = open(&g, &context, "BrokenCleanup");
    assert_eq!(Box::new(stream).close(deadline()).await, Closed::Failed);
}
