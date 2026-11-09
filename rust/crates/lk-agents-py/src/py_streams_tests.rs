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

/// Waits for the host to set its `event` (a `threading.Event`), as `log` then reads `want`.
fn signalled(g: &Py<PyAny>, event: &str, want: &str) {
    Python::attach(|py| {
        let g = g.bind(py);
        let set = g.getattr(event).unwrap().call_method1("wait", (5.0,)).unwrap();
        assert!(set.is_truthy().unwrap(), "the host never set {event}");
        assert_eq!(g.getattr("log").unwrap().str().unwrap().to_string(), want);
    });
}

// Catches: a stream its owner leaked (dropped without `close`) left open on the host: unlike
// an async generator, a stream object has no finalizer that closes it.
#[tokio::test]
async fn a_stream_dropped_unclosed_is_closed_on_its_loop() {
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let mut stream = open(&g, &context, "tracked");
    assert!(tokio::time::timeout(Duration::from_millis(30), stream.next()).await.is_err());
    drop(stream);
    signalled(&g, "tracked_closed", "['tracked closed']");
}

// Catches: a host open whose caller was cancelled before it took the returned stream,
// discarding that stream unclosed.
#[tokio::test]
async fn a_returned_stream_nobody_took_is_closed() {
    use lk_agents_abi::invoke::HostEntry as _;
    Python::initialize();
    let (g, context) = Python::attach(setup);
    let tracked = Python::attach(|py| g.bind(py).getattr("open_tracked").unwrap().unbind());
    let mut entered = StreamCall.enter(
        &context,
        None,
        Box::new(move |py| Ok((tracked.bind(py).clone(), pyo3::types::PyTuple::empty(py), None))),
    );
    // Started on its loop by the first poll, returned meanwhile, then abandoned unread.
    let first = std::future::poll_fn(|cx| std::task::Poll::Ready(entered.as_mut().poll(cx))).await;
    assert!(first.is_pending());
    signalled(&g, "tracked_returned", "[]");
    drop(entered);
    signalled(&g, "tracked_closed", "['tracked closed']");
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

mod nodes {
    //! The runtime's node legs over the generated `_AgentNodes` proxy and a live interpreter.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use lk_agents_abi::voice::io::{IoError, TextOutput};
    use lk_agents_abi::voice::nodes::{AgentNodes as _, NodeFactory};
    use lk_agents_abi::{AudioFrame, HostRef};
    use lk_agents_runtime::net::env::ProcessEnv;
    use lk_agents_runtime::voice::nodes::AgentNodes;
    use pyo3::prelude::*;

    use crate::generated::voice::SessionNodesProxy;

    struct Destination;

    impl TextOutput for Destination {
        fn label(&self) -> &str {
            "destination"
        }
        fn capture_text<'a>(
            &'a self,
            _cx: &lk_agents_abi::telemetry::Cx,
            _: &'a str,
        ) -> lk_agents_abi::BoxFuture<'a, Result<(), IoError>> {
            Box::pin(async { Ok(()) })
        }
        fn flush(&self) {}
    }

    struct BindingOwner(Arc<AtomicBool>);

    impl Drop for BindingOwner {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// The agent nodes over the host fixture's `processing` object, on this test's loop.
    fn nodes(scenario: &str) -> (Py<PyAny>, AgentNodes) {
        Python::initialize();
        let (g, captured): (Py<PyAny>, HostRef) = Python::attach(crate::py_streams::tests::setup);
        let host = Python::attach(|py| {
            crate::py_test_support::fixture(py, "install_node_types");
            crate::py_test_support::call(py, scenario, (g.bind(py),)).unwrap();
            let processing = g.bind(py).getattr("processing").unwrap();
            SessionNodesProxy::in_context(&processing, &captured).unwrap()
        });
        (g, AgentNodes::new(&ProcessEnv::current_runtime(), host))
    }

    async fn flag(g: &Py<PyAny>, name: &str) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !Python::attach(|py| {
                let event = g.bind(py).getattr(name).unwrap();
                event.call_method0("is_set").unwrap().extract::<bool>().unwrap()
            }) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the Python node did not reach the expected boundary");
    }

    // Catches: a cancelled transcription node dropping its binding owner, or settling its drain,
    // before the node's `finally` block has run to its end.
    #[tokio::test]
    async fn text_node_cancellation_retains_its_binding_until_python_finally_finishes() {
        let (g, nodes) =
            nodes("text_node_cancellation_retains_its_binding_until_python_finally_finishes");
        let node = nodes.text_node(Arc::new(Destination)).unwrap();
        node.capture_text("queued words");
        node.flush();
        flag(&g, "entered").await;
        let released = Arc::new(AtomicBool::new(false));
        node.retain_until_drained(Box::new(BindingOwner(released.clone())));
        node.cancel();
        flag(&g, "closing").await;
        let mut receipt = node.drain();
        let was_pending =
            tokio::time::timeout(Duration::from_millis(30), receipt.as_mut()).await.is_err();
        let was_retained = !released.load(Ordering::SeqCst);
        // Always release the gate, a regression failure included, so the host loop ends clean.
        Python::attach(|py| {
            let g = g.bind(py);
            let set = g.getattr("close_gate").unwrap().getattr("set").unwrap();
            let event_loop = g.getattr("loop").unwrap();
            event_loop.call_method1("call_soon_threadsafe", (set,)).unwrap();
        });
        if was_pending {
            tokio::time::timeout(Duration::from_secs(2), receipt)
                .await
                .expect("the text receipt did not settle after the node's cleanup")
                .unwrap();
        }
        flag(&g, "cleaned").await;
        assert!(was_pending, "the node completed while its Python finally block was still open");
        assert!(was_retained, "the binding owner was dropped before the node's cleanup");
        assert!(released.load(Ordering::SeqCst), "the completed node retained its binding");
    }

    // Catches: the leg awaiting the node's output before handing back its input sink, which
    // deadlocks a coroutine node (`_node_output` awaits it on the first pull) that reads its whole
    // input before it returns.
    #[tokio::test]
    async fn coroutine_node_receives_its_input_before_returning() {
        let (g, nodes) = nodes("coroutine_node_receives_its_input_before_returning");
        let (sink, mut output) = tokio::time::timeout(
            Duration::from_secs(2),
            nodes.stt_node(&lk_agents_abi::telemetry::Cx::NONE),
        )
        .await
        .expect("the leg awaited the node's output before returning its input")
        .unwrap()
        .split();
        sink.push(AudioFrame::new(vec![0u8; 320], 16_000, 1, 160)).unwrap();
        sink.end_input();
        let end = tokio::time::timeout(Duration::from_secs(2), output.next())
            .await
            .expect("the node did not return once its input ended")
            .unwrap();
        assert!(end.is_none(), "a coroutine returning None yields nothing");
        let consumed = Python::attach(|py| {
            g.bind(py).getattr("consumed").unwrap().extract::<Vec<u32>>().unwrap()
        });
        assert_eq!(consumed, [160], "the node read the input pushed before it returned");
    }
}

mod gil {
    //! The binding boundary under contention: a Python thread holding the GIL calls a generated
    //! synchronous entry whose Rust side waits for a lock, while the Rust task holding that lock waits
    //! for the GIL.

    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use lk_agents_abi::errors::ProviderError;
    use lk_agents_abi::{
        BoxFuture, StreamInput, StreamOutput, SynthesizedAudio, Tts, TtsCapabilities,
    };
    use lk_agents_abi::{Closed, Ts, TtsSink, TtsSource};
    use lk_agents_runtime::inference::fallback::{ChainPolicy, FallbackTts, IgnoreEvents};
    use lk_agents_runtime::inference::tts::stream::TtsStream;
    use pyo3::prelude::*;

    /// A host TTS as the fallback chain sees it: its stream opens once `opened` fires, and its sink
    /// takes the GIL on every push, as a host sink scheduling its drain does.
    struct HostLikeTts {
        opened: lk_agents_runtime::net::signal::Cell<bool>,
        pushing: mpsc::SyncSender<()>,
    }

    struct HostLikeSink(mpsc::SyncSender<()>);

    impl StreamInput<String> for HostLikeSink {
        fn push(&self, _item: String) -> Result<(), ProviderError> {
            let _ = self.0.try_send(());
            Python::attach(|_| ());
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    struct Silent;

    impl StreamOutput<SynthesizedAudio> for Silent {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async { Closed::Clean })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for Silent {}

    impl Tts for HostLikeTts {
        fn label(&self) -> String {
            "host-like".into()
        }
        fn capabilities(&self) -> TtsCapabilities {
            TtsCapabilities { streaming: true, aligned_transcript: false }
        }
        fn sample_rate(&self) -> u32 {
            24_000
        }
        fn num_channels(&self) -> u16 {
            1
        }
        fn synthesize<'a>(
            &'a self,
            _cx: &lk_agents_abi::telemetry::Cx,
            _text: &'a str,
        ) -> BoxFuture<'a, Result<TtsSource, ProviderError>> {
            unreachable!("the test only streams")
        }
        fn stream(
            &self,
            _cx: &lk_agents_abi::telemetry::Cx,
            _expressive: bool,
        ) -> BoxFuture<'_, Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>>
        {
            let opened = self.opened.clone();
            let pushing = self.pushing.clone();
            Box::pin(async move {
                opened.wait_for(|opened| *opened).await.unwrap();
                Ok(lk_agents_abi::DuplexStream::new(
                    Box::new(HostLikeSink(pushing)) as TtsSink,
                    Box::new(Silent) as TtsSource,
                ))
            })
        }
    }

    // Catches: a generated sync entry (`TtsStream.push_text`) keeping the GIL while it waits for the
    // fallback's token log, whose holder (the chain pump replaying text into a host TTS) waits for
    // the GIL.
    #[test]
    fn pushing_text_while_the_fallback_replays_into_a_host_tts_does_not_deadlock() {
        Python::initialize();
        let (open, opened) = lk_agents_runtime::net::signal::signal(false);
        let (pushing, pushed) = mpsc::sync_channel(1);
        let child: Arc<dyn Tts> = Arc::new(HostLikeTts { opened, pushing });
        let runtime = lk_agents_runtime::process::handle().unwrap();
        let fallback = FallbackTts::new(
            vec![child],
            None,
            ChainPolicy::tts_defaults(),
            lk_agents_runtime::process::host_clock(),
            lk_agents_runtime::process::host_spawner().unwrap(),
            Arc::new(IgnoreEvents),
        )
        .unwrap();
        let (sink, source) = runtime
            .block_on(fallback.stream(&lk_agents_abi::telemetry::Cx::NONE, false))
            .unwrap()
            .split();
        let stream =
            Arc::new(TtsStream::new(Some(sink), source, lk_agents_runtime::process::host_clock()));
        lk_agents_abi::tts::streams::TtsStream::push_text(&*stream, "replayed".into()).unwrap();

        let (attached, attaching) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            Python::attach(|py| {
                let stream = Py::new(py, crate::generated::tts::TtsStream(stream)).unwrap();
                attached.send(()).unwrap();
                // The pump now holds the log, replaying into the host TTS, which waits for the GIL.
                pushed.recv_timeout(Duration::from_secs(10)).expect("the pump replays");
                stream.bind(py).call_method1("push_text", ("live",)).unwrap();
            });
            done.send(()).unwrap();
        });
        attaching.recv().unwrap();
        open.set(true);
        assert!(finished.recv_timeout(Duration::from_secs(10)).is_ok(), "deadlocked");
    }
}
