//! Python node streams and binding behavior under GIL contention.

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
    //! for the GIL (review13's deadlock class, decision 129).

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
        opened: tokio::sync::watch::Receiver<bool>,
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
            let mut opened = self.opened.clone();
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
        let (open, opened) = tokio::sync::watch::channel(false);
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
        open.send_replace(true);
        assert!(finished.recv_timeout(Duration::from_secs(10)).is_ok(), "deadlocked");
    }
}
