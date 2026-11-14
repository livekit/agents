use super::*;

/// A pool: one stream a socket, every frame straight to it.
fn pool(max_sockets: Option<u32>, idle_timeout: Option<f64>) -> WebSocketMuxOptions {
    WebSocketMuxOptions {
        max_contexts: None,
        max_sockets,
        idle_timeout,
        keep_warm: false,
        routed: false,
    }
}

fn mux(
    clock: &Arc<VirtualClock>,
    provider: &Provider,
    connection: WebSocketConnection,
    options: WebSocketMuxOptions,
) -> Arc<Mux> {
    Arc::new(Mux::new(
        Arc::new(Contexts { connection }),
        options,
        env(clock),
        provider.transports(),
    ))
}

fn stream(
    clock: &Arc<VirtualClock>,
    mux: &Arc<Mux>,
    context: &str,
) -> (Arc<Protocol>, mpsc::UnboundedSender<Option<WebSocketInput>>, Run) {
    let (protocol, input) = Protocol::new(quiet());
    // a pooled stream's input ends with everything answered
    protocol.owes_reply.store(false, Ordering::SeqCst);
    let run = Run::spawn(Session::shared(
        Arc::clone(&protocol) as Arc<dyn WebSocketHost>,
        options(0),
        "fake".into(),
        env(clock),
        Arc::clone(mux),
        context.into(),
    ));
    (protocol, input, run)
}

// Catches: a pooled socket not reused by the next stream (a connect per utterance), reused after
// a stream left it mid-answer (its late frames then reaching the next stream), or its frames
// routed through the multiplexer.
#[tokio::test]
async fn a_pool_reuses_a_socket_only_after_a_stream_that_ended_cleanly() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), pool(None, None));

    let (first, input, run) = stream(&clock, &mux, "a");
    until("opened", || provider.count() == 1 && provider.socket(0).sent() == ["open a"]).await;
    provider.socket(0).text("hello");
    until("read", || first.read.lock().len() == 1).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    assert_eq!(*first.read.lock(), ["hello"], "frames straight to the stream");

    let (_, _input, run) = stream(&clock, &mux, "b");
    until("reused", || provider.socket(0).sent() == ["open a", "open b"]).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
    until("closed", || provider.socket(0).closed.load(Ordering::SeqCst)).await;
    assert_eq!(
        provider.socket(0).sent(),
        ["open a", "open b"],
        "an unrouted stream is not cancelled"
    );

    let (_, _input, run) = stream(&clock, &mux, "c");
    until("a new socket", || provider.count() == 2 && provider.socket(1).sent() == ["open c"])
        .await;
    run.session.aclose().await;
    mux.aclose().await;
}

// Catches: a pooled socket's provider session not set up before its stream's input, input sent
// before the provider is ready, more sockets than the cap, or an idle socket kept open forever.
#[tokio::test]
async fn a_pool_sets_up_each_socket_waits_for_ready_caps_its_sockets_and_closes_idle_ones() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let ready = WebSocketConnection {
        setup: vec![WsFrame::Text("setup".into())],
        awaits_ready: true,
        ..quiet()
    };
    let mux = mux(&clock, &provider, ready, pool(Some(1), Some(30.0)));

    let (_, input, first) = stream(&clock, &mux, "a");
    input.send(item("audio")).unwrap();
    until("the setup", || provider.count() == 1 && provider.socket(0).sent() == ["setup"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.socket(0).sent(), ["setup"], "nothing before the provider is ready");
    provider.socket(0).text("socket ready");
    until("the input", || provider.socket(0).sent() == ["setup", "open a", "audio"]).await;

    let (_, second_input, second) = stream(&clock, &mux, "b");
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.count(), 1, "at the cap, the second stream waits");
    input.send(end()).unwrap();
    first.ended().await.unwrap();
    until("the second stream", || provider.socket(0).sent().ends_with(&["open b".to_owned()]))
        .await;
    second_input.send(end()).unwrap();
    second.ended().await.unwrap();

    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("the idle socket closed", || provider.socket(0).closed.load(Ordering::SeqCst)).await;
    assert_eq!(provider.count(), 1);
    mux.aclose().await;
}

// Catches: an open context the provider closes after its inactivity limit (no per-context
// keepalive), or a warm socket the provider closed while idle leaving the next stream to pay the
// connect.
#[tokio::test]
async fn open_contexts_are_kept_alive_and_a_warm_socket_the_provider_closed_is_reopened() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let kept = WebSocketConnection { context_keepalive: Some(10.0), ..quiet() };
    let options = WebSocketMuxOptions { keep_warm: true, ..routed(None) };
    let mux = mux(&clock, &provider, kept, options);
    mux.prewarm();
    until("prewarmed", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(2)).await;
    provider.socket(0).close(Some(1000), None);
    until("warm again", || provider.count() == 2).await;

    let (_, _input, run) = stream(&clock, &mux, "a");
    until("opened", || provider.socket(1).sent() == ["open a"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    until("kept alive", || provider.socket(1).sent() == ["open a", "keep mine"]).await;
    run.session.aclose().await;
    mux.aclose().await;
}
