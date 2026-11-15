use lk_agents_abi::utils::websocket::CANCELLED_ANSWER_WAIT;

use super::*;

/// A pool: one stream a socket, every frame straight to it.
fn pool(max_sockets: Option<u32>, idle_timeout: Option<f64>) -> WebSocketMuxOptions {
    WebSocketMuxOptions {
        max_contexts: None,
        max_sockets,
        idle_timeout,
        keep_warm: false,
        routed: false,
        reuses_sockets: true,
        reads_cancel_end: false,
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

// Catches: cancellation during provider readiness keeping a reserved pool slot forever.
#[tokio::test]
async fn cancelling_before_ready_releases_the_socket_and_its_pool_slot() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, setup_then_ready(), pool(Some(1), None));
    let (_, _input, first) = stream(&clock, &mux, "a");
    until("setup sent", || provider.count() == 1 && provider.socket(0).sent() == ["setup"]).await;
    first.session.aclose().await;
    first.ended().await.unwrap();
    until("cancelled acquisition closed", || provider.socket(0).closed.load(Ordering::SeqCst))
        .await;

    let (_, input, second) = stream(&clock, &mux, "b");
    until("the pool slot is free", || provider.count() == 2).await;
    provider.socket(1).text("socket ready");
    input.send(end()).unwrap();
    second.ended().await.unwrap();
    mux.aclose().await;
}

// Catches: dropping an acquired transport while its provider setup write is still pending.
#[tokio::test]
async fn cancelling_during_setup_closes_the_acquired_transport() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    provider.held_writes.store(true, Ordering::SeqCst);
    let mux = mux(&clock, &provider, setup_then_ready(), pool(Some(1), None));
    let (_, _input, run) = stream(&clock, &mux, "a");
    until("setup write started", || {
        provider.count() == 1 && provider.socket(0).sent() == ["setup"]
    })
    .await;
    run.session.aclose().await;
    run.ended().await.unwrap();
    until("the acquired transport closed", || provider.socket(0).closed.load(Ordering::SeqCst))
        .await;
    provider.held_writes.store(false, Ordering::SeqCst);
    mux.aclose().await;
}

#[tokio::test]
async fn a_closed_pool_refuses_acquisition_without_opening_a_transport() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), pool(Some(1), None));
    mux.aclose().await;
    let (protocol, _input) = Protocol::new(quiet());
    let result = mux.acquire(&Cx::NONE, &options(0), "a".into(), protocol).await;
    assert!(result.is_err());
    assert_eq!(provider.count(), 0);
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
        ["open a", "open b", "goodbye"],
        "an unrouted stream is not cancelled; the spent socket's session is ended"
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

// Catches: a warm socket the provider closed while idle leaving the next stream to pay the
// connect.
#[tokio::test]
async fn a_warm_socket_the_provider_closed_is_reopened() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let options = WebSocketMuxOptions { keep_warm: true, ..routed(None) };
    let mux = mux(&clock, &provider, quiet(), options);
    mux.prewarm();
    until("prewarmed", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(2)).await;
    provider.socket(0).close(Some(1000), None);
    until("warm again", || provider.count() == 2).await;

    let (_, _input, run) = stream(&clock, &mux, "a");
    until("opened on the warm socket", || provider.socket(1).sent() == ["open a"]).await;
    run.session.aclose().await;
    mux.aclose().await;
}

// Catches: a provider that closes a pooled socket cleanly right after the stream's last segment
// completed, before the stream took its end, failing the stream though nothing was owed.
#[tokio::test]
async fn a_clean_close_with_nothing_owed_ends_a_pooled_stream_cleanly() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), pool(None, None));
    let (protocol, input, run) = stream(&clock, &mux, "a");
    protocol.nothing_pending.store(true, Ordering::SeqCst);
    input.send(item("text")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["open a", "text"])
        .await;
    provider.socket(0).close(Some(1000), None);
    until("the session finished", || protocol.finished_sessions.load(Ordering::SeqCst) == 1).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    assert_eq!(provider.count(), 1, "its end opens no socket of its own");
    mux.aclose().await;
}

// Catches: a stream's input-idle frames (a TTS's text idle) never sent on a multiplexed socket.
#[tokio::test]
async fn the_input_idle_frames_go_on_a_multiplexed_stream_too() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), routed(None));
    let (protocol, input, run) = stream(&clock, &mux, "a");
    *protocol.input_idle.lock() = Some(0.6);
    input.send(item("text")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["open a", "text"])
        .await;
    crate::testkit::arm_then_advance(&clock, Duration::from_millis(600)).await;
    until("the idle frames", || provider.socket(0).sent() == ["open a", "text", "idle"]).await;
    run.session.aclose().await;
    mux.aclose().await;
}

// Catches: a socket the pool closes (idle, or at its close) dropped without the provider's
// goodbye, so its session lingers against the account's concurrency limit.
#[tokio::test]
async fn a_socket_the_pool_closes_says_goodbye_first() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), pool(None, Some(30.0)));
    let (_, input, run) = stream(&clock, &mux, "a");
    until("opened", || provider.count() == 1).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("closed idle", || provider.socket(0).closed.load(Ordering::SeqCst)).await;
    assert_eq!(provider.socket(0).sent(), ["open a", "goodbye"]);

    let (_, input, run) = stream(&clock, &mux, "b");
    until("a new socket", || provider.count() == 2).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    mux.aclose().await;
    assert_eq!(provider.socket(1).sent(), ["open b", "goodbye"]);
    assert!(provider.socket(1).closed.load(Ordering::SeqCst));
}

// Catches: a parked socket's warm-keep (qwen3's empty flush) sent while a stream holds the
// socket, where its answer would reach the stream; or not sent while it is parked, so the
// provider times it out.
#[tokio::test]
async fn a_keepalive_only_while_unused_goes_only_while_no_stream_holds_the_socket() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let parked = WebSocketConnection {
        keepalive: Some(WebSocketKeepaliveFrame {
            interval: 15.0,
            frame: WsFrame::Text("keepalive".into()),
            only_while_unused: true,
        }),
        ..quiet()
    };
    let mux = mux(&clock, &provider, parked, pool(None, None));
    mux.prewarm();
    until("prewarmed", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    until("kept warm", || provider.socket(0).sent() == ["keepalive"]).await;

    let (_, input, run) = stream(&clock, &mux, "a");
    until("taken", || provider.socket(0).sent() == ["keepalive", "open a"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    assert_eq!(provider.socket(0).sent(), ["keepalive", "open a"], "not while it is used");
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    until("parked again", || provider.socket(0).sent().ends_with(&["keepalive".to_owned()])).await;
    mux.aclose().await;
}

/// A pool that cancels a stream left mid-answer and reuses its socket once the provider ended it.
fn cancelling_pool() -> WebSocketMuxOptions {
    WebSocketMuxOptions { reads_cancel_end: true, ..pool(None, None) }
}

// Catches: a barge-in spending the pooled socket (every interruption pays a new connection), or
// the next stream reusing it before the provider ended the cancelled answer, whose late audio
// it would play as its own.
#[tokio::test]
async fn a_pooled_socket_left_mid_answer_is_cancelled_and_reused_once_the_provider_ended_it() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), cancelling_pool());
    let (_, _input, abandoned) = stream(&clock, &mux, "a");
    until("opened", || provider.count() == 1).await;
    abandoned.session.aclose().await;
    abandoned.ended().await.unwrap();
    assert_eq!(provider.socket(0).sent(), ["open a", "cancel mine"]);

    let (_, _input, next) = stream(&clock, &mux, "b");
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.count(), 1, "the next stream waits for the cancel's end");
    assert_eq!(provider.socket(0).sent(), ["open a", "cancel mine"], "not before it");
    provider.socket(0).text("mine:cancelled");
    until("reused", || provider.socket(0).sent().ends_with(&["open b".to_owned()])).await;
    assert_eq!(provider.count(), 1);
    next.session.aclose().await;
    mux.aclose().await;
}

// Catches: a cancel the provider never answers holding the pooled socket (and the next stream)
// forever, or the socket then reused with the abandoned answer possibly still coming.
#[tokio::test]
async fn an_unanswered_cancel_closes_the_pooled_socket_after_its_wait() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = mux(&clock, &provider, quiet(), cancelling_pool());
    let (_, _input, abandoned) = stream(&clock, &mux, "a");
    until("opened", || provider.count() == 1).await;
    abandoned.session.aclose().await;
    abandoned.ended().await.unwrap();

    let (_, _input, next) = stream(&clock, &mux, "b");
    crate::testkit::arm_then_advance(&clock, CANCELLED_ANSWER_WAIT).await;
    until("a new socket", || provider.count() == 2 && provider.socket(1).sent() == ["open b"])
        .await;
    until("the old one closed", || provider.socket(0).closed.load(Ordering::SeqCst)).await;
    next.session.aclose().await;
    mux.aclose().await;
}

// Catches: a provider whose socket carries one session (a connection per segment) opening the
// next socket only when the next segment needs it, paying the connect and setup on the turn's
// critical path; or a used socket handed to the next session.
#[tokio::test]
async fn a_single_use_warm_pool_keeps_the_next_socket_open_as_a_standby() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let standby =
        WebSocketMuxOptions { reuses_sockets: false, keep_warm: true, ..pool(None, None) };
    let mux = mux(&clock, &provider, quiet(), standby);
    let (_, input, first) = stream(&clock, &mux, "a");
    until("the standby opened beside it", || provider.count() == 2).await;
    input.send(end()).unwrap();
    first.ended().await.unwrap();
    until("the used socket closed", || provider.socket(0).closed.load(Ordering::SeqCst)).await;

    let (_, _input, second) = stream(&clock, &mux, "b");
    until("the standby taken", || provider.socket(1).sent() == ["open b"]).await;
    until("the next standby", || provider.count() == 3).await;
    assert!(provider.socket(2).sent().is_empty());
    second.session.aclose().await;
    mux.aclose().await;
}
