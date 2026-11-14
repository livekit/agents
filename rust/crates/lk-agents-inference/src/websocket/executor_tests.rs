use super::*;

fn paced(
    protocol: &Arc<Protocol>,
    provider: &Provider,
    clock: &Arc<VirtualClock>,
    retries: u32,
    interval: Duration,
) -> Run {
    let host: Arc<dyn WebSocketHost> = Arc::clone(protocol) as Arc<dyn WebSocketHost>;
    let pacing = Arc::new(Pacing(Mutex::new(Pacer::new(Some(interval), false))));
    let session =
        Session::direct(host, options(retries), "fake".into(), env(clock), provider.transports());
    Run::spawn(session.paced(pacing))
}

/// An input item of two frames, the second paced behind the first.
fn two_frames() -> Option<WebSocketInput> {
    Some(WebSocketInput { frames: frames(&["one", "two"]), pace: 0.0 })
}

// Catches: an item taken and half written when its connection closed, lost (or split across two
// connections) instead of sent whole on the next connection.
#[tokio::test]
async fn an_item_cut_by_a_close_mid_write_goes_whole_on_the_next_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(quiet());
    let run = paced(&protocol, &provider, &clock, 1, Duration::from_millis(60));
    input.send(two_frames()).unwrap();
    until("the first frame", || provider.count() == 1 && provider.socket(0).sent() == ["one"])
        .await;
    provider.socket(0).close(Some(1000), None);
    until("reconnected", || provider.count() == 2).await;
    // the cut write's slot stays spent: the resent frames take the next ones
    for _ in 0..3 {
        crate::testkit::arm_then_advance(&clock, Duration::from_millis(60)).await;
    }
    until("the whole item", || provider.socket(1).sent() == ["one", "two"]).await;
    assert_eq!(provider.socket(0).sent(), ["one"]);
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: an item half written when its socket stopped answering pings, lost with the failed
// connection instead of sent whole on the next one.
#[tokio::test]
async fn an_item_cut_by_a_lost_ping_mid_write_goes_whole_on_the_next_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let pinged = WebSocketConnection { ping_interval: Some(30.0), ..quiet() };
    let (protocol, input) = Protocol::new(pinged);
    let run = paced(&protocol, &provider, &clock, 1, Duration::from_secs(60));
    input.send(two_frames()).unwrap();
    until("the first frame", || provider.count() == 1 && provider.socket(0).sent() == ["one"])
        .await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    until("reconnected", || provider.count() == 2).await;
    let sent = || provider.socket(1).sent().into_iter().filter(|frame| frame != "<ping>");
    for _ in 0..10 {
        crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
        provider.socket(1).pong();
    }
    until("the whole item", || sent().eq(["one", "two"])).await;
    assert_eq!(provider.socket(0).sent(), ["one", "<ping>"]);
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a reconnect asked for mid-item cutting the item (half of it on each connection), or
// the item lost.
#[tokio::test]
async fn a_reconnect_mid_item_lets_the_item_finish_first() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(quiet());
    let run = paced(&protocol, &provider, &clock, 0, Duration::from_millis(60));
    input.send(two_frames()).unwrap();
    until("the first frame", || provider.count() == 1 && provider.socket(0).sent() == ["one"])
        .await;
    run.session.reconnect("options changed");
    crate::testkit::arm_then_advance(&clock, Duration::from_millis(60)).await;
    until("the new connection", || provider.count() == 2).await;
    assert_eq!(provider.socket(0).sent(), ["one", "two"], "the item went whole");
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert!(provider.socket(1).sent().is_empty(), "the item went once");
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a failed write slept on (keepalives and timers still running against the dead socket)
// instead of waiting for the peer's close, which says why.
#[tokio::test]
async fn a_failed_write_waits_for_the_peers_close_to_say_why() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).closed.store(true, Ordering::SeqCst);
    input.send(item("audio")).unwrap();
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(5)).await;
    assert!(!run.is_finished(), "the keepalive does not fail it first");
    provider.socket(0).close(Some(1008), Some("invalid api key"));
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.error_kind, ErrorKind::Auth);
}

// Catches: a failed write whose peer never closes waiting forever.
#[tokio::test]
async fn a_failed_write_with_a_silent_peer_fails_at_the_read_limit() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(quiet());
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).closed.store(true, Ordering::SeqCst);
    input.send(item("audio")).unwrap();
    crate::testkit::arm_then_advance(&clock, DEFAULT_WAIT).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.error_kind, ErrorKind::Connection);
}

// Catches: a provider that must say it is ready waited for forever when `conn_options` sets no
// connect limit.
#[tokio::test]
async fn readiness_with_no_connect_limit_is_bounded_by_the_default_wait() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(setup_then_ready());
    let host: Arc<dyn WebSocketHost> = Arc::clone(&protocol) as Arc<dyn WebSocketHost>;
    let unlimited = ApiConnectOptions { connect_timeout: None, ..options(0) };
    let session =
        Session::direct(host, unlimited, "fake".into(), env(&clock), provider.transports());
    let run = Run::spawn(session);
    until("the setup", || provider.count() == 1 && provider.socket(0).sent() == ["setup"]).await;
    crate::testkit::arm_then_advance(&clock, DEFAULT_WAIT).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.error_kind, ErrorKind::Timeout);
}

// Catches: a connection past its maximum session replaced only at an explicit flush, so a stream
// that never flushes reaches the provider's hard limit; or one replaced while the stream has work
// in progress a new connection would cut.
#[tokio::test]
async fn a_session_past_its_limit_is_replaced_once_the_stream_can_be() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let limited = WebSocketConnection { max_session_duration: Some(60.0), ..quiet() };
    let (protocol, input) = Protocol::new(limited);
    protocol.can_reconnect.store(false, Ordering::SeqCst);
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("speech")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["speech"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(60)).await;
    input.send(item("more speech")).unwrap();
    until("sent", || provider.socket(0).sent().len() == 2).await;
    assert_eq!(provider.count(), 1, "not while the stream has work in progress");
    protocol.can_reconnect.store(true, Ordering::SeqCst);
    input.send(item("silence")).unwrap();
    until("the new connection", || provider.count() == 2).await;
    assert_eq!(provider.socket(0).sent(), ["speech", "more speech", "silence"]);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: provider sockets opened outside the job's HTTP proxy, which a locked-down network
// then refuses.
#[tokio::test]
async fn provider_sockets_open_through_the_processs_http_proxy() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let proxies = Arc::new(Mutex::new(Vec::new()));
    let transports: Transports = {
        let (provider, proxies) = (provider.clone(), Arc::clone(&proxies));
        Arc::new(move |proxy| {
            proxies.lock().push(proxy);
            Arc::new(provider.clone()) as Arc<dyn WsTransport>
        })
    };
    lk_agents_net::env::serve_http_proxy(Some("http://proxy.test:3128".into()));
    let (protocol, _input) = Protocol::new(quiet());
    let host: Arc<dyn WebSocketHost> = Arc::clone(&protocol) as Arc<dyn WebSocketHost>;
    let run = Run::spawn(Session::direct(host, options(0), "fake".into(), env(&clock), transports));
    until("connected", || provider.count() == 1).await;
    lk_agents_net::env::serve_http_proxy(None);
    assert_eq!(*proxies.lock(), [Some("http://proxy.test:3128".to_owned())]);
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: the first control message after a reconnect taken by the old connection's pending
// take and dropped, so a tool result sent right after a session change never reaches the provider.
#[tokio::test]
async fn the_first_control_message_after_a_reconnect_goes_on_the_new_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(quiet());
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    run.session.reconnect("options changed");
    until("the new connection", || provider.count() == 2).await;
    protocol.send_now("tool result");
    until("the control message", || provider.socket(1).sent() == ["tool result"]).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
}
