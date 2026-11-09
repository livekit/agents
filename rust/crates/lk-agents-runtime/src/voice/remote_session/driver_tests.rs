use tokio::sync::Notify;

use super::super::client::{ReadyError, RemoteSession, RequestError};
use super::super::{AgentInfo, ClientSink, FrameworkInfo, IoUpdate, RemoteSessionState, RtcStats};
use super::*;
use lk_agents_abi::secs;
use lk_agents_abi::{ChatItem, Clock};
use std::sync::atomic::AtomicBool;
use tokio::sync::Mutex as AsyncMutex;

/// Two linked transports: what one sends, the other receives.
struct Pipe {
    tx: mpsc::UnboundedSender<proto::AgentSessionMessage>,
    rx: AsyncMutex<mpsc::UnboundedReceiver<proto::AgentSessionMessage>>,
    fail: AtomicBool,
    closed: AtomicBool,
    sent: Mutex<Vec<proto::AgentSessionMessage>>,
}

impl SessionTransport for Pipe {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { Ok(()) })
    }
    fn send_message<'a>(
        &'a self,
        msg: &'a proto::AgentSessionMessage,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move {
            if self.fail.load(Ordering::SeqCst) {
                return Err(HostError::new("transport is down"));
            }
            self.sent.lock().push(msg.clone());
            let _ = self.tx.send(msg.clone());
            Ok(())
        })
    }
    fn recv(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>> {
        Box::pin(async move { Ok(self.rx.lock().await.recv().await) })
    }
    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            self.closed.store(true, Ordering::SeqCst);
        })
    }
}

fn pair() -> (Arc<Pipe>, Arc<Pipe>) {
    let (a_tx, a_rx) = mpsc::unbounded_channel();
    let (b_tx, b_rx) = mpsc::unbounded_channel();
    let pipe = |tx, rx| {
        Arc::new(Pipe {
            tx,
            rx: AsyncMutex::new(rx),
            fail: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            sent: Mutex::new(Vec::new()),
        })
    };
    (pipe(b_tx, a_rx), pipe(a_tx, b_rx))
}

/// A port whose `run_input` waits for a release, and whose history read fails.
struct Port {
    release: Arc<Notify>,
}

impl SessionPort for Port {
    fn chat_history(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<ChatItem>, HostError>> {
        Box::pin(async move { Err(HostError::new("no history here")) })
    }
    fn agent_info(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<AgentInfo, HostError>> {
        Box::pin(async move {
            Ok(AgentInfo {
                id: "agent-1".into(),
                instructions: None,
                tools: vec![],
                chat_ctx: vec![],
            })
        })
    }
    fn run_input<'a>(
        &'a self,
        _cx: &crate::telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<lk_agents_abi::ChatItem>, HostError>> {
        Box::pin(async move {
            if text == "fail" {
                return Err(HostError::new("input failed"));
            }
            self.release.notified().await;
            Ok(vec![])
        })
    }
    fn session_state(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<RemoteSessionState, HostError>> {
        Box::pin(async move {
            Ok(RemoteSessionState {
                agent_state: "not-a-state".into(),
                user_state: "away".into(),
                agent_id: "agent-1".into(),
                options: Default::default(),
                started_at: None,
            })
        })
    }
    fn rtc_stats(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<RtcStats>, HostError>> {
        Box::pin(async move { Ok(None) })
    }
    fn session_usage(
        &self,
        _cx: &crate::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<lk_agents_abi::usage::Usage, HostError>> {
        Box::pin(async move { Ok(Default::default()) })
    }
    fn framework_info(&self) -> FrameworkInfo {
        FrameworkInfo { sdk: "rust".into(), sdk_version: "0".into() }
    }
    fn update_io(
        &self,
        _cx: &crate::telemetry::Cx,
        _update: IoUpdate,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { Ok(()) })
    }
    fn finalize_simulation(
        &self,
        _cx: &crate::telemetry::Cx,
        provisional: lk_agents_abi::worker::SimulationVerdict,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::worker::SimulationVerdict>, HostError>,
    > {
        Box::pin(async move {
            if provisional.reason == "fail" {
                Err(HostError::new("simulation failed"))
            } else {
                Ok(Some(provisional))
            }
        })
    }
}

struct Events(Mutex<Vec<String>>);
impl ClientSink for Events {
    fn event(&self, name: &str, _event: &proto::AgentSessionEvent) {
        self.0.lock().push(name.to_owned());
    }
}

fn rt() -> (Arc<dyn Clock>, Arc<dyn Spawner>) {
    (
        Arc::new(lk_agents_net::clock::RealClock::default()),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
    )
}

async fn setup(release: Arc<Notify>) -> (SessionServer, RemoteSession, Arc<Pipe>, Arc<Events>) {
    let (ht, ct) = pair();
    let (clock, spawner) = rt();
    let host = SessionServer::new(Timebase::of_clock(Arc::clone(&clock)), Arc::clone(&spawner));
    host.set_port(Arc::new(Port { release }));
    host.start(ht.clone(), None).await.unwrap();
    let client = RemoteSession::new(clock, spawner);
    let events = Arc::new(Events(Mutex::new(Vec::new())));
    client.start(ct, events.clone()).await.unwrap();
    (host, client, ht, events)
}

async fn nap(d: Duration) {
    let clock = lk_agents_net::clock::RealClock::default();
    clock.wait_until(clock.now().saturating_add(d)).await;
}

#[tokio::test]
async fn replies_and_default_events_use_the_supplied_session_wall_clock() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let time = Timebase::from_virtual(clock);
    time.advance(Duration::from_secs(17));
    let host = SessionHost::new(Arc::new(Port { release: Arc::new(Notify::new()) }), time.clone());
    for request in [
        proto::session_request::Request::GetSessionState(Default::default()),
        proto::session_request::Request::GetSessionUsage(Default::default()),
    ] {
        let request = proto::SessionRequest { request_id: "clock".into(), request: Some(request) };
        let before = time.wall_secs();
        let response = host.handle_request(&request).await.unwrap().unwrap();
        let after = time.wall_secs();
        let Some(Message::Response(response)) = response.message else {
            panic!("response expected")
        };
        let stamp = match response.response.unwrap() {
            proto::session_response::Response::GetSessionState(state) => state.created_at,
            proto::session_response::Response::GetSessionUsage(usage) => usage.created_at,
            _ => panic!("unexpected response"),
        }
        .unwrap();
        let stamped = stamp.seconds as f64 + f64::from(stamp.nanos) / 1e9;
        assert!((before..=after).contains(&stamped), "{stamped} outside {before}..={after}");
    }
    let (transport, _peer) = pair();
    let runtime =
        SessionServer::new(time.clone(), Arc::new(lk_agents_net::task::TokioSpawner::current()));
    let before = time.wall_secs();
    runtime.send_wire(
        WireEvent::Error(lk_agents_abi::worker::remote_session::ErrorWire {
            message: "clock".into(),
        }),
        None,
    );
    runtime.start(transport.clone(), None).await.unwrap();
    runtime.aclose(Duration::from_secs(1)).await;
    let sent = transport.sent.lock();
    let Some(Message::Event(event)) = &sent[0].message else { panic!("event expected") };
    let stamp = event.created_at.as_ref().unwrap();
    let after = time.wall_secs();
    let stamped = stamp.seconds as f64 + f64::from(stamp.nanos) / 1e9;
    assert!((before..=after).contains(&stamped), "{stamped} outside {before}..={after}");
}

#[tokio::test]
async fn empty_run_input_never_enters_the_session_port() {
    use std::future::Future as _;
    let (clock, _) = rt();
    let host = SessionHost::new(
        Arc::new(Port { release: Arc::new(Notify::new()) }),
        Timebase::of_clock(clock),
    );
    let request = proto::SessionRequest {
        request_id: "empty".into(),
        request: Some(proto::session_request::Request::RunInput(Default::default())),
    };
    let mut reply = Box::pin(host.handle_request(&request));
    let ready = std::future::poll_fn(|cx| std::task::Poll::Ready(reply.as_mut().poll(cx))).await;
    let std::task::Poll::Ready(Ok(Some(reply))) = ready else {
        panic!("empty input entered the blocking session port")
    };
    let Some(Message::Response(reply)) = reply.message else { panic!("response expected") };
    assert_eq!(reply.error.as_deref(), Some("empty run_input text"));
}

#[tokio::test]
async fn input_and_simulation_failures_are_response_errors() {
    use proto::session_request::{FinalizeSimulation, Request, RunInput};
    use proto::session_response::Response;
    let (clock, _) = rt();
    let host = SessionHost::new(
        Arc::new(Port { release: Arc::new(Notify::new()) }),
        Timebase::of_clock(clock),
    );
    for (request, expected) in [
        (Request::RunInput(RunInput { text: "fail".into() }), "input failed"),
        (
            Request::FinalizeSimulation(FinalizeSimulation {
                provisional_success: false,
                provisional_reason: "fail".into(),
            }),
            "simulation failed",
        ),
    ] {
        let request = proto::SessionRequest { request_id: "failed".into(), request: Some(request) };
        let reply = host.handle_request(&request).await.unwrap().unwrap();
        let Some(Message::Response(reply)) = reply.message else { panic!("response expected") };
        assert_eq!(reply.error.as_deref(), Some(expected));
        match reply.response.unwrap() {
            Response::RunInput(input) => assert!(input.items.is_empty()),
            Response::FinalizeSimulation(simulation) => {
                assert!(simulation.user_verdict.is_none())
            }
            other => panic!("unexpected response {other:?}"),
        }
    }
}

// Catches: a failing port read left unanswered, or the exception text leaking to the peer.
#[tokio::test]
async fn a_request_whose_port_fails_is_answered_with_an_opaque_internal_error() {
    let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
    let history = proto::session_request::Request::GetChatHistory(Default::default());
    let err = client.request(history, secs(2.0)).await.unwrap_err();
    assert_eq!(
        err,
        RequestError::Failed("session request get_chat_history failed: internal error".into())
    );
    client.aclose().await;
    host.aclose(secs(1.0)).await;
}

// Catches: an unknown state literal not defaulting to AS_IDLE, or a known one defaulted.
#[tokio::test]
async fn session_state_defaults_an_unknown_literal_and_keeps_a_known_one() {
    let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
    let req = proto::session_request::Request::GetSessionState(Default::default());
    let resp = client.request(req, secs(2.0)).await.unwrap();
    let Some(proto::session_response::Response::GetSessionState(st)) = resp.response else {
        panic!("{resp:?}")
    };
    assert_eq!(st.agent_state, proto::AgentState::AsIdle as i32);
    assert_eq!(st.user_state, proto::UserState::UsAway as i32);
    assert!(st.created_at.unwrap().seconds > 0, "an unset start is stamped with the answer time");
    client.aclose().await;
    host.aclose(secs(1.0)).await;
}

// Catches: aclose cancelling in-flight handlers instead of draining them, or a straggler not
// reported by its type.
#[tokio::test]
async fn aclose_drains_a_handler_that_finishes_and_names_one_that_does_not() {
    let release = Arc::new(Notify::new());
    let (host, client, _, _) = setup(Arc::clone(&release)).await;
    let client = Arc::new(client);
    let run = || {
        proto::session_request::Request::RunInput(proto::session_request::RunInput {
            text: "hi".into(),
        })
    };
    let c = Arc::clone(&client);
    let r = run();
    let answered = tokio::spawn(async move { c.request(r, secs(3.0)).await });
    while host.in_flight().is_empty() {
        nap(secs(0.005)).await;
    }
    let rel = Arc::clone(&release);
    tokio::spawn(async move {
        nap(secs(0.05)).await;
        rel.notify_waiters();
    });
    assert!(host.aclose(secs(1.0)).await.is_empty());
    assert!(answered.await.unwrap().is_ok(), "the response of a finished run was dropped");
    client.aclose().await;

    let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
    let client = Arc::new(client);
    let c = Arc::clone(&client);
    let r = run();
    tokio::spawn(async move { c.request(r, secs(3.0)).await });
    while host.in_flight().is_empty() {
        nap(secs(0.005)).await;
    }
    assert_eq!(host.aclose(secs(0.05)).await, ["run_input"]);
    assert!(host.in_flight().is_empty());
    client.aclose().await;
}

// Catches: events out of order, dropped before start, lost at close; a closed host accepting events.
#[tokio::test]
async fn events_queue_before_start_leave_in_order_and_flush_at_close() {
    let (ht, _peer) = pair();
    let (clock, spawner) = rt();
    let host = SessionServer::new(Timebase::of_clock(clock), spawner);
    let ev = |i: usize| proto::AgentSessionEvent {
        created_at: None,
        event: Some(proto::agent_session_event::Event::Error(proto::agent_session_event::Error {
            message: i.to_string(),
        })),
    };
    host.send_event(ev(0), 1.0).unwrap();
    host.start(ht.clone(), None).await.unwrap();
    for i in 1..10 {
        host.send_event(ev(i), 1.0).unwrap();
    }
    assert!(host.send_event(ev(99), f64::NAN).is_err(), "a non-finite stamp is refused");
    host.aclose(secs(1.0)).await;
    host.send_event(ev(100), 1.0).unwrap();
    let sent: Vec<String> = ht
        .sent
        .lock()
        .iter()
        .map(|m| match &m.message {
            Some(proto::agent_session_message::Message::Event(e)) => match &e.event {
                Some(proto::agent_session_event::Event::Error(x)) => x.message.clone(),
                _ => panic!(),
            },
            _ => panic!(),
        })
        .collect();
    assert_eq!(sent, (0..10).map(|i| i.to_string()).collect::<Vec<_>>());
    assert!(ht.closed.load(Ordering::SeqCst), "the transport is closed last");
}

// Catches: a send failure reported as a bare timeout, or a late transport not polled through.
#[tokio::test]
async fn wait_for_ready_raises_the_latched_send_error_and_polls_through_a_late_transport() {
    let (ht, ct) = pair();
    let (clock, spawner) = rt();
    let client = RemoteSession::new(clock.clone(), spawner.clone());
    ct.fail.store(true, Ordering::SeqCst);
    client.start(ct.clone(), Arc::new(Events(Mutex::new(vec![])))).await.unwrap();
    let err = client.wait_for_ready(secs(0.2), secs(0.05)).await.unwrap_err();
    assert_eq!(err, ReadyError::Transport(HostError::new("transport is down")));

    let host = SessionServer::new(Timebase::of_clock(clock), spawner);
    host.set_port(Arc::new(Port { release: Arc::new(Notify::new()) }));
    host.start(ht, None).await.unwrap();
    let late = Arc::clone(&ct);
    tokio::spawn(async move {
        nap(secs(0.1)).await;
        late.fail.store(false, Ordering::SeqCst);
    });
    client.wait_for_ready(secs(2.0), secs(0.05)).await.unwrap();
    client.aclose().await;
    host.aclose(secs(1.0)).await;
}

// Catches: a timed-out request leaving its slot, two requests sharing an id, or a close
// leaving a pending caller hanging.
#[tokio::test]
async fn the_client_times_out_mints_distinct_ids_and_cancels_at_close() {
    let (_ht, ct) = pair();
    let (clock, spawner) = rt();
    let client = Arc::new(RemoteSession::new(clock, spawner));
    let events = Arc::new(Events(Mutex::new(vec![])));
    client.start(ct.clone(), events).await.unwrap();
    let ping = || proto::session_request::Request::Ping(Default::default());
    assert_eq!(client.request(ping(), secs(0.05)).await.unwrap_err(), RequestError::Timeout);
    assert!(client.shared.pending.lock().is_empty(), "a timed-out request kept its slot");

    let c = Arc::clone(&client);
    let waiting = tokio::spawn(async move { c.request(ping(), secs(5.0)).await });
    while ct.sent.lock().len() < 2 {
        nap(secs(0.005)).await;
    }
    client.aclose().await;
    assert_eq!(waiting.await.unwrap().unwrap_err(), RequestError::Cancelled);
    let ids: Vec<String> = ct
        .sent
        .lock()
        .iter()
        .map(|m| match &m.message {
            Some(proto::agent_session_message::Message::Request(r)) => r.request_id.clone(),
            _ => panic!(),
        })
        .collect();
    assert_ne!(ids[0], ids[1]);
    assert!(ids.iter().all(|i| lk_agents_net::strings::is_valid_id("req_", i)), "{ids:?}");
}
