//! Every test names the bug it catches. No socket or model: the pure decisions run on
//! [`EotSession`], the gateway over an in-memory WebSocket, the local model as a closure.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ApiErrorKind};
use lk_agents_abi::tokenize::language::{LanguageCode, ThresholdError, ThresholdOptions, TurnDetectorModel};
use lk_agents_abi::metrics::base::Metadata;
use lk_agents_abi::{ApiConnectOptions, EotInferenceMetrics};
use prost::Message as _;

use crate::eot::cloud::{
    CloudOptions, CloudTransport, HEADER_SESSION_ID, WebSocketConnector, WebSocketSink,
    WebSocketStream,
};
use crate::eot::{CancelReason, EotRequestId, EotWarning};
use crate::eot::local::{AudioRingBuffer, LocalEotModel, LocalTransport};
use crate::eot::proto;
use crate::eot::session::{
    EotAction, EotDegrade, EotFlush, EotSession, EotStartInference, EotSwapToLocal,
    TransportPrediction,
};
use crate::eot::transport::{EotTransport, PredictionSink};

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

const NOW: f64 = 1_700_000_000.0;

fn map(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
}

fn id(name: &str) -> EotRequestId {
    EotRequestId::from_string(name.to_owned())
}

/// A cloud session with the server's defaults already installed.
fn cloud_session(local_fallback: bool) -> EotSession {
    cloud_session_sharing(local_fallback).0
}

/// [`cloud_session`], plus the threshold table it shares with the caller.
fn cloud_session_sharing(local_fallback: bool) -> (EotSession, Arc<Mutex<ThresholdOptions>>) {
    let mut thresholds = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
    thresholds.update_defaults(&map(&[("en", 0.5), ("de", 0.4)]), 0.5, None, 0.0).unwrap();
    let shared = Arc::new(Mutex::new(thresholds));
    (EotSession::new(TurnDetectorModel::V1, Arc::clone(&shared), local_fallback), shared)
}

/// 16-bit little-endian PCM of `samples`, mono.
fn pcm_frame(sample_rate: u32, samples: &[i16]) -> lk_agents_abi::AudioFrame {
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    lk_agents_abi::AudioFrame::new(data, sample_rate, 1, samples.len() as u32)
}

fn resolved(actions: &[EotAction]) -> Vec<(String, f64)> {
    actions
        .iter()
        .filter_map(|a| match a {
            EotAction::Resolve(r) => {
                Some((r.id.as_str().to_owned(), r.end_of_turn_probability))
            }
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// the pure core: correlation and staleness
// ---------------------------------------------------------------------------------------------

// Catches: dropping the stale-response guard in `_resolve_prediction`, so a superseded prediction
// resolves the current turn.
#[test]
fn a_prediction_for_a_request_that_is_no_longer_in_flight_is_ignored() {
    let mut session = cloud_session(true);
    session.start(NOW, id("a"));
    let stale = session.resolve(NOW, "b", TransportPrediction::new(0.9));
    assert!(stale.is_empty(), "an id nobody is waiting on produces nothing");
    assert_eq!(session.in_flight(), Some(&id("a")), "and does not clear the live request");

    let current = session.resolve(NOW, "a", TransportPrediction::new(0.9));
    assert_eq!(resolved(&current), vec![("a".to_owned(), 0.9)]);
    assert_eq!(session.in_flight(), None, "resolved once, never twice");
}

// Catches: `predict()` not superseding (the previous future never resolves), or superseding with
// 1.0 instead of 0.0 (an abandoned request would commit the turn early).
#[test]
fn a_new_request_supersedes_the_previous_one_with_a_zero() {
    let mut session = cloud_session(true);
    session.start(NOW, id("a"));
    let actions = session.start(NOW, id("b"));

    assert_eq!(resolved(&actions), vec![("a".to_owned(), 0.0)]);
    assert_eq!(actions.last(), Some(&EotAction::StartInference(EotStartInference { id: id("b") })));
    assert_eq!(session.in_flight(), Some(&id("b")));
}

// Catches: an uncorrelated `cancel_inference`: a late timer would resolve the live request with
// 0.0 and tear down the cloud socket.
#[test]
fn cancelling_a_request_that_was_already_superseded_does_nothing() {
    let mut session = cloud_session(true);
    session.start(NOW, id("a"));
    session.start(NOW, id("b"));

    let actions = session.cancel(NOW, Some(&id("a")), CancelReason::TimedOut);

    assert!(actions.is_empty());
    assert_eq!(session.in_flight(), Some(&id("b")), "the live request survives");
    assert_eq!(session.model(), TurnDetectorModel::V1, "and so does the cloud transport");
}

// Catches: making the id mandatory on `cancel`; the host's `cancel_inference(timed_out=)` has no
// id, and `None` must still resolve the live request.
#[test]
fn an_uncorrelated_cancel_closes_whatever_is_in_flight() {
    let mut session = cloud_session(true);
    session.start(NOW, id("a"));

    let actions = session.cancel(NOW, None, CancelReason::Superseded);

    assert_eq!(resolved(&actions), vec![("a".to_owned(), 0.0)]);
    assert_eq!(session.in_flight(), None);
    // A plain supersede must not fall back: only `TimedOut` on the cloud model does.
    assert_eq!(session.model(), TurnDetectorModel::V1);
    assert!(!session.is_fallback());
}

// Catches: hoisting the timeout fallback inside the request-in-flight guard (`base.py:200-209`);
// a flushed session would then sit on a dead gateway forever.
#[test]
fn a_timeout_with_nothing_in_flight_still_falls_back() {
    let mut session = cloud_session(true);

    let actions = session.cancel(NOW, None, CancelReason::TimedOut);

    assert!(resolved(&actions).is_empty(), "nothing to resolve");
    assert_eq!(actions.last(), Some(&EotAction::SwapToLocal(EotSwapToLocal {})));
    assert_eq!(session.model(), TurnDetectorModel::V1Mini);
}

// ---------------------------------------------------------------------------------------------
// the pure core: the fallback ladder
// ---------------------------------------------------------------------------------------------

// Catches: dropping the `timed_out` leg of `cancel_inference` (`base.py:200`): a slow gateway
// never falls back. Also the session owning a copy of the threshold table instead of sharing it
// (`base.py:137`), so readers keep the cloud scale after the fallback rescale.
#[test]
fn a_prediction_timeout_on_the_cloud_model_swaps_in_the_local_one() {
    let (mut session, shared) = cloud_session_sharing(true);
    session.start(NOW, id("a"));

    let actions = session.cancel(NOW, Some(&id("a")), CancelReason::TimedOut);

    assert_eq!(resolved(&actions), vec![("a".to_owned(), 0.0)], "the cancel still resolves 0.0");
    assert!(actions.contains(&EotAction::Warn(EotWarning::CloudFailedFallingBack {
        reason: "eot prediction timed out".to_owned()
    })));
    assert_eq!(actions.last(), Some(&EotAction::SwapToLocal(EotSwapToLocal {})));
    assert_eq!(session.model(), TurnDetectorModel::V1Mini);
    assert!(session.is_fallback());
    assert!(!session.is_degraded(), "a fallback is not a degradation");
    assert_eq!(
        shared.lock().unwrap().lookup(Some(&LanguageCode::new("de"))),
        Some(0.2450),
        "and the table the caller holds moved to the local model's scale"
    );
}

// Catches: treating `_fallback_to_local` returning False as terminal after a timeout; only a
// transport failure degrades.
#[test]
fn a_prediction_timeout_with_no_fallback_leaves_the_cloud_transport_mounted() {
    let mut session = cloud_session(false);
    session.start(NOW, id("a"));

    let actions = session.cancel(NOW, Some(&id("a")), CancelReason::TimedOut);

    assert_eq!(resolved(&actions), vec![("a".to_owned(), 0.0)]);
    assert_eq!(actions.len(), 1, "no warning, no swap, no degrade");
    assert_eq!(session.model(), TurnDetectorModel::V1);
    assert!(!session.is_fallback());
    assert!(!session.is_degraded());
    assert!(!session.is_closed(), "the next turn may still get an answer");
}

// Catches: degrading without resolving the in-flight request or closing the session; the default
// is 1.0 so the turn commits on `min_endpointing_delay`.
#[test]
fn a_cloud_failure_with_no_fallback_degrades_and_answers_every_later_request() {
    let mut session = cloud_session(false);
    session.start(NOW, id("a"));

    let actions = session.transport_failed(NOW, "socket died");

    assert_eq!(resolved(&actions), vec![("a".to_owned(), 1.0)]);
    assert!(actions.contains(&EotAction::Warn(EotWarning::CloudFailedDegraded {
        reason: "socket died".to_owned()
    })));
    assert_eq!(actions.last(), Some(&EotAction::Degrade(EotDegrade {})));
    assert!(session.is_degraded());

    let later = session.start(NOW, id("b"));
    assert_eq!(resolved(&later), vec![("b".to_owned(), 1.0)], "answered without a transport");
    assert!(!later.iter().any(|a| matches!(a, EotAction::StartInference(_))));
}

// Catches: a post-fallback failure taking the cloud arm again (the arm follows the current
// model), and repeating the cloud warning on every retry.
#[test]
fn a_failure_after_the_fallback_takes_the_local_arm_and_the_cloud_warning_is_said_once() {
    let mut session = cloud_session(true);
    let first = session.transport_failed(NOW, "socket died");
    assert!(first.contains(&EotAction::SwapToLocal(EotSwapToLocal {})));

    session.start(NOW, id("a"));
    let second = session.transport_failed(NOW, "model died");

    assert!(!second.iter().any(|a| matches!(a, EotAction::SwapToLocal(_))));
    assert!(
        second.contains(&EotAction::Warn(EotWarning::LocalFailed {
            reason: "model died".to_owned()
        }))
    );
    assert!(
        !second
            .iter()
            .any(|a| matches!(a, EotAction::Warn(EotWarning::CloudFailedFallingBack { .. })))
    );
    assert_eq!(resolved(&second), vec![("a".to_owned(), 1.0)]);
    // "defaulting to 1.0 and retrying on next turn" -- nothing closed.
    assert!(!session.is_closed());
    assert!(!session.is_degraded());

    let third = session.transport_failed(NOW, "model died again");
    assert!(third.is_empty(), "the local warning is once per stream too");
}

// Catches: dropping the flush->cancel pairing at `base.py:205`.
#[test]
fn a_flush_closes_the_in_flight_request_and_is_idempotent() {
    let mut session = cloud_session(true);
    session.start(NOW, id("a"));

    let first = session.flush(NOW);
    assert_eq!(first.first(), Some(&EotAction::Flush(EotFlush {})), "the sentinel goes first");
    assert_eq!(resolved(&first), vec![("a".to_owned(), 0.0)]);

    let second = session.flush(NOW);
    assert_eq!(second, vec![EotAction::Flush(EotFlush {})], "a second flush is just another sentinel");
}

// Catches: `end_input` leaving the session open; `base.py:175` answers immediately, positively.
#[test]
fn after_end_input_every_request_is_answered_immediately_and_positively() {
    let mut session = cloud_session(true);
    session.end_input(NOW);
    assert!(session.is_closed());

    let actions = session.start(NOW, id("a"));
    assert_eq!(resolved(&actions), vec![("a".to_owned(), 1.0)]);
    assert_eq!(actions.len(), 1);
    assert!(session.flush(NOW).is_empty(), "and a flush on a closed stream is a no-op");
}

// ---------------------------------------------------------------------------------------------
// request ids
// ---------------------------------------------------------------------------------------------

// Catches: dropping the session-id prefix, or prefixing when there is no session ("None_eot_").
#[test]
fn a_request_id_carries_the_connections_session_id_or_the_fallback_prefix() {
    assert_eq!(EotRequestId::new(Some("inference_abc"), 3).as_str(), "inference_abc_eot_3");
    assert_eq!(EotRequestId::new(None, 3).as_str(), "turn_request_3");
    // `if session_id` is truthiness: an empty value is absent, not a prefix.
    assert_eq!(EotRequestId::new(Some(""), 3).as_str(), "turn_request_3");
}

// ---------------------------------------------------------------------------------------------
// the local model's buffer
// ---------------------------------------------------------------------------------------------

// Catches: evicting the newest audio instead of the oldest, or an oversized frame panicking or
// clearing the buffer (`AudioArrayBuffer` raises).
#[test]
fn the_ring_buffer_keeps_the_newest_audio() {
    let mut buf = AudioRingBuffer::new(4);
    buf.push(&[1, 2, 3]);
    buf.push(&[4, 5]);
    assert_eq!(buf.read(), vec![2, 3, 4, 5]);
    buf.push(&[6, 7, 8, 9, 10]);
    assert_eq!(buf.read(), vec![7, 8, 9, 10]);

    buf.clear();
    assert!(buf.read().is_empty(), "a flush starts the next turn from silence");
}

// ---------------------------------------------------------------------------------------------
// the gateway protocol
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct RecordingSink {
    resolves: Mutex<Vec<(String, TransportPrediction)>>,
    metrics: Mutex<Vec<EotInferenceMetrics>>,
    warnings: Mutex<Vec<EotWarning>>,
    defaults: Mutex<Vec<(BTreeMap<String, f64>, f64)>>,
    reject_defaults: bool,
}

impl PredictionSink for RecordingSink {
    fn resolve(&self, request_id: &str, prediction: TransportPrediction) {
        self.resolves.lock().unwrap().push((request_id.to_owned(), prediction));
    }

    fn session_created(
        &self,
        thresholds: &BTreeMap<String, f64>,
        default: f64,
        _backchannel_thresholds: Option<&BTreeMap<String, f64>>,
        _backchannel_default: f64,
    ) -> Result<(), ThresholdError> {
        if self.reject_defaults {
            return Err(ThresholdError::UnusableDefaults);
        }
        self.defaults.lock().unwrap().push((thresholds.clone(), default));
        Ok(())
    }

    fn metrics(&self, metrics: EotInferenceMetrics) {
        self.metrics.lock().unwrap().push(metrics);
    }

    fn warn(&self, warning: EotWarning) {
        self.warnings.lock().unwrap().push(warning);
    }

    fn metadata(&self) -> Metadata {
        Metadata {
            model_name: Some("turn-detector-v1".to_owned()),
            model_provider: Some("livekit".to_owned()),
        }
    }
}

struct FakeSink {
    sent: Arc<Mutex<Vec<Vec<u8>>>>,
}

#[async_trait]
impl WebSocketSink for FakeSink {
    async fn send(&mut self, data: Vec<u8>) -> Result<(), ApiError> {
        self.sent.lock().unwrap().push(data);
        Ok(())
    }
}

/// One connection's inbound frames, and the headers it was opened with.
type Incoming = tokio::sync::mpsc::UnboundedReceiver<Result<Vec<u8>, ApiError>>;
type Headers = Arc<Mutex<Vec<Vec<(String, String)>>>>;

struct FakeStream {
    incoming: Incoming,
}

#[async_trait]
impl WebSocketStream for FakeStream {
    async fn recv(&mut self) -> Option<Result<Vec<u8>, ApiError>> {
        self.incoming.recv().await
    }
}

struct FakeConnector {
    sent: Arc<Mutex<Vec<Vec<u8>>>>,
    urls: Arc<Mutex<Vec<String>>>,
    headers: Headers,
    incoming: Mutex<Option<Incoming>>,
    /// Never complete the handshake: a gateway that accepts the socket and says nothing.
    hang: bool,
}

#[async_trait]
impl WebSocketConnector for FakeConnector {
    async fn connect(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> Result<(Box<dyn WebSocketSink>, Box<dyn WebSocketStream>), ApiError> {
        self.urls.lock().unwrap().push(url.to_owned());
        self.headers.lock().unwrap().push(headers.to_vec());
        if self.hang {
            std::future::pending::<()>().await;
        }
        let incoming = self.incoming.lock().unwrap().take().expect("one connection per test");
        Ok((Box::new(FakeSink { sent: self.sent.clone() }), Box::new(FakeStream { incoming })))
    }
}

struct CloudFixture {
    transport: Arc<CloudTransport>,
    sent: Arc<Mutex<Vec<Vec<u8>>>>,
    urls: Arc<Mutex<Vec<String>>>,
    headers: Headers,
    server: tokio::sync::mpsc::UnboundedSender<Result<Vec<u8>, ApiError>>,
}

fn cloud_fixture(sink: Arc<RecordingSink>) -> CloudFixture {
    hanging_cloud_fixture(sink, false)
}

fn hanging_cloud_fixture(sink: Arc<RecordingSink>, hang: bool) -> CloudFixture {
    let (server, incoming) = tokio::sync::mpsc::unbounded_channel();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let urls = Arc::new(Mutex::new(Vec::new()));
    let headers = Arc::new(Mutex::new(Vec::new()));
    let connector = Arc::new(FakeConnector {
        sent: sent.clone(),
        urls: urls.clone(),
        headers: headers.clone(),
        incoming: Mutex::new(Some(incoming)),
        hang,
    });
    let transport =
        Arc::new(CloudTransport::new(cloud_options(), sink as Arc<dyn PredictionSink>, connector));
    CloudFixture { transport, sent, urls, headers, server }
}

fn cloud_options() -> CloudOptions {
    CloudOptions {
        base_url: "https://gateway.example/v1".to_owned(),
        api_key: "APIkey".to_owned(),
        api_secret: "secret".to_owned(),
        conn_options: ApiConnectOptions::default(),
        sample_rate: 16_000,
        headers: Arc::new(|| vec![(HEADER_SESSION_ID.to_owned(), "inference_abc".to_owned())]),
    }
}

// Catches: the handshake bounded on tokio's clock or not at all; a stalled handshake is the
// non-retryable `APITimeoutError`, and the cloud->local fallback is the recovery.
#[tokio::test]
async fn a_handshake_that_never_completes_times_out_on_the_session_clock_and_does_not_retry() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let fixture = hanging_cloud_fixture(Arc::new(RecordingSink::default()), true);
    let transport = Arc::try_unwrap(fixture.transport).ok().unwrap().with_clock(clock.clone());

    let (result, ()) = tokio::join!(
        transport.run(),
        crate::testkit::arm_then_advance(&clock, Duration::from_secs(10))
    );
    let error = result.expect_err("the handshake never completed");
    assert_eq!(error.kind, ApiErrorKind::Timeout);
    assert!(!error.retryable, "`_connect_ws` forces retryable=False; the fallback is the recovery");
    assert!(error.message.contains("turn detector connection timed out"), "{}", error.message);
    assert_eq!(fixture.urls.lock().unwrap().len(), 1, "dialed once, not retried");
}

/// Await `f`, but never forever, so a mutant fails the test instead of hanging it.
async fn within<T>(label: &str, f: impl std::future::Future<Output = T>) -> T {
    match tokio::time::timeout(Duration::from_secs(2), f).await {
        Ok(value) => value,
        Err(_) => panic!("timed out waiting for {label}"),
    }
}

/// Spin until `ready()`, but never forever, so a mutant fails the test instead of hanging it.
async fn until(label: &str, mut ready: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if ready() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("timed out waiting for {label}");
}

fn server_message(message: proto::server_message::Message, request_id: &str) -> Vec<u8> {
    proto::ServerMessage {
        server_created_at: None,
        request_id: Some(request_id.to_owned()),
        client_created_at: None,
        message: Some(message),
    }
    .encode_to_vec()
}

fn client_messages(sent: &Arc<Mutex<Vec<Vec<u8>>>>) -> Vec<proto::ClientMessage> {
    sent.lock()
        .unwrap()
        .iter()
        .map(|data| proto::ClientMessage::decode(data.as_slice()).expect("client message"))
        .collect()
}

// Catches: swapped or mis-scaled latency metrics, and losing `backchannel_probability`.
#[tokio::test]
async fn a_prediction_message_resolves_the_request_and_reports_its_metrics() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink.clone());

    // A second in the past, so the three arithmetic paths give distinguishable numbers.
    let sent_at =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
            as i64
            - 1;
    let prediction = proto::EotPrediction {
        probability: 0.75,
        inference_stats: Some(proto::InferenceStats {
            earliest_client_created_at: None,
            latest_client_created_at: Some(pbjson_types::Timestamp { seconds: sent_at, nanos: 0 }),
            client_e2e_latency: Some(pbjson_types::Duration { seconds: 1, nanos: 250_000_000 }),
            server_e2e_latency: Some(pbjson_types::Duration { seconds: 0, nanos: 40_000_000 }),
            preprocessing_duration: None,
            inference_duration: None,
        }),
        backend: proto::eot_prediction::EotBackend::Multimodal as i32,
        backchannel_probability: 0.125,
    };
    fixture
        .server
        .send(Ok(server_message(
            proto::server_message::Message::EotPrediction(prediction),
            "inference_abc_eot_1",
        )))
        .unwrap();
    drop(fixture.server);

    let error = within("the transport", fixture.transport.run())
        .await
        .expect_err("the peer closed unexpectedly");
    assert_eq!(error.kind, ApiErrorKind::Status);
    assert_eq!(error.status_code, Some(-1), "`ws.close_code or -1`");

    let resolves = sink.resolves.lock().unwrap();
    let (request_id, prediction) = resolves.first().expect("one resolution");
    assert_eq!(request_id, "inference_abc_eot_1");
    assert!((prediction.probability - 0.75).abs() < 1e-6);
    assert_eq!(prediction.backchannel_probability.map(|p| (p * 1000.0).round()), Some(125.0));
    assert_eq!(prediction.inference_duration, Some(0.04), "server_e2e_latency, in seconds");
    let detection_delay = prediction.detection_delay.expect("measured against latest_client");
    assert!(
        (0.9..2.0).contains(&detection_delay),
        "now minus the request's stamp, in seconds: {detection_delay}"
    );

    let metrics = sink.metrics.lock().unwrap();
    let metrics = metrics.first().expect("one metrics record");
    assert_eq!(metrics.total_duration, 1.25, "client_e2e_latency, seconds *and* nanos");
    assert_eq!(metrics.prediction_duration, 0.04);
    assert!(
        (0.9..2.0).contains(&metrics.detection_delay),
        "the same number the prediction carries: {}",
        metrics.detection_delay
    );
    assert_eq!(metrics.num_requests, 1);
    assert!(metrics.timestamp > 1_700_000_000.0, "stamped with the wall clock, in seconds");
    assert_eq!(
        metrics.metadata.as_ref().and_then(|m| m.model_name.as_deref()),
        Some("turn-detector-v1")
    );
}

// Catches: ignoring `SessionCreated`; the cloud model ships with no thresholds.
#[tokio::test]
async fn session_created_installs_the_servers_calibrated_defaults() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink.clone());

    let created = proto::SessionCreated {
        default_thresholds: [("en".to_owned(), 0.61_f32)].into_iter().collect(),
        default_threshold: 0.5,
        default_backchannel_thresholds: Default::default(),
        default_backchannel_threshold: 0.0,
    };
    fixture
        .server
        .send(Ok(server_message(proto::server_message::Message::SessionCreated(created), "")))
        .unwrap();
    drop(fixture.server);
    let _ = within("the transport", fixture.transport.run()).await;

    let defaults = sink.defaults.lock().unwrap();
    let (thresholds, default) = defaults.first().expect("one SessionCreated");
    assert_eq!(default, &0.5);
    assert_eq!(thresholds.get("en").map(|t| (t * 100.0).round()), Some(61.0));
}

// Catches: swallowing an unusable `SessionCreated`; it must end the transport, non-retryably.
#[tokio::test]
async fn an_unusable_session_created_kills_the_transport_without_a_retry() {
    let sink = Arc::new(RecordingSink { reject_defaults: true, ..Default::default() });
    let fixture = cloud_fixture(sink);

    fixture
        .server
        .send(Ok(server_message(
            proto::server_message::Message::SessionCreated(proto::SessionCreated::default()),
            "",
        )))
        .unwrap();

    let error = within("the transport", fixture.transport.run())
        .await
        .expect_err("unusable defaults are fatal");
    assert!(!error.retryable);
}

// Catches: treating the gateway's `error` message as data instead of ending the transport.
#[tokio::test]
async fn a_server_error_message_ends_the_transport_with_its_status_code() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink);

    fixture
        .server
        .send(Ok(server_message(
            proto::server_message::Message::Error(proto::InferenceError {
                message: "quota exceeded".to_owned(),
                code: 429,
            }),
            "inference_abc_eot_1",
        )))
        .unwrap();

    let error = within("the transport", fixture.transport.run())
        .await
        .expect_err("an error message is an error");
    assert_eq!(error.status_code, Some(429));
    assert_eq!(error.message, "quota exceeded");
    assert_eq!(error.request_id.as_deref(), Some("inference_abc_eot_1"));
    assert!(!error.retryable, "the reference forces retryable=False on this arm");
}

// Catches: the transport-latency warning's guards (`> 0` for unset, `> 500` for rate); the exact
// boundary is unreachable from a wall clock.
#[tokio::test]
async fn a_slow_round_trip_is_reported_and_an_unstamped_message_is_not() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink.clone());

    let two_seconds_ago =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
            as i64
            - 2;
    let mut slow = proto::ServerMessage {
        server_created_at: None,
        request_id: None,
        client_created_at: Some(pbjson_types::Timestamp { seconds: two_seconds_ago, nanos: 0 }),
        message: Some(proto::server_message::Message::InferenceStarted(proto::InferenceStarted {})),
    };
    fixture.server.send(Ok(slow.encode_to_vec())).unwrap();
    slow.client_created_at = None;
    fixture.server.send(Ok(slow.encode_to_vec())).unwrap();
    drop(fixture.server);
    let _ = within("the transport", fixture.transport.run()).await;

    let warnings = sink.warnings.lock().unwrap();
    assert_eq!(warnings.len(), 1, "the unstamped message must not be reported: {warnings:?}");
    match warnings[0] {
        EotWarning::TransportLatency { millis } => {
            // `as_secs()` truncates, so the stamp is 2 to 3 seconds back.
            assert!((1_990..3_100).contains(&millis), "{millis}ms");
        }
        ref other => panic!("expected a latency warning, got {other:?}"),
    }
}

// Catches: dropping the `closing_ws` flag, so a clean shutdown looks unexpected and falls back.
#[tokio::test]
async fn a_close_after_session_close_is_a_clean_end_and_a_close_before_it_is_not() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink);
    let transport = fixture.transport.clone();
    let sent = fixture.sent.clone();
    let running = tokio::spawn(async move { transport.run().await });

    // Wait for the handshake, so `end_input` has a live send channel to post onto.
    until("the handshake", || fixture.transport.session_id().is_some()).await;
    fixture.transport.end_input();
    until("session_close", || client_messages(&sent).len() >= 2).await;
    drop(fixture.server);

    assert!(within("the transport", running).await.unwrap().is_ok(), "we asked for this close");
    let messages = client_messages(&sent);
    assert!(matches!(messages[0].message, Some(proto::client_message::Message::SessionCreate(_))));
    assert!(matches!(messages[1].message, Some(proto::client_message::Message::SessionClose(_))));
    assert!(messages[1].created_at.is_some(), "the sender stamps what it forwards");
}

// Catches: the wire encoding of audio (the frame's `num_samples`, LE s16, no empty frames) and
// the two control messages.
#[tokio::test]
async fn audio_and_control_messages_go_out_in_the_shape_the_gateway_expects() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink);
    let transport = fixture.transport.clone();
    let sent = fixture.sent.clone();
    let running = tokio::spawn(async move { transport.run().await });
    until("the handshake", || fixture.transport.session_id().is_some()).await;

    fixture.transport.push_frame(&pcm_frame(16_000, &[1, -2]));
    fixture.transport.push_frame(&pcm_frame(16_000, &[]));
    fixture.transport.run_inference(&id("inference_abc_eot_1"));
    fixture.transport.flush();
    until("four client messages", || client_messages(&sent).len() >= 4).await;
    // `detach` drops the send channel, so nothing else reaches a socket we have given up on.
    fixture.transport.detach();
    fixture.transport.flush();
    tokio::task::yield_now().await;
    drop(fixture.server);
    let _ = within("the transport", running).await;

    let messages = client_messages(&sent);
    assert_eq!(messages.len(), 4, "the empty frame was not sent, nor anything after detach");
    match &messages[1].message {
        Some(proto::client_message::Message::InputAudio(audio)) => {
            assert_eq!(audio.audio, vec![0x01, 0x00, 0xFE, 0xFF]);
            assert_eq!(audio.num_samples, 2);
        }
        other => panic!("expected input_audio, got {other:?}"),
    }
    match &messages[2].message {
        Some(proto::client_message::Message::InferenceStart(start)) => {
            assert_eq!(start.request_id, "inference_abc_eot_1");
        }
        other => panic!("expected inference_start, got {other:?}"),
    }
    assert!(matches!(messages[3].message, Some(proto::client_message::Message::SessionFlush(_))));
}

// Catches: an `https://` gateway URL, rewriting more than the scheme, losing the bearer token, or
// minting it with the wrong grant, `sub`, `iss` or a TTL other than `ttl=600`.
#[tokio::test]
async fn the_handshake_uses_a_ws_url_a_bearer_token_and_the_hosts_session_id() {
    let sink = Arc::new(RecordingSink::default());
    let fixture = cloud_fixture(sink);
    let transport = fixture.transport.clone();
    let running = tokio::spawn(async move { transport.run().await });
    until("the handshake", || fixture.transport.session_id().is_some()).await;
    assert_eq!(
        fixture.transport.session_id().as_deref(),
        Some("inference_abc"),
        "taken from the host's headers, and it prefixes every request id"
    );
    drop(fixture.server);
    let _ = within("the transport", running).await;

    assert_eq!(fixture.urls.lock().unwrap()[0], "wss://gateway.example/v1/eot");
    let headers = fixture.headers.lock().unwrap()[0].clone();
    let token = headers
        .iter()
        .find_map(|(name, value)| (name == "Authorization").then(|| value.strip_prefix("Bearer ")))
        .flatten()
        .expect("a bearer token");
    let mut validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);
    validation.validate_aud = false;
    validation.set_required_spec_claims::<&str>(&[]);
    let claims = jsonwebtoken::decode::<serde_json::Value>(
        token,
        &jsonwebtoken::DecodingKey::from_secret(cloud_options().api_secret.as_bytes()),
        &validation,
    )
    .unwrap()
    .claims;
    assert_eq!(claims["inference"]["perform"], serde_json::json!(true));
    assert_eq!(claims["sub"], serde_json::json!("agent"));
    assert_eq!(claims["iss"], serde_json::json!(cloud_options().api_key));
    assert_eq!(claims["exp"].as_u64().unwrap() - claims["nbf"].as_u64().unwrap(), 600);
    assert_eq!(fixture.transport.session_id(), None, "cleared once the connection is gone");
}

/// Records the PCM the model was handed, so the snapshot rule is observable.
struct RecordingModel {
    seen: Arc<Mutex<Vec<Vec<i16>>>>,
    answer: Result<f64, String>,
}

impl LocalEotModel for RecordingModel {
    fn predict(&self, pcm: &[i16]) -> Result<f64, String> {
        self.seen.lock().unwrap().push(pcm.to_vec());
        self.answer.clone()
    }
}

// Catches: the local audio path: no buffering, no clear on flush, snapshot taken at run time,
// a window other than the trailing 1.2 s, or a model failure propagating instead of scoring 0.0.
#[tokio::test]
async fn the_local_transport_scores_the_audio_it_was_holding_when_the_request_was_made() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::new(RecordingSink::default());
    let transport = Arc::new(LocalTransport::new(
        Arc::new(RecordingModel { seen: seen.clone(), answer: Ok(0.9) }),
        sink.clone() as Arc<dyn PredictionSink>,
    ));
    assert_eq!(transport.session_id(), None, "there is no connection to name");

    // 1.3 s at 16 kHz: more than one window, so the model must see exactly the newest 1.2 s.
    let ramp: Vec<i16> = (0..20_800).map(|i| i as i16).collect();
    transport.push_frame(&pcm_frame(16_000, &ramp));
    transport.run_inference(&id("turn_request_1"));
    // Arrives after the request: the snapshot must not contain it.
    transport.push_frame(&pcm_frame(16_000, &[3]));
    transport.flush();
    transport.run_inference(&id("turn_request_2"));
    transport.end_input();

    within("the local transport", transport.run()).await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].len(), 19_200, "one prediction sees 1.2 s at 16 kHz");
    assert_eq!(seen[0], ramp[1_600..], "the newest 1.2 s, not the oldest");
    assert_eq!(seen[1], Vec::<i16>::new(), "flush emptied the buffer");
    let resolves = sink.resolves.lock().unwrap();
    assert_eq!(resolves.len(), 2);
    assert_eq!(resolves[0].0, "turn_request_1");
    assert_eq!(resolves[0].1.probability, 0.9);
    assert!(resolves[0].1.inference_duration.is_some());
}

// Catches: letting a model failure through instead of scoring 0.0.
#[tokio::test]
async fn a_model_that_raises_scores_the_turn_zero_and_says_so() {
    let sink = Arc::new(RecordingSink::default());
    let transport = LocalTransport::new(
        Arc::new(RecordingModel {
            seen: Arc::new(Mutex::new(Vec::new())),
            answer: Err("weights not loaded".to_owned()),
        }),
        sink.clone() as Arc<dyn PredictionSink>,
    );
    transport.run_inference(&id("turn_request_1"));
    transport.end_input();
    within("the local transport", transport.run()).await.unwrap();

    assert_eq!(sink.resolves.lock().unwrap()[0].1.probability, 0.0);
    assert_eq!(
        sink.warnings.lock().unwrap().as_slice(),
        [EotWarning::LocalPredictionFailed { reason: "weights not loaded".to_owned() }]
    );
}

// Catches: two warnings sharing a tag, or a tag the host's `_warn` arm no longer matches.
#[test]
fn every_warning_has_its_own_tag() {
    let all = [
        EotWarning::CloudFailedFallingBack { reason: "x".to_owned() },
        EotWarning::CloudFailedDegraded { reason: "x".to_owned() },
        EotWarning::LocalFailed { reason: "x".to_owned() },
        EotWarning::LocalPredictionFailed { reason: "x".to_owned() },
        EotWarning::TransportLatency { millis: 900 },
        EotWarning::UnexpectedServerMessage,
    ];
    let mut tags: Vec<&str> = all.iter().map(EotWarning::tag).collect();
    tags.sort();
    assert_eq!(
        tags,
        vec![
            "cloud_failed_degraded",
            "cloud_failed_falling_back",
            "local_failed",
            "local_prediction_failed",
            "transport_latency",
            "unexpected_server_message",
        ]
    );
}
