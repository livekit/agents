//! `_CloudTransport` (`transports.py:74`): the gateway WebSocket, protobuf both ways. The socket
//! is a [`WsTransport`]; this file owns the protocol, metrics and error taxonomy.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::{ApiConnectOptions, AudioFrame, Clock, Elapsed, EotInferenceMetrics};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::telemetry;
use lk_agents_net::transport::{
    TransportError, WsMessage, WsReceiver, WsSender, WsTransport, close_status,
};
use parking_lot::Mutex;
use prost::Message as _;

use crate::eot::proto;
use crate::eot::proto::client_message::Message;
use crate::eot::session::TransportPrediction;
use crate::eot::transport::{EotTransport, PredictionListener};
use crate::eot::{EotRequestId, EotWarning};

/// A prediction on old audio is a wrong one: past 500 ms of lag the gateway restarts on the newest.
const INPUT_POLICY: Policy = Policy::Now(std::time::Duration::from_millis(500));

/// `HEADER_SESSION_ID` (`_utils.py:21`): one per connection, and the request-id prefix.
pub const HEADER_SESSION_ID: &str = "X-LiveKit-Session-ID";

pub use crate::headers::HeaderProvider;

/// `_CloudTransportOptions` (`transports.py:63`), apart so the local transport never sees them.
#[derive(Clone)]
pub struct CloudOptions {
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    pub conn_options: ApiConnectOptions,
    pub sample_rate: u32,
    pub headers: HeaderProvider,
}

/// Every socket failure is non-retryable, as in `_connect_ws`: the cloud->local fallback is the
/// recovery.
fn socket_error(err: TransportError) -> ApiError {
    lk_agents_net::transport::connect_error("LiveKit turn detector", err).with_retryable(false)
}

/// `/eot` path (`transports.py:154`).
fn eot_url(base_url: &str) -> String {
    format!("{}/eot", lk_agents_net::transport::to_ws_url(base_url))
}

fn now_timestamp() -> pbjson_types::Timestamp {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    pbjson_types::Timestamp { seconds: now.as_secs() as i64, nanos: now.subsec_nanos() as i32 }
}

fn timestamp_millis(ts: Option<&pbjson_types::Timestamp>) -> i64 {
    // An unset `Timestamp` reads as zero, as in the reference.
    ts.map_or(0, |ts| ts.seconds * 1000 + i64::from(ts.nanos) / 1_000_000)
}

fn duration_millis(d: Option<&pbjson_types::Duration>) -> i64 {
    d.map_or(0, |d| d.seconds * 1000 + i64::from(d.nanos) / 1_000_000)
}

struct CloudState {
    /// The connection's queue: audio, and the control messages in order with it. Open from
    /// construction, so the audio heard while connecting (a pre-connect head) waits in it.
    send: Option<AudioSender<Message>>,
    /// Control messages go only to a live connection.
    connected: bool,
    session_id: Option<String>,
}

pub struct CloudTransport {
    opts: CloudOptions,
    sink: Arc<dyn PredictionListener>,
    transport: Arc<dyn WsTransport>,
    state: Mutex<CloudState>,
    /// The queue's consumer, taken by the connection.
    receiver: Mutex<Option<AudioReceiver<Message>>>,
    clock: Arc<dyn Clock>,
    drops: AudioDrops,
}

impl CloudTransport {
    pub fn new(
        opts: CloudOptions,
        sink: Arc<dyn PredictionListener>,
        transport: Arc<dyn WsTransport>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let (send, receiver) = media_queue::audio(INPUT_POLICY);
        CloudTransport {
            opts,
            sink,
            transport,
            state: Mutex::new(CloudState { send: Some(send), connected: false, session_id: None }),
            receiver: Mutex::new(Some(receiver)),
            clock,
            drops: AudioDrops::new(telemetry::AudioConsumer::EndOfTurn, INPUT_POLICY),
        }
    }

    #[cfg(test)]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// `_send_message` (`transports.py:143`): dropped if there is no live connection. A socket that
    /// stopped draining refuses it too; the prediction it carried times out, and the local model
    /// takes over.
    fn send_message(&self, message: Message) {
        let state = self.state.lock();
        if let Some(send) = state.send.as_ref().filter(|_| state.connected) {
            let _ = send.event(message);
        }
    }

    /// The `eot_prediction` arm of `_process_message` (`transports.py:200`), plus the metrics
    /// record that rides with it.
    fn on_prediction(&self, request_id: &str, prediction: &proto::EotPrediction) {
        let stats = prediction.inference_stats.as_ref();
        let request_sent_at_ms =
            timestamp_millis(stats.and_then(|s| s.latest_client_created_at.as_ref()));
        let detection_delay_ms = timestamp_millis(Some(&now_timestamp())) - request_sent_at_ms;
        let inference_duration_ms =
            duration_millis(stats.and_then(|s| s.server_e2e_latency.as_ref()));

        self.sink.resolve(
            request_id,
            TransportPrediction {
                probability: f64::from(prediction.probability),
                detection_delay: Some(detection_delay_ms as f64 / 1000.0),
                inference_duration: Some(inference_duration_ms as f64 / 1000.0),
                // Always reported by the gateway; `0.0` means "not a backchannel".
                backchannel_probability: Some(f64::from(prediction.backchannel_probability)),
            },
        );

        let client_e2e_ms = duration_millis(stats.and_then(|s| s.client_e2e_latency.as_ref()));
        self.sink.metrics(EotInferenceMetrics {
            timestamp: lk_agents_net::clock::unix_now(),
            total_duration: client_e2e_ms as f64 / 1000.0,
            prediction_duration: inference_duration_ms as f64 / 1000.0,
            detection_delay: detection_delay_ms as f64 / 1000.0,
            num_requests: EotInferenceMetrics::DEFAULT_NUM_REQUESTS,
            metadata: Some(self.sink.metadata()),
        });
    }

    /// `_warn_transport_latency` (`transports.py:190`). The `> 0` guard skips an unset
    /// `client_created_at`.
    fn warn_transport_latency(&self, client_created_at: Option<&pbjson_types::Timestamp>) {
        let stamped = timestamp_millis(client_created_at);
        let latency = timestamp_millis(Some(&now_timestamp())) - stamped;
        if latency > 500 && stamped > 0 {
            self.sink.warn(EotWarning::TransportLatency { millis: latency });
        }
    }

    fn process_message(&self, msg: proto::ServerMessage) -> Result<(), ApiError> {
        let request_id = msg.request_id.clone().unwrap_or_default();
        match msg.message {
            Some(proto::server_message::Message::EotPrediction(prediction)) => {
                self.on_prediction(&request_id, &prediction);
            }
            Some(proto::server_message::Message::SessionCreated(created)) => {
                self.warn_transport_latency(msg.client_created_at.as_ref());
                let thresholds = f32_map(&created.default_thresholds);
                let backchannel = f32_map(&created.default_backchannel_thresholds);
                self.sink
                    .session_created(
                        &thresholds,
                        f64::from(created.default_threshold),
                        Some(&backchannel),
                        f64::from(created.default_backchannel_threshold),
                    )
                    .map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
            }
            Some(proto::server_message::Message::SessionClosed(_))
            | Some(proto::server_message::Message::InferenceStarted(_))
            | Some(proto::server_message::Message::InferenceStopped(_)) => {
                self.warn_transport_latency(msg.client_created_at.as_ref());
            }
            // Never sent here (`type_settings` is unset); ignored like an unknown arm.
            Some(proto::server_message::Message::InterruptionPrediction(_)) => {}
            Some(proto::server_message::Message::Error(error)) => {
                return Err(ApiError::status(
                    error.message,
                    i64::from(error.code),
                    Some(request_id),
                    Some(false),
                ));
            }
            None => self.sink.warn(EotWarning::UnexpectedServerMessage),
        }
        Ok(())
    }

    async fn connect(&self) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), ApiError> {
        let mut headers = (self.opts.headers)();
        let token =
            lk_agents_net::token::inference_bearer(&self.opts.api_key, &self.opts.api_secret)?;
        headers.push(("Authorization".to_owned(), format!("Bearer {token}")));
        let session_id = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(HEADER_SESSION_ID))
            .map(|(_, value)| value.clone());

        let url = eot_url(&self.opts.base_url);
        let connect = self.transport.connect(&url, headers);
        let deadline = self.opts.conn_options.connect_deadline(self.clock.now());
        // On the session clock, not tokio's: the bound a virtual clock can fire and a test can see.
        let (sink, stream) = match lk_agents_abi::before(&*self.clock, deadline, connect).await {
            // Every arm is `retryable=False`, as `_connect_ws` sets it: the fallback is the
            // recovery.
            Err(Elapsed) => {
                let message = Some("turn detector connection timed out".to_owned());
                let error =
                    ApiError { phase: Some(TimeoutPhase::Connect), ..ApiError::timeout(message) };
                return Err(error.with_retryable(false));
            }
            Ok(Err(e)) => return Err(socket_error(e)),
            Ok(Ok(halves)) => halves,
        };

        // Stamped here, which `HasField("created_at")` distinguishes (`transports.py:334`).
        let session_create = proto::ClientMessage {
            created_at: Some(now_timestamp()),
            message: Some(Message::SessionCreate(proto::SessionCreate {
                settings: Some(proto::SessionSettings {
                    sample_rate: self.opts.sample_rate,
                    encoding: proto::AudioEncoding::PcmS16le as i32,
                    // The reference's EOT client sets neither; server defaults apply.
                    type_settings: None,
                }),
            })),
        };
        sink.send_binary(session_create.encode_to_vec()).await.map_err(socket_error)?;

        self.state.lock().session_id = session_id;
        Ok((sink, stream))
    }

    async fn run_once(&self) -> Result<(), ApiError> {
        let (ws_sink, mut ws_stream) = self.connect().await?;
        let mut send_rx = self.receiver.lock().take().ok_or_else(|| {
            ApiError::new("the turn detector connection already ran").with_retryable(false)
        })?;
        self.state.lock().connected = true;

        let closing = AtomicBool::new(false);
        let result = tokio::try_join!(
            self.send_audio(&mut send_rx, &*ws_sink, &closing),
            self.receive_predictions(&mut *ws_stream, &closing),
        )
        .map(|_| ());
        let mut state = self.state.lock();
        state.send = None;
        state.connected = false;
        state.session_id = None;
        result
    }

    /// Writes the queue to the gateway until it ends. A send failure ends the sender quietly:
    /// while closing it is not an error (`sender_task` swallows it).
    async fn send_audio(
        &self,
        send_rx: &mut AudioReceiver<Message>,
        ws_sink: &dyn WsSender,
        closing: &AtomicBool,
    ) -> Result<(), ApiError> {
        loop {
            self.drops.catching_up(send_rx.head(), self.clock.now());
            let Some(item) = send_rx.recv().await else { return Ok(()) };
            let message = client_message(item);
            if matches!(message, Message::SessionClose(_)) {
                closing.store(true, Ordering::SeqCst);
            }
            let msg =
                proto::ClientMessage { created_at: Some(now_timestamp()), message: Some(message) };
            if ws_sink.send_binary(msg.encode_to_vec()).await.is_err() {
                return Ok(());
            }
        }
    }

    /// Reads the gateway's predictions until the socket closes; a close this side asked for ends
    /// it cleanly.
    async fn receive_predictions(
        &self,
        ws_stream: &mut dyn WsReceiver,
        closing: &AtomicBool,
    ) -> Result<(), ApiError> {
        loop {
            match ws_stream.recv().await.map_err(socket_error)? {
                WsMessage::Binary(data) => {
                    let msg = proto::ServerMessage::decode(data.as_slice()).map_err(|e| {
                        ApiError::new(format!("malformed turn detector message: {e}"))
                            .with_retryable(false)
                    })?;
                    self.process_message(msg)?;
                }
                // Text, ping and pong are skipped (tungstenite answers pings).
                WsMessage::Text(_) | WsMessage::Other => {}
                WsMessage::Closed if closing.load(Ordering::SeqCst) => return Ok(()),
                WsMessage::Closed => {
                    return Err(ApiError::status(
                        "turn detector connection closed unexpectedly",
                        close_status(ws_stream.close_code()),
                        None,
                        Some(false),
                    ));
                }
            }
        }
    }
}

/// The wire message for one queued item.
fn client_message(item: Item<AudioFrame, Message>) -> Message {
    match item {
        Item::Media(frame) => Message::InputAudio(proto::InputAudio {
            audio: frame.data.to_vec(),
            created_at: Some(now_timestamp()),
            num_samples: frame.samples_per_channel,
        }),
        // The audio the gateway holds no longer leads up to now: start it over.
        Item::Gap(_) => Message::SessionFlush(proto::SessionFlush {}),
        Item::Event(message) => message,
    }
}

fn f32_map(map: &std::collections::HashMap<String, f32>) -> BTreeMap<String, f64> {
    map.iter().map(|(k, v)| (k.clone(), f64::from(*v))).collect()
}

#[async_trait]
impl EotTransport for CloudTransport {
    fn session_id(&self) -> Option<String> {
        self.state.lock().session_id.clone()
    }

    fn run_inference(&self, request_id: &EotRequestId) {
        self.send_message(Message::InferenceStart(proto::InferenceStart {
            request_id: request_id.as_str().to_owned(),
        }));
    }

    fn push_frame(&self, frame: &AudioFrame) {
        if frame.data.is_empty() {
            return;
        }
        let dropped = self.state.lock().send.as_ref().map(|send| send.push(frame.clone()));
        if let Some(dropped) = dropped {
            self.drops.record(dropped);
        }
    }

    fn push_head(&self, frame: &AudioFrame) {
        if let Some(send) = self.state.lock().send.as_ref() {
            send.head(frame.clone());
        }
    }

    fn observe(&self, cx: telemetry::Cx) {
        self.drops.observe(cx);
    }

    fn flush(&self) {
        self.send_message(Message::SessionFlush(proto::SessionFlush {}));
    }

    fn end_input(&self) {
        self.send_message(Message::SessionClose(proto::SessionClose {}));
        self.state.lock().send = None;
    }

    fn detach(&self) {
        self.state.lock().send = None;
    }

    /// `run` (`transports.py:270`): one attempt, since every failure is non-retryable (the
    /// cloud->local fallback is the recovery).
    async fn run(&self) -> Result<(), ApiError> {
        self.run_once().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_net::websocket::TungsteniteTransport;
    use tokio_tungstenite::tungstenite::Error as WsError;

    // Catches: a connect failure arriving as `retryable = true`, delaying the local fallback.
    #[test]
    fn every_connect_failure_is_non_retryable() {
        for err in [
            TransportError::Timeout,
            TransportError::Connect("dns".to_owned()),
            TransportError::Closed,
            TransportError::Io("reset".to_owned()),
        ] {
            assert!(!socket_error(err).retryable);
        }
    }

    // Catches: a rejected handshake losing its HTTP status.
    #[test]
    fn a_rejected_handshake_keeps_its_status_and_its_body() {
        let response = tokio_tungstenite::tungstenite::http::Response::builder()
            .status(401)
            .body(Some(b"invalid inference grant".to_vec()))
            .expect("a response");
        let err = socket_error(lk_agents_net::websocket::classify_tungstenite(WsError::Http(
            Box::new(response),
        )));
        assert_eq!(err.status_code, Some(401));
        assert!(err.message.contains("invalid inference grant"));
        assert!(!err.retryable);
    }

    #[tokio::test]
    async fn an_unusable_header_fails_the_connection_rather_than_being_skipped() {
        let error = TungsteniteTransport
            .connect("ws://localhost", vec![("invalid\nheader".into(), "value".into())])
            .await
            .map(|_| ())
            .map_err(socket_error)
            .expect_err("invalid header fails before connecting");
        assert!(error.message.contains("header"), "{}", error.message);
        assert!(!error.retryable);
    }
}
