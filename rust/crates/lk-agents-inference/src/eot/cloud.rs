//! `_CloudTransport` (`transports.py:74`): the gateway WebSocket, protobuf both ways. The socket
//! is a [`WebSocketConnector`]; this file owns the protocol, metrics and error taxonomy.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ApiErrorKind};
use lk_agents_abi::{ApiConnectOptions, AudioFrame, Clock, Elapsed, EotInferenceMetrics, RealClock};
use prost::Message as _;
use tokio::sync::mpsc;

use crate::eot::{EotRequestId, EotWarning};
use crate::eot::proto;
use crate::eot::session::TransportPrediction;
use crate::eot::transport::{EotTransport, PredictionSink};

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

pub use lk_agents_abi::{WebSocketSink, WebSocketStream};

#[async_trait]
pub trait WebSocketConnector: Send + Sync {
    async fn connect(
        &self,
        url: &str,
        headers: &[(String, String)],
    ) -> Result<(Box<dyn WebSocketSink>, Box<dyn WebSocketStream>), ApiError>;
}

/// `/eot` path (`transports.py:154`).
fn eot_url(base_url: &str) -> String {
    format!("{}/eot", crate::transport::to_ws_url(base_url))
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

fn wall_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

#[derive(Default)]
struct CloudState {
    send: Option<mpsc::UnboundedSender<proto::ClientMessage>>,
    session_id: Option<String>,
}

pub struct CloudTransport {
    opts: CloudOptions,
    sink: Arc<dyn PredictionSink>,
    connector: Arc<dyn WebSocketConnector>,
    state: Mutex<CloudState>,
    clock: Arc<dyn Clock>,
}

impl CloudTransport {
    pub fn new(
        opts: CloudOptions,
        sink: Arc<dyn PredictionSink>,
        connector: Arc<dyn WebSocketConnector>,
    ) -> Self {
        CloudTransport {
            opts,
            sink,
            connector,
            state: Mutex::new(CloudState::default()),
            clock: Arc::new(RealClock::new()),
        }
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// `_send_message` (`transports.py:143`): dropped if there is no live connection.
    fn send_message(&self, message: proto::client_message::Message) {
        let state = self.state.lock().expect("cloud state mutex");
        if let Some(send) = state.send.as_ref() {
            let _ = send.send(proto::ClientMessage { created_at: None, message: Some(message) });
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
            timestamp: wall_secs(),
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
                    .map_err(|e| ApiError {
                        kind: ApiErrorKind::Api,
                        message: e.to_string(),
                        retryable: false,
                        status_code: None,
                        request_id: None,
                    })?;
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

    async fn connect(
        &self,
    ) -> Result<(Box<dyn WebSocketSink>, Box<dyn WebSocketStream>), ApiError> {
        let mut headers = (self.opts.headers)();
        // `create_access_token(api_key, api_secret, ttl=600)` (`_utils.py:133`).
        let token = crate::token::create_access_token(
            &self.opts.api_key,
            &self.opts.api_secret,
            600,
            crate::token::now_unix(),
        )
        .map_err(|e| ApiError {
            kind: ApiErrorKind::Api,
            message: format!("failed to mint the turn detector access token: {e}"),
            retryable: false,
            status_code: None,
            request_id: None,
        })?;
        headers.push(("Authorization".to_owned(), format!("Bearer {token}")));
        let session_id = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(HEADER_SESSION_ID))
            .map(|(_, value)| value.clone());

        let url = eot_url(&self.opts.base_url);
        let connect = self.connector.connect(&url, &headers);
        let timeout = Duration::from_secs_f64(self.opts.conn_options.timeout.max(0.0));
        // On the session clock, not tokio's: the bound a virtual clock can fire and a test can see.
        let (mut sink, stream) = match lk_agents_abi::timeout(&*self.clock, timeout, connect).await
        {
            // Every arm is `retryable=False`, as `_connect_ws` sets it: the fallback is the
            // recovery.
            Err(Elapsed) => {
                return Err(ApiError::timeout(Some(
                    "turn detector connection timed out".to_owned(),
                ))
                .with_retryable(false));
            }
            Ok(Err(e)) => return Err(e),
            Ok(Ok(halves)) => halves,
        };

        // Stamped here, which `HasField("created_at")` distinguishes (`transports.py:334`).
        let session_create = proto::ClientMessage {
            created_at: Some(now_timestamp()),
            message: Some(proto::client_message::Message::SessionCreate(proto::SessionCreate {
                settings: Some(proto::SessionSettings {
                    sample_rate: self.opts.sample_rate,
                    encoding: proto::AudioEncoding::PcmS16le as i32,
                    // The reference's EOT client sets neither; server defaults apply.
                    type_settings: None,
                }),
            })),
        };
        sink.send(session_create.encode_to_vec()).await?;

        self.state.lock().expect("cloud state mutex").session_id = session_id;
        Ok((sink, stream))
    }

    async fn run_once(&self) -> Result<(), ApiError> {
        let (mut ws_sink, mut ws_stream) = self.connect().await?;
        let (send, mut send_rx) = mpsc::unbounded_channel();
        self.state.lock().expect("cloud state mutex").send = Some(send);

        let closing = AtomicBool::new(false);

        let sender = async {
            while let Some(mut msg) = send_rx.recv().await {
                if msg.created_at.is_none() {
                    msg.created_at = Some(now_timestamp());
                }
                let closes =
                    matches!(msg.message, Some(proto::client_message::Message::SessionClose(_)));
                if closes {
                    closing.store(true, Ordering::SeqCst);
                }
                // A send failure while closing is not an error (`sender_task` swallows it).
                if ws_sink.send(msg.encode_to_vec()).await.is_err() {
                    return Ok(());
                }
            }
            Ok(())
        };

        let receiver = async {
            loop {
                match ws_stream.recv().await {
                    Some(Ok(data)) => {
                        let msg = proto::ServerMessage::decode(data.as_slice()).map_err(|e| {
                            ApiError::new(format!("malformed turn detector message: {e}"))
                                .with_retryable(false)
                        })?;
                        self.process_message(msg)?;
                    }
                    Some(Err(e)) => return Err(e),
                    None => {
                        if closing.load(Ordering::SeqCst) {
                            return Ok(());
                        }
                        return Err(ApiError::status(
                            "turn detector connection closed unexpectedly",
                            // `ws.close_code or -1`: the sink has no close code.
                            -1,
                            None,
                            Some(false),
                        ));
                    }
                }
            }
        };

        let result = tokio::try_join!(sender, receiver).map(|_| ());
        let mut state = self.state.lock().expect("cloud state mutex");
        state.send = None;
        state.session_id = None;
        result
    }
}

fn f32_map(map: &std::collections::HashMap<String, f32>) -> BTreeMap<String, f64> {
    map.iter().map(|(k, v)| (k.clone(), f64::from(*v))).collect()
}

#[async_trait]
impl EotTransport for CloudTransport {
    fn session_id(&self) -> Option<String> {
        self.state.lock().expect("cloud state mutex").session_id.clone()
    }

    fn run_inference(&self, request_id: &EotRequestId) {
        self.send_message(proto::client_message::Message::InferenceStart(proto::InferenceStart {
            request_id: request_id.as_str().to_owned(),
        }));
    }

    fn push_frame(&self, frame: &AudioFrame) {
        if frame.data.is_empty() {
            return;
        }
        self.send_message(proto::client_message::Message::InputAudio(proto::InputAudio {
            audio: frame.data.to_vec(),
            created_at: Some(now_timestamp()),
            num_samples: frame.samples_per_channel,
        }));
    }

    fn flush(&self) {
        self.send_message(proto::client_message::Message::SessionFlush(proto::SessionFlush {}));
    }

    fn end_input(&self) {
        self.send_message(proto::client_message::Message::SessionClose(proto::SessionClose {}));
        self.state.lock().expect("cloud state mutex").send = None;
    }

    fn detach(&self) {
        self.state.lock().expect("cloud state mutex").send = None;
    }

    /// `run` (`transports.py:270`): one attempt, since every failure is non-retryable (the
    /// cloud->local fallback is the recovery).
    async fn run(&self) -> Result<(), ApiError> {
        self.run_once().await
    }
}
