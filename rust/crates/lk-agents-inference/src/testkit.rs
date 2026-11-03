//! Fakes for the [`lk_agents_net::transport`] seam, for tests only (`cfg(test)` or the `testkit` feature).
//! Deterministic; the only clock is the caller's `VirtualClock`.

use async_trait::async_trait;
use lk_agents_net::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use std::sync::Arc;

use parking_lot::Mutex;

/// A recorded `connect` call: the URL and its headers.
pub type ConnectRecord = (String, Vec<(String, String)>);

/// An [`ImageResolver`](lk_agents_abi::llm::provider_format::ImageResolver) that refuses every image.
pub struct NoImages;

impl lk_agents_abi::llm::provider_format::ImageResolver for NoImages {
    fn resolve(
        &self,
        _image: &lk_agents_abi::ImageContent,
    ) -> Result<lk_agents_abi::llm::provider_format::SerializedImage, String> {
        Err("no image resolver is attached to this inference client".to_owned())
    }
}

/// A scripted WebSocket: records sends, replays canned incoming messages unsynchronised with them.
type Incoming = Arc<Mutex<std::collections::VecDeque<Result<WsMessage, TransportError>>>>;

pub struct FakeWs {
    incoming: Incoming,
    sent: Arc<Mutex<Vec<String>>>,
    connects: Mutex<Vec<ConnectRecord>>,
    /// Fail the handshake instead of connecting.
    connect_error: Option<TransportError>,
    /// Park after the script instead of reporting a close.
    keep_open: bool,
    /// Never complete the handshake; the client's connect bound ends it.
    hang_connect: bool,
    /// Never complete a send: a peer that stopped reading, its TCP window full.
    stall_sends: bool,
}

impl FakeWs {
    pub fn new(incoming: Vec<Result<WsMessage, TransportError>>) -> Self {
        FakeWs {
            incoming: Arc::new(Mutex::new(incoming.into_iter().collect())),
            sent: Arc::new(Mutex::new(Vec::new())),
            connects: Mutex::new(Vec::new()),
            connect_error: None,
            keep_open: false,
            hang_connect: false,
            stall_sends: false,
        }
    }

    /// A gateway that accepts the TCP connection and never finishes the WebSocket handshake.
    pub fn hanging() -> Self {
        FakeWs { hang_connect: true, ..FakeWs::new(Vec::new()) }
    }

    /// Stay connected once the script runs out instead of reporting a close (for send-side tests).
    pub fn keep_open(mut self) -> Self {
        self.keep_open = true;
        self
    }

    /// Connected and kept open, but no send ever completes.
    pub fn stalling_sends() -> Self {
        FakeWs { stall_sends: true, ..FakeWs::new(Vec::new()).keep_open() }
    }

    pub fn failing(err: TransportError) -> Self {
        FakeWs { connect_error: Some(err), ..FakeWs::new(vec![]) }
    }

    /// Queue another incoming message after the stream is running. Only meaningful with
    /// [`keep_open`](Self::keep_open).
    pub fn feed(&self, msg: Result<WsMessage, TransportError>) {
        self.incoming.lock().push_back(msg);
    }

    pub fn sent(&self) -> Vec<String> {
        self.sent.lock().clone()
    }

    /// Panics on a non-JSON frame.
    pub fn sent_json(&self) -> Vec<serde_json::Value> {
        self.sent().iter().map(|s| serde_json::from_str(s).expect("a JSON frame")).collect()
    }

    pub fn connect_count(&self) -> usize {
        self.connects.lock().len()
    }

    pub fn last_connect(&self) -> Option<ConnectRecord> {
        self.connects.lock().last().cloned()
    }
}

#[async_trait]
impl WsTransport for FakeWs {
    async fn connect(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        self.connects.lock().push((url.to_owned(), headers));
        if let Some(err) = self.connect_error.clone() {
            return Err(err);
        }
        if self.hang_connect {
            std::future::pending::<()>().await;
        }
        let sender = Arc::new(FakeWsSender {
            sent: self.sent.clone(),
            closed: Default::default(),
            stall: self.stall_sends,
        });
        let receiver =
            Box::new(FakeWsReceiver { incoming: self.incoming.clone(), keep_open: self.keep_open });
        Ok((sender, receiver))
    }
}

struct FakeWsSender {
    sent: Arc<Mutex<Vec<String>>>,
    closed: Mutex<bool>,
    stall: bool,
}

#[async_trait]
impl WsSender for FakeWsSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if *self.closed.lock() {
            return Err(TransportError::Closed);
        }
        if self.stall {
            std::future::pending::<()>().await;
        }
        self.sent.lock().push(text);
        Ok(())
    }

    /// Recorded as its length, so a test can count the audio sent.
    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        if *self.closed.lock() {
            return Err(TransportError::Closed);
        }
        self.sent.lock().push(format!("<{} bytes>", data.len()));
        Ok(())
    }

    fn is_closed(&self) -> bool {
        *self.closed.lock()
    }

    async fn close(&self) {
        *self.closed.lock() = true;
    }
}

struct FakeWsReceiver {
    incoming: Incoming,
    keep_open: bool,
}

#[async_trait]
impl WsReceiver for FakeWsReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        loop {
            if let Some(msg) = self.incoming.lock().pop_front() {
                return msg;
            }
            if !self.keep_open {
                // The default: hang up once the script is done.
                return Ok(WsMessage::Closed);
            }
            // Yield rather than park, so `feed` can still deliver.
            tokio::task::yield_now().await;
        }
    }
}

pub fn text(json: serde_json::Value) -> Result<WsMessage, TransportError> {
    Ok(WsMessage::Text(json.to_string()))
}

/// Let the wait under test arm its deadline, then move the virtual clock past it by `by`.
pub async fn arm_then_advance(clock: &lk_agents_abi::VirtualClock, by: std::time::Duration) {
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    clock.advance(by);
}
pub mod fallback {
    //! Shared fallback provider fixtures, reused by native and composition tests.
    use crate::fallback::{ChainObserver, IgnoreEvents};
    use lk_agents_abi::errors::{ApiError, ProviderError};
    use lk_agents_abi::tokenize::language::LanguageCode;
    use lk_agents_abi::{
        ApiConnectOptions, AudioFrame, BoxFuture, Clock, Closed, Spawner, SpeechEvent,
        SpeechEventType, StreamInput, StreamOutput, Stt, SttCapabilities, SttSink, SttSource,
        SynthesizedAudio, Ts, Tts, TtsCapabilities, TtsSink, TtsSource, VirtualClock,
    };
    use lk_agents_net::task::TokioSpawner;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Every stream closed here is native and ignores the deadline.
    pub const NO_DEADLINE: Ts = Ts::from_nanos(u64::MAX);

    const LIMIT: Duration = Duration::from_secs(5);

    pub async fn within<F: std::future::Future>(what: &str, f: F) -> F::Output {
        match tokio::time::timeout(LIMIT, f).await {
            Ok(v) => v,
            Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
        }
    }

    pub async fn until(what: &str, mut cond: impl FnMut() -> bool) {
        within(what, async {
            while !cond() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await;
    }

    pub fn parts() -> (Arc<dyn Clock>, Arc<dyn Spawner>, Arc<dyn ChainObserver>) {
        (
            Arc::new(VirtualClock::new()) as Arc<dyn Clock>,
            Arc::new(TokioSpawner::current()) as Arc<dyn Spawner>,
            Arc::new(IgnoreEvents) as Arc<dyn ChainObserver>,
        )
    }

    pub fn boom(what: &str) -> ProviderError {
        ProviderError::Local(ApiError::new(what.to_owned()))
    }

    pub struct Frames {
        pub left: usize,
        pub rate: u32,
        pub channels: u16,
        pub dies: bool,
    }

    impl StreamOutput<SynthesizedAudio> for Frames {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
            Box::pin(async move {
                if self.left == 0 {
                    return if self.dies {
                        Err(boom("the provider's socket died"))
                    } else {
                        Ok(None)
                    };
                }
                self.left -= 1;
                // 10 ms of interleaved silence at the provider's own rate.
                let n = (self.rate / 100) as usize;
                let ch = self.channels as usize;
                let mut a = SynthesizedAudio::new(
                    AudioFrame::new(vec![0u8; n * ch * 2], self.rate, self.channels, n as u32),
                    "r",
                );
                a.is_final = self.left == 0 && !self.dies;
                Ok(Some(a))
            })
        }
        fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async move { Closed::Clean })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for Frames {}

    #[derive(Clone)]
    pub struct PolicyProvider {
        pub conn: ApiConnectOptions,
        pub calls: mpsc::Sender<ApiConnectOptions>,
        pub streaming: bool,
    }

    impl PolicyProvider {
        fn record(&self) {
            self.calls.try_send(self.conn).unwrap();
        }
    }

    impl Tts for PolicyProvider {
        fn label(&self) -> String {
            "policy-tts".into()
        }
        fn capabilities(&self) -> TtsCapabilities {
            TtsCapabilities { streaming: self.streaming, aligned_transcript: false }
        }
        fn sample_rate(&self) -> u32 {
            24_000
        }
        fn num_channels(&self) -> u16 {
            1
        }
        fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
            Some(Arc::new(Self { conn: *conn, ..self.clone() }))
        }
        fn synthesize<'a>(
            &'a self,
            _text: &'a str,
        ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
            Box::pin(async move {
                self.record();
                Ok(Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false })
                    as TtsSource)
            })
        }
        fn stream(
            &self,
            _: bool,
            _: lk_agents_abi::telemetry::record::TraceContext,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
        > {
            Box::pin(async move {
                self.record();
                Ok(lk_agents_abi::DuplexStream::new(
                    Box::new(DiscardText) as TtsSink,
                    Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false })
                        as TtsSource,
                ))
            })
        }
    }

    struct DiscardText;
    impl StreamInput<String> for DiscardText {
        fn push(&self, _item: String) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    impl Stt for PolicyProvider {
        fn label(&self) -> String {
            "policy-stt".into()
        }
        fn capabilities(&self) -> SttCapabilities {
            SttCapabilities { streaming: true, ..Default::default() }
        }
        fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
            Some(Arc::new(Self { conn: *conn, ..self.clone() }))
        }
        fn recognize(
            &self,
            _buffer: &[AudioFrame],
            _language: Option<&LanguageCode>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
            Box::pin(async move {
                self.record();
                Ok(SpeechEvent::new(SpeechEventType::FinalTranscript))
            })
        }
        fn stream(
            &self,
            _language: Option<&LanguageCode>,
            _offset: Option<f64>,
            _trace: lk_agents_abi::telemetry::record::TraceContext,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
        > {
            Box::pin(async move {
                self.record();
                Ok(Some(lk_agents_abi::DuplexStream::new(
                    Box::new(DiscardAudio) as SttSink,
                    Box::new(EmptySpeech) as SttSource,
                )))
            })
        }
    }

    struct DiscardAudio;
    impl StreamInput<AudioFrame> for DiscardAudio {
        fn push(&self, _item: AudioFrame) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }
    struct EmptySpeech;
    impl StreamOutput<SpeechEvent> for EmptySpeech {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
            Box::pin(async move { Ok(None) })
        }
        fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move { Closed::Clean })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for EmptySpeech {}

    pub fn policy(read: f64) -> ApiConnectOptions {
        ApiConnectOptions {
            max_retry: 7,
            retry_interval: 0.0,
            pool_timeout: None,
            connect_timeout: Some(0.75),
            write_timeout: Some(0.5),
            read_timeout: Some(read),
            total_timeout: Some(9.0),
        }
    }
}

/// The test's Tokio runtime and wall clock.
pub fn env() -> lk_agents_net::env::ProcessEnv {
    lk_agents_net::env::ProcessEnv::current_runtime()
}
