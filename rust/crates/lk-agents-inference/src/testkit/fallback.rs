//! Shared fallback provider fixtures, reused by native and composition tests.
use crate::fallback::{ChainObserver, IgnoreEvents};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::{
    ApiConnectOptions, AudioBuffer, AudioFrame, BoxFuture, Clock, Closed, Spawner, SpeechEvent,
    SpeechEventType, StreamSink, StreamSource, Stt, SttCapabilities, SttSink, SttSource,
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

impl StreamSource<SynthesizedAudio> for Frames {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if self.left == 0 {
                return if self.dies { Err(boom("the provider's socket died")) } else { Ok(None) };
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
            Ok(Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }) as TtsSource)
        })
    }
    fn stream(&self) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            self.record();
            Ok((
                Box::new(DiscardText) as TtsSink,
                Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }) as TtsSource,
            ))
        })
    }
}

struct DiscardText;
impl StreamSink<String> for DiscardText {
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
        _buffer: AudioBuffer<'_>,
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
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move {
            self.record();
            Ok(Some((Box::new(DiscardAudio) as SttSink, Box::new(EmptySpeech) as SttSource)))
        })
    }
}

struct DiscardAudio;
impl StreamSink<AudioFrame> for DiscardAudio {
    fn push(&self, _item: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}
struct EmptySpeech;
impl StreamSource<SpeechEvent> for EmptySpeech {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move { Ok(None) })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

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
