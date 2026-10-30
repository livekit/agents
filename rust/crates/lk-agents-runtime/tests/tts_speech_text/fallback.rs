//! TTS fallback integration using the production speech factory.
use super::FallbackTts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Closed, StreamSink, StreamSource, SynthesizedAudio, Ts, Tts,
    TtsCapabilities, TtsSink, TtsSource, VirtualClock,
};
use lk_agents_inference::fallback::ChainPolicy;
use lk_agents_inference::testkit::fallback::{Frames, NO_DEADLINE, boom, parts, until, within};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

async fn drain_tts(source: &mut TtsSource) -> Vec<SynthesizedAudio> {
    let mut out = Vec::new();
    for _ in 0..64 {
        match within("the next frame", source.next()).await {
            Ok(None) => return out,
            Ok(Some(a)) => out.push(a),
            Err(e) => panic!("the source failed: {e}"),
        }
    }
    panic!("the source never reported its end");
}

struct FakeTts {
    label: String,
    rate: u32,
    channels: u16,
    frames: usize,
    /// `stream()` refuses this many times before it works; `usize::MAX` never comes back.
    refusals: AtomicUsize,
    /// The source raises after its scripted frames instead of ending cleanly.
    dies: bool,
    pushed: Arc<Mutex<Vec<String>>>,
    /// Without a duplex path, `stream()` refuses.
    streaming: bool,
}

struct RecordingSink(Arc<Mutex<Vec<String>>>);

impl StreamSink<String> for RecordingSink {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        self.0.lock().push(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.0.lock().push("<flush>".into());
        Ok(())
    }
    fn end_input(&self) {
        self.0.lock().push("<end>".into());
    }
}

impl Tts for FakeTts {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: self.streaming, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn num_channels(&self) -> u16 {
        self.channels
    }
    fn synthesize<'a>(
        &'a self,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            self.pushed.lock().push(text.to_owned());
            Ok(Box::new(self.frames()) as TtsSource)
        })
    }
    fn stream(
        &self,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            if !self.streaming {
                return Err(boom("no duplex path"));
            }
            let left = self.refusals.load(Ordering::SeqCst);
            if left > 0 {
                self.refusals.store(left.saturating_sub(1), Ordering::SeqCst);
                return Err(boom("no socket"));
            }
            Ok((
                Box::new(RecordingSink(Arc::clone(&self.pushed))) as TtsSink,
                Box::new(self.frames()) as TtsSource,
            ))
        })
    }
}

impl FakeTts {
    fn frames(&self) -> Frames {
        Frames { left: self.frames, rate: self.rate, channels: self.channels, dies: self.dies }
    }
}

fn built(
    label: &str,
    rate: u32,
    channels: u16,
    frames: usize,
    refusals: usize,
    dies: bool,
) -> (Arc<FakeTts>, Arc<Mutex<Vec<String>>>) {
    let pushed = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(FakeTts {
            label: label.into(),
            rate,
            channels,
            frames,
            refusals: AtomicUsize::new(refusals),
            dies,
            pushed: Arc::clone(&pushed),
            streaming: true,
        }),
        pushed,
    )
}

// Catches: a member without a duplex path failing every streamed request (`tts:384-395`).
#[tokio::test]
async fn a_batch_only_member_streams_through_sentences() {
    let pushed = Arc::new(Mutex::new(Vec::new()));
    let batch = Arc::new(FakeTts {
        label: "batch".into(),
        rate: 24_000,
        channels: 1,
        frames: 1,
        refusals: AtomicUsize::new(0),
        dies: false,
        pushed: Arc::clone(&pushed),
        streaming: false,
    });
    let fb = tts_chain(vec![batch]);
    assert!(!fb.capabilities().streaming, "the members as given, not as adapted");
    assert_eq!(fb.label(), "tts.FallbackAdapter");

    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("Hello there, this is the first sentence. ".into()).expect("push");
    sink.push("And this one is the second sentence.".into()).expect("push");
    sink.end_input();
    assert!(!drain_tts(&mut source).await.is_empty(), "the member served the stream");
    assert_eq!(pushed.lock().len(), 2, "one synthesis per sentence");
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: dropping `_pushed_tokens` replay on failover (`tts:428-434`).
#[tokio::test]
async fn a_failover_replays_the_text_already_pushed_into_the_new_provider() {
    let (broken, _) = built("broken", 24_000, 1, 0, usize::MAX, false);
    let (good, good_pushed) = built("good", 24_000, 1, 2, 0, false);
    let fb = tts_chain(vec![broken, good]);

    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("Hello there, ".into()).expect("push");
    sink.flush().expect("flush");
    sink.push("how are you?".into()).expect("push");
    sink.end_input();

    let frames = drain_tts(&mut source).await.len();
    assert_eq!(frames, 2, "the healthy provider served the whole utterance");
    assert_eq!(
        good_pushed.lock().as_slice(),
        &["Hello there, ".to_string(), "<flush>".into(), "how are you?".into(), "<end>".into()],
        "everything pushed before the failover is replayed, in order"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

fn tts_chain(entries: Vec<Arc<dyn Tts>>) -> FallbackTts {
    let (clock, spawner, obs) = parts();
    FallbackTts::new(
        entries,
        None,
        ChainPolicy {
            max_retry_per_backend: 0,
            retry_interval: Some(0.0),
            ..ChainPolicy::tts_defaults()
        },
        clock,
        spawner,
        obs
    )
    .expect("a usable list")
}

// Catches: a streaming probe that replays nothing from `snapshot` (`tts:504-531`).
#[tokio::test]
async fn a_streaming_probe_replays_the_text_recorded_so_far() {
    let (flaky, flaky_pushed) = built("flaky", 24_000, 1, 2, 1, false);
    let (good, _) = built("good", 24_000, 1, 2, 0, false);
    let fb = tts_chain(vec![flaky.clone(), good]);

    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("Hello there".into()).expect("push");
    sink.flush().expect("flush");
    sink.end_input();
    drain_tts(&mut source).await;

    until("the probe to bring the primary back", || fb.chain().is_available(0)).await;
    assert_eq!(
        flaky_pushed.lock().as_slice(),
        &["Hello there".to_string(), "<flush>".into(), "<end>".into()],
        "the probe replays the recorded tokens, in order, into a private channel"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

/// A backup that speaks only once its input ends, like a real provider waiting for the text: the
/// request it serves cannot finish before the caller's last token.
struct SpeaksOnEnd;

struct EndSignal(Arc<tokio::sync::Notify>);

impl StreamSink<String> for EndSignal {
    fn push(&self, _item: String) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {
        self.0.notify_one();
    }
}

struct AfterEnd {
    ended: Arc<tokio::sync::Notify>,
    frames: Option<Frames>,
}

impl StreamSource<SynthesizedAudio> for AfterEnd {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if self.frames.is_none() {
                self.ended.notified().await;
                self.frames = Some(Frames { left: 1, rate: 24_000, channels: 1, dies: false });
            }
            self.frames.as_mut().expect("just set").next().await
        })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl Tts for SpeaksOnEnd {
    fn label(&self) -> String {
        "backup".into()
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
        _text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            Ok(Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }) as TtsSource)
        })
    }
    fn stream(
        &self,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            let ended = Arc::new(tokio::sync::Notify::new());
            Ok((
                Box::new(EndSignal(Arc::clone(&ended))) as TtsSink,
                Box::new(AfterEnd { ended, frames: None }) as TtsSource,
            ))
        })
    }
}

// Catches: probing a skipped entry as the scan passes it, before any text has arrived, so the
// streamed probe declines for want of text and the provider stays down for good (#6678,
// `tts:417-420`).
#[tokio::test]
async fn an_entry_skipped_on_a_streamed_request_is_probed_with_that_requests_text() {
    let (flaky, flaky_pushed) = built("flaky", 24_000, 1, 1, 2, false);
    let fb = tts_chain(vec![flaky.clone(), Arc::new(SpeaksOnEnd)]);

    // First request: the attempt and its probe are both refused, so "flaky" stays down.
    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("one".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;
    source.close(NO_DEADLINE).await;
    until("the first probe to be refused", || flaky.refusals.load(Ordering::SeqCst) == 0).await;
    assert!(!fb.chain().is_available(0));
    flaky_pushed.lock().clear();

    // Second request: "flaky" is skipped before the caller has pushed anything.
    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    tokio::time::sleep(Duration::from_millis(20)).await;
    sink.push("two".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;

    until("the skipped entry to be probed and recover", || fb.chain().is_available(0)).await;
    assert_eq!(
        flaky_pushed.lock().as_slice(),
        &["two".to_string(), "<end>".into()],
        "the probe replays the text this request received"
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: a TTS adapter `aclose` that does not reach the chain, or an exhaustion error not naming
// the list.
#[tokio::test]
async fn a_closed_tts_adapter_starts_no_more_probes_and_a_total_outage_names_the_list() {
    let (alpha, alpha_pushed) = built("alpha", 24_000, 1, 1, 1, false);
    let (beta, _) = built("beta", 24_000, 1, 1, 0, false);
    let fb = tts_chain(vec![alpha.clone(), beta]);

    fb.aclose();
    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    assert!(within("closed refusal", source.next()).await.is_err());
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert!(
        alpha_pushed.lock().is_empty(),
        "a probe started after aclose is a task nothing will ever cancel"
    );
    source.close(NO_DEADLINE).await;

    // And with every entry refusing, the error has to say who was tried.
    let (a, _) = built("alpha", 24_000, 1, 0, usize::MAX, false);
    let (b, _) = built("beta", 24_000, 1, 0, usize::MAX, false);
    let fb = tts_chain(vec![a, b]);
    let (sink, mut source) = within("stream", fb.stream(false)).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    let err = within("the failure", source.next()).await.expect_err("both entries refused");
    assert!(
        err.message().contains("all backends failed (alpha, beta)"),
        "the message must name the list, in order: {}",
        err.message()
    );
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

mod options;
mod tts;
