//! The adapter itself: each chunk its stage prepared, pushed here directly, synthesized on its
//! own. How text becomes those chunks is the stage's, tested in `stage_tests.rs`.

use super::StreamAdapter;
use lk_agents_abi::Tts;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::tts::types::{
    SpeechChunk, SynthesizedAudio, TtsCapabilities, TtsSink, TtsSource,
};
use lk_agents_abi::utils::audio::AudioFrame;
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::stream::{Closed, StreamOutput};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

// Every await is bounded: this file's bugs are hangs, and a bounded hang fails cleanly.
const GUARD: Duration = Duration::from_secs(5);

async fn bounded<F: std::future::Future>(what: &str, f: F) -> F::Output {
    match tokio::time::timeout(GUARD, f).await {
        Ok(v) => v,
        Err(_) => panic!("{what}: blocked forever -- a chunk never reached the source"),
    }
}

/// A chunk as a stage hands it over: `text` for the provider, `said` its words.
fn chunk(text: &str) -> SpeechChunk {
    SpeechChunk { text: text.to_owned(), said: text.to_owned() }
}

fn frame(samples: u32) -> AudioFrame {
    AudioFrame::new(vec![0u8; samples as usize * 2], 24_000, 1, samples)
}

/// A `TtsSource` over a fixed frame list, optionally failing partway.
struct VecSource {
    frames: VecDeque<SynthesizedAudio>,
    fail_after: Option<usize>,
    emitted: usize,
    closed: Arc<AtomicUsize>,
}

impl StreamOutput<SynthesizedAudio> for VecSource {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if self.fail_after == Some(self.emitted) {
                return Err(ApiError::status("rate limited", 429, None, None).into());
            }
            match self.frames.pop_front() {
                Some(a) => {
                    self.emitted += 1;
                    Ok(Some(a))
                }
                None => Ok(None),
            }
        })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.closed.fetch_add(1, Ordering::Relaxed);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for VecSource {}

/// A provider with no duplex path: records the text of every `synthesize()` call.
struct FakeTts {
    frames_per_call: usize,
    calls: Mutex<Vec<String>>,
    inner_closed: Arc<AtomicUsize>,
    fail_after: Option<usize>,
}

impl FakeTts {
    fn new(frames_per_call: usize) -> Arc<Self> {
        Arc::new(FakeTts {
            frames_per_call,
            calls: Mutex::new(Vec::new()),
            inner_closed: Arc::new(AtomicUsize::new(0)),
            fail_after: None,
        })
    }
    fn failing_after(frames_per_call: usize, fail_after: usize) -> Arc<Self> {
        let mut tts = FakeTts::new(frames_per_call);
        Arc::get_mut(&mut tts).expect("fresh").fail_after = Some(fail_after);
        tts
    }
    fn texts(&self) -> Vec<String> {
        self.calls.lock().clone()
    }
}

impl Tts for FakeTts {
    fn label(&self) -> String {
        "fake.FakeTTS".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: false, aligned_transcript: false }
    }
    fn sample_rate(&self) -> Result<u32, lk_agents_abi::errors::HostError> {
        Ok(24_000)
    }
    /// Stereo, so an adapter that hard-coded `1` would show.
    fn num_channels(&self) -> u16 {
        2
    }
    fn synthesize<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            self.calls.lock().push(chunk.text.clone());
            let n = self.calls.lock().len();
            let frames = (0..self.frames_per_call)
                .map(|i| {
                    let mut a = SynthesizedAudio::new(frame(240), format!("inner-{n}"));
                    // The adapter must overwrite both of these.
                    a.is_final = i + 1 == self.frames_per_call;
                    a.segment_id = format!("inner-seg-{n}");
                    a
                })
                .collect();
            Ok(Box::new(VecSource {
                frames,
                fail_after: self.fail_after,
                emitted: 0,
                closed: self.inner_closed.clone(),
            }) as TtsSource)
        })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        Box::pin(async move { Err(lk_agents_abi::tts::types::no_duplex_path(&self.label())) })
    }
}

fn adapter(tts: Arc<FakeTts>) -> StreamAdapter {
    StreamAdapter::new(tts, Arc::new(lk_agents_abi::utils::clock::VirtualClock::new()))
}

async fn open(adapter: &StreamAdapter) -> (TtsSink, TtsSource) {
    adapter.stream(&lk_agents_abi::telemetry::Cx::NONE).await.expect("opens").split()
}

/// `VecSource` closes at once, so the deadline is a formality.
const NO_DEADLINE: Ts = Ts::from_nanos(u64::MAX);

/// A source still producing after this many frames is not terminating; bounded by count
/// because a spinning `next()` has no await to time out on.
const MAX_FRAMES: usize = 64;

async fn drain(mut source: TtsSource) -> Result<Vec<SynthesizedAudio>, ProviderError> {
    bounded("drain", async move {
        let mut out = Vec::new();
        loop {
            match source.next().await? {
                Some(a) => {
                    out.push(a);
                    assert!(
                        out.len() <= MAX_FRAMES,
                        "the source is still emitting after {MAX_FRAMES} frames: it never \
                         reaches end of stream"
                    );
                }
                None => return Ok(out),
            }
        }
    })
    .await
}

// Catches: forwarding the inner `is_final` (or setting it per chunk): one TTS metric per chunk
// and wrong TTFBs.
#[tokio::test]
async fn only_the_last_frame_of_the_whole_stream_is_final() {
    let tts = FakeTts::new(2);
    let (sink, source) = open(&adapter(tts)).await;
    for text in ["one.", "two.", "three."] {
        sink.push(chunk(text)).unwrap();
    }
    sink.end_input();

    let out = drain(source).await.unwrap();
    assert_eq!(out.len(), 6, "three chunks, two frames each");
    let finals: Vec<bool> = out.iter().map(|a| a.is_final).collect();
    assert_eq!(finals, vec![false, false, false, false, false, true]);
}

fn tts_records(records: Vec<AgentMetrics>) -> Vec<(u32, bool, bool)> {
    let view = |m: AgentMetrics| match m {
        AgentMetrics::Tts(t) => (t.characters_count, t.cancelled, t.streamed),
        other => panic!("not a TTS record: {other:?}"),
    };
    records.into_iter().map(view).collect()
}

// Catches: one record for the whole stream instead of one per chunk (the one cut by the close
// as cancelled), and a failed chunk leaving one.
#[tokio::test]
async fn each_chunk_is_metered_as_its_own_call() {
    let (sink, mut source) = open(&adapter(FakeTts::new(2))).await;
    sink.push(chunk("one.")).unwrap();
    sink.push(chunk("three.")).unwrap();
    for _ in 0..3 {
        bounded("frame", source.next()).await.unwrap().expect("a frame");
    }
    // The first chunk ended when the second's first frame came.
    assert_eq!(tts_records(source.take_call_metrics(false).unwrap()), vec![(4, false, false)]);
    assert_eq!(tts_records(source.take_call_metrics(true).unwrap()), vec![(6, true, false)]);
    assert_eq!(source.take_call_metrics(true).unwrap(), vec![], "each record is read once");

    let (sink, mut source) = open(&adapter(FakeTts::failing_after(2, 1))).await;
    sink.push(chunk("one.")).unwrap();
    sink.end_input();
    assert!(drain_until_error(&mut source).await);
    assert_eq!(source.take_call_metrics(true).unwrap(), vec![]);
}

async fn drain_until_error(source: &mut TtsSource) -> bool {
    for _ in 0..MAX_FRAMES {
        match bounded("frame", source.next()).await {
            Err(_) => return true,
            Ok(None) => return false,
            Ok(Some(_)) => {}
        }
    }
    false
}

// Catches: holding every frame for the `is_final` lookahead.
#[tokio::test]
async fn a_frame_is_not_held_back_while_more_chunks_may_still_arrive() {
    let (sink, mut source) = open(&adapter(FakeTts::new(4))).await;
    sink.push(chunk("hello world.")).unwrap();

    // No `end_input`: the adapter cannot know this is the last chunk, so it must not hold.
    let first = bounded("first frame", source.next())
        .await
        .unwrap()
        .expect("the first frame must not be held back");
    assert!(!first.is_final);
    sink.end_input();
    let rest = drain(source).await.unwrap();
    assert_eq!(rest.len(), 3);
    assert!(rest.last().unwrap().is_final);
}

// Catches: reporting the adapter's own audio format instead of the wrapped provider's, or
// `capabilities.streaming` false (a second adapter would wrap this one).
#[tokio::test]
async fn the_adapter_claims_streaming_but_keeps_the_wrapped_providers_audio_format() {
    let tts = FakeTts::new(1);
    let ad = adapter(tts.clone());
    assert_eq!(
        ad.capabilities(),
        TtsCapabilities { streaming: true, aligned_transcript: true },
        "stream_adapter.py:33 verbatim"
    );
    assert_eq!(ad.sample_rate(), tts.sample_rate());
    assert_eq!(ad.num_channels(), tts.num_channels());
    assert!(ad.label().contains(&tts.label()), "the label must name what it wraps: {}", ad.label());
    assert_ne!(ad.label(), tts.label(), "and must not impersonate it -- the chain keys on labels");
}

// Catches: synthesizing a whitespace-only chunk, or a skipped chunk stealing the final marker.
#[tokio::test]
async fn a_whitespace_only_chunk_is_not_synthesized() {
    let tts = FakeTts::new(1);
    let (sink, source) = open(&adapter(tts.clone())).await;
    sink.push(chunk("real text.")).unwrap();
    sink.push(chunk("   ")).unwrap();
    sink.end_input();

    let out = drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["real text."], "the blank chunk must not be synthesized");
    assert!(out.last().unwrap().is_final, "a queued blank must not steal the final marker");
}

// Catches: a wrong `start_time` clock, attaching a chunk's words to the previous frame, or
// aligning the provider's text (its markup) instead of the words it says.
#[tokio::test]
async fn a_chunks_words_ride_its_first_frame_at_the_audio_before_it() {
    let (sink, source) = open(&adapter(FakeTts::new(2))).await; // 2 x 240 samples = 20 ms
    sink.push(SpeechChunk { text: "[laugh] one.".into(), said: "one.".into() }).unwrap();
    sink.push(chunk("two.")).unwrap();
    sink.end_input();

    let out = drain(source).await.unwrap();
    let stamped: Vec<(usize, &str, Option<f64>)> = out
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            a.timed_transcript.iter().map(move |t| (i, t.text.as_str(), t.start_time))
        })
        .collect();
    assert_eq!(stamped, vec![(0, "one.", Some(0.0)), (2, "two.", Some(0.02))]);
}

// Catches: letting the inner provider's `request_id`/`segment_id` through.
#[tokio::test]
async fn every_frame_carries_the_adapters_ids_not_the_inner_providers() {
    let (sink, source) = open(&adapter(FakeTts::new(2))).await;
    sink.push(chunk("one.")).unwrap();
    sink.push(chunk("two.")).unwrap();
    sink.end_input();

    let out = drain(source).await.unwrap();
    let first = &out[0];
    assert!(!first.request_id.is_empty() && !first.segment_id.is_empty());
    assert_ne!(first.request_id, first.segment_id, "two ids, minted separately");
    for a in &out {
        assert_eq!(a.request_id, first.request_id);
        assert_eq!(a.segment_id, first.segment_id);
        assert!(!a.segment_id.starts_with("inner-"), "the inner segment id leaked");
    }
}

// Catches: a push after `end_input` raising, or being synthesized.
#[tokio::test]
async fn a_push_after_end_input_is_a_silent_no_op() {
    let tts = FakeTts::new(1);
    let (sink, source) = open(&adapter(tts.clone())).await;
    sink.push(chunk("first.")).unwrap();
    sink.end_input();
    assert!(sink.push(chunk("too late.")).is_ok());
    assert!(sink.flush().is_ok());

    drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["first."]);
}

// Catches: swallowing an inner failure, emitting the held frame after it, or leaving a spent
// chunk retryable (the engine would retry the whole reply on top of its budget).
#[tokio::test]
async fn an_inner_failure_surfaces_as_a_final_error() {
    let (sink, mut source) = open(&adapter(FakeTts::failing_after(3, 1))).await;
    sink.push(chunk("boom.")).unwrap();
    sink.end_input();

    // The first frame is held, so the error is the first thing the caller sees.
    let err = bounded("failing frame", source.next())
        .await
        .expect_err("the 429 must not become a clean EOS");
    assert!(!err.retryable());
    assert_eq!(err.status_code(), Some(429));
}

fn retrying_adapter(tts: Arc<FakeTts>) -> StreamAdapter {
    let budget = lk_agents_abi::ApiConnectOptions {
        max_retry: 2,
        retry_interval: 0.0,
        ..lk_agents_abi::ApiConnectOptions::default()
    };
    StreamAdapter::new(tts, Arc::new(lk_agents_net::clock::SimClock::new())).with_budget(budget)
}

// Catches: a chunk whose synthesis failed before any audio not synthesized again within the
// budget, or retried past it.
#[tokio::test(start_paused = true)]
async fn a_chunk_failing_before_audio_is_synthesized_again_within_its_budget() {
    let tts = FakeTts::failing_after(3, 0);
    let (sink, source) = open(&retrying_adapter(tts.clone())).await;
    sink.push(chunk("first.")).unwrap();
    sink.push(chunk("second.")).unwrap();
    sink.end_input();
    let err = drain(source).await.expect_err("every attempt fails");
    assert!(!err.retryable());
    assert_eq!(tts.texts(), vec!["first.", "first.", "first."]);
}

// Catches: resynthesizing a chunk whose audio was already delivered: the listener would hear its
// beginning twice. The delivered frame stays delivered and the stream fails.
#[tokio::test(start_paused = true)]
async fn a_partial_failure_never_resynthesizes_delivered_audio() {
    let tts = FakeTts::failing_after(3, 1);
    let (sink, mut source) = open(&retrying_adapter(tts.clone())).await;
    sink.push(chunk("first.")).unwrap();
    sink.push(chunk("second.")).unwrap();
    sink.end_input();
    let delivered = bounded("first frame", source.next()).await.unwrap();
    assert!(delivered.is_some(), "the first frame of the chunk reaches the listener");
    let err = bounded("failure", source.next()).await.expect_err("partial output fails");
    assert!(!err.retryable());
    assert_eq!(err.status_code(), Some(429));
    assert_eq!(tts.texts(), vec!["first."], "delivered audio is never synthesized again");
}

// Catches: dropping the inner `TtsSource` instead of closing it (leaks a host task per chunk).
#[tokio::test]
async fn every_inner_stream_is_aclosed_including_on_teardown() {
    let tts = FakeTts::new(1);
    let closed = tts.inner_closed.clone();
    let (sink, mut source) = open(&adapter(tts)).await;
    sink.push(chunk("one.")).unwrap();
    sink.push(chunk("two.")).unwrap();
    bounded("frame 1", source.next()).await.unwrap(); // the only frame of "one."
    bounded("frame 2", source.next()).await.unwrap(); // exhausts "one.", closes it, starts "two."
    assert_eq!(closed.load(Ordering::Relaxed), 1, "an exhausted inner stream is closed");

    // Tear down mid-reply, the interruption shape.
    let how = bounded("close", source.close(NO_DEADLINE)).await;
    assert_eq!(closed.load(Ordering::Relaxed), 2, "the in-flight inner stream is closed too");
    assert_eq!(how, Closed::Clean, "every inner stream acknowledged, so the adapter does too");
}

// Catches: a sink that cannot be shared between the pump and drain tasks.
#[tokio::test]
async fn the_two_halves_drive_from_two_tasks_at_once() {
    let tts = FakeTts::new(1);
    let (sink, source) = open(&adapter(tts.clone())).await;
    let sink: Arc<TtsSink> = Arc::new(sink);
    let feeder = {
        let sink = sink.clone();
        tokio::spawn(async move {
            for i in 0..8 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                sink.push(chunk(&format!("s{i}."))).unwrap();
            }
            sink.end_input();
        })
    };
    let out = drain(source).await.unwrap();
    bounded("feeder", feeder).await.unwrap();
    assert_eq!(out.len(), 8, "nothing dropped between the two tasks");
    assert_eq!(tts.texts().len(), 8);
}

// Catches: a source that loops on its input without an await (a chunk that is never audio),
// wedging the worker it runs on. A wedged executor can only be reported from outside it: hence
// the second runtime, `shutdown_timeout`, and the explicit exit.
#[test]
fn a_source_cannot_wedge_the_worker_it_runs_on() {
    let rt =
        tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let outcome = rt.block_on(async {
        let (sink, source) = open(&adapter(FakeTts::new(1))).await;
        sink.push(chunk("   ")).unwrap();
        sink.push(chunk("one sentence.")).unwrap();
        sink.end_input();
        let drained = tokio::spawn(drain(source));
        // Its own generous bound: failure takes the process down.
        tokio::time::timeout(Duration::from_secs(10), drained).await
    });
    rt.shutdown_timeout(Duration::from_millis(0));

    let Ok(drained) = outcome else {
        eprintln!(
            "a_source_cannot_wedge_the_worker_it_runs_on: the drain never yielded -- \
             `raw_next` is spinning on a chunk that is never audio, and every other test in this \
             binary is stuck behind it"
        );
        // 101 is libtest's own failure code. Panicking would be correct and invisible.
        std::process::exit(101);
    };
    let out = drained.expect("the drain task panicked").expect("the drain failed");
    assert_eq!(out.len(), 1, "one chunk, one frame, and then end of stream");
}

// Catches: a chunk's TTFB measured from the stream's open (or its first push) instead of from
// its submission to the provider. https://github.com/livekit/agents/pull/6300
#[tokio::test]
async fn ttfb_runs_from_the_chunks_submission() {
    let clock = Arc::new(lk_agents_abi::utils::clock::VirtualClock::new());
    let adapter = StreamAdapter::new(FakeTts::new(2), clock.clone());
    let (sink, mut source) = open(&adapter).await;
    // the stage took 2 s to prepare the chunk
    clock.advance(Duration::from_secs(2));
    sink.push(chunk("one two.")).unwrap();
    sink.end_input();
    while bounded("frame", source.next()).await.unwrap().is_some() {}

    let ttfbs: Vec<Option<f64>> = source
        .take_call_metrics(true)
        .unwrap()
        .into_iter()
        .map(|m| match m {
            AgentMetrics::Tts(t) => t.ttfb,
            other => panic!("not a TTS record: {other:?}"),
        })
        .collect();
    assert_eq!(ttfbs.len(), 1, "{ttfbs:?}");
    let ttfb = ttfbs[0].expect("a first frame");
    assert!(ttfb < 0.5, "TTFB {ttfb} includes the 2 s before the chunk was submitted");
}
