//! Sentence adapter conformance with the production speech text transformations.

use lk_agents_abi::Tts;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::tts::stream_adapter::SentenceSplitter;
use lk_agents_abi::tts::stream_adapter::SplitterFactory;
use lk_agents_abi::tts::tts::{SynthesizedAudio, TtsCapabilities, TtsSink, TtsSource};
use lk_agents_abi::utils::audio::AudioFrame;
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::stream::{Closed, StreamSource};
use lk_agents_inference::tts::sentence_adapter::StreamAdapter;
use lk_agents_voice::tts::markup::{Markup, MarkupExt};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use std::sync::atomic::AtomicUsize;
use std::time::Duration;

// Every await is bounded: this file's bugs are hangs, and a bounded hang fails cleanly.
const GUARD: Duration = Duration::from_secs(5);

async fn bounded<F: std::future::Future>(what: &str, f: F) -> F::Output {
    match tokio::time::timeout(GUARD, f).await {
        Ok(v) => v,
        Err(_) => panic!("{what}: blocked forever -- a token never reached the source"),
    }
}

// ---------------------------------------------------------------- fakes

/// Splits on '.', keeping the delimiter, and buffers the tail. Records every call so a test
/// can assert what the sink forwarded.
#[derive(Default)]
struct DotSplitter {
    buf: String,

    seen: Arc<Mutex<Vec<String>>>,

    flushes: Arc<AtomicUsize>,
}

impl SentenceSplitter for DotSplitter {
    fn push(&mut self, text: &str) -> Vec<String> {
        self.seen.lock().unwrap().push(text.to_owned());
        self.buf.push_str(text);
        let mut out = Vec::new();
        while let Some(i) = self.buf.find('.') {
            out.push(self.buf.drain(..=i).collect());
        }
        out
    }
    fn flush(&mut self) -> Vec<String> {
        self.flushes.fetch_add(1, Ordering::Relaxed);
        if self.buf.is_empty() { Vec::new() } else { vec![std::mem::take(&mut self.buf)] }
    }
}

/// A `StreamAdapter` plus the two observation points its splitter exposes.
struct Observed {
    adapter: StreamAdapter,

    seen: Arc<Mutex<Vec<String>>>,

    flushes: Arc<AtomicUsize>,
}

fn observed(tts: Arc<FakeTts>) -> Observed {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let flushes = Arc::new(AtomicUsize::new(0));
    let (s, f) = (seen.clone(), flushes.clone());
    Observed {
        adapter: StreamAdapter::new(
            tts,
            Arc::new(move |_| {
                Box::new(DotSplitter { buf: String::new(), seen: s.clone(), flushes: f.clone() })
            }),
            lk_agents_voice::tts::markup::text_processor_factory(),
            Arc::new(lk_agents_abi::utils::clock::VirtualClock::new()),
        ),
        seen,
        flushes,
    }
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

impl StreamSource<SynthesizedAudio> for VecSource {
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

struct FakeTts {
    /// Frames per `synthesize()` call, and the text it was called with.
    frames_per_call: usize,

    calls: Mutex<Vec<String>>,

    inner_closed: Arc<AtomicUsize>,
    fail_after: Option<usize>,
    /// `markup._provider_key()`.
    markup: &'static str,
}

impl FakeTts {
    fn new(frames_per_call: usize) -> Arc<Self> {
        Arc::new(FakeTts {
            frames_per_call,
            calls: Mutex::new(Vec::new()),
            inner_closed: Arc::new(AtomicUsize::new(0)),
            fail_after: None,
            markup: "",
        })
    }
    fn texts(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl Tts for FakeTts {
    fn label(&self) -> String {
        "fake.FakeTTS".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: false, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
    /// Stereo, so an adapter that hard-coded `1` would show.
    fn num_channels(&self) -> u16 {
        2
    }
    fn markup_provider(&self) -> String {
        self.markup.to_owned()
    }
    fn synthesize<'a>(
        &'a self,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push(text.to_owned());
            let n = self.calls.lock().unwrap().len();
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
        _: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move { Err(lk_agents_abi::tts::tts::no_duplex_path(&self.label())) })
    }
}

fn adapter(tts: Arc<FakeTts>) -> StreamAdapter {
    StreamAdapter::new(
        tts,
        Arc::new(|_| Box::new(DotSplitter::default())),
        lk_agents_voice::tts::markup::text_processor_factory(),
        Arc::new(lk_agents_abi::utils::clock::VirtualClock::new()),
    )
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

// ---------------------------------------------------------------- tests

// Catches: forwarding the inner `is_final` (or setting it per sentence): one TTS metric per
// sentence and wrong TTFBs.
#[tokio::test]
async fn only_the_last_frame_of_the_whole_stream_is_final() {
    let tts = FakeTts::new(2);
    let (sink, source) = adapter(tts).stream(false).await.unwrap();
    sink.push("one. two. three.".into()).unwrap();
    sink.end_input();

    let out = drain(source).await.unwrap();
    assert_eq!(out.len(), 6, "three sentences, two frames each");
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

// Catches: one record for the whole stream: the reference's adapter forwards each sentence's
// `ChunkedStream` metrics (one non-streamed TTSMetrics per sentence, the one cut by the close
// as cancelled), and a failed sentence leaves none.
#[tokio::test]
async fn each_sentence_is_metered_as_its_own_call() {
    let tts = FakeTts::new(2);
    let (sink, mut source) = adapter(tts).stream(false).await.unwrap();
    sink.push("one. three.".into()).unwrap();
    for _ in 0..3 {
        bounded("frame", source.next()).await.unwrap().expect("a frame");
    }
    // The first sentence ended when the second's first frame came.
    assert_eq!(tts_records(source.take_call_metrics(false).unwrap()), vec![(4, false, false)]);
    assert_eq!(tts_records(source.take_call_metrics(true).unwrap()), vec![(6, true, false)]);
    assert_eq!(source.take_call_metrics(true).unwrap(), vec![], "each record is read once");

    let mut failing = FakeTts::new(2);
    Arc::get_mut(&mut failing).unwrap().fail_after = Some(1);
    let (sink, mut source) = adapter(failing).stream(false).await.unwrap();
    sink.push("one.".into()).unwrap();
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
async fn a_frame_is_not_held_back_while_more_text_may_still_arrive() {
    let tts = FakeTts::new(4);
    let (sink, mut source) = adapter(tts).stream(false).await.unwrap();
    sink.push("hello world.".into()).unwrap();

    // No `end_input`: the adapter cannot know this is the last sentence, so it must not hold.
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

// Catches: closing the input channel before flushing the splitter (last partial sentence lost).
#[tokio::test]
async fn end_input_flushes_the_partial_sentence_before_closing() {
    let tts = FakeTts::new(1);
    let (sink, source) = adapter(tts.clone()).stream(false).await.unwrap();
    sink.push("complete. trailing without a dot".into()).unwrap();
    sink.end_input();

    drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["complete.", "trailing without a dot"]);
}

// Catches: treating an empty push as data.
#[tokio::test]
async fn an_empty_push_never_reaches_the_splitter() {
    let tts = FakeTts::new(1);
    let obs = observed(tts.clone());
    let (sink, source) = obs.adapter.stream(false).await.unwrap();
    sink.push("half".into()).unwrap();
    sink.push(String::new()).unwrap();
    sink.push(" a sentence.".into()).unwrap();
    sink.end_input();

    drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["half a sentence."]);
    // What the output alone cannot show: an empty push reaching the splitter.
    assert_eq!(*obs.seen.lock().unwrap(), vec!["half", " a sentence."]);
}

// Catches: a no-op `flush()`, and a `push` guard of `&&` instead of `||`.
#[tokio::test]
async fn flush_forces_the_buffered_partial_out_mid_stream() {
    let tts = FakeTts::new(1);
    let obs = observed(tts.clone());
    let (sink, source) = obs.adapter.stream(false).await.unwrap();
    sink.push("no delimiter here".into()).unwrap();
    sink.flush().unwrap();
    sink.push("second".into()).unwrap();
    sink.end_input();

    drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["no delimiter here", "second"]);
    // The explicit `flush()`, `end_input()`'s, and the source's safety-net flush on disconnect.
    assert_eq!(obs.flushes.load(Ordering::Relaxed), 3);
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

// Catches: synthesizing a whitespace-only sentence, and dropping its alignment.
#[tokio::test]
async fn a_whitespace_only_sentence_is_aligned_but_not_synthesized() {
    let tts = FakeTts::new(1);
    let (sink, source) = adapter(tts.clone()).stream(false).await.unwrap();
    sink.push("real text.".into()).unwrap();
    sink.push("   ".into()).unwrap(); // no delimiter: `end_input`'s flush emits it as-is
    sink.end_input();

    let out = drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["real text."], "the blank sentence must not be synthesized");
    let words: Vec<&str> =
        out.iter().flat_map(|a| a.timed_transcript.iter().map(|t| t.text.as_str())).collect();
    assert_eq!(words, vec!["real text.", "   "], "both tokens must reach the alignment");
    assert!(out.last().unwrap().is_final, "a queued blank must not steal the final marker");
}

// Catches: a wrong `start_time` clock, or attaching the transcript to the previous frame.
#[tokio::test]
async fn the_timed_transcript_rides_the_next_frame_and_carries_the_preceding_duration() {
    let tts = FakeTts::new(2); // 2 x 240 samples @ 24 kHz = 20 ms per sentence
    let (sink, source) = adapter(tts).stream(false).await.unwrap();
    sink.push("one. two.".into()).unwrap();
    sink.end_input();

    let out = drain(source).await.unwrap();
    let stamped: Vec<(usize, &str, Option<f64>)> = out
        .iter()
        .enumerate()
        .flat_map(|(i, a)| {
            a.timed_transcript.iter().map(move |t| (i, t.text.as_str(), t.start_time))
        })
        .collect();
    assert_eq!(stamped, vec![(0, "one.", Some(0.0)), (2, " two.", Some(0.02))]);
}

// Catches: letting the inner provider's `request_id`/`segment_id` through.
#[tokio::test]
async fn every_frame_carries_the_adapters_ids_not_the_inner_providers() {
    let tts = FakeTts::new(2);
    let (sink, source) = adapter(tts).stream(false).await.unwrap();
    sink.push("one. two.".into()).unwrap();
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

// Catches: two streams from one adapter sharing a splitter.
#[tokio::test]
async fn each_stream_gets_its_own_splitter_state() {
    let tts = FakeTts::new(1);
    let ad = adapter(tts.clone());

    let (sink_a, source_a) = ad.stream(false).await.unwrap();
    sink_a.push("leftover with no delimiter".into()).unwrap();
    drop(sink_a); // dropped, not ended: the source flushes what it can and stops
    drain(source_a).await.unwrap();

    let (sink_b, source_b) = ad.stream(false).await.unwrap();
    sink_b.push("fresh.".into()).unwrap();
    sink_b.end_input();
    drain(source_b).await.unwrap();

    assert_eq!(tts.texts(), vec!["leftover with no delimiter", "fresh."]);
}

// Catches: a push after `end_input` raising, or being synthesized.
#[tokio::test]
async fn a_push_after_end_input_is_a_silent_no_op() {
    let tts = FakeTts::new(1);
    let (sink, source) = adapter(tts.clone()).stream(false).await.unwrap();
    sink.push("first.".into()).unwrap();
    sink.end_input();
    assert!(sink.push("too late.".into()).is_ok());
    assert!(sink.flush().is_ok());

    drain(source).await.unwrap();
    assert_eq!(tts.texts(), vec!["first."]);
}

// Catches: swallowing an inner failure, emitting the held frame after it, or leaving a spent
// sentence retryable (the engine would retry the whole reply on top of its budget).
#[tokio::test]
async fn an_inner_failure_surfaces_as_a_final_error() {
    let mut tts = FakeTts::new(3);
    Arc::get_mut(&mut tts).unwrap().fail_after = Some(1);
    let (sink, mut source) = adapter(tts).stream(false).await.unwrap();
    sink.push("boom.".into()).unwrap();
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
    let clock = Arc::new(lk_agents_net::clock::SimClock::new());
    let splitter: SplitterFactory = Arc::new(|_| Box::new(DotSplitter::default()));
    StreamAdapter::new(tts, splitter, lk_agents_voice::tts::markup::text_processor_factory(), clock)
        .with_budget(budget)
}

// Catches: a sentence whose synthesis failed before any audio not synthesized again within
// the budget, or retried past it.
#[tokio::test(start_paused = true)]
async fn a_sentence_failing_before_audio_is_synthesized_again_within_its_budget() {
    let mut tts = FakeTts::new(3);
    Arc::get_mut(&mut tts).unwrap().fail_after = Some(0);
    let (sink, source) = retrying_adapter(tts.clone()).stream(false).await.unwrap();
    sink.push("first. second.".into()).unwrap();
    sink.end_input();
    let err = drain(source).await.expect_err("every attempt fails");
    assert!(!err.retryable());
    assert_eq!(tts.texts(), vec!["first.", "first.", "first."]);
}

// Catches: resynthesizing a sentence whose audio was already delivered (F-10): the listener
// would hear its beginning twice. The delivered frame stays delivered and the stream fails.
#[tokio::test(start_paused = true)]
async fn sentence_adapter_partial_failure_never_resynthesizes_delivered_audio() {
    let mut tts = FakeTts::new(3);
    Arc::get_mut(&mut tts).unwrap().fail_after = Some(1);
    let (sink, mut source) = retrying_adapter(tts.clone()).stream(false).await.unwrap();
    sink.push("first. second.".into()).unwrap();
    sink.end_input();
    let delivered = bounded("first frame", source.next()).await.unwrap();
    assert!(delivered.is_some(), "the first frame of the sentence reaches the listener");
    let err = bounded("failure", source.next()).await.expect_err("partial output fails");
    assert!(!err.retryable());
    assert_eq!(err.status_code(), Some(429));
    assert_eq!(tts.texts(), vec!["first."], "delivered audio is never synthesized again");
}

// Catches: dropping the inner `TtsSource` instead of closing it (leaks a host task per sentence).
#[tokio::test]
async fn every_inner_stream_is_aclosed_including_on_teardown() {
    let tts = FakeTts::new(1);
    let closed = tts.inner_closed.clone();
    let (sink, mut source) = adapter(tts).stream(false).await.unwrap();
    sink.push("one. two.".into()).unwrap();
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
    let (sink, source) = adapter(tts.clone()).stream(false).await.unwrap();
    let sink: Arc<TtsSink> = Arc::new(sink);
    let feeder = {
        let sink = sink.clone();
        tokio::spawn(async move {
            for i in 0..8 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                sink.push(format!("s{i}.")).unwrap();
            }
            sink.end_input();
        })
    };
    let out = drain(source).await.unwrap();
    bounded("feeder", feeder).await.unwrap();
    assert_eq!(out.len(), 8, "nothing dropped between the two tasks");
    assert_eq!(tts.texts().len(), 8);
}

// Catches: a `next_sentence` that always returns a sentence, which spins `raw_next` with no
// await. A wedged executor can only be reported from outside it: hence the second runtime,
// `shutdown_timeout`, and the explicit exit.

#[test]
fn a_source_that_never_reaches_a_real_sentence_cannot_wedge_the_worker_it_runs_on() {
    let rt =
        tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let outcome = rt.block_on(async {
        let tts = FakeTts::new(1);
        let (sink, source) = adapter(tts).stream(false).await.unwrap();
        sink.push("one sentence.".into()).unwrap();
        sink.end_input();
        let drained = tokio::spawn(drain(source));
        // Its own generous bound: failure takes the process down.
        tokio::time::timeout(Duration::from_secs(10), drained).await
    });
    rt.shutdown_timeout(Duration::from_millis(0));

    let Ok(drained) = outcome else {
        eprintln!(
            "a_source_that_never_reaches_a_real_sentence_cannot_wedge_the_worker_it_runs_on: \
             the drain never yielded -- `raw_next` is spinning on a sentence that is never \
             audio, and every other test in this binary is stuck behind it"
        );
        // 101 is libtest's own failure code. Panicking would be correct and invisible.
        std::process::exit(101);
    };
    let out = drained.expect("the drain task panicked").expect("the drain failed");
    assert_eq!(out.len(), 1, "one sentence, one frame, and then end of stream");
}

// Catches: markup spoken aloud by a non-streaming TTS under expressive (the adapter is its
// only lowering path), a sentence split inside a marker (the splitter not asked to be
// xml-aware), and a synthesis with expressive off lowered.

#[tokio::test]
async fn under_expressive_each_sentence_is_lowered_to_the_providers_markup() {
    let said = "<expr type=\"emotion\" label=\"happy\"/>Hi there.";
    let markup = Markup::from_provider_key("cartesia");
    for on in [true, false] {
        let tts =
            Arc::new(FakeTts { markup: "cartesia", ..Arc::into_inner(FakeTts::new(1)).unwrap() });
        let xml_aware = Arc::new(AtomicBool::new(!on));
        let seen = xml_aware.clone();
        let adapter = StreamAdapter::new(
            tts.clone(),
            Arc::new(move |xml| {
                seen.store(xml, Ordering::Relaxed);
                Box::new(DotSplitter::default())
            }),
            lk_agents_voice::tts::markup::text_processor_factory(),
            Arc::new(lk_agents_abi::utils::clock::VirtualClock::new()),
        );
        assert!(adapter.lowers_markup());
        assert_eq!(adapter.markup_provider(), "cartesia");
        let (sink, source) = adapter.stream(on).await.unwrap();
        sink.push(said.into()).unwrap();
        sink.end_input();
        let out = drain(source).await.unwrap();
        assert_eq!(xml_aware.load(Ordering::Relaxed), on, "xml-aware only while markup flows");
        let want = if on {
            markup.convert(&markup.normalize(said)).trim().to_owned()
        } else {
            said.to_owned()
        };
        assert_eq!(tts.texts(), vec![want], "expressive={on}");
        // The transcript keeps the framework's markup; the sinks strip it downstream.
        assert_eq!(out[0].timed_transcript[0].text, said);
        if on {
            assert_ne!(tts.texts()[0], said, "cartesia lowers `expr` to its own tags");
        }
    }
}
