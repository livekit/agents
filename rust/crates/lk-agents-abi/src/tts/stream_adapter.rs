//! [`StreamAdapter`] (`tts/stream_adapter.py`): a non-streaming TTS as a duplex one. Frames pass
//! through; `is_final` needs a one-frame hold, taken only once no further sentence can arrive.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::TimedString;
use crate::errors::{ApiError, ProviderError};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::tts::tts::{SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource};
use crate::types::identity::ProviderId;
use crate::utils::clock::{Clock, Ts};
use crate::utils::stream::{Closed, HOST_TEARDOWN_TIMEOUT, StreamSink, StreamSource};

/// No default implementation on purpose: segmentation boundaries are audible, and the reference's
/// default is blingfire's, not a hand-rolled punctuation splitter.
pub trait SentenceSplitter: Send {
    /// Partial text stays buffered.
    fn push(&mut self, text: &str) -> Vec<String>;
    /// Hard boundary: whatever is buffered, complete or not.
    fn flush(&mut self) -> Vec<String>;
}

pub type SplitterFactory = Arc<dyn Fn() -> Box<dyn SentenceSplitter> + Send + Sync>;

/// `utils.shortuuid()`'s shape; ids only need to be distinct within a session.
fn stream_uid() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("{:012x}", NEXT.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------- StreamAdapter

pub struct StreamAdapter {
    tts: Arc<dyn Tts>,
    splitter: SplitterFactory,
    /// Bounds the per-sentence inner closes done inside `next()`.
    clock: Arc<dyn Clock>,
}

impl StreamAdapter {
    pub fn new(tts: Arc<dyn Tts>, splitter: SplitterFactory, clock: Arc<dyn Clock>) -> Self {
        StreamAdapter { tts, splitter, clock }
    }

    /// Itself when it has a duplex path, else adapted; decided per instance by its capabilities.
    pub fn adapted(
        tts: Arc<dyn Tts>,
        splitter: SplitterFactory,
        clock: Arc<dyn Clock>,
    ) -> Arc<dyn Tts> {
        if tts.capabilities().streaming {
            return tts;
        }
        Arc::new(StreamAdapter::new(tts, splitter, clock))
    }
}

/// Each push is one sentence: the host ran its own tokenizer and pacer.
struct PreSplit;

impl SentenceSplitter for PreSplit {
    fn push(&mut self, text: &str) -> Vec<String> {
        vec![text.to_owned()]
    }
    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// `tts.StreamAdapter`'s engine: `provider` under `conn_options(max_retry, interval, timeout)`,
/// fed sentences the host already split, answering the duplex stream.
#[crate::host(export(name = "tts_adapter_stream", stream = crate::tts::tts::FallbackTtsStream))]
pub async fn adapter_stream(
    provider: Arc<dyn Tts>,
    maximum: u32,
    interval: f64,
    timeout: f64,
    #[host(ambient)] clock: Arc<dyn Clock>,
) -> Result<(TtsSink, TtsSource), ProviderError> {
    let options = crate::ApiConnectOptions::new(maximum, interval, timeout).map_err(|e| {
        ProviderError::Local(ApiError { retryable: false, ..ApiError::new(e.to_string()) })
    })?;
    let provider = provider.with_conn_options(&options).unwrap_or(provider);
    StreamAdapter::new(provider, Arc::new(|| Box::new(PreSplit)), clock).stream().await
}

#[async_trait]
impl Tts for StreamAdapter {
    fn label(&self) -> String {
        format!("StreamAdapter[{}]", self.tts.label())
    }

    /// The adapter is transparent to the metrics.
    fn id(&self) -> ProviderId {
        self.tts.id()
    }

    fn prewarm(&self) {
        self.tts.prewarm();
    }

    /// Verbatim from `stream_adapter.py:33`.
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: true }
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn num_channels(&self) -> u16 {
        self.tts.num_channels()
    }

    /// `stream_adapter.py:58`.
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        self.tts.synthesize(text).await
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        // Unbounded on purpose: text -> TTS may not drop, a lost token mangles speech.
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = AdapterSink { tx: Mutex::new(Some(tx)), closed: AtomicBool::new(false) };
        let source = AdapterSource {
            tts: self.tts.clone(),
            clock: Arc::clone(&self.clock),
            rx,
            splitter: (self.splitter)(),
            pending: VecDeque::new(),
            current: None,
            inner_timed_out: false,
            input_done: false,
            duration: 0.0,
            pending_transcript: Vec::new(),
            held: None,
            request_id: stream_uid(),
            segment_id: stream_uid(),
        };
        Ok((Box::new(sink) as TtsSink, Box::new(source) as TtsSource))
    }
}

enum AdapterInput {
    Text(String),
    Flush,
}

struct AdapterSink {
    tx: Mutex<Option<mpsc::UnboundedSender<AdapterInput>>>,
    closed: AtomicBool,
}

impl AdapterSink {
    fn send(&self, item: AdapterInput) -> Result<(), ProviderError> {
        let guard = self.tx.lock().expect("adapter sink poisoned");
        match guard.as_ref() {
            // The source was dropped without `aclose`. Nothing will ever read this.
            None => Ok(()),
            Some(tx) => tx
                .send(item)
                .map_err(|_| ApiError::new("stream adapter: the source half is gone").into()),
        }
    }
}

impl StreamSink<String> for AdapterSink {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        // `tts/tts.py:744`: an empty token or a closed input is a silent no-op.
        if item.is_empty() || self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        self.send(AdapterInput::Text(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        // `tts/tts.py:771`: silent once the input channel is closed.
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        self.send(AdapterInput::Flush)
    }

    fn end_input(&self) {
        // `tts/tts.py:785`: `flush()` then `close()`, in that order.
        let _ = self.flush();
        self.closed.store(true, Ordering::Release);
        drop(self.tx.lock().expect("adapter sink poisoned").take());
    }
}

struct AdapterSource {
    tts: Arc<dyn Tts>,
    clock: Arc<dyn Clock>,
    rx: mpsc::UnboundedReceiver<AdapterInput>,
    splitter: Box<dyn SentenceSplitter>,
    pending: VecDeque<String>,
    current: Option<TtsSource>,
    /// Folded into `close`.
    inner_timed_out: bool,
    input_done: bool,
    /// The `start_time` stamped on each sentence's `TimedString`.
    duration: f64,
    /// Ride on the next frame sent (`tts/tts.py:1104`).
    pending_transcript: Vec<TimedString>,
    held: Option<SynthesizedAudio>,
    /// One request and one segment for the whole stream (`stream_adapter.py:96`).
    request_id: String,
    segment_id: String,
}

impl AdapterSource {
    /// Bounded on the session clock; a timeout is remembered for `close`.
    async fn close_current(&mut self) {
        if let Some(done) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            match done.close(deadline).await {
                Closed::Clean => {}
                Closed::TimedOut => self.inner_timed_out = true,
            }
        }
    }

    /// Without awaiting. Stands in for the reference's concurrent `_forward_input`: otherwise the
    /// adapter would learn input ended only after the last frame, and never mark it final.
    fn drain_input(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(AdapterInput::Text(t)) => self.pending.extend(self.splitter.push(&t)),
                Ok(AdapterInput::Flush) => self.pending.extend(self.splitter.flush()),
                Err(mpsc::error::TryRecvError::Empty) => return,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if !self.input_done {
                        self.input_done = true;
                        self.pending.extend(self.splitter.flush());
                    }
                    return;
                }
            }
        }
    }

    async fn next_sentence(&mut self) -> Option<String> {
        loop {
            self.drain_input();
            if let Some(s) = self.pending.pop_front() {
                return Some(s);
            }
            if self.input_done {
                return None;
            }
            match self.rx.recv().await {
                Some(AdapterInput::Text(t)) => self.pending.extend(self.splitter.push(&t)),
                Some(AdapterInput::Flush) => self.pending.extend(self.splitter.flush()),
                None => {
                    // Sender dropped. A sink dropped without `end_input` never flushed, so flush
                    // again rather than lose the last half-sentence.
                    self.input_done = true;
                    self.pending.extend(self.splitter.flush());
                }
            }
        }
    }

    async fn raw_next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        loop {
            if let Some(cur) = self.current.as_mut() {
                match cur.next().await {
                    Ok(Some(a)) => return Ok(Some(a)),
                    Ok(None) => {
                        // Dropping a host stream does not cancel its task; close it, bounded.
                        self.close_current().await;
                        continue;
                    }
                    // Close a failed inner source too, or it counts as open after the retry
                    // decision.
                    Err(e) => {
                        self.close_current().await;
                        return Err(e);
                    }
                }
            }
            let Some(sentence) = self.next_sentence().await else { return Ok(None) };

            // The transcript uses the unstripped token and precedes the empty check
            // (`stream_adapter.py:125`).
            self.pending_transcript.push(TimedString {
                text: sentence.clone(),
                start_time: Some(self.duration),
                ..TimedString::default()
            });
            // Whitespace-only is falsy in Python; several plugins raise on empty input.
            if sentence.trim_matches(crate::tokenize::is_py_space).is_empty() {
                continue;
            }
            self.current = Some(
                self.tts.synthesize(sentence.trim_matches(crate::tokenize::is_py_space)).await?,
            );
        }
    }
}

#[async_trait]
impl StreamSource<SynthesizedAudio> for AdapterSource {
    async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        loop {
            match self.raw_next().await {
                Ok(Some(mut frame)) => {
                    self.drain_input();
                    self.duration += frame.frame.duration();
                    // A sentence boundary is not a segment boundary (`tts/tts.py:1183`): the inner
                    // provider's ids and `is_final` are not ours.
                    frame.is_final = false;
                    frame.request_id.clone_from(&self.request_id);
                    frame.segment_id.clone_from(&self.segment_id);
                    if !self.pending_transcript.is_empty() {
                        frame.timed_transcript = std::mem::take(&mut self.pending_transcript);
                    }
                    // Hold only once no further audio can arrive; a queued whitespace-only sentence
                    // is not audio. Once true it stays true.
                    let more_audio_coming = !self.input_done
                        || self
                            .pending
                            .iter()
                            .any(|s| !s.trim_matches(crate::tokenize::is_py_space).is_empty());
                    if !more_audio_coming {
                        if let Some(previous) = self.held.replace(frame) {
                            return Ok(Some(previous));
                        }
                        continue;
                    }
                    return Ok(Some(frame));
                }
                Ok(None) => {
                    return match self.held.take() {
                        Some(mut last) => {
                            last.is_final = true;
                            // Alignment for trailing sentences that produced no audio rides on the
                            // real last frame (no silence marker, which is `AudioEmitter`'s job).
                            last.timed_transcript.append(&mut self.pending_transcript);
                            Ok(Some(last))
                        }
                        None => Ok(None),
                    };
                }
                Err(e) => {
                    // Drop the hold and report now: `FallbackChain` needs the error to decide
                    // `retry_on_chunk_sent`.
                    self.held = None;
                    return Err(e);
                }
            }
        }
    }

    /// `TimedOut` if any inner stream did not ack.
    async fn close(mut self: Box<Self>, deadline: Ts) -> Closed {
        let current = match self.current.take() {
            Some(cur) => cur.close(deadline).await,
            None => Closed::Clean,
        };
        self.rx.close();
        match (self.inner_timed_out, current) {
            (false, Closed::Clean) => Closed::Clean,
            (true, Closed::Clean) | (false, Closed::TimedOut) | (true, Closed::TimedOut) => {
                Closed::TimedOut
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::audio::AudioFrame;
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
                Arc::new(move || {
                    Box::new(DotSplitter {
                        buf: String::new(),
                        seen: s.clone(),
                        flushes: f.clone(),
                    })
                }),
                Arc::new(crate::utils::clock::VirtualClock::new()),
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

    #[async_trait]
    impl StreamSource<SynthesizedAudio> for VecSource {
        async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
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
        }
        async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
            self.closed.fetch_add(1, Ordering::Relaxed);
            Closed::Clean
        }
    }

    struct FakeTts {
        /// Frames per `synthesize()` call, and the text it was called with.
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
        fn texts(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
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
        async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
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
        }
        async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
            Err(crate::tts::tts::no_duplex_path(&self.label()))
        }
    }

    fn adapter(tts: Arc<FakeTts>) -> StreamAdapter {
        StreamAdapter::new(
            tts,
            Arc::new(|| Box::new(DotSplitter::default())),
            Arc::new(crate::utils::clock::VirtualClock::new()),
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
        let (sink, source) = adapter(tts).stream().await.unwrap();
        sink.push("one. two. three.".into()).unwrap();
        sink.end_input();

        let out = drain(source).await.unwrap();
        assert_eq!(out.len(), 6, "three sentences, two frames each");
        let finals: Vec<bool> = out.iter().map(|a| a.is_final).collect();
        assert_eq!(finals, vec![false, false, false, false, false, true]);
    }

    // Catches: holding every frame for the `is_final` lookahead.
    #[tokio::test]
    async fn a_frame_is_not_held_back_while_more_text_may_still_arrive() {
        let tts = FakeTts::new(4);
        let (sink, mut source) = adapter(tts).stream().await.unwrap();
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
        let (sink, source) = adapter(tts.clone()).stream().await.unwrap();
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
        let (sink, source) = obs.adapter.stream().await.unwrap();
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
        let (sink, source) = obs.adapter.stream().await.unwrap();
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
        assert!(
            ad.label().contains(&tts.label()),
            "the label must name what it wraps: {}",
            ad.label()
        );
        assert_ne!(
            ad.label(),
            tts.label(),
            "and must not impersonate it -- the chain keys on labels"
        );
    }

    // Catches: synthesizing a whitespace-only sentence, and dropping its alignment.
    #[tokio::test]
    async fn a_whitespace_only_sentence_is_aligned_but_not_synthesized() {
        let tts = FakeTts::new(1);
        let (sink, source) = adapter(tts.clone()).stream().await.unwrap();
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
        let (sink, source) = adapter(tts).stream().await.unwrap();
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
        let (sink, source) = adapter(tts).stream().await.unwrap();
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

        let (sink_a, source_a) = ad.stream().await.unwrap();
        sink_a.push("leftover with no delimiter".into()).unwrap();
        drop(sink_a); // dropped, not ended: the source flushes what it can and stops
        drain(source_a).await.unwrap();

        let (sink_b, source_b) = ad.stream().await.unwrap();
        sink_b.push("fresh.".into()).unwrap();
        sink_b.end_input();
        drain(source_b).await.unwrap();

        assert_eq!(tts.texts(), vec!["leftover with no delimiter", "fresh."]);
    }

    // Catches: a push after `end_input` raising, or being synthesized.
    #[tokio::test]
    async fn a_push_after_end_input_is_a_silent_no_op() {
        let tts = FakeTts::new(1);
        let (sink, source) = adapter(tts.clone()).stream().await.unwrap();
        sink.push("first.".into()).unwrap();
        sink.end_input();
        assert!(sink.push("too late.".into()).is_ok());
        assert!(sink.flush().is_ok());

        drain(source).await.unwrap();
        assert_eq!(tts.texts(), vec!["first."]);
    }

    // Catches: swallowing an inner failure, or emitting the held frame after it.
    #[tokio::test]
    async fn an_inner_failure_surfaces_as_an_error_with_its_retryability_intact() {
        let mut tts = FakeTts::new(3);
        Arc::get_mut(&mut tts).unwrap().fail_after = Some(1);
        let (sink, mut source) = adapter(tts).stream().await.unwrap();
        sink.push("boom.".into()).unwrap();
        sink.end_input();

        // The first frame is held, so the error is the first thing the caller sees.
        let err = bounded("failing frame", source.next())
            .await
            .expect_err("the 429 must not become a clean EOS");
        assert!(err.retryable());
        assert_eq!(err.status_code(), Some(429));
    }

    // Catches: dropping the inner `TtsSource` instead of closing it (leaks a host task per sentence).
    #[tokio::test]
    async fn every_inner_stream_is_aclosed_including_on_teardown() {
        let tts = FakeTts::new(1);
        let closed = tts.inner_closed.clone();
        let (sink, mut source) = adapter(tts).stream().await.unwrap();
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
        let (sink, source) = adapter(tts.clone()).stream().await.unwrap();
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
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let outcome = rt.block_on(async {
            let tts = FakeTts::new(1);
            let (sink, source) = adapter(tts).stream().await.unwrap();
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
}
