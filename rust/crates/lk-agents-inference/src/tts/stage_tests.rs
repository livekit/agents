//! The stage, driven through the real path: `tts::stream` over a `StreamAdapter` over a fake
//! provider, so each chunk the stage prepares is one `synthesize()` call the fake records.

use crate::tts::sentence_adapter::StreamAdapter;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tokenize::bindings::{SentenceSettings, SentenceTokenizerLeg};
use lk_agents_abi::tts::stream_adapter::HostSentenceTokenizer;
use lk_agents_abi::tts::stream_pacer::{HostSentenceStream, SentenceStreamPacer};
use lk_agents_abi::tts::types::{
    SpeechChunk, SynthesizedAudio, TextUnit, Tts, TtsCapabilities, TtsOptions, TtsSink, TtsSource,
};
use lk_agents_abi::utils::audio::AudioFrame;
use lk_agents_abi::utils::clock::{Clock, Ts};
use lk_agents_abi::utils::stream::{Closed, StreamOutput};
use lk_agents_abi::{ApiConnectOptions, TextSink};
use lk_agents_core::markup::{Markup, MarkupExt};
use lk_agents_net::signal;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

const GUARD: Duration = Duration::from_secs(120);

/// One frame per chunk, of `seconds` of audio.
struct OneFrame(Option<SynthesizedAudio>);

impl StreamOutput<SynthesizedAudio> for OneFrame {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move { Ok(self.0.take()) })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for OneFrame {}

/// A provider with no duplex path: what it was asked to synthesize, and when.
struct Provider {
    markup: &'static str,
    unit: TextUnit,
    limit: Option<usize>,
    options: TtsOptions,
    /// Seconds of audio each chunk synthesizes to.
    seconds: u32,
    asked: Mutex<Vec<(SpeechChunk, tokio::time::Instant)>>,
}

/// Splits sentence by sentence (one word is enough), unpaced, no dialect.
fn provider() -> Provider {
    Provider {
        markup: "",
        unit: TextUnit::Sentences,
        limit: None,
        options: TtsOptions { sentence_tokenizer: sentences(), pacing: None },
        seconds: 1,
        asked: Mutex::new(Vec::new()),
    }
}

fn sentences() -> SentenceTokenizerLeg {
    SentenceTokenizerLeg::Native(Arc::new(SentenceSettings {
        min_words: 1,
        ..SentenceSettings::default()
    }))
}

impl Provider {
    fn texts(&self) -> Vec<String> {
        self.asked.lock().iter().map(|(chunk, _)| chunk.text.clone()).collect()
    }
    fn times(&self) -> Vec<tokio::time::Instant> {
        self.asked.lock().iter().map(|(_, at)| *at).collect()
    }
}

impl Tts for Provider {
    fn label(&self) -> String {
        "fake.Provider".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: false, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        1_000
    }
    fn num_channels(&self) -> u16 {
        1
    }
    fn markup_provider(&self) -> String {
        self.markup.to_owned()
    }
    fn text_unit(&self) -> TextUnit {
        self.unit
    }
    fn max_text_len(&self) -> Option<usize> {
        self.limit
    }
    fn options(&self) -> Arc<dyn lk_agents_abi::tts::types::SharedTtsOptions> {
        Arc::new(self.options.clone())
    }
    fn synthesize<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            self.asked.lock().push((chunk.clone(), tokio::time::Instant::now()));
            let samples = self.seconds * 1_000;
            let frame = AudioFrame::new(vec![0u8; samples as usize * 2], 1_000, 1, samples);
            Ok(Box::new(OneFrame(Some(SynthesizedAudio::new(frame, "r")))) as TtsSource)
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

/// A synthesis request over `provider`, as a session opens one.
async fn request(provider: &Arc<Provider>) -> (TextSink, TtsSource) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::RealClock::new());
    let adapter = Arc::new(StreamAdapter::new(provider.clone(), clock.clone()));
    let cx = lk_agents_abi::telemetry::Cx::NONE;
    let opened = crate::tts::stream(adapter, &cx, ApiConnectOptions::default(), clock, None);
    opened.await.expect("opens").split()
}

async fn drain(mut source: TtsSource) -> Vec<SynthesizedAudio> {
    let drained = tokio::time::timeout(GUARD, async move {
        let mut out = Vec::new();
        while let Some(frame) = source.next().await.expect("synthesized") {
            out.push(frame);
        }
        source.close(Ts::from_nanos(u64::MAX)).await;
        out
    });
    drained.await.expect("the request never ended")
}

/// What `provider` is asked to synthesize for `pushed`, each `None` a flush.
async fn asked(provider: Provider, pushed: &[Option<&str>]) -> Vec<String> {
    let provider = Arc::new(provider);
    let (sink, source) = request(&provider).await;
    for push in pushed {
        match push {
            Some(text) => sink.push((*text).to_owned()).expect("pushed"),
            None => sink.flush().expect("flushed"),
        }
    }
    sink.end_input();
    drain(source).await;
    provider.texts()
}

fn words(frames: &[SynthesizedAudio]) -> Vec<String> {
    frames.iter().flat_map(|f| f.timed_transcript.iter().map(|t| t.text.clone())).collect()
}

// Catches: a paced stream sending every sentence at once, sending before the audio already
// synthesized played out, or losing a held sentence.
#[tokio::test(start_paused = true)]
async fn the_first_sentence_goes_at_once_then_the_rest_batch_as_audio_plays_out() {
    let pacing = SentenceStreamPacer { min_remaining_audio: 0.5, max_text_length: 300 };
    let provider = Arc::new(Provider {
        options: TtsOptions { sentence_tokenizer: sentences(), pacing: Some(pacing) },
        ..provider()
    });
    let (sink, source) = request(&provider).await;
    let started = tokio::time::Instant::now();
    sink.push("One is here. Two is here. Three is here.".into()).unwrap();
    sink.end_input();
    drain(source).await;
    assert_eq!(provider.texts(), ["One is here.", "Two is here. Three is here."]);
    // the batch waited for the first sentence's second of audio to play out
    let times = provider.times();
    assert_eq!(times[0], started, "the first sentence goes at once");
    assert!(times[1] - started >= Duration::from_millis(500), "{:?}", times[1] - started);
}

// Catches: a `Flush` that does not end the segment (the next segment's first chunk paced behind
// the previous segment's audio, or joined to it).
#[tokio::test(start_paused = true)]
async fn a_flush_ends_a_segment_and_the_next_segments_first_chunk_goes_at_once() {
    let pacing = SentenceStreamPacer { min_remaining_audio: 1.0, max_text_length: 300 };
    let provider = Arc::new(Provider {
        options: TtsOptions { sentence_tokenizer: sentences(), pacing: Some(pacing) },
        seconds: 10,
        ..provider()
    });
    let (sink, source) = request(&provider).await;
    sink.push("First of one. Second of one.".into()).unwrap();
    sink.flush().unwrap();
    sink.push("First of two. Second of two.".into()).unwrap();
    sink.end_input();
    drain(source).await;
    assert_eq!(
        provider.texts(),
        ["First of one.", "Second of one.", "First of two.", "Second of two."],
        "a segment's chunks are never batched with another's"
    );
    let times = provider.times();
    // within a segment the second chunk waits for the first's audio to (nearly) play out
    assert!(times[1] - times[0] >= Duration::from_secs(9), "{:?}", times[1] - times[0]);
    // the next segment starts afresh once the flush went: its first chunk at once
    assert_eq!(times[2], times[1], "the next segment's first chunk waited for audio");
    assert!(times[3] - times[2] >= Duration::from_secs(9), "{:?}", times[3] - times[2]);
}

// Catches: a new segment's pacer taking the audio synthesized before it as if it had only now
// started playing: after a flush, every later chunk waits for the earlier segments' audio again.
#[tokio::test(start_paused = true)]
async fn a_segments_pacing_counts_only_the_audio_still_to_play() {
    let pacing = SentenceStreamPacer { min_remaining_audio: 1.0, max_text_length: 300 };
    let provider = Arc::new(Provider {
        options: TtsOptions { sentence_tokenizer: sentences(), pacing: Some(pacing) },
        seconds: 10,
        ..provider()
    });
    let (sink, source) = request(&provider).await;
    sink.push("First of one. Second of one.".into()).unwrap();
    sink.flush().unwrap();
    sink.push("First of two. Second of two.".into()).unwrap();
    sink.end_input();
    drain(source).await;
    let times = provider.times();
    assert_eq!(times.len(), 4, "{:?}", provider.texts());
    // three chunks of 10 s, the first started at once: the audio before the last plays out by
    // 30 s, so the last goes at most a second before that
    let last = times[3] - times[0];
    assert!(last <= Duration::from_secs(30), "the last chunk waited until {last:?}");
}

// Catches: the input's end closing the request before the splitter's partial sentence went out.
#[tokio::test]
async fn end_input_flushes_the_partial_sentence() {
    let texts = asked(provider(), &[Some("Complete. Trailing without a dot")]).await;
    assert_eq!(texts, ["Complete.", "Trailing without a dot"]);
}

// Catches: short sentences sent one request each instead of batched up to the tokenizer's floor.
#[tokio::test]
async fn short_sentences_are_batched_up_to_the_tokenizers_floor() {
    let options = TtsOptions { pacing: None, ..TtsOptions::default() };
    let pushed = [Some("One. "), Some("Two! "), Some("Three? ")];
    let texts = asked(Provider { options, ..provider() }, &pushed).await;
    assert_eq!(texts, ["One. Two! Three?"]);
}

// Catches: an empty push treated as data (a boundary splitting the sentence around it).
#[tokio::test]
async fn an_empty_push_changes_nothing() {
    let texts = asked(provider(), &[Some("half"), Some(""), Some(" a sentence.")]).await;
    assert_eq!(texts, ["half a sentence."]);
}

// Catches: a no-op flush, holding the buffered partial until the input ends or joining it to
// the text after it.
#[tokio::test]
async fn a_flush_forces_the_buffered_partial_out() {
    let texts = asked(provider(), &[Some("no delimiter here"), None, Some("second")]).await;
    assert_eq!(texts, ["no delimiter here", "second"]);
}

// Catches: two requests over one TTS sharing a splitter: one's leftover spoken by the other.
#[tokio::test]
async fn each_request_splits_on_its_own() {
    let provider = Arc::new(provider());
    let (sink_a, source_a) = request(&provider).await;
    sink_a.push("leftover with no delimiter".into()).unwrap();
    let (sink_b, source_b) = request(&provider).await;
    sink_b.push("Fresh.".into()).unwrap();
    sink_b.end_input();
    drain(source_b).await;
    assert_eq!(provider.texts(), ["Fresh."]);
    source_a.close(Ts::from_nanos(u64::MAX)).await;
}

// Catches: markup spoken aloud by a TTS the stage did not lower for, a sentence split inside a
// marker, or the provider's syntax aligned as the words said.
#[tokio::test]
async fn each_chunk_is_lowered_to_the_providers_markup() {
    let written = "<expr type=\"expression\" label=\"happy\"/>Hi there.";
    let provider = Arc::new(Provider { markup: "cartesia", ..provider() });
    let (sink, source) = request(&provider).await;
    sink.push(written.into()).unwrap();
    sink.end_input();
    let frames = drain(source).await;
    let markup = Markup::from_provider_key("cartesia");
    let lowered = markup.lower(&markup.normalize(written));
    assert_eq!(provider.texts(), [lowered.text.trim()]);
    assert_ne!(provider.texts()[0], written, "cartesia lowers `expr` to its own tags");
    assert_eq!(words(&frames), ["Hi there."]);
}

// Catches: a chunk's `said` keeping markup the provider performs rather than says, so the
// transcript (and what the history says was heard) reads the tag.
#[tokio::test]
async fn a_chunks_words_exclude_its_markup() {
    let written = "Ha <expr type=\"sound\" label=\"laugh\"/>that was funny.";
    let provider = Arc::new(Provider { markup: "inworld", ..provider() });
    let (sink, source) = request(&provider).await;
    sink.push(written.into()).unwrap();
    sink.end_input();
    let frames = drain(source).await;
    assert_eq!(provider.texts(), ["Ha [laugh] that was funny."]);
    let said = words(&frames).concat();
    assert!(!said.contains("laugh"), "the transcript reads the tag: {said:?}");
    assert!(said.starts_with("Ha") && said.ends_with("that was funny."), "{said:?}");
}

// Catches: a word provider sent sentences (it buffers text itself, and waits for words).
#[tokio::test]
async fn a_word_provider_is_sent_words() {
    let texts =
        asked(Provider { unit: TextUnit::Words, ..provider() }, &[Some("Hello there my friend.")])
            .await;
    assert_eq!(texts, ["Hello", "there", "my", "friend."]);
}

#[derive(Default)]
struct Stream {
    sentences: VecDeque<String>,
    ended: bool,
}

/// A user's host sentence stream: the sentences it was scripted to yield, whatever the text
/// pushed, its end once its input ended.
struct UserSentences(signal::Shared<Stream>);

impl HostSentenceStream for UserSentences {
    fn push_text(&self, _text: String) {}
    fn flush(&self) {}
    fn end_input(&self) {
        self.0.modify(|s| s.ended = true);
    }
    fn next(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::tokenize::TokenData>, ProviderError>,
    > {
        Box::pin(async move {
            let token = self
                .0
                .wait_until(|s| match s.sentences.pop_front() {
                    Some(token) => Some(Some(token)),
                    None => s.ended.then_some(None),
                })
                .await;
            Ok(token.map(|token| lk_agents_abi::tokenize::TokenData {
                segment_id: String::new(),
                token,
            }))
        })
    }
    fn aclose(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async { Ok(()) })
    }
}

/// A user's sentence tokenizer whose one stream yields `sentences`.
struct UserTokenizer(Mutex<Option<Vec<String>>>);

impl HostSentenceTokenizer for UserTokenizer {
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Arc<dyn HostSentenceStream>, lk_agents_abi::errors::HostError>,
    > {
        let sentences = self.0.lock().take().expect("one segment, one stream");
        let stream = Stream { sentences: sentences.into(), ended: false };
        Box::pin(async move {
            Ok(Arc::new(UserSentences(signal::Shared::new(stream))) as Arc<dyn HostSentenceStream>)
        })
    }
}

fn user_tokenizer(sentences: &[&str]) -> TtsOptions {
    let sentences = sentences.iter().map(|s| (*s).to_owned()).collect();
    let tokenizer = UserTokenizer(Mutex::new(Some(sentences)));
    TtsOptions { sentence_tokenizer: SentenceTokenizerLeg::Host(Arc::new(tokenizer)), pacing: None }
}

// Catches: a user's sentence tokenizer cutting inside a voice tag. Each half is lowered alone,
// so the voice reads the second half of the tag aloud; a sentence of markup alone is a synthesis
// with nothing to say.
#[tokio::test]
async fn a_user_tokenizer_never_splits_the_voices_tags() {
    let user = [
        "Ha <expr type=\"sound\" ",
        "label=\"laugh\"/> that was funny.",
        "<expr type=\"expression\" label=\"happy\"/>",
        "See you soon.",
    ];
    let provider = Provider { markup: "cartesia", options: user_tokenizer(&user), ..provider() };
    let texts = asked(provider, &[Some("whatever the user tokenizer makes of it")]).await;
    let markup = Markup::from_provider_key("cartesia");
    let lowered = |s: &str| markup.lower(&markup.normalize(s)).text.trim().to_owned();
    assert_eq!(
        texts,
        [
            lowered("Ha <expr type=\"sound\" label=\"laugh\"/> that was funny."),
            lowered("<expr type=\"expression\" label=\"happy\"/>See you soon."),
        ]
    );
}

// Catches: a user tokenizer's sentences held to the end of the reply behind a wrapper the
// model never closed.
#[tokio::test]
async fn an_unclosed_wrapper_does_not_hold_a_user_tokenizers_reply() {
    let mut user = vec!["<expr type=\"prosody\" label=\"slow\">Slowly now.".to_owned()];
    user.extend((0..40).map(|n| format!(" This is sentence number {n} of the reply.")));
    let user: Vec<&str> = user.iter().map(String::as_str).collect();
    let provider = Provider { markup: "cartesia", options: user_tokenizer(&user), ..provider() };
    let texts = asked(provider, &[Some("the reply")]).await;
    assert!(texts.len() > 1, "the whole reply waited for its end: {texts:?}");
}

// Catches: a user tokenizer's sentence longer than the provider's request limit sent whole (the
// provider refuses it), or cut by dropping or reordering its words.
#[tokio::test]
async fn a_user_sentence_past_the_provider_limit_is_cut_within_it() {
    let long = "This sentence runs on, and on, and on well past what one request may carry.";
    let provider = Provider { limit: Some(30), options: user_tokenizer(&[long]), ..provider() };
    let texts = asked(provider, &[Some(long)]).await;
    assert!(texts.len() > 1, "{texts:?}");
    assert!(texts.iter().all(|t| t.chars().count() <= 30), "{texts:?}");
    let joined: Vec<&str> = texts.iter().flat_map(|t| t.split_whitespace()).collect();
    assert_eq!(joined, long.split_whitespace().collect::<Vec<_>>());
}
