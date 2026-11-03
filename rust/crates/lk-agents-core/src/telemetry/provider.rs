//! Provider metric accumulation and emission eligibility, independent of host event loops.

use std::collections::VecDeque;

use lk_agents_abi::metrics::base::{
    ConnectionAcquired, LlmMetrics, Metadata, SttMetrics, TtsMetrics,
};
use lk_agents_abi::telemetry::provider::{HostContext, RecognitionUsageMissing};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{LlmChunk, SpeechEvent, SpeechEventType, SynthesizedAudio};

impl Context {
    pub fn from_host((timestamp, label, cancelled, model, provider): HostContext) -> Self {
        Context {
            timestamp,
            label,
            cancelled,
            metadata: Some(Metadata { model_name: Some(model), model_provider: Some(provider) }),
        }
    }
}

pub struct ProviderMetrics;

impl lk_agents_abi::telemetry::provider::ProviderMetrics for ProviderMetrics {
    fn stt_batch_metrics(
        event: SpeechEvent,
        start: f64,
        now: f64,
        audio_durations: Vec<f64>,
        host: HostContext,
    ) -> SttMetrics {
        let context = Context::from_host(host);
        SttMetrics {
            label: context.label,
            request_id: event.request_id,
            timestamp: context.timestamp,
            duration: Some(now - start),
            audio_duration: Some(audio_durations.iter().sum()),
            streamed: false,
            tokens: TokenUsage::default(),
            acquire_time: None,
            connection_reused: false,
            metadata: context.metadata,
        }
    }

    fn stt_stream_metrics(
        event: SpeechEvent,
        host: HostContext,
    ) -> Result<(Option<SttMetrics>, bool), RecognitionUsageMissing> {
        let context = Context::from_host(host);

        if event.kind == SpeechEventType::FinalTranscript {
            return Ok((None, true));
        }
        if event.kind != SpeechEventType::RecognitionUsage {
            return Ok((None, false));
        }
        let usage = event.recognition_usage.as_ref().ok_or(RecognitionUsageMissing)?;
        Ok((
            Some(SttMetrics {
                label: context.label,
                request_id: event.request_id.clone(),
                timestamp: context.timestamp,
                duration: None,
                audio_duration: usage.audio_duration,
                streamed: true,
                tokens: usage.tokens.clone(),
                acquire_time: usage.connection.map(|c| c.acquire_time),
                connection_reused: usage.connection.is_some_and(|c| c.reused),
                metadata: context.metadata,
            }),
            false,
        ))
    }
}

pub struct Context {
    pub timestamp: f64,
    pub label: String,
    pub cancelled: bool,
    pub metadata: Option<Metadata>,
}

pub struct LlmMetricState {
    start: f64,
    first: Option<f64>,
    received: bool,
    request_id: String,
    usage: Option<TokenUsage>,
}
/// `LLMStream`'s metric fold as the host drives it (`_core.LlmMetricState`).
impl lk_agents_abi::telemetry::provider::LlmMetricState for LlmMetricState {
    fn new(start: f64) -> Self {
        LlmMetricState::new(start)
    }

    fn observe(&mut self, now: f64, chunk: LlmChunk) -> bool {
        LlmMetricState::observe(self, &chunk, || now)
    }

    fn finish(&self, now: f64, failed: bool, host: HostContext) -> Option<LlmMetrics> {
        LlmMetricState::finish(self, failed, || (now, Context::from_host(host)))
    }
}

/// The fold itself, for the host and for voice's native streams alike (`llm.py:336`).
impl LlmMetricState {
    pub fn new(start: f64) -> Self {
        Self { start, first: None, received: false, request_id: String::new(), usage: None }
    }

    /// Whether `chunk` is the first with a response. `now` is read only then. Every chunk's id is
    /// the request's, an empty one included (`request_id = ev.id`).
    pub fn observe(&mut self, chunk: &LlmChunk, now: impl FnOnce() -> f64) -> bool {
        self.received = true;
        let first = self.first.is_none() && chunk.has_response();
        if first {
            self.first = Some(now() - self.start);
        }
        self.request_id.clone_from(&chunk.id);
        if chunk.usage.is_some() {
            self.usage = chunk.usage.as_deref().cloned();
        }
        first
    }

    /// Time to the first chunk with a response.
    pub fn ttft(&self) -> Option<f64> {
        self.first
    }

    /// `None` for a failed attempt or one that yielded no chunk; `stamp` (the clock and the
    /// context) is read only for a record.
    pub fn finish(
        &self,
        failed: bool,
        stamp: impl FnOnce() -> (f64, Context),
    ) -> Option<LlmMetrics> {
        if failed || !self.received {
            return None;
        }
        let (now, context) = stamp();
        let duration = now - self.start;
        let tokens = self.usage.clone().unwrap_or_default();
        Some(LlmMetrics {
            label: context.label,
            request_id: self.request_id.clone(),
            timestamp: context.timestamp,
            duration,
            ttft: self.first,
            cancelled: context.cancelled,
            // The reference divides unguarded (`completion_tokens / duration`); zero is "no
            // interval".
            tokens_per_second: if duration > 0.0 {
                tokens.output_tokens as f64 / duration
            } else {
                0.0
            },
            tokens,
            speech_id: None,
            metadata: context.metadata,
        })
    }
}

pub struct TtsMetricState {
    streamed: bool,
    first: Option<f64>,
    audio: f64,
    request_id: String,
    segment_id: String,
    texts: VecDeque<String>,
    /// What the provider reported billing, from the frames that carried it.
    tokens: TokenUsage,
    characters: Option<u64>,
    /// The connection the provider acquired for the request, from the frame that carried it.
    connection: Option<ConnectionAcquired>,
}
/// The TTS streams' metric fold as the host drives it (`_core.TtsMetricState`).
impl lk_agents_abi::telemetry::provider::TtsMetricState for TtsMetricState {
    fn new(streamed: bool) -> Self {
        TtsMetricState::new(streamed)
    }

    fn queue_text(&mut self, text: String) {
        self.texts.push_back(text);
    }

    fn observe(&mut self, now: f64, start: f64, audio: SynthesizedAudio) -> bool {
        TtsMetricState::observe(self, &audio, || now - start)
    }

    fn finish(
        &mut self,
        now: f64,
        start: f64,
        failed: bool,
        batch_text: String,
        host: HostContext,
    ) -> Option<TtsMetrics> {
        if failed || (self.streamed && start == 0.0) {
            return None;
        }
        let text = if self.streamed { self.texts.pop_front()? } else { batch_text };
        let context = Context::from_host(host);
        TtsMetricState::finish(self, now, start, &text, context)
    }
}

/// The fold itself, for the host and for voice's native streams alike (`tts.py:327`, `:683`).
impl TtsMetricState {
    pub fn new(streamed: bool) -> Self {
        Self {
            streamed,
            first: None,
            audio: 0.0,
            request_id: String::new(),
            segment_id: String::new(),
            texts: VecDeque::new(),
            tokens: TokenUsage::default(),
            characters: None,
            connection: None,
        }
    }

    /// Whether `audio` ends its segment. `ttfb` is read only for the first frame. Every frame's
    /// ids are the request's, empty ones included (`request_id = ev.request_id`).
    pub fn observe(&mut self, audio: &SynthesizedAudio, ttfb: impl FnOnce() -> f64) -> bool {
        if self.first.is_none() {
            self.first = Some(ttfb());
        }
        self.audio += audio.frame.duration();
        self.request_id.clone_from(&audio.request_id);
        self.segment_id.clone_from(&audio.segment_id);
        if let Some(usage) = &audio.usage {
            self.tokens.add(usage);
        }
        if let Some(n) = audio.characters_count {
            *self.characters.get_or_insert(0) += n;
        }
        if audio.connection.is_some() {
            self.connection = audio.connection;
        }
        audio.is_final
    }

    /// Time to the first frame.
    pub fn ttfb(&self) -> Option<f64> {
        self.first
    }

    /// The record for `text`, then the figures reset. A streamed segment with no text has none.
    /// The characters are the provider's billed count when it reported one, else `text`'s.
    pub fn finish(
        &mut self,
        now: f64,
        start: f64,
        text: &str,
        context: Context,
    ) -> Option<TtsMetrics> {
        if self.streamed && text.is_empty() {
            return None;
        }
        let connection = self.connection.take();
        let metrics = TtsMetrics {
            label: context.label,
            request_id: self.request_id.clone(),
            timestamp: context.timestamp,
            ttfb: self.first,
            duration: now - start,
            audio_duration: self.audio,
            cancelled: context.cancelled,
            characters_count: self
                .characters
                .take()
                .map_or(text.chars().count() as u32, |n| u32::try_from(n).unwrap_or(u32::MAX)),
            tokens: std::mem::take(&mut self.tokens),
            streamed: self.streamed,
            acquire_time: connection.map(|c| c.acquire_time),
            connection_reused: connection.is_some_and(|c| c.reused),
            segment_id: self.streamed.then(|| self.segment_id.clone()),
            speech_id: None,
            metadata: context.metadata,
        };
        self.first = None;
        self.audio = 0.0;
        self.request_id.clear();
        Some(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{ChoiceDelta, FunctionToolCall};

    fn context() -> Context {
        Context { timestamp: 9.0, label: "p".into(), cancelled: false, metadata: None }
    }

    fn chunk(id: &str, delta: Option<ChoiceDelta>) -> LlmChunk {
        LlmChunk { id: id.into(), delta, ..LlmChunk::default() }
    }

    // Catches: an empty chunk id keeping the previous one, or ttft waiting for text past a tool
    // call -- voice's own fold did both (`llm.py:349` takes every `ev.id`; `:354` starts the clock
    // on `has_response()`).
    #[test]
    fn llm_fold_takes_every_id_and_times_the_first_response() {
        let mut fold = LlmMetricState::new(1.0);
        let call = FunctionToolCall {
            name: "f".into(),
            arguments: "{}".into(),
            call_id: "c".into(),
            extra: None,
        };
        let tool = ChoiceDelta { tool_calls: vec![call], ..ChoiceDelta::default() };
        assert!(fold.observe(&chunk("req-1", Some(tool)), || 1.5));
        let text = ChoiceDelta { content: Some("hi".into()), ..ChoiceDelta::default() };
        assert!(!fold.observe(&chunk("req-1", Some(text)), || unreachable!("read once")));
        fold.observe(&chunk("", None), || unreachable!());
        let record = fold.finish(false, || (3.0, context())).expect("a chunk arrived");
        assert_eq!(
            (record.request_id.as_str(), record.ttft, record.duration),
            ("", Some(0.5), 2.0)
        );
    }

    // Catches: a metadata-only delta (a gateway deployment stamp) starting the clock, or a
    // response that generated nothing losing its token counts.
    #[test]
    fn a_response_without_output_reports_its_tokens_and_no_ttft() {
        let mut fold = LlmMetricState::new(1.0);
        let stamp = ChoiceDelta { extra: Some(Default::default()), ..ChoiceDelta::default() };
        assert!(!fold.observe(&chunk("req-1", Some(stamp)), || unreachable!("not a response")));
        let usage = TokenUsage { input_tokens: 11, output_tokens: 7, ..Default::default() };
        fold.observe(
            &LlmChunk { usage: Some(Box::new(usage)), ..chunk("req-1", None) },
            || unreachable!(),
        );
        let record = fold.finish(false, || (3.0, context())).expect("chunks arrived");
        assert_eq!(
            (record.ttft, record.tokens.output_tokens, record.tokens.total_tokens()),
            (None, 7, 18)
        );
    }

    // Catches: empty ids keeping the previous request's, or an empty segment id dropped to `None`
    // (`tts.py:733` takes every `ev.request_id`/`ev.segment_id`, and passes the string through).
    #[test]
    fn tts_fold_takes_every_id_and_keeps_an_empty_segment() {
        let mut fold = TtsMetricState::new(true);
        let frame = || lk_agents_abi::voice::rtc::silence_frame(240, 24_000, 1);
        let first = SynthesizedAudio {
            segment_id: "seg-1".into(),
            ..SynthesizedAudio::new(frame(), "req-1")
        };
        fold.observe(&first, || 0.25);
        fold.observe(&SynthesizedAudio::new(frame(), ""), || unreachable!("read once"));
        let record = fold.finish(2.0, 1.0, "hello", context()).expect("text fed");
        assert_eq!((record.request_id.as_str(), record.segment_id.as_deref()), ("", Some("")));
        assert_eq!((record.ttfb, record.characters_count), (Some(0.25), 5));
        assert!(fold.finish(2.0, 1.0, "", context()).is_none(), "no text");
    }

    // Catches: an acquisition lost, repeated on every record of the connection, or dropped by a
    // segment without text before the record that reports it.
    #[test]
    fn tts_fold_reports_the_connection_once_on_the_next_record() {
        let mut fold = TtsMetricState::new(true);
        let frame = || lk_agents_abi::voice::rtc::silence_frame(240, 24_000, 1);
        let acquired = ConnectionAcquired { acquire_time: 0.2, reused: false };
        fold.observe(
            &SynthesizedAudio { connection: Some(acquired), ..SynthesizedAudio::new(frame(), "r") },
            || 0.1,
        );
        assert!(fold.finish(2.0, 1.0, "", context()).is_none(), "no text, no record");
        let first = fold.finish(2.0, 1.0, "hi", context()).expect("text fed");
        assert_eq!((first.acquire_time, first.connection_reused), (Some(0.2), false));
        fold.observe(&SynthesizedAudio::new(frame(), "r"), || 0.1);
        let second = fold.finish(3.0, 2.0, "again", context()).expect("text fed");
        assert_eq!((second.acquire_time, second.connection_reused), (None, false));
    }
}
