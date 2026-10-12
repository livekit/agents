//! Provider metric accumulation and emission eligibility, independent of host event loops.
use crate::metrics::base::{LlmMetrics, Metadata, SttMetrics, TtsMetrics};
use crate::{CompletionUsage, LlmChunk, SynthesizedAudio};
use crate::{SpeechEvent, SpeechEventType};
use std::collections::VecDeque;

/// The host's context tuple, `(timestamp, label, cancelled, model, provider)`, as the folds'.
pub type HostContext = (f64, String, bool, String, String);

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

/// `STT.recognize`'s metrics for one batch recognition.
#[crate::host(export(name = "stt_batch_metrics", pid = false))]
pub fn stt_batch_for_host(event: SpeechEvent, start: f64, now: f64, audio_durations: Vec<f64>, host: HostContext) -> SttMetrics {
    let context = Context::from_host(host);
    SttMetrics {
        label: context.label,
        request_id: event.request_id,
        timestamp: context.timestamp,
        duration: now - start,
        audio_duration: audio_durations.iter().sum(),
        input_tokens: 0,
        output_tokens: 0,
        streamed: false,
        acquire_time: 0.0,
        connection_reused: false,
        metadata: context.metadata,
    }
}

/// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
#[crate::host(export(name = "stt_stream_metrics", pid = false, err = "AssertionError"))]
pub fn stt_stream_for_host(event: SpeechEvent, host: HostContext) -> Result<(Option<SttMetrics>, bool), &'static str> {
    let context = Context::from_host(host);

    if event.kind == SpeechEventType::FinalTranscript {
        return Ok((None, true));
    }
    if event.kind != SpeechEventType::RecognitionUsage {
        return Ok((None, false));
    }
    let usage = event
        .recognition_usage
        .as_ref()
        .ok_or("recognition_usage must be provided for RECOGNITION_USAGE event")?;
    Ok((
        Some(SttMetrics {
            label: context.label,
            request_id: event.request_id.clone(),
            timestamp: context.timestamp,
            duration: 0.0,
            audio_duration: usage.audio_duration,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            streamed: true,
            acquire_time: 0.0,
            connection_reused: false,
            metadata: context.metadata,
        }),
        false,
    ))
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
    usage: Option<CompletionUsage>,
}
/// `LLMStream`'s metric fold as the host drives it (`_core.LlmMetricState`).
#[crate::host(object(cell = "mutex", pid = false))]
impl LlmMetricState {
    #[host(new)]
    pub fn for_host(start: f64) -> Self {
        Self { start, first: None, received: false, request_id: String::new(), usage: None }
    }

    #[host(method(name = "observe"))]
    pub fn observe_host(&mut self, now: f64, chunk: LlmChunk) -> bool {
        self.observe(&chunk, || now)
    }

    #[host(method(name = "finish"))]
    pub fn finish_host(&self, now: f64, failed: bool, host: HostContext) -> Option<LlmMetrics> {
        self.finish(failed, || (now, Context::from_host(host)))
    }
}

/// The fold itself, for the host and for voice's native streams alike (`llm.py:336`).
impl LlmMetricState {
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
            self.usage.clone_from(&chunk.usage);
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
        let usage = self.usage.clone().unwrap_or_default();
        Some(LlmMetrics {
            label: context.label,
            request_id: self.request_id.clone(),
            timestamp: context.timestamp,
            duration,
            ttft: self.first.unwrap_or(-1.0),
            cancelled: context.cancelled,
            completion_tokens: usage.completion_tokens,
            prompt_tokens: usage.prompt_tokens,
            prompt_cached_tokens: usage.prompt_cached_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            total_tokens: usage.total_tokens,
            // The reference divides unguarded (`completion_tokens / duration`); zero is "no
            // interval".
            tokens_per_second: if duration > 0.0 {
                usage.completion_tokens as f64 / duration
            } else {
                0.0
            },
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
}
/// The TTS streams' metric fold as the host drives it (`_core.TtsMetricState`).
#[crate::host(object(cell = "mutex", pid = false))]
impl TtsMetricState {
    #[host(new)]
    pub fn for_host(streamed: bool) -> Self {
        Self {
            streamed,
            first: None,
            audio: 0.0,
            request_id: String::new(),
            segment_id: String::new(),
            texts: VecDeque::new(),
        }
    }

    #[host(method(name = "queue_text"))]
    pub fn queue_text_host(&mut self, text: String) {
        self.texts.push_back(text);
    }

    #[host(method(name = "observe"))]
    pub fn observe_host(&mut self, now: f64, start: f64, audio: SynthesizedAudio) -> bool {
        self.observe(&audio, || now - start)
    }

    #[host(method(name = "finish"))]
    #[allow(clippy::too_many_arguments)]
    pub fn finish_host(
        &mut self,
        now: f64,
        start: f64,
        failed: bool,
        batch_text: &str,
        input_tokens: u32,
        output_tokens: u32,
        acquire_time: f64,
        connection_reused: bool,
        host: HostContext,
    ) -> Option<TtsMetrics> {
        if failed || (self.streamed && start == 0.0) {
            return None;
        }
        let text = if self.streamed { self.texts.pop_front()? } else { batch_text.to_owned() };
        let context = Context::from_host(host);
        self.finish(
            now,
            start,
            &text,
            input_tokens,
            output_tokens,
            acquire_time,
            connection_reused,
            context,
        )
    }
}

/// The fold itself, for the host and for voice's native streams alike (`tts.py:327`, `:683`).
impl TtsMetricState {
    /// Whether `audio` ends its segment. `ttfb` is read only for the first frame. Every frame's
    /// ids are the request's, empty ones included (`request_id = ev.request_id`).
    pub fn observe(&mut self, audio: &SynthesizedAudio, ttfb: impl FnOnce() -> f64) -> bool {
        if self.first.is_none() {
            self.first = Some(ttfb());
        }
        self.audio += audio.frame.duration();
        self.request_id.clone_from(&audio.request_id);
        self.segment_id.clone_from(&audio.segment_id);
        audio.is_final
    }

    /// Time to the first frame.
    pub fn ttfb(&self) -> Option<f64> {
        self.first
    }

    /// The record for `text`, then the figures reset. A streamed segment with no text has none.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        now: f64,
        start: f64,
        text: &str,
        input_tokens: u32,
        output_tokens: u32,
        acquire_time: f64,
        connection_reused: bool,
        context: Context,
    ) -> Option<TtsMetrics> {
        if self.streamed && text.is_empty() {
            return None;
        }
        let metrics = TtsMetrics {
            label: context.label,
            request_id: self.request_id.clone(),
            timestamp: context.timestamp,
            ttfb: self.first.unwrap_or(-1.0),
            duration: now - start,
            audio_duration: self.audio,
            cancelled: context.cancelled,
            characters_count: text.chars().count() as u32,
            input_tokens,
            output_tokens,
            streamed: self.streamed,
            acquire_time,
            connection_reused,
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
    use crate::{ChoiceDelta, FunctionToolCall};

    fn context() -> Context {
        Context { timestamp: 9.0, label: "p".into(), cancelled: false, metadata: None }
    }

    fn chunk(id: &str, delta: Option<ChoiceDelta>) -> LlmChunk {
        LlmChunk { id: id.into(), delta, usage: None, flush: false }
    }

    // Catches: an empty chunk id keeping the previous one, or ttft waiting for text past a tool
    // call -- voice's own fold did both (`llm.py:349` takes every `ev.id`; `:354` starts the clock
    // on `has_response()`).
    #[test]
    fn llm_fold_takes_every_id_and_times_the_first_response() {
        let mut fold = LlmMetricState::for_host(1.0);
        let call =
            FunctionToolCall { name: "f".into(), arguments: "{}".into(), call_id: "c".into(), extra: None };
        let tool = ChoiceDelta { tool_calls: vec![call], ..ChoiceDelta::default() };
        assert!(fold.observe(&chunk("req-1", Some(tool)), || 1.5));
        let text = ChoiceDelta { content: Some("hi".into()), ..ChoiceDelta::default() };
        assert!(!fold.observe(&chunk("req-1", Some(text)), || unreachable!("read once")));
        fold.observe(&chunk("", None), || unreachable!());
        let record = fold.finish(false, || (3.0, context())).expect("a chunk arrived");
        assert_eq!((record.request_id.as_str(), record.ttft, record.duration), ("", 0.5, 2.0));
    }

    // Catches: empty ids keeping the previous request's, or an empty segment id dropped to `None`
    // (`tts.py:733` takes every `ev.request_id`/`ev.segment_id`, and passes the string through).
    #[test]
    fn tts_fold_takes_every_id_and_keeps_an_empty_segment() {
        let mut fold = TtsMetricState::for_host(true);
        let frame = || crate::voice::rtc::silence_frame(240, 24_000, 1);
        let first =
            SynthesizedAudio { segment_id: "seg-1".into(), ..SynthesizedAudio::new(frame(), "req-1") };
        fold.observe(&first, || 0.25);
        fold.observe(&SynthesizedAudio::new(frame(), ""), || unreachable!("read once"));
        let record = fold.finish(2.0, 1.0, "hello", 0, 0, 0.0, false, context()).expect("text fed");
        assert_eq!((record.request_id.as_str(), record.segment_id.as_deref()), ("", Some("")));
        assert_eq!((record.ttfb, record.characters_count), (0.25, 5));
        assert!(fold.finish(2.0, 1.0, "", 0, 0, 0.0, false, context()).is_none(), "no text");
    }
}
