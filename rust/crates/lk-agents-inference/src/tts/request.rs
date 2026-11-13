use super::fed::{Fed, TextLog};
use super::stage::Stage;
use crate::{retry, telemetry};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::Entry;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Clock, DuplexStream, StreamInput, SynthesizedAudio, TextSink,
    Tts, TtsSource,
};
use lk_agents_net::signal;
use lk_agents_net::source::{Gated, InputGate};
use std::sync::Arc;

/// A synthesis of `text`, whole: one request retried within `options`, or, when it passes the
/// provider's request limit, one per chunk, each retried on its own.
pub async fn synthesize(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    text: &str,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    observer: Option<retry::Observer>,
) -> Result<TtsSource, ProviderError> {
    options.validate().map_err(|error| {
        lk_agents_abi::errors::ApiError::new(
            lk_agents_abi::errors::ErrorKind::InvalidRequest,
            error.to_string(),
        )
    })?;
    let mut chunks = super::stage::one_shot(tts.as_ref(), text);
    if chunks.len() > 1 {
        return in_chunks(tts, cx, chunks, options, clock);
    }
    let chunk = chunks.pop().unwrap_or_default();
    let open = synthesis(tts.clone(), cx, chunk, options);
    let (_, options) = attempts(&tts, options);
    Ok(retry::stream(
        tts.label(),
        &options,
        clock,
        open,
        |_| true,
        report(tts, observer),
        tokio_util::sync::CancellationToken::new(),
    )
    .await)
}

/// One synthesis of `chunks` in turn, each its own request retried within `budget`.
pub(crate) fn in_chunks(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    chunks: Vec<lk_agents_abi::SpeechChunk>,
    budget: ApiConnectOptions,
    clock: Arc<dyn Clock>,
) -> Result<TtsSource, ProviderError> {
    let adapter = super::sentence_adapter::StreamAdapter::new(tts, clock).with_budget(budget);
    let (sink, source) = adapter.open(cx)?.split();
    for chunk in chunks {
        sink.push(chunk)?;
    }
    sink.end_input();
    Ok(source)
}

pub(crate) fn synthesis(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    chunk: lk_agents_abi::SpeechChunk,
    options: ApiConnectOptions,
) -> retry::Open<SynthesizedAudio> {
    let (provider, _) = attempts(&tts, options);
    let cx = cx.clone();
    Arc::new(move || {
        let (provider, cx, chunk) = (provider.clone(), cx.clone(), chunk.clone());
        Box::pin(async move { provider.synthesize(&cx, &chunk).await })
    })
}

/// A synthesis of the text pushed: its input logged, each attempt reading the log through a
/// stage of its own (see [`Stage`]), the attempts retried within `options`.
pub async fn stream(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    observer: Option<retry::Observer>,
) -> Result<DuplexStream<TextSink, TtsSource>, ProviderError> {
    let (sink, opening) = start(tts, cx, options, clock, observer)?;
    Ok(DuplexStream::new(sink, opening.await))
}

/// [`stream`], its input taking text at once and its first attempt opened by the future.
pub fn start(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    observer: Option<retry::Observer>,
) -> Result<(TextSink, BoxFuture<'static, TtsSource>), ProviderError> {
    options.validate().map_err(|error| {
        lk_agents_abi::errors::ApiError::new(
            lk_agents_abi::errors::ErrorKind::InvalidRequest,
            error.to_string(),
        )
    })?;
    let (provider, options) = attempts(&tts, options);
    let gate = InputGate::default();
    let input = Arc::new(Input { entries: signal::Shared::new(Vec::new()), gate: gate.clone() });
    let (cx, pending, staged, stage_clock) =
        (cx.clone(), input.clone(), tts.clone(), clock.clone());
    let open = Arc::new(move || {
        let (provider, cx, input) = (provider.clone(), cx.clone(), pending.clone());
        let (staged, clock) = (staged.clone(), stage_clock.clone());
        Box::pin(async move {
            let (sink, source) = provider.stream(&cx).await?.split();
            // each attempt reads the log from its start through a stage of its own
            let stage = Stage::new(staged.as_ref(), input, true, &cx, clock);
            Ok(Box::new(Fed::new(stage, sink, source)) as TtsSource)
        }) as BoxFuture<'static, _>
    });
    let subject = tts.label();
    let report = report(tts, observer);
    // the request's failure, once it gave up, is what its input answers from then on
    let opening = async move {
        let stop = tokio_util::sync::CancellationToken::new();
        let source = retry::stream(subject, &options, clock, open, |_| true, report, stop).await;
        Box::new(Gated::new(source, gate)) as TtsSource
    };
    Ok((Box::new(Sink(input)) as TextSink, Box::pin(opening)))
}

/// The provider each attempt runs on, and the request's own retry policy ([`retry::owner`]).
fn attempts(tts: &Arc<dyn Tts>, options: ApiConnectOptions) -> (Arc<dyn Tts>, ApiConnectOptions) {
    let (given, request) = retry::owner(tts.retries_itself(), options);
    (tts.with_conn_options(&given).unwrap_or_else(|| tts.clone()), request)
}

fn report(tts: Arc<dyn Tts>, observer: Option<retry::Observer>) -> retry::Observer {
    Arc::new(move |error, retry, attempt| {
        tts.failed(error.clone(), retry.is_some());
        if let Some(observer) = &observer {
            observer(error, retry, attempt);
        }
    })
}

struct Input {
    entries: signal::Shared<Vec<Entry<String>>>,
    /// Ended by the end of input or the request's close; failed by its failure.
    gate: InputGate,
}

impl Input {
    fn send(&self, entry: Entry<String>) -> Result<(), ProviderError> {
        self.entries.modify(|entries| {
            self.gate.admit()?;
            if entry == Entry::EndInput {
                self.gate.end();
            }
            entries.push(entry);
            Ok(())
        })
    }
}

impl TextLog for Input {
    fn entry(&self, index: usize) -> BoxFuture<'_, Entry<String>> {
        Box::pin(self.entries.wait_until(move |entries| entries.get(index).cloned()))
    }
}

struct Sink(Arc<Input>);

/// A request whose input is dropped has no more text: what it was given is still spoken.
impl Drop for Sink {
    fn drop(&mut self) {
        let _ = self.0.send(Entry::EndInput);
    }
}

impl StreamInput<String> for Sink {
    fn push(&self, text: String) -> Result<(), ProviderError> {
        self.0.send(Entry::Item(text))
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.0.send(Entry::Flush)
    }
    fn end_input(&self) {
        let _ = self.0.send(Entry::EndInput);
    }
}
