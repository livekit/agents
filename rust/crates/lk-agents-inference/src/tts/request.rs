use super::fed::{Fed, TextLog};
use crate::{retry, telemetry};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::Entry;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Clock, DuplexStream, StreamInput, SynthesizedAudio, Tts, TtsSink,
    TtsSource,
};
use lk_agents_net::signal;
use std::sync::Arc;

pub async fn synthesize(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    text: &str,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    observer: Option<retry::Observer>,
) -> Result<TtsSource, ProviderError> {
    options.validate().map_err(|error| {
        lk_agents_abi::errors::ApiError::new(error.to_string()).with_retryable(false)
    })?;
    let open = synthesis(tts.clone(), cx, text, options);
    Ok(retry::stream(
        &options,
        clock,
        open,
        |_| true,
        report(tts, observer),
        tokio_util::sync::CancellationToken::new(),
    )
    .await)
}

pub(crate) fn synthesis(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    text: &str,
    options: ApiConnectOptions,
) -> retry::Open<SynthesizedAudio> {
    let attempt = ApiConnectOptions { max_retry: 0, ..options };
    let provider = tts.with_conn_options(&attempt).unwrap_or(tts);
    let (cx, text) = (cx.clone(), text.to_owned());
    Arc::new(move || {
        let (provider, cx, text) = (provider.clone(), cx.clone(), text.clone());
        Box::pin(async move { provider.synthesize(&cx, &text).await })
    })
}

pub async fn stream(
    tts: Arc<dyn Tts>,
    cx: &telemetry::Cx,
    expressive: bool,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    observer: Option<retry::Observer>,
) -> Result<DuplexStream<TtsSink, TtsSource>, ProviderError> {
    options.validate().map_err(|error| {
        lk_agents_abi::errors::ApiError::new(error.to_string()).with_retryable(false)
    })?;
    let attempt = ApiConnectOptions { max_retry: 0, ..options };
    let provider = tts.with_conn_options(&attempt).unwrap_or_else(|| tts.clone());
    let stop = tokio_util::sync::CancellationToken::new();
    let input = Arc::new(Input { entries: signal::Shared::new(Vec::new()), stop: stop.clone() });
    let (cx, pending) = (cx.clone(), input.clone());
    let open = Arc::new(move || {
        let (provider, cx, input) = (provider.clone(), cx.clone(), pending.clone());
        Box::pin(async move {
            let (sink, source) = provider.stream(&cx, expressive).await?.split();
            Ok(Box::new(Fed::new(input, sink, source)) as TtsSource)
        }) as BoxFuture<'static, _>
    });
    let output = retry::stream(&options, clock, open, |_| true, report(tts, observer), stop).await;
    Ok(DuplexStream::new(Box::new(Sink(input)) as TtsSink, output))
}

fn report(tts: Arc<dyn Tts>, observer: Option<retry::Observer>) -> retry::Observer {
    Arc::new(move |error, retry, attempt| {
        tts.failed(error.clone(), matches!(retry, lk_agents_core::request::Retry::At(_)));
        if let Some(observer) = &observer {
            observer(error, retry, attempt);
        }
    })
}

struct Input {
    entries: signal::Shared<Vec<Entry<String>>>,
    stop: tokio_util::sync::CancellationToken,
}

impl Input {
    fn send(&self, entry: Entry<String>) -> Result<(), ProviderError> {
        self.entries.modify(|entries| {
            if self.stop.is_cancelled() || matches!(entries.last(), Some(Entry::EndInput)) {
                return Err(lk_agents_abi::errors::ApiError::new("synthesis input is closed")
                    .with_retryable(false)
                    .into());
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
