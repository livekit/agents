use crate::{retry, telemetry};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{Entry, ProviderStream};
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Clock, Closed, DuplexStream, StreamInput, StreamOutput,
    SynthesizedAudio, Ts, Tts, TtsSink, TtsSource,
};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::Notify;

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
    let input = Arc::new(Input {
        state: Mutex::new(InputState { entries: Vec::new(), sink: None, error: None }),
        changed: Notify::new(),
        stop: stop.clone(),
    });
    let (cx, pending) = (cx.clone(), input.clone());
    let open = Arc::new(move || {
        let (provider, cx, input) = (provider.clone(), cx.clone(), pending.clone());
        Box::pin(async move {
            let (sink, source) = provider.stream(&cx, expressive).await?.split();
            let sink: Arc<dyn StreamInput<String>> = Arc::from(sink);
            Ok(Box::new(Attempt { input, sink, source, installed: false }) as TtsSource)
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
    state: Mutex<InputState>,
    changed: Notify,
    stop: tokio_util::sync::CancellationToken,
}

struct InputState {
    entries: Vec<Entry<String>>,
    sink: Option<Arc<dyn StreamInput<String>>>,
    error: Option<ProviderError>,
}

impl Input {
    fn send(&self, entry: Entry<String>) -> Result<(), ProviderError> {
        let mut state = self.state.lock();
        if self.stop.is_cancelled() || matches!(state.entries.last(), Some(Entry::EndInput)) {
            return Err(lk_agents_abi::errors::ApiError::new("synthesis input is closed")
                .with_retryable(false)
                .into());
        }
        state.entries.push(entry.clone());
        let sink = state.sink.clone();
        drop(state);
        let Some(sink) = sink else { return Ok(()) };
        let sent = match entry {
            Entry::Item(text) => sink.push(text),
            Entry::Flush => sink.flush(),
            Entry::EndInput => {
                sink.end_input();
                Ok(())
            }
        };
        if let Err(error) = sent {
            let previous = self.state.lock().error.replace(error);
            drop(previous);
            self.changed.notify_one();
        }
        Ok(())
    }

    async fn install(&self, sink: Arc<dyn StreamInput<String>>) -> Result<(), ProviderError> {
        let mut cursor = 0;
        loop {
            let entries = {
                let mut state = self.state.lock();
                if cursor == state.entries.len() {
                    let previous_error = state.error.take();
                    let previous = state.sink.replace(sink);
                    drop(state);
                    drop(previous);
                    drop(previous_error);
                    return Ok(());
                }
                state.entries[cursor..].to_vec()
            };
            for entry in entries {
                sink.send(entry).await?;
                cursor += 1;
            }
        }
    }

    async fn failure(&self) -> ProviderError {
        loop {
            let changed = self.changed.notified();
            let error = self.state.lock().error.take();
            if let Some(error) = error {
                return error;
            }
            changed.await;
        }
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

struct Attempt {
    input: Arc<Input>,
    sink: Arc<dyn StreamInput<String>>,
    source: TtsSource,
    installed: bool,
}

impl StreamOutput<SynthesizedAudio> for Attempt {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if !self.installed {
                self.input.install(self.sink.clone()).await?;
                self.installed = true;
            }
            tokio::select! {
                biased;
                error = self.input.failure() => Err(error),
                next = self.source.next() => next,
            }
        })
    }
    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        let previous = self.input.state.lock().sink.take();
        drop(previous);
        self.sink.end_input();
        self.source.close(deadline)
    }
}
impl ProviderStream<SynthesizedAudio> for Attempt {
    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }
    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.source.take_call_metrics(in_flight)
    }
    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        self.source.call_ledger()
    }
    fn report_retries(&mut self, report: lk_agents_abi::utils::stream::RetryReport) {
        self.source.report_retries(report);
    }
}
