use crate::telemetry;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::ProviderStream;
use lk_agents_abi::{BoxFuture, Closed, LlmChunk, StreamOutput, SynthesizedAudio, Ts};

pub(super) trait Trace: Send {
    type Item: Send + 'static;

    fn observe(&mut self, item: &Self::Item);
    fn finish(self, error: Option<&ProviderError>);
}

/// Ends an attempt's trace with its stream; dropping it preserves trace cancellation.
pub(super) struct Traced<S: Trace> {
    pub source: Box<dyn ProviderStream<S::Item>>,
    pub trace: Option<S>,
}

impl<S: Trace + 'static> StreamOutput<S::Item> for Traced<S> {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<S::Item>, ProviderError>> {
        Box::pin(async move {
            let next = self.source.next().await;
            match &next {
                Ok(Some(item)) => {
                    if let Some(trace) = &mut self.trace {
                        trace.observe(item);
                    }
                }
                Ok(None) => {
                    if let Some(trace) = self.trace.take() {
                        trace.finish(None);
                    }
                }
                Err(error) => {
                    if let Some(trace) = self.trace.take() {
                        trace.finish(Some(error));
                    }
                }
            }
            next
        })
    }

    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.source.close(deadline)
    }
}

impl<S: Trace + 'static> ProviderStream<S::Item> for Traced<S> {
    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }
}

pub(super) struct Llm {
    pub trace: telemetry::LlmTrace,
    pub run: Option<telemetry::ScopedSpan<telemetry::LlmRequestRun>>,
}

impl Trace for Llm {
    type Item = LlmChunk;

    fn observe(&mut self, item: &LlmChunk) {
        self.trace.observe(item);
    }

    fn finish(mut self, error: Option<&ProviderError>) {
        self.trace.end_attempt(self.run, error);
        match error {
            Some(error) => self.trace.fail(error),
            None => {
                self.trace.end(Default::default());
            }
        }
    }
}

pub(super) struct Tts {
    pub trace: telemetry::TtsTrace,
    pub run: Option<telemetry::ScopedSpan<telemetry::TtsRequestRun>>,
    pub req: super::TtsReq,
}

impl Trace for Tts {
    type Item = SynthesizedAudio;

    fn observe(&mut self, item: &SynthesizedAudio) {
        self.trace.observe(item);
    }

    fn finish(mut self, error: Option<&ProviderError>) {
        self.trace.end_attempt(self.run, error);
        match error {
            Some(error) => self.trace.fail(error),
            None => self.trace.end(self.req.spoken(), None),
        }
    }
}
