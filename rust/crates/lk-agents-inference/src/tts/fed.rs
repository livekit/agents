//! A synthesis attempt's output, its text fed meanwhile: what the request logged is handed to the
//! attempt's input one entry at a time, each awaited, so a slow input (a host TTS whose loop is
//! busy) holds the feed instead of refusing text. The log keeps the backlog.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{CallLedger, Entry, ProviderStream, RetryReport};
use lk_agents_abi::{BoxFuture, Closed, StreamOutput, SynthesizedAudio, Ts, TtsSink, TtsSource};
use std::sync::Arc;

/// The text a synthesis request received, in order: every attempt's input replays it from the
/// start.
pub(crate) trait TextLog: Send + Sync {
    /// The entry at `index`, once the request logged it.
    fn entry(&self, index: usize) -> BoxFuture<'_, Entry<String>>;
}

pub(crate) struct Fed {
    feeder: Feeder,
    source: TtsSource,
    /// The input failed: reported once the audio already produced is read.
    failed: Option<ProviderError>,
}

struct Feeder {
    log: Arc<dyn TextLog>,
    sink: TtsSink,
    /// The next entry to hand over.
    cursor: usize,
    sending: Option<BoxFuture<'static, Result<(), ProviderError>>>,
    /// The input took the log's end.
    ended: bool,
}

impl Fed {
    pub(crate) fn new(log: Arc<dyn TextLog>, sink: TtsSink, source: TtsSource) -> Self {
        Fed {
            feeder: Feeder { log, sink, cursor: 0, sending: None, ended: false },
            source,
            failed: None,
        }
    }
}

impl Feeder {
    /// Feeds the input until it fails. Cancel-safe: an entry in flight stays in flight.
    async fn feed(&mut self) -> ProviderError {
        loop {
            if let Some(sending) = &mut self.sending {
                let sent = sending.await;
                self.sending = None;
                if let Err(error) = sent {
                    return error;
                }
            }
            if self.ended {
                return std::future::pending().await;
            }
            let entry = self.log.entry(self.cursor).await;
            self.cursor += 1;
            self.ended = matches!(entry, Entry::EndInput);
            self.sending = Some(self.sink.send(entry));
        }
    }
}

impl StreamOutput<SynthesizedAudio> for Fed {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            // The feed goes first: a provider with audio ready would otherwise get no more text
            // until its audio ran out. Its failure waits for the audio already produced.
            loop {
                tokio::select! {
                    biased;
                    error = self.feeder.feed(), if self.failed.is_none() => {
                        self.failed = Some(error);
                    }
                    next = self.source.next() => return next,
                    () = std::future::ready(()), if self.failed.is_some() => {
                        return Err(self.failed.take().expect("guarded by the branch"));
                    }
                }
            }
        })
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        let Fed { feeder, source, .. } = *self;
        let Feeder { sink, sending, ended, .. } = feeder;
        // Ended unless the log's end never reached the input.
        if !ended || sending.is_some() {
            drop(sending);
            sink.end_input();
        }
        source.close(deadline)
    }
}

impl ProviderStream<SynthesizedAudio> for Fed {
    fn started_time(&self) -> Option<f64> {
        self.source.started_time()
    }
    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.source.take_call_metrics(in_flight)
    }
    fn call_ledger(&self) -> Option<CallLedger> {
        self.source.call_ledger()
    }
    fn report_retries(&mut self, report: RetryReport) {
        self.source.report_retries(report);
    }
}
