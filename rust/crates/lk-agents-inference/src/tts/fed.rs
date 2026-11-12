//! A synthesis attempt's output, its text fed meanwhile: what the request logged goes through the
//! attempt's [`Stage`] and each chunk is handed to the attempt's input, each awaited, so a slow
//! input (a host TTS whose loop is busy) holds the feed instead of refusing text. The log keeps
//! the backlog.

use super::stage::Stage;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{CallLedger, Entry, ProviderStream, RetryReport};
use lk_agents_abi::{BoxFuture, Closed, StreamOutput, SynthesizedAudio, Ts, TtsSink, TtsSource};

/// A log written in full before it is read: a probe's replay.
pub(crate) struct FixedLog(pub(crate) Vec<Entry<String>>);

impl TextLog for FixedLog {
    fn entry(&self, index: usize) -> BoxFuture<'_, Entry<String>> {
        match self.0.get(index) {
            Some(entry) => Box::pin(std::future::ready(entry.clone())),
            None => Box::pin(std::future::pending()),
        }
    }
}

/// The text a synthesis request received, in order: every attempt's input replays it from the
/// start.
pub(crate) trait TextLog: Send + Sync {
    /// The entry at `index`, once the request logged it.
    fn entry(&self, index: usize) -> BoxFuture<'_, Entry<String>>;
}

pub(crate) struct Fed {
    feeder: Feeder,
    source: TtsSource,
    /// Seconds of audio the attempt produced: what the stage paces against.
    audio: f64,
    /// The input failed: reported once the audio already produced is read.
    failed: Option<ProviderError>,
}

struct Feeder {
    stage: Stage,
    sink: TtsSink,
    sending: Option<BoxFuture<'static, Result<(), ProviderError>>>,
    /// The input took the log's end.
    ended: bool,
}

impl Fed {
    pub(crate) fn new(stage: Stage, sink: TtsSink, source: TtsSource) -> Self {
        Fed {
            feeder: Feeder { stage, sink, sending: None, ended: false },
            source,
            audio: 0.0,
            failed: None,
        }
    }
}

impl Feeder {
    /// Feeds the input until it fails. Cancel-safe: an entry in flight stays in flight.
    async fn feed(&mut self, audio: f64) -> ProviderError {
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
            match self.stage.next(audio).await {
                Ok(Some(entry)) => {
                    self.ended = matches!(entry, Entry::EndInput);
                    self.sending = Some(self.sink.send(entry));
                }
                Ok(None) => self.ended = true,
                Err(error) => return error,
            }
        }
    }
}

impl StreamOutput<SynthesizedAudio> for Fed {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            // The feed goes first: a provider with audio ready would otherwise get no more text
            // until its audio ran out. Its failure waits for the audio already produced.
            loop {
                let audio = self.audio;
                tokio::select! {
                    biased;
                    error = self.feeder.feed(audio), if self.failed.is_none() => {
                        self.failed = Some(error);
                    }
                    next = self.source.next() => {
                        if let Ok(Some(frame)) = &next {
                            self.audio += frame.frame.duration();
                        }
                        return next;
                    }
                    () = std::future::ready(()), if self.failed.is_some() => {
                        return Err(self.failed.take().expect("guarded by the branch"));
                    }
                }
            }
        })
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        let Fed { feeder, source, .. } = *self;
        let Feeder { mut stage, sink, sending, ended } = feeder;
        // Ended unless the log's end never reached the input.
        if !ended || sending.is_some() {
            drop(sending);
            sink.end_input();
        }
        Box::pin(async move {
            stage.close(deadline).await;
            source.close(deadline).await
        })
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
