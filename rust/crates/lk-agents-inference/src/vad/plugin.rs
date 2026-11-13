use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{Entry, ProviderStream};
use lk_agents_abi::{
    AudioFrame, BoxFuture, Clock, Closed, DuplexStream, StreamInput, StreamOutput, Ts, VadEvent,
    VadSink, VadSource,
};
use lk_agents_media::media_queue::{AudioReceiver, Item};
use lk_agents_net::source::{Gated, InputGate};
use lk_agents_telemetry as telemetry;

use super::stream::{Flush, Input};

// A plugin that falls behind holds the pump, so the queue drops to the newest audio instead of
// the stream failing; a gap is a flush, so the detector restarts on current audio.

/// The wrapper a host VAD's `_core_stream` is handed out through.
pub struct PluginInput;

impl PluginInput {
    /// `stream` with its input queued; drops and the head's drain are measured under `cx`'s job.
    pub fn wrap(
        stream: DuplexStream<VadSink, VadSource>,
        cx: &telemetry::Cx,
        clock: Arc<dyn Clock>,
    ) -> DuplexStream<VadSink, VadSource> {
        let (sink, events) = stream.split();
        let gate = InputGate::default();
        let (input, audio) = Input::open(gate.clone(), cx, clock);
        let pump = Pump { audio, sink, sending: None, ended: false };
        DuplexStream::new(Box::new(input), Box::new(Gated::new(Source { events, pump }, gate)))
    }
}

struct Pump {
    audio: AudioReceiver<Flush>,
    sink: Box<dyn StreamInput<AudioFrame>>,
    sending: Option<BoxFuture<'static, Result<(), ProviderError>>>,
    ended: bool,
}

impl Pump {
    /// Feeds the plugin until its input fails. Cancel-safe: an entry in flight stays in flight.
    async fn run(&mut self) -> ProviderError {
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
            let entry = match self.audio.recv().await {
                Some(Item::Media(frame)) => Entry::Item(frame),
                Some(Item::Gap(_) | Item::Event(Flush)) => Entry::Flush,
                None => {
                    self.ended = true;
                    Entry::EndInput
                }
            };
            self.sending = Some(self.sink.send(entry));
        }
    }
}

struct Source {
    events: VadSource,
    pump: Pump,
}

impl StreamOutput<VadEvent> for Source {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
        Box::pin(async move {
            tokio::select! {
                biased;
                event = self.events.next() => event,
                error = self.pump.run() => Err(error),
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        self.events.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        let Source { events, pump } = *self;
        drop(pump);
        events.close(deadline)
    }
}

impl ProviderStream<VadEvent> for Source {}

#[cfg(test)]
#[path = "plugin_tests.rs"]
mod tests;
