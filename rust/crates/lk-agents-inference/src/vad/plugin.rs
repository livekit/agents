//! A plugin VAD's stream: its audio waits on the `Now` media queue a native detection uses, and a
//! pump hands it to the plugin one entry at a time, awaiting each. A plugin that falls behind
//! holds the pump, so the queue drops to the newest audio instead of the stream failing. The pump
//! runs while the reader pulls events; a gap is a flush, so the detector restarts on current audio.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::stream::{Entry, ProviderStream};
use lk_agents_abi::{
    AudioFrame, BoxFuture, Clock, Closed, DuplexStream, StreamInput, StreamOutput, Ts, VadEvent,
    VadSink, VadSource,
};
use lk_agents_media::media_queue::{AudioReceiver, Item};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::telemetry;
use tokio_util::sync::CancellationToken;

use super::stream::{Flush, Input};

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
        let cancelled = CancellationToken::new();
        let (input, audio) = Input::open(cancelled.clone(), cx);
        let pump =
            Pump { audio, sink, sending: None, ended: false, drops: input.drops.clone(), clock };
        DuplexStream::new(Box::new(input), Box::new(Source { events, pump, cancelled }))
    }
}

struct Pump {
    audio: AudioReceiver<Flush>,
    sink: Box<dyn StreamInput<AudioFrame>>,
    sending: Option<BoxFuture<'static, Result<(), ProviderError>>>,
    ended: bool,
    drops: Arc<AudioDrops>,
    clock: Arc<dyn Clock>,
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
            self.drops.catching_up(self.audio.head(), self.clock.now());
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
    cancelled: CancellationToken,
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
        let Source { events, pump, cancelled } = *self;
        cancelled.cancel();
        drop(pump);
        events.close(deadline)
    }
}

impl ProviderStream<VadEvent> for Source {}

#[cfg(test)]
#[path = "plugin_tests.rs"]
mod tests;
