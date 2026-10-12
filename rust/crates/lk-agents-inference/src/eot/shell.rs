//! The host-facing shell around one mounted [`EotTransport`]. [`PredictionSink`] methods run on
//! the transport's task, so all but `session_created` only post to a channel; taking the host's
//! lock there is how deadlocks happen.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lk_agents_abi::EotInferenceMetrics;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::turn::eot::{MetricsEvent, PredictionEvent};
use lk_agents_abi::metrics::base::Metadata;
use lk_agents_abi::tokenize::language::{ThresholdError, ThresholdOptions};
use tokio::sync::mpsc;

use crate::eot::{EotTransport, EotWarning, PredictionSink, TransportPrediction};

/// `provider` (`base.py:65`): both native backends are the same vendor.
pub const PROVIDER: &str = "livekit";

/// One thing a transport reported, on its way to the host's `_apply` chain.
pub enum SinkEvent {
    Prediction(PredictionEvent),
    Metrics(Box<MetricsEvent>),
    Warn(EotWarning),
    /// The transport has stopped reporting. Posted through the channel so it arrives after every
    /// queued event.
    End,
}

pub struct ChannelSink {
    events: mpsc::UnboundedSender<SinkEvent>,
    thresholds: Arc<Mutex<ThresholdOptions>>,
    /// The model this transport is, for [`EotInferenceMetrics::metadata`]; constant per transport.
    model: &'static str,
}

impl PredictionSink for ChannelSink {
    fn resolve(&self, request_id: &str, prediction: TransportPrediction) {
        let _ = self.events.send(SinkEvent::Prediction(PredictionEvent {
            id: request_id.to_owned(),
            probability: prediction.probability,
            detection_delay: prediction.detection_delay,
            inference_duration: prediction.inference_duration,
            backchannel_probability: prediction.backchannel_probability,
        }));
    }

    /// Synchronous, unlike the rest: a rejection fails the connection, and the thresholds land on
    /// the shared table at once.
    fn session_created(
        &self,
        thresholds: &std::collections::BTreeMap<String, f64>,
        default: f64,
        backchannel_thresholds: Option<&std::collections::BTreeMap<String, f64>>,
        backchannel_default: f64,
    ) -> Result<(), ThresholdError> {
        self.thresholds.lock().expect("thresholds mutex").update_defaults(
            thresholds,
            default,
            backchannel_thresholds,
            backchannel_default,
        )
    }

    fn metrics(&self, metrics: EotInferenceMetrics) {
        let _ = self.events.send(SinkEvent::Metrics(Box::new(MetricsEvent::of(&metrics))));
    }

    fn warn(&self, warning: EotWarning) {
        let _ = self.events.send(SinkEvent::Warn(warning));
    }

    fn metadata(&self) -> Metadata {
        Metadata {
            model_name: Some(self.model.to_owned()),
            model_provider: Some(PROVIDER.to_owned()),
        }
    }
}

pub struct TransportShell {
    pub transport: Arc<dyn EotTransport>,
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<SinkEvent>>,
    /// The shell's own handle on the sink's channel, for posting [`SinkEvent::End`].
    end: mpsc::UnboundedSender<SinkEvent>,
    /// Makes `run()` return on `detach()`; the cloud receive half would otherwise stay parked on
    /// an open socket.
    stopped: tokio::sync::Notify,
    ended: AtomicBool,
}

impl TransportShell {
    /// Mount the transport `build` makes over a sink reporting as `model`. One owner holds both
    /// channel ends, so [`SinkEvent::End`] follows the last report.
    pub fn mount(
        model: &'static str,
        thresholds: Arc<Mutex<ThresholdOptions>>,
        build: impl FnOnce(Arc<dyn PredictionSink>) -> Arc<dyn EotTransport>,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let sink = Arc::new(ChannelSink { events: tx.clone(), thresholds, model });
        Arc::new(TransportShell {
            transport: build(sink),
            events: tokio::sync::Mutex::new(rx),
            end: tx,
            stopped: tokio::sync::Notify::new(),
            ended: AtomicBool::new(false),
        })
    }

    /// Idempotent, and called from both ends: a transport that finished on its own and a host
    /// that detached one. Whichever happens first ends the host's pump.
    pub fn finish(&self) {
        if !self.ended.swap(true, Ordering::SeqCst) {
            let _ = self.end.send(SinkEvent::End);
        }
        self.stopped.notify_waiters();
    }

    /// Stop, without waiting for anything in flight.
    pub fn detach(&self) {
        self.transport.detach();
        self.finish();
    }

    /// Run the transport for its whole lifetime. `Ok` is "the audio ended"; `Err` goes to the
    /// session's `transport_failed`.
    pub async fn run(&self) -> Result<(), ApiError> {
        if self.ended.load(Ordering::SeqCst) {
            return Ok(());
        }
        let result = tokio::select! {
            result = self.transport.run() => result,
            // `detach()` while parked: not an error.
            () = self.stopped.notified() => Ok(()),
        };
        self.finish();
        result
    }

    /// The next thing the transport reported, or `None` after [`SinkEvent::End`] (everything
    /// queued ahead of it is delivered first).
    pub async fn next_event(&self) -> Option<SinkEvent> {
        match self.events.lock().await.recv().await {
            Some(SinkEvent::End) | None => None,
            Some(event) => Some(event),
        }
    }
}
