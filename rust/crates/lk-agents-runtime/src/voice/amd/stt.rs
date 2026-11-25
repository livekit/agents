//! AMD's own STT leg (`AMDTranscriber`).
//!
//! **Feed.** The stream's sink sits on the audio router's [`AudioChannel::Amd`] slot, a raw
//! channel on the real side of the gate: it hears the participant's frames while the session's
//! STT hears silence (AEC warmup, uninterruptible speech), as Python's `push_audio` feeds AMD
//! before the substitution. The router drops every frame at its source until the gate applies
//! `Activated`, so AMD hears nothing before the activation instant, as the session does not.
//!
//! The stream opens when the run starts listening, as Python's opens on the first frame, and its
//! sink is installed once it opened; frames before that are not heard. Each non-blank final is
//! reported. A stream that will not open, fails or ends is reported once as failed: AMD then
//! uses the session's transcripts alone.

use super::executor::LegReport;
use crate::telemetry;
use crate::voice::legs::{OpeningLeg, PendingLeg};
use crate::voice::resources::Resources;
use crate::voice::session::Timebase;
use lk_agents_abi::{SpeechEvent, SpeechEventType, Stt};
use lk_agents_net::owned::Background;
use lk_agents_voice::audio::AudioChannel;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Opens `stt`'s stream with the session's `stt_conn_options` and reports it, opened or refused. A
/// stream opened after nobody listens is closed by its owner.
pub(super) async fn open(
    (stt, options): (Arc<dyn Stt>, lk_agents_abi::ApiConnectOptions),
    cx: telemetry::Cx,
    time: Timebase,
    resources: Resources,
    background: Background,
    cancel: CancellationToken,
    report: impl Fn(LegReport) -> bool,
) {
    let open = lk_agents_inference::stt::stream(stt, options, time.clock(), &cx, None, None);
    // Abandoning an open is safe: a host open dropped unsettled is cancelled, and a stream it
    // returned unread is closed.
    let opened = tokio::select! {
        biased;
        () = cancel.cancelled() => return,
        opened = open => opened,
    };
    let Ok(Some(pair)) = opened else {
        report(LegReport::SttFailed);
        return;
    };
    let (sink, source) = pair.split();
    let leg = OpeningLeg::new(
        AudioChannel::Amd,
        Arc::from(sink),
        PendingLeg::Stt(source),
        resources,
        &background,
    );
    // A refused report drops the leg, which closes it.
    report(LegReport::SttOpened(Box::new(leg)));
}

/// The final's words, if it has any: a blank final does not consume a turn's wait.
pub(super) fn final_text(event: &SpeechEvent) -> Option<String> {
    if event.kind != SpeechEventType::FinalTranscript {
        return None;
    }
    let text = event.alternatives.first()?.text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Reads the stream's finals until it ends, fails or the run stops it, then closes it. A stream
/// that ends on its own is a failure: no final can come any more.
pub(super) async fn drain(
    mut leg: OpeningLeg,
    cancel: CancellationToken,
    report: impl Fn(LegReport) -> bool,
) {
    let failed = match leg.stt_source() {
        None => true,
        Some(source) => loop {
            let next = tokio::select! {
                biased;
                () = cancel.cancelled() => break false,
                item = source.next() => item,
            };
            match next {
                Ok(Some(event)) => {
                    if let Some(text) = final_text(&event)
                        && !report(LegReport::AmdFinal(text))
                    {
                        break false;
                    }
                }
                Ok(None) | Err(_) => break true,
            }
        },
    };
    if failed && !cancel.is_cancelled() {
        report(LegReport::SttFailed);
    }
    let closed = leg.close().await;
    if closed != lk_agents_abi::Closed::Clean {
        log::debug!(target: lk_agents_abi::LOG_TARGET, outcome:? = closed; "AMD STT cleanup incomplete");
    }
}
