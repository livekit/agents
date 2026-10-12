//! `@livekit/agents-core` -- the napi-rs binding.
//!
//! The Node twin of `lk-agents-py`, and deliberately its mirror image wherever the two languages
//! differ. Read that crate first; every decision here is either the same decision or a documented
//! departure from it, and the departures are the interesting part.
//!
//! # What this crate is, in one sentence
//!
//! Layer 1 of the three-layer anti-drift structure: the shared Rust core is layer 1, the
//! *generated* TypeScript (`index.d.ts` from the `#[napi]` surface below, plus `@livekit/protocol`
//! for the wire types) is layer 2, and `node/@livekit-agents/src` -- the only place drift can
//! happen -- is layer 3.
//!
//! # Three things this crate proves end to end
//!
//! 1. **The ambient-frame round trip** ([`frame`]). Rust hands a
//!    [`FrameId`] out as *data*; the shim `als.run(...)`s it; JS hands it
//!    back on the way in. Never inherited across the FFI, always installed -- which is why the
//!    `agents-js` `warm_transfer.ts:176-180` bug is not reproducible here by construction.
//! 2. **The telemetry drain** ([`telemetry`]). Completed records, pulled by the host, materialised
//!    in the host's own OTel SDK. `HrTime` pairs, not integers -- see
//!    [`lk_agents_abi::telemetry`] for why the integer form is not merely awkward but lossy.
//! 3. **One provider through its trait** ([`providers`]). A JavaScript TTS driven through
//!    `lk_agents_abi::Tts`, both paths, with `FallbackChain` unable to tell it from a native one.
//!
//! # Where the shared halves of (1) and (2) live now
//!
//! Not here. The frame slab, its RAII scope and the four-way inbound resolution are
//! [`lk_agents_abi::utils::frame`]; the record shapes, the bounded drain and the span builder are
//! [`lk_agents_abi::telemetry`]. Both were written in this crate first, napi-free, and were
//! hoisted the moment `lk-agents-py` needed the same protocol -- two private copies of a contract
//! whose only job is to make the two hosts agree is the drift this project exists to remove.
//! What remains in [`frame`] and [`telemetry`] is the **FFI encoding**: `FrameView`, the `Either4`
//! span attribute, and the tagged `TelemetryRecord` that stands in for an enum napi cannot carry.
//!
//! # Importing this module must start nothing
//!
//! R13, §1.13 -- unchanged from `lk-agents-py`, and easier to honour: there is no `#[napi]`
//! initialiser here at all. Everything that starts lives behind constructing an [`AgentCore`].
//!
//! **There is no pid guard.** `lk-agents-py`'s `check_pid()` exists because a forked Python child
//! inherits a core with threads that did not survive. Node's `child_process.fork()` is a fresh
//! process with a fresh addon load, and `worker_threads` get their own env and their own
//! `AgentCore`; `posix.fork()` is not reachable from JS. So the hazard does not exist, and adding
//! a guard for it would be cargo-culted rather than reasoned.

use std::sync::{Arc, Mutex};

use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, RealClock, StreamSource, Tts};
use lk_agents_abi::FrameId;
use lk_agents_abi::utils::frame::{Frame, FrameTable, InboundFrame};
use lk_agents_abi::telemetry::{
    LogRecord, Severity, SpanBuilder, TelemetryBuffer, attr_bool, attr_num, attr_str, attrs,
};

use napi::bindgen_prelude::*;
use napi_derive::napi;

pub mod bridge;
pub mod errors;
pub mod frame;
/// The generated wrappers: `#[host(..)]` items and records, from `napi.rs` in `lk-agents-codegen`.
pub mod generated;
pub mod host_error;
pub mod host_table;
pub mod providers;
pub mod remote_session;
pub mod session;
pub mod telemetry;

use crate::errors::JsErrorOut;
use crate::frame::FrameView;
use crate::providers::NodeTts;
use crate::providers::host::TtsCallbacks;
use crate::telemetry::TelemetryRecord;

/// One session's worth of core state. **Per session, not global** -- §1.13's table is explicit
/// that the frame slab is per session because thread mode shares a process.
#[napi]
pub struct AgentCore {
    frames: Arc<FrameTable>,
    telemetry: Arc<TelemetryBuffer>,
    tts: Mutex<Option<Arc<dyn Tts>>>,
    /// The core's clock: the one every host stream's `close` deadline is read against, here and
    /// in the provider shim it is handed to.
    clock: Arc<dyn Clock>,
}

#[napi]
impl AgentCore {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        AgentCore {
            frames: Arc::new(FrameTable::new()),
            telemetry: Arc::new(TelemetryBuffer::default()),
            tts: Mutex::new(None),
            clock: Arc::new(RealClock::new()),
        }
    }

    // ------------------------------------------------------------------ §8e, the frame slab

    /// Allocate a frame. In the real driver this is internal -- `plan_reply` mints one per speech
    /// -- and is exposed here so the shim's tests can drive both legs without a session.
    #[napi]
    pub fn alloc_frame(
        &self,
        activity: i64,
        speech: Option<i64>,
        function_call: Option<String>,
        inline_ok: bool,
        traceparent: Option<String>,
    ) -> i64 {
        self.frames.alloc(Frame::new(activity, speech, function_call, inline_ok, traceparent)).raw()
    }

    /// What the shim needs to install its `AsyncLocalStorage` store. `None` for
    /// `FrameId::ROOT` and for an id this session never minted -- both of which are ordinary, and
    /// neither of which is an error.
    #[napi]
    pub fn frame_view(&self, id: i64) -> Option<FrameView> {
        let id = FrameId::from_raw(id);
        self.frames.get(id).map(|f| FrameView::of(id, &f))
    }

    /// The speech finished, or the activity was swapped. Later inbound calls carrying this id are
    /// *stale* -- still readable, still not an error.
    #[napi]
    pub fn retire_frame(&self, id: i64) {
        self.frames.retire(FrameId::from_raw(id));
    }

    /// §8e's inbound half, and the only shape it is allowed to have: **the host reads its own
    /// ambient storage and passes the id in.**
    ///
    /// There is deliberately no `currentFrame()` for JS to call and no way for Rust to read
    /// `AsyncLocalStorage`. The concrete thing this models is `_on_metrics_collected`
    /// (`agent_activity.py:1923-1928` / `agent_activity.ts:1580-1582`), which stamps `speech_id`
    /// on metrics emitted from inside a third-party plugin's own task -- the one case where only
    /// the host knows which speech an event belongs to.
    ///
    /// Returns the resolution so the caller can see staleness rather than guess: `"live"`,
    /// `"stale"`, `"root"`, `"unknown"`.
    #[napi]
    pub fn note_metrics(&self, frame: i64, metric: String) -> String {
        let frame = FrameId::from_raw(frame);
        let resolved = self.frames.resolve_inbound(frame);
        // The four spellings come from the core (`InboundFrame::kind`), not from a `match` here:
        // the Python binding reports the same four strings and they must not drift apart.
        let kind = resolved.kind();
        let speech = match &resolved {
            InboundFrame::Root | InboundFrame::Unknown => None,
            InboundFrame::Live(f) | InboundFrame::Stale(f) => f.speech,
        };
        let mut span = SpanBuilder::start("metrics", frame, None);
        span.attr(attrs::node::METRIC, attr_str(metric));
        span.attr(attrs::node::FRAME_RESOLUTION, attr_str(kind));
        if let Some(s) = speech {
            // The whole point of the inbound leg: without it this stamp is silently absent.
            span.attr(attrs::SPEECH_ID, attr_num(s as f64));
        }
        span.end(&self.telemetry);
        kind.to_owned()
    }

    /// How many frames the slab is holding. Test-only in practice; a slab that never shrinks is a
    /// per-speech leak for the whole session.
    #[napi]
    pub fn frame_count(&self) -> u32 {
        self.frames.len() as u32
    }

    // ------------------------------------------------------------------ §1.10, the drain

    /// Take everything buffered since the last call. The host calls this on a timer and once more
    /// during the closure sequence -- *drain, then flush the provider, then exit* -- which is
    /// §1.10's flush-ordering problem solved by construction.
    #[napi]
    pub fn drain_telemetry(&self) -> Vec<TelemetryRecord> {
        self.telemetry.drain().into_iter().map(Into::into).collect()
    }

    /// Records lost to a host that stopped draining. Non-zero is a bug in the host loop, and it
    /// being visible at all is the difference between bounded loss and silent loss.
    #[napi]
    pub fn dropped_telemetry(&self) -> i64 {
        self.telemetry.dropped() as i64
    }

    // ------------------------------------------------------------------ §1.11, the side tables

    /// Ids the shim may delete from its payload map.
    #[napi]
    pub fn drain_host_refs(&self) -> Vec<i64> {
        host_table::drain_released().into_iter().map(|i| i as i64).collect()
    }

    /// Ids the shim may delete from its *error* map.
    #[napi]
    pub fn drain_error_refs(&self) -> Vec<i64> {
        errors::drain_released()
    }

    /// Captured exceptions the shim is still holding for us. The leak detector.
    #[napi]
    pub fn live_error_count(&self) -> u32 {
        errors::live_count() as u32
    }

    /// Take ownership of a payload the shim has put in its map, and drop it again immediately --
    /// which is how a test asserts the RAII half works from JavaScript.
    #[napi]
    pub fn intern_and_drop_media(&self, id: i64) {
        drop(host_table::intern_media(id as u64));
    }

    // ------------------------------------------------------------------ §1.0, the provider seam

    /// Wrap a JavaScript TTS and hold it as an `Arc<dyn Tts>`.
    ///
    /// From here on nothing in the core knows it is talking to JavaScript, which is the entire
    /// claim being tested: the same `Arc<dyn Tts>` is what `FallbackChain` and the driver take.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn register_tts(
        &self,
        #[napi(
            ts_arg_type = "(frame: number, kind: 'synthesize' | 'stream', text: string | null) => Promise<JsOpenOutcome>"
        )]
        open: providers::host::OpenFn,
        #[napi(
            ts_arg_type = "(frame: number, stream: number, op: 'next' | 'push' | 'flush' | 'endInput' | 'close', text: string | null) => Promise<JsNextOutcome> | void"
        )]
        stream: providers::host::StreamFn,
        label: String,
        streaming: bool,
        aligned_transcript: bool,
        sample_rate: u32,
        num_channels: u16,
    ) {
        let tts = NodeTts::new(
            self.frames.clone(),
            TtsCallbacks { open, stream },
            label,
            streaming,
            aligned_transcript,
            sample_rate,
            num_channels,
            Arc::clone(&self.clock),
        );
        *self.tts.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(tts));
    }

    /// The four static properties, read back through the trait. If these came from the class and
    /// not the instance, six real plugins would be misrouted.
    #[napi]
    pub fn tts_info(&self) -> Result<TtsInfo> {
        let tts = self.tts()?;
        let caps = tts.capabilities();
        Ok(TtsInfo {
            label: tts.label(),
            streaming: caps.streaming,
            aligned_transcript: caps.aligned_transcript,
            sample_rate: tts.sample_rate(),
            num_channels: tts.num_channels(),
        })
    }

    /// Drive `Tts::synthesize` -- the simplex path, the one method all 39 packages implement.
    ///
    /// `frame` is the id the caller wants installed for the duration; pass `0` for
    /// `FrameId::ROOT`. The scope is entered here, on the driver, and every crossing inside picks
    /// it up without naming it.
    #[napi]
    pub async fn drive_synthesize(&self, frame: i64, text: String) -> Result<DriveOutcome> {
        let frame = FrameId::from_raw(frame);
        let tts = self.tts()?;
        let mut span = SpanBuilder::start("tts_synthesize", frame, None);
        span.attr(attrs::TTS_LABEL, attr_str(tts.label()));
        span.attr(attrs::TTS_STREAMING, attr_bool(tts.capabilities().streaming));

        let _f = self.frames.enter(frame);
        let result = async {
            let mut source = tts.synthesize(&text).await?;
            let out = collect(&mut *source).await;
            source.close(self.teardown_deadline()).await;
            out
        }
        .await;
        drop(_f);

        self.finish(frame, span, result)
    }

    /// Drive `Tts::stream` -- the duplex path, which is where the sink/source split earns itself.
    ///
    /// An instance whose capabilities say it has no duplex path is reported as `noDuplexPath`
    /// rather than asked to open one: it is an instance to wrap in `StreamAdapter`, not a
    /// failure. Every other instance is asked, and an `Err` is the real failure it says it is --
    /// including the `no_duplex_path` error a shim answers for an instance whose capabilities
    /// claimed a duplex path it turned out not to have.
    #[napi]
    pub async fn drive_stream(&self, frame: i64, tokens: Vec<String>) -> Result<DriveOutcome> {
        let frame = FrameId::from_raw(frame);
        let tts = self.tts()?;
        let mut span = SpanBuilder::start("tts_stream", frame, None);
        span.attr(attrs::TTS_LABEL, attr_str(tts.label()));

        if !tts.capabilities().streaming {
            span.attr(attrs::node::TTS_NO_DUPLEX_PATH, attr_bool(true));
            span.end(&self.telemetry);
            return Ok(DriveOutcome { no_duplex_path: true, ..DriveOutcome::default() });
        }

        let _f = self.frames.enter(frame);
        let result = match tts.stream().await {
            Err(err) => Err(err),
            Ok((sink, mut source)) => {
                let pushed = (|| {
                    for t in &tokens {
                        sink.push(t.clone())?;
                    }
                    sink.flush()?;
                    Ok::<(), lk_agents_abi::errors::ProviderError>(())
                })();
                sink.end_input();
                match pushed {
                    Err(e) => Err(e),
                    Ok(()) => {
                        let out = collect(&mut *source).await;
                        source.close(self.teardown_deadline()).await;
                        out
                    }
                }
            }
        };
        drop(_f);
        self.finish(frame, span, result)
    }

    /// When a host stream's `close` stops being waited for: [`HOST_TEARDOWN_TIMEOUT`] from now on
    /// the core's clock. The outcome has no trace to land in here, so it is not kept.
    fn teardown_deadline(&self) -> lk_agents_abi::Ts {
        self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)
    }

    fn tts(&self) -> Result<Arc<dyn Tts>> {
        self.tts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or_else(|| Error::from_reason("no TTS registered; call registerTts first"))
    }

    /// One exit point for both drivers: end the span, and turn a `ProviderError` into the scalars
    /// the shim needs to re-raise the original.
    fn finish(
        &self,
        span_frame: FrameId,
        mut span: SpanBuilder,
        result: std::result::Result<Collected, lk_agents_abi::errors::ProviderError>,
    ) -> Result<DriveOutcome> {
        match result {
            Ok(c) => {
                span.attr(attrs::node::TTS_FRAMES, attr_num(c.frames as f64));
                span.attr(attrs::node::TTS_BYTES, attr_num(c.bytes as f64));
                span.end(&self.telemetry);
                Ok(DriveOutcome {
                    frames: c.frames as i64,
                    bytes: c.bytes as i64,
                    text: c.text,
                    request_id: c.request_id,
                    started_time: c.started_time,
                    words: c.words,
                    ..DriveOutcome::default()
                })
            }
            Err(err) => {
                span.error(err.message().to_owned());
                span.attr(attrs::node::ERROR_RETRYABLE, attr_bool(err.retryable()));
                span.end(&self.telemetry);
                // The log half of §1.10, on the one record shape that exists before
                // `SessionReport` does: a provider failure. Nested attributes on purpose --
                // `LogAttributes` is `AnyValueMap` on both sides, so this half needs no
                // flattening and the session summary can be a document rather than 40 dotted keys.
                self.telemetry.push_log(
                    span_frame,
                    LogRecord::now(
                        "livekit.agents.tts",
                        err.message(),
                        Severity::Error,
                        serde_json::json!({
                            "error": {
                                "retryable": err.retryable(),
                                "statusCode": err.status_code(),
                                "kind": err.api_kind().map(|k| format!("{k:?}")),
                            },
                        }),
                    ),
                );
                let out = JsErrorOut::from(&err);
                // Hand the entry back: the shim is about to re-raise the original, so nothing
                // downstream needs it and a table that only grows is the leak this detects.
                errors::release(&err);
                Ok(DriveOutcome { error: Some(out), ..DriveOutcome::default() })
            }
        }
    }
}

#[derive(Default)]
struct Collected {
    frames: usize,
    bytes: usize,
    text: String,
    request_id: String,
    started_time: Option<f64>,
    words: Vec<String>,
}

/// Pull a `StreamSource` to exhaustion, the way the driver's `tts_node` does.
async fn collect(
    source: &mut dyn StreamSource<lk_agents_abi::SynthesizedAudio>,
) -> std::result::Result<Collected, lk_agents_abi::errors::ProviderError> {
    let mut c = Collected::default();
    while let Some(item) = source.next().await? {
        c.frames += 1;
        c.bytes += item.frame.data.len();
        c.text.push_str(&item.delta_text);
        if c.request_id.is_empty() {
            c.request_id = item.request_id.clone();
        }
        // R2b: `0.0` is an anchor, not an absence, so this is `is_none()` and not `unwrap_or(0.0)`.
        if c.started_time.is_none() {
            c.started_time = item.started_time;
        }
        c.words.extend(item.timed_transcript.iter().map(|t| t.text.clone()));
    }
    Ok(c)
}

#[napi(object)]
pub struct TtsInfo {
    pub label: String,
    pub streaming: bool,
    pub aligned_transcript: bool,
    pub sample_rate: u32,
    pub num_channels: u16,
}

#[napi(object)]
#[derive(Default)]
pub struct DriveOutcome {
    pub frames: i64,
    pub bytes: i64,
    pub text: String,
    pub request_id: String,
    pub started_time: Option<f64>,
    pub words: Vec<String>,
    /// The instance has no duplex path: it is one to wrap in `StreamAdapter`, not a failed turn.
    pub no_duplex_path: bool,
    pub error: Option<JsErrorOut>,
}
