//! The machinery `NodeTts` (and, when they are written, `NodeLlm`/`NodeStt`/`NodeVad`) share:
//! the threadsafe-function bundle, the two stream halves, and the frame that rides every call.

use async_trait::async_trait;
use lk_agents_abi::{AudioFrame, Clock, Closed, StreamSink, StreamSource, SynthesizedAudio, Ts};
use lk_agents_abi::FrameId;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::TimedString;
use napi::Status;
use napi::bindgen_prelude::{FnArgs, Promise, Uint8Array};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

use crate::errors::JsProviderError;

// --------------------------------------------------------------------- the wire types
//
// Everything below the `#[napi(object)]` line is what §1.10/§Context call the *generated* layer:
// one definition, rendered into `index.d.ts` by the napi CLI. Hand-writing the TypeScript twin of
// these is exactly the drift the three-layer structure exists to prevent, so nothing in
// `node/@livekit-agents/src` redeclares them -- it imports them from the generated `.d.ts`.

/// `rtc.AudioFrame`, flattened. `Uint8Array` and not `Buffer`: the shim may be handed either, and
/// a `Buffer` *is* a `Uint8Array` in JS, so this accepts both.
#[napi(object)]
pub struct JsAudioFrame {
    pub data: Uint8Array,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub samples_per_channel: u32,
}

// The word timing is the declared `TimedString`'s generated twin (`generated::records`), not a
// hand-written copy of its six fields. Named as itself: napi writes `index.d.ts` from the Rust
// spelling, so an alias here would name a type the `.d.ts` never declares.

/// `tts.SynthesizedAudio`, plus the two fields the Python path has to dig out of
/// `rtc.AudioFrame.userdata`. **They are ordinary named fields here**, because there is no
/// back-compat obligation on the JS side and `USERDATA_TIMED_TRANSCRIPT` /
/// `USERDATA_TTS_STARTED_TIME` are a Python-specific smuggling channel, not a wire format.
#[napi(object)]
pub struct JsSynthesizedAudio {
    pub frame: JsAudioFrame,
    pub request_id: String,
    pub is_final: bool,
    pub segment_id: String,
    pub delta_text: String,
    /// R2b: `0.0` is a legitimate anchor and must not collapse into "absent". `Option` in Rust,
    /// `number | null` in TS, and the shim must not use `??` to default it.
    pub started_time: Option<f64>,
    pub timed_transcript: Vec<crate::generated::records::TimedString>,
}

/// The result of a factory call. Exactly one of `stream`/`error` is set.
#[napi(object)]
pub struct JsOpenOutcome {
    /// The shim's id for the stream object it just constructed.
    pub stream: Option<i64>,
    pub error: Option<JsProviderError>,
    /// `true` when the failure means "this instance has no duplex path" -- the base class's
    /// `NotImplementedError` equivalent, reported as `no_duplex_path` like a `streaming: false`
    /// capability, so the instance is wrapped in `StreamAdapter` where it enters the engine.
    /// Conflating it with a failed open is a real bug: a websocket plugin that cannot open its
    /// socket must fail loudly rather than be silently downgraded to sentence-at-a-time
    /// synthesis.
    pub no_duplex_path: bool,
}

/// One pull. All three fields absent is a clean end of stream.
#[napi(object)]
pub struct JsNextOutcome {
    pub item: Option<JsSynthesizedAudio>,
    pub error: Option<JsProviderError>,
}

// --------------------------------------------------------------------- the tsfn bundle

/// `CalleeHandled = false`: the shim's functions are ordinary `(args) => value`, not
/// node-style `(err, ...args)` callbacks.
///
/// A JS `throw` is still delivered to Rust as `Err` -- napi captures the pending exception in
/// `call_js_cb` before the `CalleeHandled` branch is reached, and retains the original thrown
/// value behind a refcounted `ErrorRef`. The `fatal_exception` path the `call_async` doc warns
/// about is only reachable when the *Rust-side argument marshalling* fails, and every marshaller
/// here builds primitives.
/// `FnArgs<(..)>` and not a bare tuple: a bare tuple is `ToNapiValue` and therefore arrives in
/// JavaScript as a **single array argument**, silently. `FnArgs` is the marker that spreads it
/// into real parameters. This cost a debugging round -- the failure is a `NumberExpected` from the
/// *next* call, not from the crossing that got it wrong.
type Tsfn<Args, Ret> = ThreadsafeFunction<FnArgs<Args>, Ret, FnArgs<Args>, Status, false>;

/// Open a factory: `(frame, kind, text) => Promise<JsOpenOutcome>`. `kind` is `"synthesize"` or
/// `"stream"`; `text` is present only for the former.
pub type OpenFn = Tsfn<(i64, String, Option<String>), Promise<JsOpenOutcome>>;

/// **Every operation on a stream goes through one threadsafe function**, discriminated by `op`:
/// `"next"`, `"push"`, `"flush"`, `"endInput"`, `"close"`.
///
/// # This is a correctness requirement, not a tidiness one, and it cost a flaky test to find
///
/// napi gives **one queue per threadsafe function**, and there is no ordering guarantee *between*
/// two of them. With `push` on its own tsfn and `next` on another, a pull can be delivered to
/// JavaScript before the pushes that preceded it -- observed as a duplex synthesis that returned
/// zero frames, roughly one run in four.
///
/// Python does not have this hazard and it is worth being precise about why: every crossing there
/// goes through `loop.call_soon_threadsafe` onto **the loop's single `_ready` queue**, so FIFO
/// across all operations is free. Node's equivalent of that single queue is *one tsfn*. Splitting
/// operations across several is the same class of mistake as calling a Python stream method
/// directly instead of through the loop -- and it fails the same way, silently and intermittently.
///
/// The sink/source split (§1.0) is unaffected: they are still two objects with two lifetimes, and
/// `next` still takes `&mut self` while `push` takes `&self`. They share a *transport*, not a
/// state machine, and Rust still never fuses a `push` and a `next` into one `select!`.
pub type StreamFn = Tsfn<(i64, i64, String, Option<String>), Promise<JsNextOutcome>>;

/// The two functions a TTS host provides.
///
/// **Not `#[napi(object)]`.** A `ThreadsafeFunction` is `FromNapiValue` but not `ToNapiValue` --
/// it is a one-way door by construction -- and `#[napi(object)]` derives both. So they arrive as
/// two parameters of `registerTts`, which the generated `.d.ts` types individually anyway.
pub struct TtsCallbacks {
    pub open: OpenFn,
    pub stream: StreamFn,
}

/// Everything a shim needs to talk to one host provider, plus the frame source.
///
/// **`Arc`, and cloned into every stream.** A `ThreadsafeFunction` keeps the JS environment alive
/// for as long as it exists, which is what makes the whole `TaskLocals` capture dance from
/// `lk-agents-py` unnecessary here: there is no "loop that might not be running yet" to snapshot.
pub struct NodeHandle {
    /// `Arc`, because a `ThreadsafeFunction` is deliberately not `Clone`: cloning one would
    /// duplicate the ref that keeps the JS environment alive, and napi makes you say which one
    /// owns it. Every stream shares the provider's six.
    pub cb: std::sync::Arc<TtsCallbacks>,
    /// Which stream on the host side. `-1` for the provider itself, which has no stream.
    pub stream: i64,
}

impl NodeHandle {
    /// §1.6's single choke point, and the reason it is a method rather than six call sites: the
    /// frame is injected **here**, so no caller can forget it. This is the Node twin of the
    /// `port!` macro (§2.1) -- `port!(io, call_tool(...))` expands to
    /// `io.call_tool(self.current_frame(), ...)` for exactly the same reason.
    pub(crate) async fn open(
        &self,
        frame: FrameId,
        kind: &'static str,
        text: Option<String>,
    ) -> Result<JsOpenOutcome, ProviderError> {
        // `.raw()` here and nowhere else: the newtype is what keeps `frame` from being confused
        // with `stream`, `activity` or `speech` -- all `i64` too -- everywhere above this line.
        let promise = self
            .cb
            .open
            .call_async((frame.raw(), kind.to_owned(), text).into())
            .await
            .map_err(loop_gone)?;
        promise.await.map_err(loop_gone)
    }
}

/// A napi transport failure -- the environment shutting down, the queue closing. Not a provider
/// error, and not something a caller can act on beyond ending the stream.
fn loop_gone(e: napi::Error) -> ProviderError {
    ApiError::new(format!("the host environment stopped responding: {e}")).into()
}

// --------------------------------------------------------------------- conversion

/// The other direction, for a proxy handing a frame to JS (one copy into the JS-owned buffer).
pub fn audio_frame_to_js(f: &AudioFrame) -> JsAudioFrame {
    JsAudioFrame {
        data: Uint8Array::from(f.data.to_vec()),
        sample_rate: f.sample_rate,
        num_channels: f.num_channels,
        samples_per_channel: f.samples_per_channel,
    }
}

/// `AudioEncoding` as one string: `"pcm"`, or the encoded container's MIME type.
pub fn encoding_to_js(e: &lk_agents_abi::AudioEncoding) -> String {
    match e {
        lk_agents_abi::AudioEncoding::Pcm => "pcm".to_owned(),
        lk_agents_abi::AudioEncoding::Encoded { mime } => mime.clone(),
    }
}

pub fn encoding_from_js(s: String) -> lk_agents_abi::AudioEncoding {
    if s == "pcm" { lk_agents_abi::AudioEncoding::Pcm } else { lk_agents_abi::AudioEncoding::Encoded { mime: s } }
}

pub fn audio_frame_from_js(f: JsAudioFrame) -> AudioFrame {
    // Copied out of the `Uint8Array` immediately and the JS handle dropped: holding a `napi_ref`
    // on a tokio worker for the life of a frame would pin the backing store and route every drop
    // back through the custom-GC threadsafe function at frame rate.
    AudioFrame {
        data: f.data.to_vec().into(),
        sample_rate: f.sample_rate,
        num_channels: f.num_channels,
        samples_per_channel: f.samples_per_channel,
    }
}

pub fn synthesized_audio_from_js(a: JsSynthesizedAudio) -> SynthesizedAudio {
    SynthesizedAudio {
        frame: audio_frame_from_js(a.frame),
        request_id: a.request_id,
        is_final: a.is_final,
        segment_id: a.segment_id,
        delta_text: a.delta_text,
        started_time: a.started_time,
        timed_transcript: a
            .timed_transcript
            .into_iter()
            .map(TimedString::from)
            .collect(),
        // A JS plugin hands over decoded PCM (`AudioFrame` is PCM).
        encoding: lk_agents_abi::AudioEncoding::Pcm,
    }
}

// --------------------------------------------------------------------- the two halves

/// The drain half: one pull plus a converter. One type for every source, exactly as
/// `lk-agents-py`'s `HostSource` is -- the five bodies came out identical there and would here.
pub struct NodeSource<T> {
    handle: std::sync::Arc<NodeHandle>,
    convert: fn(JsSynthesizedAudio) -> T,
    frame: FrameId,
    /// The clock the `deadline` handed to [`StreamSource::close`] is on -- the core's, held by the
    /// provider that minted this source.
    clock: std::sync::Arc<dyn Clock>,
}

impl<T> NodeSource<T> {
    pub fn new(
        handle: std::sync::Arc<NodeHandle>,
        frame: FrameId,
        convert: fn(JsSynthesizedAudio) -> T,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Self {
        NodeSource { handle, convert, frame, clock }
    }
}

impl NodeHandle {
    /// Fire and forget, in FIFO order behind everything already queued on [`StreamFn`].
    ///
    /// **`Ok(())` means "handed over", not "the host accepted it"** (§1.2d, unchanged in Node). A
    /// failure raised *inside* the JS callback surfaces on the source, which is the authoritative
    /// error path.
    fn signal(&self, frame: FrameId, op: &'static str, text: Option<String>) -> Status {
        self.cb.stream.call(
            (frame.raw(), self.stream, op.to_owned(), text).into(),
            ThreadsafeFunctionCallMode::NonBlocking,
        )
    }

    async fn ask(&self, frame: FrameId, op: &'static str) -> Result<JsNextOutcome, ProviderError> {
        let promise = self
            .cb
            .stream
            .call_async((frame.raw(), self.stream, op.to_owned(), None).into())
            .await
            .map_err(loop_gone)?;
        promise.await.map_err(loop_gone)
    }
}

#[async_trait]
impl<T: Send> StreamSource<T> for NodeSource<T> {
    /// `Err` is a provider failure; `Ok(None)` is a clean end of stream. The two must not collapse
    /// -- §1.0's reason holds in JS too: an async iterator that rejects and one that completes are
    /// two different answers arriving on the same call.
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        let out = self.handle.ask(self.frame, "next").await?;
        if let Some(err) = out.error {
            return Err(err.into());
        }
        Ok(out.item.map(self.convert))
    }

    /// §1.2c: the only teardown. Dropping the future does not stop the JS side any more than it
    /// stops a Python task -- an async generator parked on an `await` stays parked. The host's
    /// `close` is awaited until `deadline` on the core's clock; past it the caller traces
    /// `TimedOut`. Consuming, so "closed twice" is not expressible and needs no flag.
    async fn close(self: Box<Self>, deadline: Ts) -> Closed {
        Closed::bounded(&*self.clock, deadline, async {
            let _ = self.handle.ask(self.frame, "close").await;
        })
        .await
    }
}

/// The push half. `&self` and `Sync`, so the pump can hold it while the source is drained
/// elsewhere -- the split §1.0 insists on, for the measured data-loss reason.
pub struct NodeSink {
    handle: std::sync::Arc<NodeHandle>,
    frame: FrameId,
}

impl NodeSink {
    pub fn new(handle: std::sync::Arc<NodeHandle>, frame: FrameId) -> Self {
        NodeSink { handle, frame }
    }
}

impl StreamSink<String> for NodeSink {
    /// `NonBlocking` is deliberate: this is the text->TTS leg, which may not drop, and blocking a
    /// tokio worker on a full tsfn queue would be worse than the unbounded queue napi gives us by
    /// default (`MaxQueueSize = 0`).
    fn push(&self, item: String) -> Result<(), ProviderError> {
        match self.handle.signal(self.frame, "push", Some(item)) {
            Status::Ok => Ok(()),
            status => {
                Err(ApiError::new(format!("push into the host stream failed: {status}")).into())
            }
        }
    }

    fn flush(&self) -> Result<(), ProviderError> {
        match self.handle.signal(self.frame, "flush", None) {
            Status::Ok => Ok(()),
            status => {
                Err(ApiError::new(format!("flush of the host stream failed: {status}")).into())
            }
        }
    }

    fn end_input(&self) {
        // Returns nothing on purpose: a failure here means the environment is already gone, which
        // the source reports as the end of the stream.
        let _ = self.handle.signal(self.frame, "endInput", None);
    }
}
