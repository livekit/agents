//! Python channel, metadata and loop callbacks for the generated TTS emitter class.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use std::time::Duration;

use lk_agents_abi::SynthesizedAudio as CoreAudio;
use lk_agents_abi::tts::emitter::{
    AudioEmitter as Machine, Cmd, EmitterDriver, EmitterError, EmitterHost, HostEffect, PumpError,
    SegmentDump,
};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyList, PyType};

type Finalization = tokio::sync::watch::Receiver<Option<Result<(), Arc<PyErr>>>>;

fn emitter_error_to_py(e: EmitterError) -> PyErr {
    lk_agents_abi::python::HostException::to_py_err(&e)
}

/// Resolved lazily: `livekit.agents.tts` imports `_core`, so resolving at module init is a cycle.
fn cached<'py>(
    slot: &'static PyOnceLock<Py<PyType>>,
    py: Python<'py>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    slot.get_or_try_init(py, || {
        py.import(module)?
            .getattr(name)?
            .cast_into::<PyType>()
            .map(|t| t.unbind())
            .map_err(PyErr::from)
    })
    .map(|t| t.bind(py))
}

// A plugin calls this at frame rate while a loop timer can re-enter it (FFI hazard #9): the class
// is `frozen` with state behind `Mutex`, and no Python call happens while the machine lock is held.

const USERDATA_TIMED_TRANSCRIPT: &str = lk_agents_abi::USERDATA_TIMED_TRANSCRIPT;
const ATTR_PROVIDER_REQUEST_IDS: &str = lk_agents_abi::telemetry::attrs::PROVIDER_REQUEST_IDS;

/// `LK_DUMP_TTS`, read once; a non-integer value disables dumping.
fn dump_default() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LK_DUMP_TTS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .is_some_and(|v| v != 0)
    })
}

fn synthesized_audio_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static T: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    cached(&T, py, "livekit.agents.tts", "SynthesizedAudio")
}

fn timed_string_type(py: Python<'_>) -> Option<&Bound<'_, PyType>> {
    static T: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    cached(&T, py, "livekit.agents.types", "TimedString").ok()
}

fn logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("livekit.agents.log")?.getattr("logger")
}

pub(crate) struct State {
    context: Mutex<lk_agents_abi::HostRef>,
    compressed: Mutex<bool>,
    closed: AtomicBool,
    finalization: Mutex<Option<Finalization>>,
    /// Effects come back through [`PyHost`] with no lock held.
    driver: EmitterDriver,
    dst_ch: Py<PyAny>,
    label: String,
    /// Host `TimedString`s kept verbatim so alignment fields survive; drained as the machine assigns.
    words: Mutex<VecDeque<Py<PyAny>>>,
    /// Rebuilt per `initialize`, so one request never gets the tail of the previous one.
    decoder: Mutex<Option<lk_agents_runtime::media::Decoder>>,
    timer: Mutex<Option<Py<PyAny>>>,
}

fn pump_err(e: PumpError<PyErr>) -> PyErr {
    match e {
        PumpError::Machine(e) => emitter_error_to_py(e),
        PumpError::Host(e) => e,
    }
}

/// Called with no lock held, so a `send_nowait` that re-enters the emitter just queues.
struct PyHost<'a>(&'a Arc<State>);

impl EmitterHost for PyHost<'_> {
    type Error = PyErr;

    fn deliver(&mut self, effect: HostEffect) -> PyResult<()> {
        run_now(self.0, move |state, py| match effect {
            HostEffect::Send(audio) => state.send(py, *audio),
            HostEffect::Arm(delay) => State::arm_timer(state, py, delay),
            HostEffect::Cancel => state.cancel_timer(py),
            HostEffect::Dump(dump) => state.write_dump(py, &dump),
            HostEffect::Dropped(bytes) => {
                logger(py)?.call_method1(
                    "warning",
                    (format!(
                        "incomplete PCM sample at end of segment, discarding {bytes} trailing bytes"
                    ),),
                )?;
                Ok(())
            }
            HostEffect::ProviderIds(ids) => state.set_provider_ids(py, &ids),
        })
    }
}

impl State {
    /// No `check_pid()`: built per synthesis in whatever process imported the package, and the
    /// reference has no init requirement for `TTS`.
    pub(crate) fn new(py: Python<'_>, label: &str, dst_ch: Py<PyAny>) -> PyResult<Self> {
        let mut machine = Machine::new(label);
        machine.set_dump(dump_default());
        Ok(State {
            context: Mutex::new(capture(py)?),
            compressed: Mutex::new(false),
            closed: AtomicBool::new(false),
            finalization: Mutex::new(None),
            driver: EmitterDriver::new(machine),
            dst_ch,
            label: label.to_string(),
            words: Mutex::new(VecDeque::new()),
            decoder: Mutex::new(None),
            timer: Mutex::new(None),
        })
    }

    pub(crate) fn ensure_unstarted(&self) -> PyResult<()> {
        if self.driver.machine().started() {
            return Err(emitter_error_to_py(EmitterError::AlreadyStarted));
        }
        Ok(())
    }

    pub(crate) fn initialize(
        slf: &Arc<Self>,
        py: Python<'_>,
        request_id: &str,
        sample_rate: u32,
        num_channels: u16,
        mime_type: &str,
        frame_size_ms: u32,
        stream: bool,
    ) -> PyResult<()> {
        let me = slf;
        me.ensure_unstarted()?;

        // `audio/pcm` and `audio/raw` build no decoder; other containers are decoded. Unknown mime
        // types are refused by `lk_agents_runtime::media::decode_plan`, as the reference's
        // `NotImplementedError`.
        let decoder = match lk_agents_runtime::media::decode_plan(mime_type)
            .map_err(|e| PyNotImplementedError::new_err(format!("AudioEmitter: {e}")))?
        {
            lk_agents_runtime::media::DecodePlan::Passthrough => None,
            lk_agents_runtime::media::DecodePlan::Decode(mt) => Some(
                lk_agents_runtime::media::Decoder::with_workers(
                    &lk_agents_runtime::process::workers()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                    &mt,
                    Some(sample_rate),
                    num_channels,
                )
                .map_err(|e| PyNotImplementedError::new_err(format!("AudioEmitter: {e}")))?,
            ),
        };
        me.driver
            .machine()
            .initialize(request_id, sample_rate, num_channels, frame_size_ms, stream)
            .map_err(emitter_error_to_py)?;
        *me.compressed.lock().unwrap_or_else(|e| e.into_inner()) = decoder.is_some()
            && lk_agents_runtime::media::mime_to_av_format(Some(mime_type)) != Some("wav");
        me.closed.store(false, Ordering::Release);
        *me.decoder.lock().unwrap_or_else(|e| e.into_inner()) = decoder;

        if request_id.is_empty() {
            // Verbatim, %-style, as the reference logs it.
            logger(py)?
                .call_method1("warning", ("no request_id provided for TTS %s", &me.label))?;
        }

        Self::drain_on_output(slf, py)?;

        // `stream=False` opens a segment here; drain so no stale queue is left.
        Self::pump(slf, py)
    }

    pub(crate) fn start_segment(slf: &Arc<Self>, py: Python<'_>, segment_id: &str) -> PyResult<()> {
        Self::submit(slf, py, Cmd::StartSegment(segment_id.to_string()))
    }

    pub(crate) fn end_segment(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        Self::submit(slf, py, Cmd::EndSegment)
    }

    /// `bytes | bytearray | memoryview`, as the reference takes.
    pub(crate) fn push(slf: &Arc<Self>, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::submit(slf, py, Cmd::Push(Self::decoded(slf, py, pcm_arg(data)?)?))
    }

    /// An already framed `rtc.AudioFrame`, sent as it is.
    pub(crate) fn push_frame(
        slf: &Arc<Self>,
        py: Python<'_>,
        frame: lk_agents_abi::AudioFrame,
    ) -> PyResult<()> {
        let samples_per_channel = frame.samples_per_channel;
        let frame = lk_agents_abi::Frame { data: frame.data.to_vec(), samples_per_channel };
        Self::submit(slf, py, Cmd::Frame(frame))
    }

    /// `TimedString | list[TimedString]`.
    pub(crate) fn push_timed_transcript(
        slf: &Arc<Self>,
        py: Python<'_>,
        delta_text: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // A `TimedString` is a `str`, and iterating it would push one word per character.
        let items: Vec<Bound<'_, PyAny>> = match timed_string_type(py) {
            Some(t) if delta_text.is_instance(t)? => vec![delta_text.clone()],
            _ if delta_text.is_instance_of::<PyList>() => {
                delta_text.try_iter()?.collect::<PyResult<_>>()?
            }
            Some(_) => vec![delta_text.clone()],
            // No `TimedString` class (Rust unit tests only).
            None if delta_text.is_instance_of::<pyo3::types::PyString>() => {
                vec![delta_text.clone()]
            }
            None => delta_text.try_iter()?.collect::<PyResult<_>>()?,
        };

        // `generation.py:432` drops non-`TimedString` entries; filter once so the host queue and
        // the machine's list keep the same length.
        let kept: Vec<Bound<'_, PyAny>> = match timed_string_type(py) {
            Some(t) => items
                .into_iter()
                .filter_map(|i| i.is_instance(t).ok().and_then(|b| b.then_some(i)))
                .collect(),
            None => items,
        };
        if kept.is_empty() {
            return Ok(());
        }
        let rust: Vec<lk_agents_abi::TimedString> = kept
            .iter()
            .map(|w| lk_agents_abi::python::extract_generated::<lk_agents_abi::TimedString>(w))
            .collect::<PyResult<_>>()?;

        {
            let me = slf;
            let mut q = me.words.lock().unwrap_or_else(|e| e.into_inner());
            for k in kept {
                q.push_back(k.unbind());
            }
        }
        Self::submit(slf, py, Cmd::Words(rust))
    }

    pub(crate) fn flush(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        Self::submit(slf, py, Cmd::Flush)
    }

    pub(crate) fn end_input(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        if *slf.compressed.lock().unwrap_or_else(|e| e.into_inner()) {
            return Self::finish_compressed(slf);
        }
        // Flush the decoder first, or the last packet's tail is lost. An empty tail is not pushed:
        // the machine counts pushes, and PCM streams never had that boundary.
        // The decoder goes with its tail, and with it the callback holding this emitter.
        let tail = {
            let decoder = slf.decoder.lock().unwrap_or_else(|e| e.into_inner()).take();
            match decoder {
                None => Vec::new(),
                Some(mut d) => py
                    .detach(move || d.flush())
                    .map(|pcm| lk_agents_runtime::media::pcm_bytes(&pcm))
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("AudioEmitter: decode flush failed: {e}"))
                    })?,
            }
        };
        if !tail.is_empty() {
            Self::submit(slf, py, Cmd::Push(tail))?;
        }
        Self::submit(slf, py, Cmd::EndInput)
    }

    fn finish_compressed(slf: &Arc<Self>) -> PyResult<()> {
        let pending = slf.finalization.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_some() {
            return Ok(());
        }
        drop(pending);
        let runtime = lk_agents_runtime::process::handle()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let receipt = slf
            .decoder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
            .and_then(|decoder| decoder.end_input());
        Self::track_finalization(
            slf,
            async move {
                if let Some(receipt) = receipt {
                    receipt.await?;
                }
                Ok(())
            },
            |state, ()| {
                let mut decoder = state.decoder.lock().unwrap_or_else(|e| e.into_inner());
                match decoder.as_mut() {
                    Some(decoder) => {
                        decoder.flush().map(|pcm| lk_agents_runtime::media::pcm_bytes(&pcm))
                    }
                    None => Ok(Vec::new()),
                }
            },
            &runtime,
        );
        Ok(())
    }

    fn track_finalization<T: Send + 'static>(
        slf: &Arc<Self>,
        receipt: impl Future<Output = Result<T, lk_agents_runtime::media::workers::WorkerPanicked>>
        + Send
        + 'static,
        drain: impl FnOnce(&Arc<Self>, T) -> lk_agents_runtime::media::Result<Vec<u8>> + Send + 'static,
        runtime: &tokio::runtime::Handle,
    ) {
        let (settled, receiver) = tokio::sync::watch::channel(None);
        *slf.finalization.lock().unwrap_or_else(|e| e.into_inner()) = Some(receiver);
        let state = slf.clone();
        runtime.spawn(async move {
            let result = match receipt.await {
                Ok(_) if state.closed.load(Ordering::Acquire) => Ok(()),
                Ok(native) => {
                    run(&state, Default::default(), move |state, py| {
                        if state.closed.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        let tail = drain(state, native).map_err(|error| {
                            PyRuntimeError::new_err(format!(
                                "AudioEmitter: decode flush failed: {error}"
                            ))
                        })?;
                        if !tail.is_empty() {
                            Self::submit(state, py, Cmd::Push(tail))?;
                        }
                        Self::submit(state, py, Cmd::EndInput)
                    })
                    .await
                }
                Err(error) => Err(PyRuntimeError::new_err(error.to_string())),
            };
            settled.send_replace(Some(result.map_err(Arc::new)));
        });
    }

    pub(crate) async fn finalized(&self) -> PyResult<()> {
        let pending = self.finalization.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(mut pending) = pending else {
            return Ok(());
        };
        loop {
            if let Some(result) = pending.borrow_and_update().as_ref() {
                return match result {
                    Ok(()) => Ok(()),
                    Err(error) => Err(Python::attach(|py| error.clone_ref(py))),
                };
            }
            pending
                .changed()
                .await
                .map_err(|_| PyRuntimeError::new_err("audio finalizer ended without settlement"))?;
        }
    }

    pub(crate) fn join(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        if !slf.driver.machine().started() {
            return Err(emitter_error_to_py(EmitterError::NotStarted));
        }
        Self::pump(slf, py)
    }

    pub(crate) fn aclose(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        slf.closed.store(true, Ordering::Release);
        slf.driver.stop();
        let closing = slf
            .decoder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .and_then(|mut decoder| decoder.close());
        slf.words.lock().unwrap_or_else(|e| e.into_inner()).clear();
        slf.cancel_timer(py)?;
        if let Some(closing) = closing {
            let runtime = lk_agents_runtime::process::handle()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Self::track_finalization(
                slf,
                async move { closing.await.map(|()| Ok(Vec::new())) },
                |_, result| result,
                &runtime,
            );
        }
        Ok(())
    }

    pub(crate) fn pushed_duration(&self, idx: i64) -> f64 {
        self.driver.machine().pushed_duration(idx)
    }

    pub(crate) fn num_segments(&self) -> usize {
        self.driver.machine().num_segments()
    }

    /// `_note_provider_request_id`: a plugin may learn the provider's id after `start_segment`.
    pub(crate) fn _note_provider_request_id(
        slf: &Arc<Self>,
        py: Python<'_>,
        context_id: &str,
    ) -> PyResult<()> {
        let me = slf;
        let (noted, ids) = {
            let mut m = me.driver.machine();
            (m.note_provider_request_id(context_id), m.provider_request_ids().to_vec())
        };
        if noted {
            me.set_provider_ids(py, &ids)?;
        }
        Ok(())
    }

    pub(crate) fn output_closed(&self, py: Python<'_>) -> PyResult<bool> {
        self.dst_ch.bind(py).getattr("closed")?.extract()
    }
}

impl State {
    /// A container decodes on its own thread: each time it has PCM, the loop drains it into the
    /// emitter, so decoded audio goes out as it is ready rather than at the next push.
    fn drain_on_output(slf: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        let context = slf.context.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let (event_loop, _) = super::captured_pair(py, &context)?;
        if event_loop.is_none() {
            return Ok(());
        }
        let state = Arc::downgrade(slf);
        let notifier = super::Notifier::default();
        let on_output: lk_agents_runtime::media::decoder::OnOutput = Arc::new(move || {
            let state = state.clone();
            lk_agents_runtime::host::notify(
                &notifier,
                "tts.AudioEmitter.decode",
                &context,
                &[],
                false,
                Box::new(move |py| {
                    let drain =
                        pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
                            let Some(state) = state.upgrade() else {
                                return Ok(());
                            };
                            let py = args.py();
                            let pcm = State::drained(&state)?;
                            if pcm.is_empty() {
                                return Ok(());
                            }
                            State::submit(&state, py, Cmd::Push(pcm))
                        })?;
                    Ok((drain.into_any(), pyo3::types::PyTuple::empty(py)))
                }),
            );
        });
        if let Some(decoder) = slf.decoder.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            decoder.set_on_output(Some(on_output));
        }
        Ok(())
    }

    /// Framing undecoded container bytes would be audible noise, so they pass through unchanged.
    fn decoded(slf: &Arc<Self>, _py: Python<'_>, bytes: Vec<u8>) -> PyResult<Vec<u8>> {
        let mut guard = slf.decoder.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            None => Ok(bytes),
            Some(decoder) => decoded_pcm(decoder.push(&bytes)),
        }
    }

    fn drained(slf: &Arc<Self>) -> PyResult<Vec<u8>> {
        let mut decoder = slf.decoder.lock().unwrap_or_else(|e| e.into_inner());
        match decoder.as_mut() {
            Some(decoder) => decoded_pcm(decoder.drain()),
            None => Ok(Vec::new()),
        }
    }

    fn submit(slf: &Arc<Self>, _py: Python<'_>, cmd: Cmd) -> PyResult<()> {
        slf.driver.submit(cmd, &mut PyHost(slf)).map_err(pump_err)
    }

    fn pump(slf: &Arc<Self>, _py: Python<'_>) -> PyResult<()> {
        slf.driver.pump(&mut PyHost(slf)).map_err(pump_err)
    }

    fn send(&self, py: Python<'_>, audio: CoreAudio) -> PyResult<()> {
        let frame = (&audio.frame).into_pyobject(py)?;

        // The words this frame covers. The key is always set, even to `[]`.
        let n = audio.timed_transcript.len();
        let words: Vec<Py<PyAny>> = {
            let mut q = self.words.lock().unwrap_or_else(|e| e.into_inner());
            (0..n).filter_map(|_| q.pop_front()).collect()
        };
        frame.getattr("userdata")?.set_item(USERDATA_TIMED_TRANSCRIPT, PyList::new(py, words)?)?;

        let kwargs = lk_agents_abi::voice::calls::SynthesizedAudioKwargs {
            frame: (),
            request_id: audio.request_id.clone(),
            segment_id: audio.segment_id.clone(),
            is_final: audio.is_final,
        }
        .host_dict(py, frame)?;
        let ev = synthesized_audio_type(py)?.call((), Some(&kwargs))?;

        self.dst_ch.bind(py).call_method1("send_nowait", (ev,))?;
        Ok(())
    }

    fn arm_timer(slf: &Arc<Self>, py: Python<'_>, delay: Duration) -> PyResult<()> {
        let me = slf;
        me.cancel_timer(py)?;
        let context = me.context.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let (ev_loop, _) = super::captured_pair(py, &context)?;
        if ev_loop.is_none() {
            return Ok(());
        }
        let state = Arc::downgrade(slf);
        let cb = pyo3::types::PyCFunction::new_closure(
            py,
            Some(c"_audio_emitter_drift_flush"),
            None,
            move |_args, _kwargs| -> PyResult<()> {
                let Some(state) = state.upgrade() else {
                    return Ok(());
                };
                run_now(&state, |state, py| {
                    *state.timer.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    logger(py)?.call_method1(
                        "debug",
                        ("flush audio emitter due to slow audio generation",),
                    )?;
                    State::submit(state, py, Cmd::Drift)
                })
            },
        )?;
        let handle = ev_loop.call_method1("call_later", (delay.as_secs_f64(), cb))?;
        *me.timer.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.unbind());
        Ok(())
    }

    fn cancel_timer(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self.timer.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            h.bind(py).call_method0("cancel")?;
        }
        Ok(())
    }

    /// A missing `opentelemetry` is skipped: tracing must never kill a synthesis.
    fn set_provider_ids(&self, py: Python<'_>, ids: &[String]) -> PyResult<()> {
        let Ok(trace) = py.import("opentelemetry.trace") else {
            return Ok(());
        };
        let span = trace.call_method0("get_current_span")?;
        if span.call_method0("is_recording")?.extract::<bool>()? {
            span.call_method1("set_attribute", (ATTR_PROVIDER_REQUEST_IDS, PyList::new(py, ids)?))?;
        }
        Ok(())
    }

    /// `dump_segment()` (`tts/tts.py:1195-1207`).
    fn write_dump(&self, py: Python<'_>, d: &SegmentDump) -> PyResult<()> {
        let stamp: String = py
            .import("datetime")?
            .getattr("datetime")?
            .call_method0("now")?
            .call_method1("strftime", ("%Y-%m-%d_%H-%M-%S",))?
            .extract()?;
        // Creates the directory (the reference would raise inside the emitter task).
        std::fs::create_dir_all(SegmentDump::DIR)?;
        std::fs::write(d.path(&stamp), d.wav_bytes())?;
        Ok(())
    }
}

fn decoded_pcm(result: lk_agents_runtime::media::Result<Vec<i16>>) -> PyResult<Vec<u8>> {
    result
        .map(|pcm| lk_agents_runtime::media::pcm_bytes(&pcm))
        .map_err(|error| PyRuntimeError::new_err(format!("AudioEmitter: decode failed: {error}")))
}

/// `bytes | bytearray | memoryview` -> owned PCM. The copy is needed: the bytes outlive the call.
fn pcm_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = obj.cast::<PyByteArray>() {
        return Ok(b.to_vec());
    }
    // any other buffer (memoryview, array): `memoryview(obj).tobytes()`, since the buffer API is
    // not in the stable ABI before 3.11
    let view = obj.py().import("builtins")?.getattr("memoryview")?.call1((obj,))?;
    Ok(view.call_method0("tobytes")?.cast::<PyBytes>()?.as_bytes().to_vec())
}

/// Synchronous TTS entry also works before an asyncio loop is running.
fn capture(py: Python<'_>) -> PyResult<lk_agents_abi::HostRef> {
    let event_loop = py
        .import("asyncio")?
        .call_method0("get_running_loop")
        .unwrap_or_else(|_| py.None().into_bound(py));
    let context = py.import("contextvars")?.call_method0("copy_context")?;
    let pair = pyo3::types::PyTuple::new(py, [event_loop, context])?;
    Ok(lk_agents_abi::python::intern_host(pair.into_any().unbind()))
}

impl State {
    pub(crate) fn recapture(&self) -> PyResult<()> {
        *self.context.lock().unwrap_or_else(|p| p.into_inner()) = Python::attach(capture)?;
        Ok(())
    }
}

/// Emitter callbacks require no framework scope: standalone TTS synthesis is supported.
/// Every entry carries the context captured by the owning synthesis initialization.
pub(crate) fn run_now<T: Send + 'static>(
    state: &Arc<State>,
    call: impl FnOnce(&Arc<State>, Python<'_>) -> PyResult<T> + Send + 'static,
) -> PyResult<T> {
    let context = state.context.lock().unwrap_or_else(|p| p.into_inner()).clone();
    let result = Arc::new(Mutex::new(None));
    lk_agents_runtime::host::call_now_in(
        &super::Entry,
        "tts.AudioEmitter",
        &context,
        &[],
        callback(state.clone(), result.clone(), call),
    )?;
    let value = result
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .expect("a successful host entry returns its result");
    Ok(value)
}

/// Coroutine methods return to the owning Python loop before accessing the channel or timers.
pub(crate) async fn run<T: Send + 'static>(
    state: &Arc<State>,
    entry: lk_agents_abi::invoke::CallContext,
    call: impl FnOnce(&Arc<State>, Python<'_>) -> PyResult<T> + Send + 'static,
) -> PyResult<T> {
    let context = if state.driver.machine().started() {
        state.context.lock().unwrap_or_else(|p| p.into_inner()).clone()
    } else {
        entry.host_context.ok_or_else(|| {
            PyRuntimeError::new_err("emitter coroutine requires a running event loop")
        })?
    };
    let result = Arc::new(Mutex::new(None));
    lk_agents_runtime::host::call(
        &super::Entry,
        "tts.AudioEmitter",
        &context,
        &[],
        callback(state.clone(), result.clone(), call),
    )
    .await?;
    let value = result
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
        .expect("a successful host entry returns its result");
    Ok(value)
}

fn callback<T: Send + 'static>(
    state: Arc<State>,
    output: Arc<Mutex<Option<T>>>,
    call: impl FnOnce(&Arc<State>, Python<'_>) -> PyResult<T> + Send + 'static,
) -> super::HostCall {
    let call = Mutex::new(Some(call));
    Box::new(move |py| {
        let callback = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
            let call = call
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take()
                .expect("an emitter callback is entered once");
            let value = call(&state, args.py())?;
            *output.lock().unwrap_or_else(|p| p.into_inner()) = Some(value);
            Ok::<_, PyErr>(())
        })?;
        Ok((callback.into_any(), pyo3::types::PyTuple::empty(py)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    #[test]
    fn pcm_arg_takes_any_buffer() {
        attach(|py| {
            for expr in [
                "b'\\x01\\x02'",
                "bytearray(b'\\x01\\x02')",
                "memoryview(b'\\x01\\x02')",
                "__import__('array').array('B', [1, 2])",
            ] {
                let obj = py.eval(&std::ffi::CString::new(expr).unwrap(), None, None).unwrap();
                assert_eq!(pcm_arg(&obj).unwrap(), vec![1, 2], "{expr}");
            }
            assert!(pcm_arg(&py.eval(c"3", None, None).unwrap()).is_err());
        });
    }

    // ---------------------------------------------------------------------------- the emitter

    /// Stand-ins for the host modules the emitter touches. The real package cannot be imported
    /// here, and each nextest process caches these in its `PyOnceLock`s.
    const STUBS: &str = r#"
import sys, types as _t

def _mod(name):
    m = _t.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m

lk = sys.modules.get('livekit') or _mod('livekit')
ag = sys.modules.get('livekit.agents') or _mod('livekit.agents'); lk.agents = ag
rtc = sys.modules.get('livekit.rtc') or _mod('livekit.rtc'); lk.rtc = rtc
tp = sys.modules.get('livekit.agents.types') or _mod('livekit.agents.types'); ag.types = tp
log = sys.modules.get('livekit.agents.log') or _mod('livekit.agents.log'); ag.log = log
tts_mod = _mod('livekit.agents.tts'); ag.tts = tts_mod

import logging
log.logger = logging.getLogger('livekit.agents')

class NotGiven:
    def __bool__(self): return False
NOT_GIVEN = NotGiven()

if not hasattr(tp, 'TimedString'):
    class TimedString(str):
        def __new__(cls, text, start_time=NOT_GIVEN, end_time=NOT_GIVEN, confidence=NOT_GIVEN,
                    start_time_offset=NOT_GIVEN, speaker_id=None):
            o = super().__new__(cls, text)
            o.start_time = start_time; o.end_time = end_time; o.confidence = confidence
            o.start_time_offset = start_time_offset; o.speaker_id = speaker_id
            return o
    tp.TimedString = TimedString

if not hasattr(rtc, 'AudioFrame'):
    class AudioFrame:
        def __init__(self, data, sample_rate, num_channels, samples_per_channel):
            self._d = bytes(data); self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.samples_per_channel = samples_per_channel
            self.userdata = {}
        @property
        def data(self): return memoryview(self._d)
    rtc.AudioFrame = AudioFrame

class SynthesizedAudio:
    def __init__(self, frame, request_id='', is_final=False, segment_id='', delta_text=''):
        self.frame = frame; self.request_id = request_id; self.is_final = is_final
        self.segment_id = segment_id; self.delta_text = delta_text
tts_mod.SynthesizedAudio = SynthesizedAudio

otel = sys.modules.get('opentelemetry') or _mod('opentelemetry')
otrace = sys.modules.get('opentelemetry.trace') or _mod('opentelemetry.trace')
otel.trace = otrace

class _Span:
    def __init__(self): self.attrs = {}
    def is_recording(self): return True
    def set_attribute(self, k, v): self.attrs[k] = v
SPAN = _Span()
otrace.get_current_span = lambda: SPAN

class Chan:
    def __init__(self): self.items = []
    def send_nowait(self, ev): self.items.append(ev)

# A channel that calls straight back into the emitter from inside `send_nowait`, which is what a
# real consumer's callback chain can do.
class ReentrantChan(Chan):
    emitter = None
    def send_nowait(self, ev):
        self.items.append(ev)
        if self.emitter is not None:
            self.emitter.num_segments
            self.emitter.pushed_duration()
            self.emitter.flush()
"#;

    fn env(py: Python<'_>) -> Bound<'_, PyDict> {
        let g = PyDict::new(py);
        py.run(std::ffi::CString::new(STUBS).unwrap().as_c_str(), Some(&g), None).unwrap();
        g
    }

    fn make<'py>(
        py: Python<'py>,
        g: &Bound<'py, PyDict>,
        chan: &str,
    ) -> (Bound<'py, PyAny>, Bound<'py, PyAny>) {
        let ch = g.get_item(chan).unwrap().unwrap().call0().unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("label", "test").unwrap();
        kwargs.set_item("dst_ch", &ch).unwrap();
        let em =
            py.get_type::<crate::generated::media::AudioEmitter>().call((), Some(&kwargs)).unwrap();
        (em, ch)
    }

    fn init(em: &Bound<'_, PyAny>, stream: bool) {
        let py = em.py();
        let kw = PyDict::new(py);
        kw.set_item("request_id", "req").unwrap();
        kw.set_item("sample_rate", 1000).unwrap();
        kw.set_item("num_channels", 1).unwrap();
        kw.set_item("mime_type", "audio/pcm").unwrap();
        kw.set_item("frame_size_ms", 100).unwrap();
        kw.set_item("stream", stream).unwrap();
        em.call_method("initialize", (), Some(&kw)).unwrap();
    }

    #[test]
    fn delivery_uses_the_synthesis_context_without_mutating_the_caller() {
        attach(|py| {
            let g = env(py);
            py.run(
                c"
import contextvars
owner = contextvars.ContextVar('emitter_owner', default='unset')
seen = []
class ContextChan(Chan):
    def send_nowait(self, event):
        seen.append(owner.get())
        owner.set('callback')
        super().send_nowait(event)
owner.set('synthesis')
",
                Some(&g),
                None,
            )
            .unwrap();
            let (em, _ch) = make(py, &g, "ContextChan");
            init(&em, false);
            let owner = g.get_item("owner").unwrap().unwrap();
            owner.call_method1("set", ("caller",)).unwrap();
            em.call_method1("push", (PyBytes::new(py, &[0; 600]),)).unwrap();
            em.call_method0("end_input").unwrap();
            let seen = g.get_item("seen").unwrap().unwrap().extract::<Vec<String>>().unwrap();
            assert!(!seen.is_empty(), "audio must reach the consumer");
            assert!(seen.iter().all(|value| value == "synthesis"));
            assert_eq!(owner.call_method0("get").unwrap().extract::<String>().unwrap(), "caller");
        });
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_waits_for_native_finalization_without_blocking_control() {
        let state = attach(|py| {
            let g = env(py);
            let channel = g.get_item("Chan").unwrap().unwrap().call0().unwrap().unbind();
            Arc::new(State::new(py, "test", channel).unwrap())
        });
        state.closed.store(true, Ordering::Release);
        let workers =
            lk_agents_runtime::media::workers::MediaWorkers::new(Default::default()).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let receipt = workers
            .try_spawn(lk_agents_runtime::media::workers::MediaWorkerClass::Decoder, move || {
                wait.recv().unwrap();
            })
            .unwrap();
        State::track_finalization(
            &state,
            async move { receipt.await.map(|()| Ok(Vec::new())) },
            |_, result| result,
            &tokio::runtime::Handle::current(),
        );
        let completed = Arc::new(AtomicBool::new(false));
        let control = completed.clone();
        tokio::spawn(async move {
            control.store(true, Ordering::Release);
            release.send(()).unwrap();
        });
        state.finalized().await.unwrap();
        assert!(completed.load(Ordering::Acquire));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_reinitialize_does_not_resurrect_a_closed_synthesis() {
        let (state, channel) = attach(|py| {
            let g = env(py);
            let channel = g.get_item("Chan").unwrap().unwrap().call0().unwrap();
            let state = Arc::new(State::new(py, "test", channel.clone().unbind()).unwrap());
            State::initialize(&state, py, "original", 1000, 1, "audio/pcm", 100, false).unwrap();
            State::aclose(&state, py).unwrap();
            (state, channel.unbind())
        });
        let workers =
            lk_agents_runtime::media::workers::MediaWorkers::new(Default::default()).unwrap();
        let (release, wait) = std::sync::mpsc::channel();
        let receipt = workers
            .try_spawn(lk_agents_runtime::media::workers::MediaWorkerClass::Formatting, move || {
                wait.recv().unwrap();
                Ok(vec![0; 200])
            })
            .unwrap();
        State::track_finalization(
            &state,
            receipt,
            |_, result| result,
            &tokio::runtime::Handle::current(),
        );
        attach(|py| {
            let error =
                State::initialize(&state, py, "replacement", 1000, 1, "audio/pcm", 100, false)
                    .expect_err("an emitter accepts one synthesis");
            assert!(error.to_string().contains("already started"));
        });
        release.send(()).unwrap();
        state.finalized().await.unwrap();
        attach(|py| {
            assert_eq!(
                channel.bind(py).getattr("items").unwrap().len().unwrap(),
                0,
                "late native audio must remain discarded after close and rejected reinitialization"
            );
        });
    }

    #[cfg(feature = "ffmpeg")]
    #[tokio::test(flavor = "current_thread")]
    async fn close_reaches_a_decoder_held_after_eof() {
        use std::sync::atomic::AtomicUsize;
        let mut encoder = lk_agents_runtime::media::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
        let mut encoded = encoder.push(&vec![100; 4800]).unwrap();
        encoded.extend(encoder.finish().unwrap());
        let (entered, waiting) = tokio::sync::oneshot::channel();
        let entered = Mutex::new(Some(entered));
        let (release, gate) = std::sync::mpsc::channel();
        let gate = Mutex::new(gate);
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = calls.clone();
        let state = attach(|py| {
            let g = env(py);
            let channel = g.get_item("Chan").unwrap().unwrap().call0().unwrap().unbind();
            let state = Arc::new(State::new(py, "cancel", channel).unwrap());
            State::initialize(&state, py, "original", 48_000, 1, "audio/ogg", 20, false).unwrap();
            {
                let mut decoder = state.decoder.lock().unwrap();
                let decoder = decoder.as_mut().unwrap();
                decoder.set_on_output(Some(Arc::new(move || {
                    callback_calls.fetch_add(1, Ordering::AcqRel);
                    if let Some(entered) = entered.lock().unwrap().take() {
                        let _ = entered.send(());
                        let _ = gate.lock().unwrap().recv();
                    }
                })));
                decoder.push(&encoded).unwrap();
            }
            State::finish_compressed(&state).unwrap();
            state
        });
        tokio::time::timeout(Duration::from_secs(10), waiting)
            .await
            .expect("the EOF decoder must still own its readiness callback")
            .unwrap();
        attach(|py| State::aclose(&state, py)).unwrap();
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), state.finalized())
            .await
            .expect("native cancellation must settle after its held callback returns")
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "close must disconnect native readiness before returning ownership"
        );
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn compressed_streams_finish_while_formatting_capacity_is_exhausted() {
        use lk_agents_runtime::media::workers::{MediaWorkerClass, MediaWorkerLimits};
        let workers = lk_agents_runtime::process::workers().unwrap();
        let formatting = MediaWorkerLimits::default().formatting;
        let _reserved = (0..formatting)
            .map(|_| workers.try_admit(MediaWorkerClass::Formatting).unwrap())
            .collect::<Vec<_>>();
        let mut encoder = lk_agents_runtime::media::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
        let mut encoded = encoder.push(&vec![100; 4800]).unwrap();
        encoded.extend(encoder.finish().unwrap());
        attach(|py| {
            let g = env(py);
            g.set_item("Emitter", py.get_type::<crate::generated::media::AudioEmitter>()).unwrap();
            g.set_item("encoded", PyBytes::new(py, &encoded)).unwrap();
            g.set_item("streams", formatting + 1).unwrap();
            py.run(c"
import asyncio, contextvars
owner = contextvars.ContextVar('stream_owner', default='unset')
class ContextChan(Chan):
    def __init__(self, expected):
        super().__init__()
        self.expected = expected
    def send_nowait(self, event):
        assert owner.get() == self.expected, (owner.get(), self.expected)
        super().send_nowait(event)
async def exercise():
    pending = []
    for index in range(streams):
        owner.set(f'synthesis-{index}')
        channel = ContextChan(owner.get())
        emitter = Emitter(label='concurrent', dst_ch=channel)
        emitter.initialize(request_id=str(index), sample_rate=48000, num_channels=1, mime_type='audio/ogg', frame_size_ms=20)
        emitter.push(encoded)
        pending.append((emitter, channel))
    owner.set('caller')
    for emitter, channel in pending:
        emitter.end_input()
        try:
            emitter.push(encoded)
        except RuntimeError:
            pass
        else:
            raise AssertionError('input after EOF must be rejected before framing')
    await asyncio.gather(*(emitter.join() for emitter, channel in pending))
    for emitter, channel in pending:
        assert channel.items, 'every decoded stream must reach its consumer'
        await emitter.aclose()
    assert owner.get() == 'caller'
asyncio.run(exercise())
", Some(&g), None).unwrap();
        });
    }

    // Catches: `pushed_duration()` defaulting to `0` rather than `-1` (newest segment); retries
    // use it to avoid replaying audio already heard.
    #[test]
    fn pushed_duration_defaults_to_the_newest_segment() {
        attach(|py| {
            let g = env(py);
            let (em, _ch) = make(py, &g, "Chan");
            init(&em, true);
            for (seg, n) in [("a", 200usize), ("b", 400usize)] {
                let kw = PyDict::new(py);
                kw.set_item("segment_id", seg).unwrap();
                em.call_method("start_segment", (), Some(&kw)).unwrap();
                em.call_method1("push", (PyBytes::new(py, &vec![0xffu8; n]),)).unwrap();
                em.call_method0("end_segment").unwrap();
            }
            let bare = em.call_method0("pushed_duration").unwrap().extract::<f64>().unwrap();
            let first = em.call_method1("pushed_duration", (0,)).unwrap().extract::<f64>().unwrap();
            assert!((bare - 0.2).abs() < 1e-9, "the newest segment");
            assert!((first - 0.1).abs() < 1e-9);
            assert_eq!(em.getattr("num_segments").unwrap().extract::<usize>().unwrap(), 2);
        });
    }

    // Catches: FFI hazard #9, `RuntimeError: Already borrowed` when a consumer re-enters from
    // `send_nowait`.
    #[test]
    fn a_consumer_that_re_enters_from_send_nowait_is_not_already_borrowed() {
        attach(|py| {
            let g = env(py);
            let (em, ch) = make(py, &g, "ReentrantChan");
            ch.setattr("emitter", &em).unwrap();
            init(&em, true);
            let kw = PyDict::new(py);
            kw.set_item("segment_id", "s1").unwrap();
            em.call_method("start_segment", (), Some(&kw)).unwrap();

            em.call_method1("push", (PyBytes::new(py, &[0xffu8; 600]),)).unwrap();
            em.call_method0("end_segment").unwrap();

            // The re-entrant `flush()` is picked up by the outer drain: nothing stranded or doubled.
            let items = ch.getattr("items").unwrap();
            let mut real = 0u32;
            let mut finals = 0usize;
            for i in 0..items.len().unwrap() {
                let ev = items.get_item(i).unwrap();
                let n = ev
                    .getattr("frame")
                    .unwrap()
                    .getattr("samples_per_channel")
                    .unwrap()
                    .extract::<u32>()
                    .unwrap();
                if ev.getattr("is_final").unwrap().extract::<bool>().unwrap() {
                    finals += 1;
                    // The segment closes on the synthetic 10 ms marker.
                    assert_eq!(n, 10);
                } else {
                    real += n;
                }
            }
            assert_eq!(real, 300, "every pushed sample came out exactly once");
            assert_eq!(finals, 1);
        });
    }

    // Catches: accepting a mime type nothing can decode.
    #[test]
    fn a_mime_type_no_decoder_understands_is_still_refused() {
        attach(|py| {
            let g = env(py);
            let (em, _ch) = make(py, &g, "Chan");
            let kw = PyDict::new(py);
            kw.set_item("request_id", "req").unwrap();
            kw.set_item("sample_rate", 1000).unwrap();
            kw.set_item("num_channels", 1).unwrap();
            kw.set_item("mime_type", "audio/x-not-a-real-container").unwrap();
            let err = em.call_method("initialize", (), Some(&kw)).unwrap_err();
            assert!(err.is_instance_of::<PyNotImplementedError>(py), "{err}");
        });
    }

    // Catches: `lk.provider_request_ids` written per segment instead of per new id, or lost.
    #[test]
    fn provider_request_ids_reach_the_span_deduped() {
        attach(|py| {
            let g = env(py);
            let (em, _ch) = make(py, &g, "Chan");
            init(&em, true);
            for seg in ["p1", "p1", "p2"] {
                let kw = PyDict::new(py);
                kw.set_item("segment_id", seg).unwrap();
                // One provider request can span several segments.
                let _ = em.call_method("start_segment", (), Some(&kw));
                let _ = em.call_method0("end_segment");
            }
            let span = g.get_item("SPAN").unwrap().unwrap();
            let attrs = span.getattr("attrs").unwrap();
            let ids = attrs.get_item(ATTR_PROVIDER_REQUEST_IDS).unwrap();
            assert_eq!(ids.extract::<Vec<String>>().unwrap(), vec!["p1", "p2"]);
        });
    }

    // Catches: double `initialize()` or push-before-initialize not raising `RuntimeError`.
    #[test]
    fn the_reference_runtime_errors_survive_the_crossing() {
        attach(|py| {
            let g = env(py);
            let (em, _ch) = make(py, &g, "Chan");
            let err = em.call_method1("push", (PyBytes::new(py, b"ab"),)).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("isn't started"));

            init(&em, true);
            let kw = PyDict::new(py);
            kw.set_item("segment_id", "s").unwrap();
            em.call_method("start_segment", (), Some(&kw)).unwrap();
            let err = em.call_method("start_segment", (), Some(&kw)).unwrap_err();
            assert!(err.to_string().contains("previous segment was ended"));
        });
    }
}
