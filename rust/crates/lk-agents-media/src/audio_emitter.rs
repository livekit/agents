//! `tts.AudioEmitter`: the pyclass over `lk_agents_abi::tts::emitter`'s machine plus the container
//! decoder in front of it.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use lk_agents_abi::SynthesizedAudio as CoreAudio;
use lk_agents_abi::tts::emitter::{
    AudioEmitter as Machine, Cmd, EmitterDriver, EmitterError, EmitterHost, HostEffect, PumpError, SegmentDump,
};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyList, PyType};

fn emitter_error_to_py(e: EmitterError) -> PyErr {
    lk_agents_abi::python::HostException::to_py_err(&e)
}

/// Resolved lazily: `livekit.agents.tts.tts` imports `_core`, so resolving at module init is a cycle.
fn cached<'py>(
    slot: &'static PyOnceLock<Py<PyType>>,
    py: Python<'py>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    slot.get_or_try_init(py, || {
        py.import(module)?.getattr(name)?.cast_into::<PyType>().map(|t| t.unbind()).map_err(PyErr::from)
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
    cached(&T, py, "livekit.agents.tts.tts", "SynthesizedAudio")
}

fn timed_string_type(py: Python<'_>) -> Option<&Bound<'_, PyType>> {
    static T: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    cached(&T, py, "livekit.agents.types", "TimedString").ok()
}

fn logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("livekit.agents.log")?.getattr("logger")
}

#[pyclass(frozen, subclass, name = "AudioEmitter", module = "livekit.agents.tts.tts")]
pub struct AudioEmitter {
    /// Effects come back through [`PyHost`] with no lock held.
    driver: EmitterDriver,
    dst_ch: Py<PyAny>,
    label: String,
    /// Host `TimedString`s kept verbatim so alignment fields survive; drained as the machine assigns.
    words: Mutex<VecDeque<Py<PyAny>>>,
    event_loop: Mutex<Option<Py<PyAny>>>,
    /// Rebuilt per `initialize`, so one request never gets the tail of the previous one.
    decoder: Mutex<Option<crate::Decoder>>,
    timer: Mutex<Option<Py<PyAny>>>,
}

fn pump_err(e: PumpError<PyErr>) -> PyErr {
    match e {
        PumpError::Machine(e) => emitter_error_to_py(e),
        PumpError::Host(e) => e,
    }
}

/// Called with no lock held, so a `send_nowait` that re-enters the emitter just queues.
struct PyHost<'a, 'py>(&'a Bound<'py, AudioEmitter>);

impl EmitterHost for PyHost<'_, '_> {
    type Error = PyErr;

    fn deliver(&mut self, effect: HostEffect) -> PyResult<()> {
        let slf = self.0;
        let py = slf.py();
        let me = slf.get();
        match effect {
            HostEffect::Send(audio) => me.send(py, *audio),
            HostEffect::Arm(delay) => AudioEmitter::arm_timer(slf, delay),
            HostEffect::Cancel => me.cancel_timer(py),
            HostEffect::Dump(d) => me.write_dump(py, &d),
            HostEffect::Dropped(n) => {
                logger(py)?.call_method1(
                    "warning",
                    (format!("AudioEmitter: incomplete frame during flush, dropping {n} bytes"),),
                )?;
                Ok(())
            }
            HostEffect::ProviderIds(ids) => me.set_provider_ids(py, &ids),
        }
    }
}

#[pymethods]
impl AudioEmitter {
    /// No `check_pid()`: built per synthesis in whatever process imported the package, and the
    /// reference has no init requirement for `TTS`.
    #[new]
    #[pyo3(signature = (*, label, dst_ch))]
    fn new(label: &str, dst_ch: Py<PyAny>) -> Self {
        let mut machine = Machine::new(label);
        machine.set_dump(dump_default());
        AudioEmitter {
            driver: EmitterDriver::new(machine),
            dst_ch,
            label: label.to_string(),
            words: Mutex::new(VecDeque::new()),
            event_loop: Mutex::new(None),
            decoder: Mutex::new(None),
            timer: Mutex::new(None),
        }
    }

    #[pyo3(signature = (*, request_id, sample_rate, num_channels, mime_type, frame_size_ms=200, stream=false))]
    fn initialize(
        slf: &Bound<'_, Self>,
        request_id: &str,
        sample_rate: u32,
        num_channels: u16,
        mime_type: &str,
        frame_size_ms: u32,
        stream: bool,
    ) -> PyResult<()> {
        let py = slf.py();
        let me = slf.get();

        // `audio/pcm` and `audio/raw` build no decoder; other containers are decoded. Unknown mime
        // types are refused by `crate::decode_plan`, as the reference's
        // `NotImplementedError`.
        let decoder = match crate::decode_plan(mime_type)
            .map_err(|e| PyNotImplementedError::new_err(format!("AudioEmitter: {e}")))?
        {
            crate::DecodePlan::Passthrough => None,
            crate::DecodePlan::Decode(mt) => Some(
                crate::Decoder::new(&mt, sample_rate, num_channels)
                    .map_err(|e| PyNotImplementedError::new_err(format!("AudioEmitter: {e}")))?,
            ),
        };
        *me.decoder.lock().unwrap_or_else(|e| e.into_inner()) = decoder;

        if request_id.is_empty() {
            // Verbatim, %-style, as the reference logs it.
            logger(py)?
                .call_method1("warning", ("no request_id provided for TTS %s", &me.label))?;
        }

        // Without a running loop the drift watchdog simply never fires.
        *me.event_loop.lock().unwrap_or_else(|e| e.into_inner()) =
            py.import("asyncio")?.call_method0("get_running_loop").ok().map(|l| l.unbind());

        me.driver
            .machine()
            .initialize(request_id, sample_rate, num_channels, frame_size_ms, stream)
            .map_err(emitter_error_to_py)?;
        // `stream=False` opens a segment here; drain so no stale queue is left.
        Self::pump(slf)
    }

    #[pyo3(signature = (*, segment_id))]
    fn start_segment(slf: &Bound<'_, Self>, segment_id: &str) -> PyResult<()> {
        Self::submit(slf, Cmd::StartSegment(segment_id.to_string()))
    }

    fn end_segment(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::submit(slf, Cmd::EndSegment)
    }

    /// `bytes | bytearray | memoryview`, as the reference takes.
    fn push(slf: &Bound<'_, Self>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::submit(slf, Cmd::Push(Self::decoded(slf, pcm_arg(data)?)?))
    }

    /// `TimedString | list[TimedString]`.
    fn push_timed_transcript(slf: &Bound<'_, Self>, delta_text: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
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
            let me = slf.get();
            let mut q = me.words.lock().unwrap_or_else(|e| e.into_inner());
            for k in kept {
                q.push_back(k.unbind());
            }
        }
        Self::submit(slf, Cmd::Words(rust))
    }

    fn flush(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::submit(slf, Cmd::Flush)
    }

    fn end_input(slf: &Bound<'_, Self>) -> PyResult<()> {
        // Flush the decoder first, or the last packet's tail is lost. An empty tail is not pushed:
        // the machine counts pushes, and PCM streams never had that boundary.
        let tail = {
            let mut guard = slf.get().decoder.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_mut() {
                None => Vec::new(),
                Some(d) => d.flush().map(|pcm| crate::pcm_bytes(&pcm)).map_err(|e| {
                    PyRuntimeError::new_err(format!("AudioEmitter: decode flush failed: {e}"))
                })?,
            }
        };
        if !tail.is_empty() {
            Self::submit(slf, Cmd::Push(tail))?;
        }
        Self::submit(slf, Cmd::EndInput)
    }

    /// `await emitter.join()`. Resolves on the next loop tick so the consumer gets to run.
    fn join<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !slf.get().driver.machine().started() {
            return Err(emitter_error_to_py(EmitterError::NotStarted));
        }
        Self::pump(slf)?;
        slf.py().import("asyncio")?.call_method1("sleep", (0.0,))
    }

    /// `await emitter.aclose()`. Safe before `initialize()`: `ChunkedStream` calls it from a
    /// `finally`.
    fn aclose<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let me = slf.get();
        me.driver.stop();
        me.words.lock().unwrap_or_else(|e| e.into_inner()).clear();
        me.cancel_timer(py)?;
        py.import("asyncio")?.call_method1("sleep", (0.0,))
    }

    #[pyo3(signature = (idx=-1))]
    fn pushed_duration(&self, idx: i64) -> f64 {
        self.driver.machine().pushed_duration(idx)
    }

    #[getter]
    fn num_segments(&self) -> usize {
        self.driver.machine().num_segments()
    }

    /// `_note_provider_request_id`: a plugin may learn the provider's id after `start_segment`.
    fn _note_provider_request_id(slf: &Bound<'_, Self>, context_id: &str) -> PyResult<()> {
        let me = slf.get();
        let (noted, ids) = {
            let mut m = me.driver.machine();
            (m.note_provider_request_id(context_id), m.provider_request_ids().to_vec())
        };
        if noted {
            me.set_provider_ids(slf.py(), &ids)?;
        }
        Ok(())
    }

    #[getter]
    fn output_closed(&self, py: Python<'_>) -> PyResult<bool> {
        self.dst_ch.bind(py).getattr("closed")?.extract()
    }
}

impl AudioEmitter {
    /// Framing undecoded container bytes would be audible noise, so they pass through unchanged.
    fn decoded(slf: &Bound<'_, Self>, bytes: Vec<u8>) -> PyResult<Vec<u8>> {
        let mut guard = slf.get().decoder.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            None => Ok(bytes),
            Some(d) => d
                .push(&bytes)
                .map(|pcm| crate::pcm_bytes(&pcm))
                .map_err(|e| PyRuntimeError::new_err(format!("AudioEmitter: decode failed: {e}"))),
        }
    }

    fn submit(slf: &Bound<'_, Self>, cmd: Cmd) -> PyResult<()> {
        slf.get().driver.submit(cmd, &mut PyHost(slf)).map_err(pump_err)
    }

    fn pump(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.get().driver.pump(&mut PyHost(slf)).map_err(pump_err)
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

    fn arm_timer(slf: &Bound<'_, Self>, delay: Duration) -> PyResult<()> {
        let py = slf.py();
        let me = slf.get();
        me.cancel_timer(py)?;
        let ev_loop = me
            .event_loop
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|l| l.clone_ref(py));
        let Some(ev_loop) = ev_loop else {
            return Ok(());
        };
        // Clearing the handle as it fires breaks the emitter -> handle -> callback -> emitter cycle.
        let this: Py<AudioEmitter> = slf.clone().unbind();
        let cb = pyo3::types::PyCFunction::new_closure(
            py,
            Some(c"_audio_emitter_drift_flush"),
            None,
            move |args, _kwargs| -> PyResult<()> {
                let py = args.py();
                let bound = this.bind(py);
                *bound.get().timer.lock().unwrap_or_else(|e| e.into_inner()) = None;
                logger(py)?
                    .call_method1("debug", ("flush audio emitter due to slow audio generation",))?;
                AudioEmitter::submit(bound, Cmd::Drift)
            },
        )?;
        let handle = ev_loop.bind(py).call_method1("call_later", (delay.as_secs_f64(), cb))?;
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
        d.write(&stamp)?;
        Ok(())
    }
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


lk_agents_abi::register_host!(class AudioEmitter);

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
tts_pkg = _mod('livekit.agents.tts'); ag.tts = tts_pkg
tts_mod = _mod('livekit.agents.tts.tts'); tts_pkg.tts = tts_mod

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
        let em = Py::new(py, AudioEmitter::new("test", ch.clone().unbind()))
            .unwrap()
            .into_bound(py)
            .into_any();
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
