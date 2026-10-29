//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `media` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/media/emitter.rs.
type AudioEmitterImpl = <crate::Selected as super::Implementations>::AudioEmitter;

fn _extract_audio_emitter_initialize_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")
}

fn _extract_audio_emitter_initialize_num_channels(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "num_channels")
}

fn _extract_audio_emitter_initialize_frame_size_ms(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "frame_size_ms")
}

fn _extract_audio_emitter_push_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "data") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
}

fn _extract_audio_emitter_pushed_duration_idx(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "idx")
}

#[pyclass(frozen, subclass, name = "AudioEmitter", module = "livekit.agents.tts")]
pub struct AudioEmitter(::std::sync::Arc<AudioEmitterImpl>);

#[pymethods]
impl AudioEmitter {
    #[new]
    #[pyo3(signature = (*, label, dst_ch))]
    fn new(label: ::std::string::String, dst_ch: Bound<'_, PyAny>) -> PyResult<AudioEmitter> {
        ::lk_agents_abi::python::check_pid()?;
        let dst_ch = crate::generated::media::EmitterOutputProxy::from_host(&dst_ch)?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::new(label, dst_ch).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AudioEmitter(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, request_id, sample_rate, num_channels, mime_type, frame_size_ms = 200, stream = false))]
    fn initialize(&self, request_id: ::std::string::String, #[pyo3(from_py_with = _extract_audio_emitter_initialize_sample_rate)] sample_rate: u32, #[pyo3(from_py_with = _extract_audio_emitter_initialize_num_channels)] num_channels: u16, mime_type: ::std::string::String, #[pyo3(from_py_with = _extract_audio_emitter_initialize_frame_size_ms)] frame_size_ms: u32, stream: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::initialize(&self.0, request_id, sample_rate, num_channels, mime_type, frame_size_ms, stream).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (*, segment_id))]
    fn start_segment(&self, segment_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::start_segment(&self.0, segment_id).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_segment(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_segment(&self.0).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (data))]
    fn push(&self, #[pyo3(from_py_with = _extract_audio_emitter_push_data)] data: ::std::vec::Vec<u8>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push(&self.0, &data).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (frame))]
    fn push_frame(&self, frame: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_frame(&self.0, frame).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (delta_text))]
    fn push_timed_transcript(&self, delta_text: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let delta_text = (if delta_text.is_instance_of::<::pyo3::types::PyList>() { delta_text.try_iter().and_then(|items| items.map(|item| { let item = item?; item.extract::<::lk_agents_abi::TimedString>() }).collect::<PyResult<Vec<_>>>()) } else { (delta_text.extract::<::lk_agents_abi::TimedString>()).map(|item| vec![item]) })?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_timed_transcript(&self.0, delta_text).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::flush(&self.0).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_input(&self.0).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// Waits for compressed input to finish decoding; reports a decode failure.
    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::join(&this, )
                .await.map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Discards pending audio and observes native settlement; never reports a synthesis failure.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (idx = -1))]
    fn pushed_duration(&self, #[pyo3(from_py_with = _extract_audio_emitter_pushed_duration_idx)] idx: i64) -> PyResult<f64> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::pushed_duration(&self.0, idx);
        Ok(out)
    }

    #[getter]
    fn num_segments(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::num_segments(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (context_id))]
    fn _note_provider_request_id(&self, context_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::_note_provider_request_id(&self.0, context_id).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn output_closed(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::output_closed(&self.0).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.
type AudioDecoderImpl = <crate::Selected as super::Implementations>::AudioDecoder;

fn _extract_audio_decoder_new_sample_rate(obj: &Bound<'_, PyAny>) -> PyResult<::std::option::Option<u32>> {
    if obj.is_none() { Ok(None) } else { (::lk_agents_abi::python::number_from_py::<u32>(&obj, "sample_rate")).map(Some) }
}

fn _extract_audio_decoder_new_num_channels(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "num_channels")
}

fn _extract_audio_decoder_push_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "data") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
}

/// Container bytes decoded on one admitted native worker: input is queued without waiting for
/// the decoder, and decoded frames are awaited.
#[pyclass(frozen, name = "AudioDecoder", module = "livekit.agents._core")]
pub struct AudioDecoder(::std::sync::Arc<AudioDecoderImpl>);

#[pymethods]
impl AudioDecoder {
    #[new]
    #[pyo3(signature = (mime, *, sample_rate = Some(48000), num_channels = 1))]
    fn new(mime: ::std::string::String, #[pyo3(from_py_with = _extract_audio_decoder_new_sample_rate)] sample_rate: ::std::option::Option<u32>, #[pyo3(from_py_with = _extract_audio_decoder_new_num_channels)] num_channels: u16) -> PyResult<AudioDecoder> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::new(mime, sample_rate, num_channels).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AudioDecoder(::std::sync::Arc::new(out)))
    }

    /// Refused after `end_input`, `aclose` or a decode failure.
    #[pyo3(signature = (data))]
    fn push(&self, #[pyo3(from_py_with = _extract_audio_decoder_push_data)] data: ::std::vec::Vec<u8>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::push(&self.0, &data).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// No more input: `next` returns `None` once everything pushed is decoded.
    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::end_input(&self.0);
        Ok(())
    }

    /// The next decoded frame, at the requested format; `None` at the end of input.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::next(&this, )
                .await.map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Cancel decoding and observe release of the native decoder's worker admission.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.
type StreamBufferImpl = <crate::Selected as super::Implementations>::StreamBuffer;

fn _extract_stream_buffer_write_data(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "data") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
}

fn _extract_stream_buffer_read_size(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    ::lk_agents_abi::python::number_from_py::<isize>(&obj, "size")
}

/// A byte pipe: normal EOF drains buffered bytes; close discards them immediately.
#[pyclass(frozen, name = "StreamBuffer", module = "livekit.agents._core")]
pub struct StreamBuffer(::std::sync::Arc<StreamBufferImpl>);

#[pymethods]
impl StreamBuffer {
    #[new]
    fn new() -> PyResult<StreamBuffer> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::new();
        Ok(StreamBuffer(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (data))]
    fn write(&self, #[pyo3(from_py_with = _extract_stream_buffer_write_data)] data: ::std::vec::Vec<u8>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::write(&self.0, &data);
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::end_input(&self.0);
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::close(&self.0);
        Ok(())
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::len(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (size = -1))]
    fn read<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_stream_buffer_read_size)] size: isize) -> PyResult<::std::vec::Vec<u8>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = py.detach(|| <StreamBufferImpl as ::lk_agents_abi::media::codecs::StreamBuffer>::read(&self.0, size));
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.
type MediaTokenizationImpl = <crate::Selected as super::Implementations>::MediaTokenization;

fn _extract_media_tokenization_split_sentences_min_sentence_len(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "min_sentence_len")
}

#[pyfunction]
#[pyo3(signature = (text, min_sentence_len = 20, *, retain_format = false))]
pub fn split_sentences<'py>(py: Python<'py>, text: ::std::string::String, #[pyo3(from_py_with = _extract_media_tokenization_split_sentences_min_sentence_len)] min_sentence_len: usize, retain_format: bool) -> PyResult<::std::vec::Vec<(::std::string::String, usize, usize)>> {
    let out = py.detach(|| <MediaTokenizationImpl as ::lk_agents_abi::media::codecs::MediaTokenization>::split_sentences(&text, min_sentence_len, retain_format)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/media/emitter.rs.

/// [`::lk_agents_abi::media::emitter::EmitterOutput`] over a host `livekit.agents.utils.aio.Chan`.
#[derive(Clone)]
pub struct EmitterOutputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl EmitterOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::emitter::EmitterOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::media::emitter::EmitterOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::media::emitter::EmitterOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let view = crate::host::construct(&obj.py().import("livekit.agents.tts")?.getattr("_EmitterOutput")?, (obj,), None)?;
        let obj = &view;
        Ok(EmitterOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::media::emitter::EmitterOutput>(|obj| EmitterOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::media::emitter::EmitterOutput for EmitterOutputProxy {
    fn send(&self, frame: ::lk_agents_abi::AudioFrame, request_id: ::std::string::String, segment_id: ::std::string::String, is_final: bool, timed_transcript: ::std::vec::Vec<::lk_agents_abi::TimedString>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let request_id_arg = request_id;
        let segment_id_arg = segment_id;
        let is_final_arg = is_final;
        let timed_transcript_arg = timed_transcript;
        ::lk_agents_runtime::host::notify(&self.notifier, "EmitterOutput.send", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 5] = [::pyo3::IntoPyObjectExt::into_bound_py_any(frame_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(segment_id_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(is_final_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(timed_transcript_arg, py)?];
                Ok((host.bind(py).getattr("send")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn span_attribute(&self, key: ::std::string::String, values: ::std::vec::Vec<::std::string::String>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let key_arg = key;
        let values_arg = values;
        ::lk_agents_runtime::host::notify(&self.notifier, "EmitterOutput.span_attribute", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(key_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(values_arg, py)?];
                Ok((host.bind(py).getattr("span_attribute")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn dump(&self, name: ::std::string::String, wav: ::std::vec::Vec<u8>) {
        let host = ::std::sync::Arc::clone(&self.host);
        let name_arg = name;
        let wav_arg = wav;
        ::lk_agents_runtime::host::notify(&self.notifier, "EmitterOutput.dump", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [::pyo3::IntoPyObjectExt::into_bound_py_any(name_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(wav_arg, py)?];
                Ok((host.bind(py).getattr("dump")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }

    fn closed(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("closed")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
