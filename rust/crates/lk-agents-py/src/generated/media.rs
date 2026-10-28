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

fn _extract_audio_emitter_pushed_duration_idx(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "idx")
}

#[pyclass(frozen, subclass, name = "AudioEmitter", module = "livekit.agents.tts")]
pub struct AudioEmitter(::std::sync::Arc<AudioEmitterImpl>);

#[pymethods]
impl AudioEmitter {
    #[new]
    #[pyo3(signature = (*, label, dst_ch))]
    fn new(label: ::std::string::String, dst_ch: ::lk_agents_abi::HostRef) -> PyResult<AudioEmitter> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::new(label, dst_ch).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(AudioEmitter(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, request_id, sample_rate, num_channels, mime_type, frame_size_ms = 200, stream = false))]
    fn initialize(&self, request_id: ::std::string::String, #[pyo3(from_py_with = _extract_audio_emitter_initialize_sample_rate)] sample_rate: u32, #[pyo3(from_py_with = _extract_audio_emitter_initialize_num_channels)] num_channels: u16, mime_type: ::std::string::String, #[pyo3(from_py_with = _extract_audio_emitter_initialize_frame_size_ms)] frame_size_ms: u32, stream: bool) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::initialize(&self.0, request_id, sample_rate, num_channels, mime_type, frame_size_ms, stream).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (*, segment_id))]
    fn start_segment(&self, segment_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::start_segment(&self.0, segment_id).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_segment(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_segment(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (data))]
    fn push(&self, data: ::lk_agents_abi::HostRef) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push(&self.0, data).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (frame))]
    fn push_frame(&self, frame: ::lk_agents_abi::AudioFrame) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_frame(&self.0, frame).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (delta_text))]
    fn push_timed_transcript(&self, delta_text: ::lk_agents_abi::HostRef) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_timed_transcript(&self.0, delta_text).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::flush(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn end_input(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_input(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let context = crate::host::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::join(&this, context)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let context = crate::host::entered();
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::aclose(&this, context)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
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
        <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::_note_provider_request_id(&self.0, context_id).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn output_closed(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::output_closed(&self.0).map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
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
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { let bytes = ::lk_agents_abi::python::retained_bytes(&obj, "data", None)?; Ok(obj.py().detach(move || bytes.to_vec())) } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
}

/// Incremental container bytes to little-endian interleaved PCM.
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

    #[getter]
    fn output_format(&self) -> PyResult<::std::option::Option<(u32, u16)>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::output_format(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (data))]
    fn push<'py>(&self, py: Python<'py>, #[pyo3(from_py_with = _extract_audio_decoder_push_data)] data: ::std::vec::Vec<u8>) -> PyResult<::std::vec::Vec<u8>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = py.detach(|| <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::push(&self.0, &data)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Blocks only the invoking native thread while the admitted decoder drains.
    fn flush<'py>(&self, py: Python<'py>) -> PyResult<::std::vec::Vec<u8>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = py.detach(|| <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::flush(&self.0)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    #[pyo3(signature = (on_output = None))]
    fn set_on_output(&self, on_output: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let on_output = on_output.map(|v| crate::generated::media::DecoderOutputProxy::from_host(&v)).transpose()?;
        <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::set_on_output(&self.0, on_output);
        Ok(())
    }

    /// Cancel pending input and observe release of the native decoder's worker admission.
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
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { let bytes = ::lk_agents_abi::python::retained_bytes(&obj, "data", None)?; Ok(obj.py().detach(move || bytes.to_vec())) } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "data") }).collect::<PyResult<Vec<_>>>()) } }
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

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.

/// [`::lk_agents_abi::media::codecs::DecoderOutput`] over a host `typing.Callable`.
#[derive(Clone)]
pub struct DecoderOutputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::host::Notifier,
}

impl DecoderOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::codecs::DecoderOutput>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::media::codecs::DecoderOutput>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::media::codecs::DecoderOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(DecoderOutputProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: crate::host::capture_context(obj.py())?, notifier: ::std::default::Default::default() })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::media::codecs::DecoderOutput>(|obj| DecoderOutputProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::media::codecs::DecoderOutput for DecoderOutputProxy {
    fn ready(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        ::lk_agents_runtime::host::notify(&self.notifier, "DecoderOutput.__call__", &self.context, &[], false, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
    }
}
