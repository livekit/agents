//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `media` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/media/images.rs.
type FrameReaderInstallImpl =
    <crate::generated::implementations::Selected as super::Implementations>::FrameReaderInstall;

/// Hands the runtime the host's frame reader; images of host frames need it.
#[pyfunction]
#[pyo3(signature = (reader))]
pub fn _core_install_frame_reader(py: Python<'_>, reader: Bound<'_, PyAny>) -> PyResult<()> {
    let reader = crate::generated::media::FrameReaderProxy::from_host(reader.as_any())?;
    py.detach(|| <FrameReaderInstallImpl as ::lk_agents_abi::media::images::FrameReaderInstall>::install_frame_reader(reader));
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/media/images.rs.
type ImageFunctionsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ImageFunctions;

fn _extract_image_functions_encode_image_quality(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ImageFunctionsEncodeImageQualityInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u8>(obj.as_any(), "quality")).map(Some)
    }
}
type ImageFunctionsEncodeImageQualityInput = ::std::option::Option<u8>;

fn _extract_image_functions_encode_image_resize_width(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ImageFunctionsEncodeImageResizeWidthInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "resize_width")).map(Some)
    }
}
type ImageFunctionsEncodeImageResizeWidthInput = ::std::option::Option<u32>;

fn _extract_image_functions_encode_image_resize_height(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ImageFunctionsEncodeImageResizeHeightInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "resize_height")).map(Some)
    }
}
type ImageFunctionsEncodeImageResizeHeightInput = ::std::option::Option<u32>;

fn _extract_image_functions_serialize_image_frame_inference_width(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ImageFunctionsSerializeImageFrameInferenceWidthInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "inference_width")).map(Some)
    }
}
type ImageFunctionsSerializeImageFrameInferenceWidthInput = ::std::option::Option<u32>;

fn _extract_image_functions_serialize_image_frame_inference_height(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ImageFunctionsSerializeImageFrameInferenceHeightInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "inference_height")).map(Some)
    }
}
type ImageFunctionsSerializeImageFrameInferenceHeightInput = ::std::option::Option<u32>;

#[pyfunction]
#[pyo3(signature = (frame, *, format = "JPEG".to_owned(), quality = None, resize_width = None, resize_height = None, resize_strategy = None))]
pub fn encode_image<'py>(
    py: Python<'py>,
    frame: Bound<'_, PyAny>,
    format: ::std::string::String,
    #[pyo3(from_py_with = _extract_image_functions_encode_image_quality)]
    quality: ImageFunctionsEncodeImageQualityInput,
    #[pyo3(from_py_with = _extract_image_functions_encode_image_resize_width)]
    resize_width: ImageFunctionsEncodeImageResizeWidthInput,
    #[pyo3(from_py_with = _extract_image_functions_encode_image_resize_height)]
    resize_height: ImageFunctionsEncodeImageResizeHeightInput,
    resize_strategy: ::std::option::Option<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let frame = crate::py_objects::object_ref_from_py(frame.as_any())?;
    crate::py_calls::future_into_py(py, async move {
        let out = <ImageFunctionsImpl as ::lk_agents_abi::media::images::ImageFunctions>::encode_image(&env, frame, format, quality, resize_width, resize_height, resize_strategy).await.map_err(|e: ::lk_agents_abi::media::images::ImageError| <::lk_agents_abi::media::images::ImageError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

/// A frame as JPEG bytes, resized to its inference size when it has one.
#[pyfunction]
#[pyo3(signature = (frame, inference_width = None, inference_height = None))]
pub fn serialize_image_frame<'py>(
    py: Python<'py>,
    frame: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_image_functions_serialize_image_frame_inference_width)]
    inference_width: ImageFunctionsSerializeImageFrameInferenceWidthInput,
    #[pyo3(from_py_with = _extract_image_functions_serialize_image_frame_inference_height)]
    inference_height: ImageFunctionsSerializeImageFrameInferenceHeightInput,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    let frame = crate::py_objects::object_ref_from_py(frame.as_any())?;
    crate::py_calls::future_into_py(py, async move {
        let out = <ImageFunctionsImpl as ::lk_agents_abi::media::images::ImageFunctions>::serialize_image_frame(&env, frame, inference_width, inference_height).await.map_err(|e: ::lk_agents_abi::media::images::ImageError| <::lk_agents_abi::media::images::ImageError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

/// A URL's inline bytes and MIME type, or the URL itself when it is external.
#[pyfunction]
#[pyo3(signature = (url, mime_type))]
pub fn serialize_image_url<'py>(
    py: Python<'py>,
    url: ::std::string::String,
    mime_type: ::std::option::Option<::std::string::String>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::check_pid()?;
    let env = crate::process::env();
    crate::py_calls::future_into_py(py, async move {
        let out = <ImageFunctionsImpl as ::lk_agents_abi::media::images::ImageFunctions>::serialize_image_url(&env, url, mime_type).await.map_err(|e: ::lk_agents_abi::media::images::ImageError| <::lk_agents_abi::media::images::ImageError as crate::py_errors::HostException>::to_py_err(&e))?;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
    })
}

// Declared in rust/crates/lk-agents-abi/src/media/emitter.rs.
type AudioEmitterImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AudioEmitter;

fn _extract_audio_emitter_initialize_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioEmitterInitializeSampleRateInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")
}
type AudioEmitterInitializeSampleRateInput = u32;

fn _extract_audio_emitter_initialize_num_channels(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioEmitterInitializeNumChannelsInput> {
    crate::py_values::number_from_py::<u16>(obj.as_any(), "num_channels")
}
type AudioEmitterInitializeNumChannelsInput = u16;

fn _extract_audio_emitter_initialize_frame_size_ms(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioEmitterInitializeFrameSizeMsInput> {
    crate::py_values::number_from_py::<u32>(obj.as_any(), "frame_size_ms")
}
type AudioEmitterInitializeFrameSizeMsInput = u32;

fn _extract_audio_emitter_push_data(obj: &Bound<'_, PyAny>) -> PyResult<AudioEmitterPushDataInput> {
    if obj.is_instance_of::<::pyo3::types::PyBytes>()
        || obj.is_instance_of::<::pyo3::types::PyByteArray>()
        || obj.is_instance_of::<::pyo3::types::PyMemoryView>()
    {
        crate::py_values::owned_bytes(obj.as_any(), "data")
    } else {
        obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::number_from_py::<u8>(item.as_any(), "data")
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }
}
type AudioEmitterPushDataInput = ::std::vec::Vec<u8>;

fn _extract_audio_emitter_report_usage_characters_count(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioEmitterReportUsageCharactersCountInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u64>(obj.as_any(), "characters_count")).map(Some)
    }
}
type AudioEmitterReportUsageCharactersCountInput = ::std::option::Option<u64>;

fn _extract_audio_emitter_pushed_duration_idx(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioEmitterPushedDurationIdxInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "idx")
}
type AudioEmitterPushedDurationIdxInput = i64;

#[pyclass(frozen, subclass, name = "AudioEmitter", module = "livekit.agents.tts")]
pub struct AudioEmitter(pub(crate) ::std::sync::Arc<AudioEmitterImpl>);

#[pymethods]
impl AudioEmitter {
    #[new]
    #[pyo3(signature = (*, label, dst_ch))]
    fn new(
        py: Python<'_>,
        label: ::std::string::String,
        dst_ch: Bound<'_, PyAny>,
    ) -> PyResult<AudioEmitter> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let dst_ch = crate::generated::media::EmitterOutputProxy::from_host(dst_ch.as_any())?;
        let out = py
            .detach(|| {
                <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::new(
                    &env, label, dst_ch,
                )
            })
            .map_err(|e: ::lk_agents_abi::EmitterError| {
                <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(AudioEmitter(::std::sync::Arc::new(out)))
    }

    #[pyo3(signature = (*, request_id, sample_rate, num_channels, mime_type, frame_size_ms = 200, stream = false))]
    fn initialize(
        &self,
        py: Python<'_>,
        request_id: ::std::string::String,
        #[pyo3(from_py_with = _extract_audio_emitter_initialize_sample_rate)]
        sample_rate: AudioEmitterInitializeSampleRateInput,
        #[pyo3(from_py_with = _extract_audio_emitter_initialize_num_channels)]
        num_channels: AudioEmitterInitializeNumChannelsInput,
        mime_type: ::std::string::String,
        #[pyo3(from_py_with = _extract_audio_emitter_initialize_frame_size_ms)]
        frame_size_ms: AudioEmitterInitializeFrameSizeMsInput,
        stream: bool,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::initialize(
                &self.0,
                request_id,
                sample_rate,
                num_channels,
                mime_type,
                frame_size_ms,
                stream,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    #[pyo3(signature = (*, segment_id))]
    fn start_segment(&self, py: Python<'_>, segment_id: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::start_segment(
                &self.0, segment_id,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    fn end_segment(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_segment(
                &self.0,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    #[pyo3(signature = (data))]
    fn push(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_audio_emitter_push_data)] data: AudioEmitterPushDataInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push(
                &self.0, &data,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    #[pyo3(signature = (frame))]
    fn push_frame(&self, py: Python<'_>, frame: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let frame = crate::generated::records::rtc_frames::audio_frame_from_py(frame.as_any())?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_frame(
                &self.0, frame,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    #[pyo3(signature = (delta_text))]
    fn push_timed_transcript(&self, py: Python<'_>, delta_text: Bound<'_, PyAny>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let delta_text = (if delta_text.is_instance_of::<::pyo3::types::PyList>() {
            delta_text.try_iter().and_then(|items| {
                items
                    .map(|item| {
                        let item = item?;
                        crate::generated::records::transcript::transcript_chunk_from_py(
                            item.as_any(),
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
        } else {
            (crate::generated::records::transcript::transcript_chunk_from_py(delta_text.as_any()))
                .map(|item| vec![item])
        })?;
        py.detach(|| <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::push_timed_transcript(&self.0, delta_text)).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// What the provider reported it billed for the open segment, or the next one when none is
    /// open, carried on the segment's final frame. Reports add up, one per provider request;
    /// providers that report nothing keep the framework's own measure (characters sent, audio
    /// received).
    #[pyo3(signature = (usage = None, *, characters_count = None))]
    fn report_usage(
        &self,
        py: Python<'_>,
        usage: Option<Bound<'_, PyAny>>,
        #[pyo3(from_py_with = _extract_audio_emitter_report_usage_characters_count)]
        characters_count: AudioEmitterReportUsageCharactersCountInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let usage = usage
            .map(|v| crate::generated::records::usage::token_usage_from_py(v.as_any()))
            .transpose()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::report_usage(
                &self.0,
                usage,
                characters_count,
            )
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::flush(&self.0)
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    fn end_input(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::end_input(&self.0)
        })
        .map_err(|e: ::lk_agents_abi::EmitterError| {
            <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
        })?;
        Ok(())
    }

    /// Waits for compressed input to finish decoding; reports a decode failure.
    fn join<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::join(
                &this,
            )
            .await
            .map_err(|e: ::lk_agents_abi::EmitterError| {
                <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    /// Discards pending audio and observes native settlement; never reports a synthesis failure.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::aclose(
                &this,
            )
            .await
            .map_err(|e: ::lk_agents_abi::EmitterError| {
                <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }

    #[pyo3(signature = (idx = -1))]
    fn pushed_duration(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_audio_emitter_pushed_duration_idx)]
        idx: AudioEmitterPushedDurationIdxInput,
    ) -> PyResult<f64> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::pushed_duration(
                &self.0, idx,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[getter]
    fn num_segments(&self, py: Python<'_>) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::num_segments(
                &self.0,
            )
        });
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (context_id))]
    fn _note_provider_request_id(
        &self,
        py: Python<'_>,
        context_id: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::_note_provider_request_id(&self.0, context_id)).map_err(|e: ::lk_agents_abi::EmitterError| <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[getter]
    fn output_closed(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py
            .detach(|| {
                <AudioEmitterImpl as ::lk_agents_abi::media::emitter::AudioEmitter>::output_closed(
                    &self.0,
                )
            })
            .map_err(|e: ::lk_agents_abi::EmitterError| {
                <::lk_agents_abi::EmitterError as crate::py_errors::HostException>::to_py_err(&e)
            })?;
        ::pyo3::PyResult::Ok(out)
    }
}

// Wherever the host passes a `::lk_agents_abi::tts::stream_pacer::PushedAudio`, an instance of this class is itself.
inventory::submit! {
    crate::py_objects::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::tts::stream_pacer::PushedAudio>(),
        read: |obj| {
            let this = obj.cast::<AudioEmitter>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::tts::stream_pacer::PushedAudio> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.
type AudioDecoderImpl =
    <crate::generated::implementations::Selected as super::Implementations>::AudioDecoder;

fn _extract_audio_decoder_new_sample_rate(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioDecoderNewSampleRateInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u32>(obj.as_any(), "sample_rate")).map(Some)
    }
}
type AudioDecoderNewSampleRateInput = ::std::option::Option<u32>;

fn _extract_audio_decoder_new_num_channels(
    obj: &Bound<'_, PyAny>,
) -> PyResult<AudioDecoderNewNumChannelsInput> {
    crate::py_values::number_from_py::<u16>(obj.as_any(), "num_channels")
}
type AudioDecoderNewNumChannelsInput = u16;

fn _extract_audio_decoder_push_data(obj: &Bound<'_, PyAny>) -> PyResult<AudioDecoderPushDataInput> {
    if obj.is_instance_of::<::pyo3::types::PyBytes>()
        || obj.is_instance_of::<::pyo3::types::PyByteArray>()
        || obj.is_instance_of::<::pyo3::types::PyMemoryView>()
    {
        crate::py_values::owned_bytes(obj.as_any(), "data")
    } else {
        obj.try_iter().and_then(|items| {
            items
                .map(|item| {
                    let item = item?;
                    crate::py_values::number_from_py::<u8>(item.as_any(), "data")
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }
}
type AudioDecoderPushDataInput = ::std::vec::Vec<u8>;

/// Container bytes decoded on one admitted native worker: input is queued without waiting for
/// the decoder, and decoded frames are awaited.
#[pyclass(frozen, name = "AudioDecoder", module = "livekit.agents._core")]
pub struct AudioDecoder(pub(crate) ::std::sync::Arc<AudioDecoderImpl>);

#[pymethods]
impl AudioDecoder {
    #[new]
    #[pyo3(signature = (mime, *, sample_rate = Some(48000), num_channels = 1))]
    fn new(
        py: Python<'_>,
        mime: ::std::string::String,
        #[pyo3(from_py_with = _extract_audio_decoder_new_sample_rate)]
        sample_rate: AudioDecoderNewSampleRateInput,
        #[pyo3(from_py_with = _extract_audio_decoder_new_num_channels)]
        num_channels: AudioDecoderNewNumChannelsInput,
    ) -> PyResult<AudioDecoder> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::new(&env, mime, sample_rate, num_channels)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        ::pyo3::PyResult::Ok(AudioDecoder(::std::sync::Arc::new(out)))
    }

    /// Refused after `end_input`, `aclose` or a decode failure.
    #[pyo3(signature = (data))]
    fn push(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_audio_decoder_push_data)] data: AudioDecoderPushDataInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::push(&self.0, &data)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as crate::py_errors::HostException>::to_py_err(&e))?;
        Ok(())
    }

    /// No more input: `next` returns `None` once everything pushed is decoded.
    fn end_input(&self, py: Python<'_>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::end_input(&self.0)
        });
        Ok(())
    }

    /// The next decoded frame, at the requested format; `None` at the end of input.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let out = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::next(&this, )
                .await.map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| {
                ::pyo3::IntoPyObjectExt::into_py_any(
                    out.map(|value| {
                        crate::generated::records::rtc_frames::audio_frame_to_py(py, value)
                    })
                    .transpose()?,
                    py,
                )
            })
        })
    }

    /// Cancel decoding and observe release of the native decoder's worker admission.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::py_calls::future_into_py(py, async move {
            let () = <AudioDecoderImpl as ::lk_agents_abi::media::codecs::AudioDecoder>::aclose(&this, )
                .await.map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as crate::py_errors::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/codecs.rs.
type MediaTokenizationImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MediaTokenization;
type MediaTokenizationSplitSentencesOutput = ::std::vec::Vec<(::std::string::String, usize, usize)>;

fn _extract_media_tokenization_split_sentences_min_sentence_len(
    obj: &Bound<'_, PyAny>,
) -> PyResult<MediaTokenizationSplitSentencesMinSentenceLenInput> {
    crate::py_values::number_from_py::<usize>(obj.as_any(), "min_sentence_len")
}
type MediaTokenizationSplitSentencesMinSentenceLenInput = usize;

#[pyfunction]
#[pyo3(signature = (text, min_sentence_len = 20, *, retain_format = false))]
pub fn split_sentences(
    py: Python<'_>,
    text: ::std::string::String,
    #[pyo3(from_py_with = _extract_media_tokenization_split_sentences_min_sentence_len)]
    min_sentence_len: MediaTokenizationSplitSentencesMinSentenceLenInput,
    retain_format: bool,
) -> PyResult<MediaTokenizationSplitSentencesOutput> {
    let out = py.detach(|| <MediaTokenizationImpl as ::lk_agents_abi::media::codecs::MediaTokenization>::split_sentences(&text, min_sentence_len, retain_format)).map_err(|e: ::lk_agents_abi::media::codecs::CodecFailure| <::lk_agents_abi::media::codecs::CodecFailure as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/media/images.rs.

/// [`::lk_agents_abi::media::images::FrameReader`] over a host `livekit.agents.utils.media._FrameReader`.
#[derive(Clone)]
pub struct FrameReaderProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl FrameReaderProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::images::FrameReader>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::media::images::FrameReader>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::images::FrameReader>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::media::images::FrameReader>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::media::images::FrameReader>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let _ = context;
        Ok(FrameReaderProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

impl ::lk_agents_abi::media::images::FrameReader for FrameReaderProxy {
    fn rgba(
        &self,
        frame: &::lk_agents_abi::MediaRef,
    ) -> ::std::result::Result<
        ::lk_agents_abi::utils::video::VideoFrame,
        ::lk_agents_abi::errors::HostError,
    > {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = ::std::clone::Clone::clone(frame);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] =
                    [crate::py_objects::object_ref_to_py(py, frame_arg)?];
                Ok((
                    host.bind(py).getattr("_core_rgba")?,
                    ::pyo3::types::PyTuple::new(py, args)?,
                    None,
                ))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (crate::generated::records::rtc_frames::video_frame_from_py(o.as_any())).map_err(|e| <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/media/emitter.rs.

/// [`::lk_agents_abi::media::emitter::EmitterOutput`] over a host `livekit.agents.utils.aio.Chan[livekit.agents.tts.SynthesizedAudio]`.
#[derive(Clone)]
pub struct EmitterOutputProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: crate::py_calls::Notifier,
}

impl EmitterOutputProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::emitter::EmitterOutput>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::media::emitter::EmitterOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(
        obj: &Bound<'_, PyAny>,
        context: &::lk_agents_abi::utils::host_ref::HostRef,
    ) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::media::emitter::EmitterOutput>> {
        if let Some(native) =
            crate::py_objects::native::<dyn ::lk_agents_abi::media::emitter::EmitterOutput>(obj)
        {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::media::emitter::EmitterOutput>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(
        obj: &Bound<'_, PyAny>,
        context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>,
    ) -> PyResult<Self> {
        let view = crate::py_calls::call(
            &obj.py().import("livekit.agents.tts")?.getattr("_EmitterOutput")?,
            (obj,),
            None,
        )?;
        let obj = &view;
        let context = match context {
            Some(context) => ::std::clone::Clone::clone(context),
            None => crate::py_calls::capture_context(obj.py())?,
        };
        Ok(EmitterOutputProxy {
            host: ::std::sync::Arc::new(obj.clone().unbind()),
            context,
            notifier: ::std::default::Default::default(),
        })
    }
}

impl ::lk_agents_abi::media::emitter::EmitterOutput for EmitterOutputProxy {
    fn send(
        &self,
        frame: ::lk_agents_abi::AudioFrame,
        request_id: ::std::string::String,
        segment_id: ::std::string::String,
        is_final: bool,
        timed_transcript: ::std::vec::Vec<::lk_agents_abi::TranscriptChunk>,
        usage: ::std::option::Option<::lk_agents_abi::usage::TokenUsage>,
        characters_count: ::std::option::Option<u64>,
        provider_request_ids: ::std::vec::Vec<::std::string::String>,
    ) {
        let host = ::std::sync::Arc::clone(&self.host);
        let frame_arg = frame;
        let request_id_arg = request_id;
        let segment_id_arg = segment_id;
        let is_final_arg = is_final;
        let timed_transcript_arg = timed_transcript;
        let usage_arg = usage;
        let characters_count_arg = characters_count;
        let provider_request_ids_arg = provider_request_ids;
        ::lk_agents_runtime::host::notify(
            &self.notifier,
            "EmitterOutput.send",
            &self.context,
            &[],
            false,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 8] = [
                    crate::generated::records::rtc_frames::audio_frame_to_py(py, frame_arg)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(request_id_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(segment_id_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(is_final_arg, py)?,
                    ::pyo3::types::PyList::new(
                        py,
                        timed_transcript_arg
                            .into_iter()
                            .map(|x| {
                                crate::generated::records::transcript::transcript_chunk_to_py(py, x)
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    )?
                    .into_any(),
                    match usage_arg {
                        None => py.None().into_bound(py),
                        Some(x) => crate::generated::records::usage::token_usage_to_py(py, x)?,
                    },
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(characters_count_arg, py)?,
                    ::pyo3::IntoPyObjectExt::into_bound_py_any(provider_request_ids_arg, py)?,
                ];
                Ok((host.bind(py).getattr("send")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
    }

    fn closed(&self) -> ::std::result::Result<bool, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let out = ::lk_agents_runtime::host::call_now(
            &crate::py_calls::Entry,
            ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
                Ok((host.bind(py).getattr("closed")?, ::pyo3::types::PyTuple::new(py, args)?, None))
            }),
        );
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.value.bind(py);
            let out = (o.extract::<bool>()).map_err(|e| {
                <::lk_agents_abi::errors::HostError as crate::py_errors::FromHostErr>::from_host(
                    py, &e,
                )
            })?;
            Ok(out)
        })
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::media::images::ImageError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::EmitterError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        #[allow(unreachable_patterns)]
        match self {
            Self::UnsupportedMime(..) => ::pyo3::exceptions::PyNotImplementedError::new_err(
                ::std::string::ToString::to_string(self),
            ),
            _ => ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(
                self,
            )),
        }
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::media::codecs::CodecFailure {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyRuntimeError::new_err(::std::string::ToString::to_string(self))
    }
}
