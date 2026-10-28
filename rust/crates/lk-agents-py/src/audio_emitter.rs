//! Implementation selected by the generated TTS emitter adapter.

use crate::host::emitter::{State, run_now};
use pyo3::prelude::*;
use std::sync::Arc;

/// Python metadata and callback adapter behind the ABI-declared, generated emitter class.
pub struct AudioEmitter {
    state: Arc<State>,
}

impl AudioEmitter {
    fn on_host<T: Send + 'static>(
        &self,
        call: impl FnOnce(&Arc<State>, Python<'_>) -> PyResult<T> + Send + 'static,
    ) -> Result<T, lk_agents_abi::errors::ProviderError> {
        run_now(&self.state, call).map_err(lk_agents_abi::python::captured)
    }
}

impl lk_agents_abi::media::emitter::AudioEmitter for AudioEmitter {
    fn new(
        label: String,
        dst_ch: lk_agents_abi::HostRef,
    ) -> Result<Self, lk_agents_abi::errors::ProviderError> {
        Python::attach(|py| {
            let channel = dst_ch.into_pyobject(py)?.unbind();
            Ok(Self { state: Arc::new(State::new(py, &label, channel)?) })
        })
        .map_err(lk_agents_abi::python::captured)
    }
    fn initialize(
        &self,
        request_id: String,
        sample_rate: u32,
        num_channels: u16,
        mime_type: String,
        frame_size_ms: u32,
        stream: bool,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.state.ensure_unstarted().map_err(lk_agents_abi::python::captured)?;
        self.state.recapture().map_err(lk_agents_abi::python::captured)?;
        self.on_host(move |state, py| {
            State::initialize(
                state,
                py,
                &request_id,
                sample_rate,
                num_channels,
                &mime_type,
                frame_size_ms,
                stream,
            )
        })
    }
    fn start_segment(
        &self,
        segment_id: String,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| State::start_segment(state, py, &segment_id))
    }
    fn end_segment(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(State::end_segment)
    }
    fn push(
        &self,
        data: lk_agents_abi::HostRef,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| State::push(state, py, &data.into_pyobject(py)?))
    }
    fn push_frame(
        &self,
        frame: lk_agents_abi::AudioFrame,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| State::push_frame(state, py, frame))
    }
    fn push_timed_transcript(
        &self,
        delta_text: lk_agents_abi::HostRef,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| {
            State::push_timed_transcript(state, py, &delta_text.into_pyobject(py)?)
        })
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(State::flush)
    }
    fn end_input(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(State::end_input)
    }
    async fn join(
        &self,
        context: lk_agents_abi::invoke::CallContext,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.state.finalized().await.map_err(lk_agents_abi::python::captured)?;
        crate::host::emitter::run(&self.state, context, State::join)
            .await
            .map_err(lk_agents_abi::python::captured)
    }
    async fn aclose(
        &self,
        context: lk_agents_abi::invoke::CallContext,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        crate::host::emitter::run(&self.state, context, State::aclose)
            .await
            .map_err(lk_agents_abi::python::captured)?;
        // Closing discards synthesis errors, but still owns the native finalizer through settlement.
        let _ = self.state.finalized().await;
        Ok(())
    }
    fn pushed_duration(&self, idx: i64) -> f64 {
        self.state.pushed_duration(idx)
    }
    fn num_segments(&self) -> usize {
        self.state.num_segments()
    }
    fn _note_provider_request_id(
        &self,
        context_id: String,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| State::_note_provider_request_id(state, py, &context_id))
    }
    fn output_closed(&self) -> Result<bool, lk_agents_abi::errors::ProviderError> {
        self.on_host(move |state, py| state.output_closed(py))
    }
}
