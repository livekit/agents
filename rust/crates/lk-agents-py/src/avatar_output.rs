//! SDK-room adaptation for the ABI-declared avatar sink. Runtime owns its lifecycle.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::io::{AudioOutput, PlaybackTracker, Result};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, HostRef};
use lk_agents_runtime::net::env::ProcessEnv;
use lk_agents_runtime::net::task::TokioSpawner;
use std::sync::Arc;

pub struct DataStreamAudioOutput(lk_agents_runtime::avatar::output::DataStreamAudioOutput);

impl lk_agents_abi::voice::avatar::DataStreamAudioOutput for DataStreamAudioOutput {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        room: HostRef,
        destination_identity: String,
        sample_rate: Option<u32>,
        wait_remote_track: Option<i32>,
        clear_buffer_timeout: Option<f64>,
        wait_playback_start: bool,
    ) -> std::result::Result<Self, ProviderError> {
        use lk_agents_abi::python::FromHostErr;
        use pyo3::prelude::*;

        let (room, timeout) = Python::attach(|py| -> PyResult<_> {
            let timeout = clear_buffer_timeout
                .map(|seconds| {
                    lk_agents_abi::python::duration_from_secs(seconds, "clear_buffer_timeout")
                })
                .transpose()?;
            let room = crate::conversions::host_ref_to_py(py, room)?;
            let factory =
                py.import("livekit.agents.voice.room_io")?.getattr("Room")?.getattr("of")?;
            let room = crate::host::construct(&factory, (room,), None)?;
            Ok((crate::generated::voice::RtcRoomProxy::from_host(&room)?, timeout))
        })
        .map_err(|error| Python::attach(|py| ProviderError::from_host(py, &error)))?;
        let handle = (env.runtime)()
            .map_err(|error| ProviderError::from(lk_agents_abi::errors::ApiError::new(error)))?;
        let spawner = TokioSpawner::new(handle);
        Ok(Self(lk_agents_runtime::avatar::output::DataStreamAudioOutput::new(
            room,
            destination_identity,
            sample_rate,
            wait_remote_track,
            timeout,
            wait_playback_start,
            env.clock.clone(),
            Arc::new(spawner),
        )))
    }

    fn label(&self) -> String {
        self.0.label().to_owned()
    }
    fn sample_rate(&self) -> Option<u32> {
        self.0.sample_rate()
    }
    async fn aclose(&self) {
        self.0.aclose().await;
    }
}

impl AudioOutput for DataStreamAudioOutput {
    fn label(&self) -> &str {
        self.0.label()
    }
    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        self.0.tracker()
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        self.0.capabilities()
    }
    fn sample_rate(&self) -> Option<u32> {
        self.0.sample_rate()
    }
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.0.capture(frame).await })
    }
    fn flush_segment(&self) {
        self.0.flush_segment();
    }
    fn clear_buffer(&self) {
        self.0.clear_buffer();
    }
    fn pause(&self) {
        self.0.pause();
    }
    fn resume(&self) {
        self.0.resume();
    }
}
