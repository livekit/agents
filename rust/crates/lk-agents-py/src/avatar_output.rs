//! SDK-room adaptation for the ABI-declared avatar sink. Runtime owns its lifecycle.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::io::{AudioOutput, PlaybackTracker, Result};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, HostRef, RealClock, TokioSpawner};
use std::sync::Arc;

pub struct DataStreamAudioOutput(lk_agents_runtime::avatar_output::DataStreamAudioOutput);

impl lk_agents_abi::voice::avatar::DataStreamAudioOutput for DataStreamAudioOutput {
    fn new(
        room: HostRef,
        destination_identity: String,
        sample_rate: Option<u32>,
        wait_remote_track: Option<i32>,
        clear_buffer_timeout: Option<f64>,
        wait_playback_start: bool,
    ) -> std::result::Result<Self, ProviderError> {
        use lk_agents_abi::python::{FromHostErr, HostProxy};
        use pyo3::prelude::*;

        let (room, timeout) = Python::attach(|py| -> PyResult<_> {
            let timeout = clear_buffer_timeout
                .map(|seconds| {
                    lk_agents_abi::python::duration_from_secs(seconds, "clear_buffer_timeout")
                })
                .transpose()?;
            let room = (&room).into_pyobject(py)?;
            Ok((<dyn lk_agents_rtc::RtcRoom as HostProxy>::from_host(&room, None)?, timeout))
        })
        .map_err(|error| Python::attach(|py| ProviderError::from_host(py, &error)))?;
        let handle = lk_agents_runtime::process::handle().map_err(|error| {
            ProviderError::from(lk_agents_abi::errors::ApiError::new(error.to_string()))
        })?;
        let spawner = TokioSpawner::new(handle);
        let output = lk_agents_voice::avatar::DataStreamAudioOutput::new(
            room,
            destination_identity,
            sample_rate,
            wait_remote_track,
            timeout,
            wait_playback_start,
            lk_agents_runtime::process::host_clock(),
            Arc::new(spawner.clone()),
        );
        Ok(Self(lk_agents_runtime::avatar_output::DataStreamAudioOutput::from_native(
            output,
            spawner,
            Arc::new(RealClock::new()),
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
    fn tracker(&self) -> &Arc<PlaybackTracker> {
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

#[cfg(test)]
mod tests {
    use lk_agents_abi::voice::io::AudioOutput;
    use lk_agents_abi::voice::io::AudioOutputExt;
    use lk_agents_abi::{AudioFrame, VirtualClock};
    use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, RealClock, TokioSpawner};
    use lk_agents_rtc::{FakeRoom, FakeRtc, ParticipantKind};
    use lk_agents_runtime::avatar_output::DataStreamAudioOutput;
    use std::sync::Arc;

    fn fixture(name: &str) -> (DataStreamAudioOutput, Arc<FakeRoom>, Arc<VirtualClock>) {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room(name);
        room.add_participant("avatar", ParticipantKind::Agent);
        let spawner = TokioSpawner::current();
        let output = lk_agents_voice::avatar::DataStreamAudioOutput::new(
            room.clone(),
            "avatar".into(),
            None,
            None,
            None,
            false,
            clock.clone(),
            Arc::new(spawner.clone()),
        );
        let output = DataStreamAudioOutput::from_native(output, spawner, clock.clone());
        std::mem::forget(rtc);
        (output, room, clock)
    }

    async fn settle() {
        for _ in 0..40 {
            tokio::task::yield_now().await;
        }
    }

    fn frame() -> AudioFrame {
        AudioFrame::new(vec![0; 480], 24_000, 1, 240)
    }

    #[tokio::test]
    async fn dropping_the_adapter_retains_cleanup_in_the_native_executor() {
        let (output, room, _clock) = fixture("runtime_avatar_drop");
        output.capture_frame(&frame()).await.unwrap();
        room.local_participant().flow_byte_closes(false);
        drop(output);
        settle().await;
        assert!(!room.local_participant().byte_streams()[0].closed);
        room.local_participant().flow_byte_closes(true);
        settle().await;
        assert!(room.local_participant().byte_streams()[0].closed);
    }

    #[tokio::test]
    async fn a_deadline_records_incomplete_cleanup_and_keeps_the_writer_until_release() {
        use lk_agents_abi::telemetry::work::TelemetryWorker;
        fn incomplete_count() -> u64 {
            lk_agents_runtime::telemetry::work::SdkWorker
                .diagnostics()
                .into_iter()
                .find(|(key, _)| key == "incomplete_cleanup")
                .unwrap()
                .1
        }
        let (output, room, clock) = fixture("runtime_avatar_deadline");
        output.capture_frame(&frame()).await.unwrap();
        room.local_participant().flow_byte_closes(false);
        let before = incomplete_count();
        let (sent, received) = tokio::sync::oneshot::channel();
        let output = Arc::new(output);
        let closing = Arc::clone(&output);
        lk_agents_abi::SpawnerExt::spawn_unboxed(&TokioSpawner::current(), async move {
            closing.aclose().await;
            let _ = sent.send(());
        });
        settle().await;
        clock.advance(HOST_TEARDOWN_TIMEOUT);
        received.await.unwrap();
        assert!(!room.local_participant().byte_streams()[0].closed);
        assert!(incomplete_count() > before);
        room.local_participant().flow_byte_closes(true);
        settle().await;
        assert!(room.local_participant().byte_streams()[0].closed);
        output.aclose().await;
    }
    #[test]
    fn drop_cleanup_runs_after_the_callers_async_runtime_has_closed() {
        let host_loop = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let native = lk_agents_runtime::process::handle().unwrap();
        let (rtc, _) = FakeRtc::with_virtual_clock();
        let room = rtc.room("runtime_avatar_closed_host");
        room.add_participant("avatar", ParticipantKind::Agent);
        let clock = Arc::new(RealClock::new());
        let spawner = TokioSpawner::new(native.clone());
        let output = lk_agents_voice::avatar::DataStreamAudioOutput::new(
            room.clone(),
            "avatar".into(),
            None,
            None,
            None,
            false,
            clock.clone(),
            Arc::new(spawner.clone()),
        );
        let output = DataStreamAudioOutput::from_native(output, spawner, clock.clone());
        host_loop.block_on(output.capture_frame(&frame())).unwrap();
        drop(host_loop);
        drop(output);
        native.block_on(async {
            lk_agents_abi::timeout(&*clock, HOST_TEARDOWN_TIMEOUT, async {
                while !room.local_participant().byte_streams()[0].closed {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        });
        assert!(room.local_participant().byte_streams()[0].closed);
    }
}

#[cfg(test)]
#[path = "avatar_output_tests.rs"]
mod ordering_tests;
