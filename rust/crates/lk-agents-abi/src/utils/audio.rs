//! Owned PCM, not a [`MediaRef`](crate::MediaRef): native VAD and TTS read the samples and frames
//! live milliseconds, so the binding copies; the [`Arc`] makes the STT/VAD fan-out free.

use std::sync::Arc;


/// Interleaved signed 16-bit little-endian PCM, with a one-dimensional contiguous payload.
/// `samples_per_channel` is carried, not derived: a plugin frame may disagree with its data length.
/// The field names are `rtc.AudioFrame(..)`'s keyword arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// `bytes`, not a list of ints (which `rtc.AudioFrame` would read as garbage).
    pub data: Arc<[u8]>,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub samples_per_channel: u32,
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &AudioFrame {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        static CLASS: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("data", pyo3::types::PyBytes::new(py, &self.data))?;
        kwargs.set_item("sample_rate", self.sample_rate)?;
        kwargs.set_item("num_channels", self.num_channels)?;
        kwargs.set_item("samples_per_channel", self.samples_per_channel)?;
        CLASS.import(py, "livekit.rtc", "AudioFrame")?.call((), Some(&kwargs))
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for AudioFrame {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

/// `data` is snapshotted once through the buffer protocol. The
/// scalars are read first, so an object that is not an audio frame (a video frame tried as one)
/// fails before its buffer is copied.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AudioFrame {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr_with, number_from_py};
        const OWNER: &str = "AudioFrameFromPy";
        let obj = &*obj;
        let sample_rate = attr_with(obj, OWNER, "sample_rate", |v| number_from_py(v, "AudioFrame.sample_rate"))?;
        let num_channels = attr_with(obj, OWNER, "num_channels", |v| number_from_py(v, "AudioFrame.num_channels"))?;
        let samples_per_channel = attr_with(obj, OWNER, "samples_per_channel", |v| number_from_py(v, "AudioFrame.samples_per_channel"))?;
        Ok(AudioFrame {
            data: attr_with(obj, OWNER, "data", |v| {
                crate::python::snapshot_buffer(v, crate::python::BufferFormat::I16Le, "AudioFrame.data")
            })?,
            sample_rate,
            num_channels,
            samples_per_channel,
        })
    }
}

impl AudioFrame {
    pub fn new(
        data: impl Into<Arc<[u8]>>,
        sample_rate: u32,
        num_channels: u16,
        samples_per_channel: u32,
    ) -> Self {
        AudioFrame { data: data.into(), sample_rate, num_channels, samples_per_channel }
    }

    /// Zero when `sample_rate` is zero.
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        f64::from(self.samples_per_channel) / f64::from(self.sample_rate)
    }
}

/// `utils.AudioBuffer`: a slice covers one frame or a list.
pub type AudioBuffer<'a> = &'a [AudioFrame];

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: computing duration from `data.len()` instead of the declared sample count, or a
    // divide-by-zero panic on a host-supplied zero sample rate.
    #[test]
    fn duration_uses_the_declared_sample_count() {
        let f = AudioFrame::new(vec![0u8; 4], 48_000, 1, 480);
        assert!((f.duration() - 0.01).abs() < 1e-12);
        assert_eq!(AudioFrame::new(vec![0u8; 4], 0, 1, 480).duration(), 0.0);
    }


}
