//! Video as plain data, and the frames an avatar worker moves: records with their bytes, never the
//! host's objects. A frame's buffer is shared (`Bytes`), so moving it through the core copies
//! nothing; crossing in from a host `bytes` buffer copies nothing either, and crossing out is one
//! copy into the host's `bytes`.

use lk_agents_host_derive::Host;

use crate::AudioFrame;

/// `rtc.VideoFrame(width, height, type, data)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// `rtc.VideoBufferType`, as the SDK numbers it.
    pub buffer_type: i32,
    pub data: bytes::Bytes,
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &VideoFrame {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::types::PyAnyMethods as _;
        static CLASS: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        let data = pyo3::types::PyBytes::new(py, &self.data);
        CLASS.import(py, "livekit.rtc", "VideoFrame")?.call1((
            self.width,
            self.height,
            self.buffer_type,
            data,
        ))
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for VideoFrame {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

/// `data` held as the host's own `bytes` when the frame's buffer is one, else copied once
/// through the buffer protocol. Read-only views of mutable exporters still copy.
#[cfg(feature = "python")]
impl VideoFrame {
    /// Extract once, optionally refusing oversized host buffers before borrowing or copying them.
    pub fn from_host(
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        max_bytes: Option<usize>,
    ) -> pyo3::PyResult<Self> {
        use crate::python::{attr_with, number_from_py};
        const OWNER: &str = "VideoFrameFromPy";
        let width = attr_with(obj, OWNER, "width", |v| number_from_py(v, "VideoFrame.width"))?;
        let height = attr_with(obj, OWNER, "height", |v| number_from_py(v, "VideoFrame.height"))?;
        let buffer_type = attr_with(obj, OWNER, "type", |v| number_from_py(v, "VideoFrame.type"))?;
        let data = attr_with(obj, OWNER, "data", |view| {
            crate::python::retained_bytes(view, "VideoFrame.data", max_bytes)
        })?;
        Ok(VideoFrame { width, height, buffer_type, data })
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for VideoFrame {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        Self::from_host(&obj, None)
    }
}

/// A frame the SDK's synchronizer takes: audio or video.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, extract, untagged)]
#[crate::host(handwritten(extract, emit, host = "rtc.AudioFrame | rtc.VideoFrame"))]
pub enum MediaFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
}

/// What an avatar's audio receiver or video generator yields: a frame, or the end of a segment
/// (`AudioSegmentEnd`).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, extract, untagged)]
#[crate::host(handwritten(extract, emit, host = "rtc.AudioFrame | rtc.VideoFrame | AudioSegmentEnd"))]
pub enum AvatarFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
    #[binding(class = "livekit.agents.voice.avatar.AudioSegmentEnd")]
    SegmentEnd,
}
