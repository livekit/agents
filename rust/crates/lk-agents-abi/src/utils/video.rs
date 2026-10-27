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
        static CLASS: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new();
        let data = pyo3::types::PyBytes::new(py, &self.data);
        CLASS.import(py, "livekit.rtc", "VideoFrame")?.call1((self.width, self.height, self.buffer_type, data))
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
/// (`memoryview.tobytes()`).
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for VideoFrame {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr, attr_with};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "VideoFrameFromPy";
        let obj = &*obj;
        let width = attr(obj, OWNER, "width")?;
        let height = attr(obj, OWNER, "height")?;
        let buffer_type = attr(obj, OWNER, "type")?;
        let data = attr_with(obj, OWNER, "data", |view| {
            let whole = view.getattr("obj").ok().filter(|b| {
                b.is_instance_of::<pyo3::types::PyBytes>()
                    && b.len().ok() == view.getattr("nbytes").and_then(|n| n.extract::<usize>()).ok()
            });
            let buffer = match whole {
                Some(b) => b,
                None => view.call_method0("tobytes")?,
            };
            Ok(bytes::Bytes::from_owner(buffer.extract::<pyo3::pybacked::PyBackedBytes>()?))
        })?;
        Ok(VideoFrame { width, height, buffer_type, data })
    }
}

/// A frame the SDK's synchronizer takes: audio or video.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, extract, untagged)]
pub enum MediaFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
}

/// What an avatar's audio receiver or video generator yields: a frame, or the end of a segment
/// (`AudioSegmentEnd`).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit, extract, untagged)]
pub enum AvatarFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
    #[binding(class = "livekit.agents.voice.avatar.AudioSegmentEnd")]
    SegmentEnd,
}
