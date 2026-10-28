//! Retained buffers: only exact immutable bytes may be borrowed. Other exporters are snapshotted
//! under Python's buffer protocol while attached; no reference to mutable host storage escapes.

use std::sync::Arc;

use pyo3::buffer::{PyBuffer, PyUntypedBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};

/// Byte buffers have one-dimensional C layout. PCM additionally accepts signed 16-bit little
/// endian samples (the fixed AudioFrame format); it never reinterprets floats or big-endian PCM.
#[derive(Clone, Copy)]
pub enum BufferFormat {
    Bytes,
    I16Le,
    /// AudioByteStream accepts chunks that split a sample across calls.
    PcmChunk,
}

fn layout_error(path: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{path}: expected a one-dimensional contiguous byte or i16le buffer"
    ))
}

/// Exactly one copy from host storage into the shared allocation. Fan-out clones the Arc.
pub fn snapshot_buffer(
    obj: &Bound<'_, PyAny>,
    format: BufferFormat,
    path: &str,
) -> PyResult<Arc<[u8]>> {
    // The builtin view acquires the exporter once. All subsequent metadata access and casts use
    // that same view, so a custom exporter cannot hand out different buffers between checks.
    let view = PyMemoryView::from(obj).map_err(|e| super::at_path(obj.py(), e, path))?;
    let buffer = PyUntypedBuffer::get(view.as_any())?;
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(layout_error(path));
    }
    let pcm = matches!(format, BufferFormat::I16Le | BufferFormat::PcmChunk);
    let valid_format = match buffer.format().to_bytes() {
        b"B" | b"b" | b"c" => true,
        b"<h" => pcm,
        b"h" | b"@h" | b"=h" => pcm && cfg!(target_endian = "little"),
        _ => false,
    };
    if !valid_format {
        return Err(layout_error(path));
    }
    if matches!(format, BufferFormat::I16Le) && buffer.len_bytes() % 2 != 0 {
        return Err(PyValueError::new_err(format!("{path}: i16le data has a partial sample")));
    }
    // Byte casting is a zero-copy buffer operation. Copy the validated representation directly;
    // no aligned i16 Rust references or mutable host references escape the GIL-held snapshot.
    let byte_view = view.call_method1("cast", ("B",))?;
    let bytes = PyBuffer::<u8>::get(&byte_view)?;
    let cells = bytes.as_slice(obj.py()).ok_or_else(|| layout_error(path))?;
    Ok(copy_bytes(cells.iter().map(|v| v.get())))
}

fn copy_bytes(bytes: impl ExactSizeIterator<Item = u8>) -> Arc<[u8]> {
    let mut output = Arc::<[u8]>::new_uninit_slice(bytes.len());
    for (slot, byte) in
        Arc::get_mut(&mut output).expect("new allocation is unique").iter_mut().zip(bytes)
    {
        slot.write(byte);
    }
    // SAFETY: the exact-size iterator initializes every byte of this unique allocation. All
    // callers use slice iterators, with no host calls or fallible work during initialization.
    unsafe { output.assume_init() }
}

/// `Bytes` owns the snapshot or retains an exact bytes object for its entire native lifetime.
/// A read-only memoryview can still have writable aliases and therefore always takes a copy.
pub fn retained_bytes(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<bytes::Bytes> {
    if obj.is_exact_instance_of::<PyBytes>() {
        return Ok(bytes::Bytes::from_owner(obj.extract::<pyo3::pybacked::PyBackedBytes>()?));
    }
    Ok(bytes::Bytes::from_owner(snapshot_buffer(obj, BufferFormat::Bytes, path)?))
}
