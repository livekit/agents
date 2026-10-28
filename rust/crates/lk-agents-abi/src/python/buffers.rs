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
    max_bytes: Option<usize>,
) -> PyResult<Arc<[u8]>> {
    // The builtin view acquires the exporter once. All subsequent metadata access and casts use
    // that same view, so a custom exporter cannot hand out different buffers between checks.
    let view = PyMemoryView::from(obj).map_err(|e| super::at_path(obj.py(), e, path))?;
    let buffer = PyUntypedBuffer::get(view.as_any())?;
    if let Some(limit) = max_bytes
        && buffer.len_bytes() > limit
    {
        return Err(PyValueError::new_err(format!(
            "{path}: buffer has {} bytes; limit is {limit}",
            buffer.len_bytes()
        )));
    }
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
    Ok(copy_bytes(obj.py(), &bytes))
}

fn copy_bytes(py: Python<'_>, bytes: &PyBuffer<u8>) -> Arc<[u8]> {
    let len = bytes.len_bytes();
    let mut output = if len >= 1024 * 1024 {
        // Only the destination length crosses this detached section. The acquired exporter
        // remains pinned; its mutable bytes are first read after Python is attached again.
        py.detach(move || {
            let mut output = Arc::<[u8]>::new_uninit_slice(len);
            let destination = Arc::get_mut(&mut output).expect("new allocation is unique");
            // Commit destination pages outside the GIL. The final byte covers an unaligned
            // allocation's trailing page; 4 KiB strides cover supported larger page sizes too.
            // SAFETY: each address is an allocated byte of this unique destination.
            // Volatile writes preserve the page touches when a later memcpy overwrites them.
            unsafe {
                for page in destination.chunks_mut(4096) {
                    page[0].as_mut_ptr().write_volatile(0);
                }
                if let Some(last) = destination.last_mut() {
                    last.as_mut_ptr().write_volatile(0);
                }
            }
            output
        })
    } else {
        Arc::<[u8]>::new_uninit_slice(len)
    };
    let destination = Arc::get_mut(&mut output).expect("new allocation is unique");
    // SAFETY: the retained buffer has been validated as contiguous bytes. Python is attached
    // again before the source is read; no mutable host reference escapes. The unique
    // destination is disjoint from the exporter and has exactly `len` writable bytes.
    // A bulk copy keeps large SDK buffers from holding the GIL in a per-byte Rust loop.
    unsafe {
        if len != 0 {
            std::ptr::copy_nonoverlapping(
                bytes.buf_ptr().cast::<u8>(),
                destination.as_mut_ptr().cast::<u8>(),
                len,
            );
        }
        output.assume_init()
    }
}

/// `Bytes` owns the snapshot or retains an exact bytes object for its entire native lifetime.
/// A read-only memoryview can still have writable aliases and therefore always takes a copy.
pub fn retained_bytes(
    obj: &Bound<'_, PyAny>,
    path: &str,
    max_bytes: Option<usize>,
) -> PyResult<bytes::Bytes> {
    if obj.is_exact_instance_of::<PyBytes>() {
        let size = obj.cast::<PyBytes>()?.as_bytes().len();
        if let Some(limit) = max_bytes
            && size > limit
        {
            return Err(PyValueError::new_err(format!(
                "{path}: buffer has {size} bytes; limit is {limit}"
            )));
        }
        return Ok(bytes::Bytes::from_owner(obj.extract::<pyo3::pybacked::PyBackedBytes>()?));
    }
    Ok(bytes::Bytes::from_owner(snapshot_buffer(obj, BufferFormat::Bytes, path, max_bytes)?))
}
