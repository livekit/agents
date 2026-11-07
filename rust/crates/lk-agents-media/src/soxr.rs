//! Sample-rate conversion through libsoxr, also used by `rtc.AudioResampler`.
//! Quality recipes retain the SDK's ordinal mapping, including High and VeryHigh.

#[allow(
    clippy::disallowed_types,
    reason = "guards libsoxr's process-global FFT cache across pool and host threads"
)]
use std::{
    ffi::{CStr, c_char},
    os::raw::{c_ulong, c_void},
    sync::RwLock,
};

#[allow(
    clippy::disallowed_types,
    reason = "guards libsoxr's process-global FFT cache across pool and host threads"
)]
/// libsoxr's FFT coefficient cache is process-global and unlocked without OpenMP; growing it under
/// a running transform crashes or silently corrupts output. Only `soxr_create` grows it, so `new`
/// takes this for writing and `push`/`flush` (`soxr_clear`) for reading.
static FFT_CACHE: RwLock<()> = RwLock::new(());

/// `Quality`'s discriminants are protobuf ordinals, not soxr recipes (see module docs).
use lk_agents_abi::{Quality, ResampleError};

/// Not `Sync`; distinct handles on distinct threads are safe only because of `FFT_CACHE`.
pub struct Resampler {
    ptr: soxr_sys::soxr_t,
    out: Vec<i16>,
    input_rate: f64,
    output_rate: f64,
    num_channels: u32,
}

unsafe impl Send for Resampler {}

impl Resampler {
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        num_channels: u32,
        quality: Quality,
    ) -> Result<Self, ResampleError> {
        if input_rate == 0 || output_rate == 0 || num_channels == 0 {
            return Err(ResampleError::Create("rates and channel count must be non-zero".into()));
        }
        let input_rate = f64::from(input_rate);
        let output_rate = f64::from(output_rate);

        // A real out-param, unlike upstream.
        let mut err: *const c_char = std::ptr::null();

        // The writer. A poisoned lock guards no state of ours, so it is not a reason to refuse.
        let _cache = FFT_CACHE.write().unwrap_or_else(|e| e.into_inner());

        let ptr = unsafe {
            let mut io = soxr_sys::soxr_io_spec(
                soxr_sys::soxr_datatype_t_SOXR_INT16_I,
                soxr_sys::soxr_datatype_t_SOXR_INT16_I,
            );
            // Default integer-output dithering uses a seed derived from wall time and the handle
            // address. Media transforms must reproduce the same PCM from the same supplied input.
            io.flags |= soxr_sys::SOXR_NO_DITHER as c_ulong;
            // Quality flags 0 and 1 thread: what `livekit-ffi` hard-codes.
            let q = soxr_sys::soxr_quality_spec(quality as c_ulong, 0);
            let rt = soxr_sys::soxr_runtime_spec(1);
            soxr_sys::soxr_create(input_rate, output_rate, num_channels, &mut err, &io, &q, &rt)
        };

        if !err.is_null() {
            return Err(ResampleError::Create(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        if ptr.is_null() {
            // soxr documents that it sets `err`; a null handle would otherwise segfault on `push`.
            return Err(ResampleError::Create("soxr_create returned null".into()));
        }

        Ok(Self {
            ptr,
            // A valid pointer for soxr even before the first resize.
            out: Vec::with_capacity(output_rate as usize / 100),
            input_rate,
            output_rate,
            num_channels,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate as u32
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate as u32
    }

    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }

    /// May return an empty slice: soxr buffers.
    pub fn push(&mut self, input: &[i16]) -> Result<&[i16], ResampleError> {
        let nch = self.num_channels as usize;
        if !input.len().is_multiple_of(nch) {
            return Err(ResampleError::PartialFrame {
                len: input.len(),
                num_channels: self.num_channels,
            });
        }
        let in_frames = input.len() / nch;
        let ratio = self.output_rate / self.input_rate;

        // A reader of the FFT cache.
        let _cache = FFT_CACHE.read().unwrap_or_else(|e| e.into_inner());

        // `soxr_delay` is the filter's residual; sizing on ratio alone makes soxr silently emit less.
        let delay = unsafe { soxr_sys::soxr_delay(self.ptr) };
        // `+ 1` is a margin against a `ceil` rounding edge, not load-bearing.
        let max_out = ((in_frames as f64 * ratio).ceil() as usize) + (delay.ceil() as usize) + 1;

        let needed = max_out * nch;
        if self.out.len() < needed {
            self.out.resize(needed, 0);
        }

        let mut idone: usize = 0;
        let mut odone: usize = 0;
        let err = unsafe {
            soxr_sys::soxr_process(
                self.ptr,
                input.as_ptr() as *const c_void,
                in_frames,
                &mut idone,
                self.out.as_mut_ptr() as *mut c_void,
                max_out,
                &mut odone,
            )
        };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        Ok(&self.out[..odone * nch])
    }

    /// Flushes and clears, as the host SDK does, so no filter history carries across a silence.
    pub fn flush(&mut self) -> Result<&[i16], ResampleError> {
        let nch = self.num_channels as usize;
        if self.out.is_empty() {
            // Nothing pushed: no tail, but soxr still needs a real buffer, not a dangling pointer.
            self.out.resize(self.output_rate as usize / 100 * nch, 0);
        }
        let cap = self.out.len() / nch;

        // A reader: `soxr_clear` re-initialises at lengths construction already reached.
        let _cache = FFT_CACHE.read().unwrap_or_else(|e| e.into_inner());

        let mut odone: usize = 0;
        let err = unsafe {
            soxr_sys::soxr_process(
                self.ptr,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                self.out.as_mut_ptr() as *mut c_void,
                cap,
                &mut odone,
            )
        };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }

        let err = unsafe { soxr_sys::soxr_clear(self.ptr) };
        if !err.is_null() {
            return Err(ResampleError::Process(
                unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned(),
            ));
        }
        Ok(&self.out[..odone * nch])
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe { soxr_sys::soxr_delete(self.ptr) }
    }
}

#[cfg(test)]
#[path = "soxr_tests.rs"]
mod tests;
