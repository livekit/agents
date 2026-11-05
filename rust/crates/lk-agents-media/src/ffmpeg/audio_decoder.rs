//! Demux and decode to s16 at the source rate, then resample through the shared soxr path.

use std::ffi::CStr;

use super::*;
use crate::audio_decoder::DecodeLimits;
use crate::resample::AudioResampler;

/// Decode one container through caller-owned I/O. This function creates no thread or queue.
/// The runtime calls it on an admitted blocking worker because FFmpeg's AVIO calls `Read`.
/// Native handles and callback pointers remain local to this invocation.
pub fn decode_stream<R, F>(
    format_short_name: Option<&str>,
    out_rate: u32,
    out_channels: u16,
    limits: DecodeLimits,
    input: &mut R,
    mut output: F,
) -> Result<AudioFormat>
where
    R: std::io::Read,
    F: FnMut(AudioFormat, &[i16]) -> Result<()>,
{
    if !limits.is_valid()
        || out_channels == 0
        || out_channels > limits.max_channels
        || (out_rate != 0 && !(limits.min_sample_rate..=limits.max_sample_rate).contains(&out_rate))
    {
        return Err(CodecError::DecodeLimit("invalid target format or transform limits"));
    }
    let name = format_short_name.map(CString::new).transpose().map_err(|_| CodecError::Ffmpeg {
        op: "container name",
        detail: "contains a NUL byte".into(),
    })?;
    // The reader outlives the native handle, which is dropped first.
    let mut reader = Reader { input, error: None };
    let mut error = ErrBuf::new();
    let decoder = Handle::open(name.as_deref(), out_channels, &limits, &mut reader, &mut error)?;
    let source = unsafe {
        AudioFormat::new(
            lk_dec_src_rate(decoder.0).max(0) as u32,
            lk_dec_src_channels(decoder.0).max(0) as u16,
        )
    };
    let format =
        AudioFormat::new(if out_rate == 0 { source.sample_rate } else { out_rate }, out_channels);
    limits.validate(source, Some(format.sample_rate), 0)?;
    let mut output = Output {
        callback: &mut output,
        source: AudioFormat::new(source.sample_rate, out_channels),
        format,
        resampler: AudioResampler::decoded(Some(format.sample_rate)),
        limits,
        error: None,
    };
    let rc = unsafe {
        lk_dec_run(
            decoder.0,
            pcm::<F>,
            (&mut output as *mut Output<'_, F>).cast(),
            error.as_mut_ptr(),
            ERRLEN as c_int,
        )
    };
    if let Some(error) = reader.error.take().or(output.error.take()) {
        return Err(error);
    }
    if rc < 0 {
        return Err(error.take("decode"));
    }
    output.flush()?;
    Ok(format)
}

/// The caller's input, read by FFmpeg's AVIO callback.
struct Reader<'a, R> {
    input: &'a mut R,
    error: Option<CodecError>,
}

extern "C" fn read<R: std::io::Read>(opaque: *mut c_void, data: *mut u8, size: c_int) -> c_int {
    if size <= 0 {
        return 0;
    }
    // SAFETY: `reader` lives until the native decoder is freed. FFmpeg calls this synchronously
    // on the invoking thread; its output buffer is valid for `size` bytes.
    let reader = unsafe { &mut *(opaque as *mut Reader<'_, R>) };
    let bytes = unsafe { std::slice::from_raw_parts_mut(data, size as usize) };
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.input.read(bytes)));
    let detail = match result {
        Ok(Ok(n)) if n <= bytes.len() => return n as c_int,
        Ok(Err(error)) => error.to_string(),
        Err(_) => "input callback panicked".into(),
        Ok(Ok(_)) => "input callback returned more bytes than its supplied buffer".into(),
    };
    reader.error = Some(CodecError::Ffmpeg { op: "read", detail });
    -1
}

/// Resamples decoded frames and hands them to the caller's callback.
struct Output<'a, F> {
    callback: &'a mut F,
    source: AudioFormat,
    format: AudioFormat,
    resampler: AudioResampler,
    limits: DecodeLimits,
    error: Option<CodecError>,
}

impl<F: FnMut(AudioFormat, &[i16]) -> Result<()>> Output<'_, F> {
    fn push(&mut self, samples: &[i16]) -> Result<()> {
        self.limits.validate(self.source, Some(self.format.sample_rate), samples.len())?;
        let samples = self.resampler.push_samples(self.source, samples)?;
        self.emit(&samples)
    }

    /// Hands the resampler's tail to the callback, still catching its panic.
    fn flush(&mut self) -> Result<()> {
        let tail = self.resampler.flush_samples()?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.emit(&tail)))
            .map_err(|_| output_panicked())?
    }

    fn emit(&mut self, samples: &[i16]) -> Result<()> {
        if samples.len() > self.limits.max_samples_per_frame {
            return Err(CodecError::DecodeLimit("resampled frame samples"));
        }
        if !samples.is_empty() {
            (self.callback)(self.format, samples)?;
        }
        Ok(())
    }
}

extern "C" fn pcm<F>(opaque: *mut c_void, data: *const i16, size: c_int) -> c_int
where
    F: FnMut(AudioFormat, &[i16]) -> Result<()>,
{
    if size <= 0 {
        return 0;
    }
    // SAFETY: `output` lives through lk_dec_run and native teardown. PCM is borrowed only
    // for this synchronous callback. Catching panics prevents unwinding across the C ABI.
    let output = unsafe { &mut *(opaque as *mut Output<'_, F>) };
    let samples = unsafe { std::slice::from_raw_parts(data, size as usize) };
    let error =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.push(samples))) {
            Ok(Ok(())) => return 0,
            Ok(Err(error)) => error,
            Err(_) => output_panicked(),
        };
    output.error = Some(error);
    -1
}

fn output_panicked() -> CodecError {
    CodecError::Ffmpeg { op: "output", detail: "output callback panicked".into() }
}

struct Handle(*mut LkDecoder);

impl Handle {
    /// Opens the container, reading through `reader`, which must outlive the handle.
    fn open<R: std::io::Read>(
        name: Option<&CStr>,
        out_channels: u16,
        limits: &DecodeLimits,
        reader: &mut Reader<'_, R>,
        error: &mut ErrBuf,
    ) -> Result<Self> {
        // SAFETY: the reader and container name outlive the native handle; callbacks do not
        // escape lk_dec_open/lk_dec_run. The opaque pointer is borrowed exclusively on this thread.
        let decoder = unsafe {
            lk_dec_open(
                name.map_or(std::ptr::null(), |s| s.as_ptr()),
                out_channels as c_int,
                limits.min_sample_rate as c_int,
                limits.max_sample_rate as c_int,
                limits.max_channels as c_int,
                limits.max_streams as c_int,
                limits.max_samples_per_frame as c_int,
                read::<R>,
                (reader as *mut Reader<'_, R>).cast(),
                error.as_mut_ptr(),
                ERRLEN as c_int,
            )
        };
        if decoder.is_null() {
            return Err(reader.error.take().unwrap_or_else(|| error.take("open")));
        }
        Ok(Handle(decoder))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: lk_dec_open returned this sole handle and no callback retains it.
        unsafe { lk_dec_free(self.0) };
    }
}
