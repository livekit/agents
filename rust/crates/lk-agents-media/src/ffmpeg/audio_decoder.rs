//! Demux and decode to s16 at the source rate, then resample through the shared soxr path.

use super::*;
use crate::resample::AudioResampler;

/// Decode one container through caller-owned I/O. This function creates no thread or queue.
/// The runtime calls it on an admitted blocking worker because FFmpeg's AVIO calls `Read`.
/// Native handles and callback pointers remain local to this invocation.
pub fn decode_stream<R, F>(
    format_short_name: Option<&str>,
    out_rate: u32,
    out_channels: u16,
    limits: crate::audio_decoder::DecodeLimits,
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
    struct Reader<'a, R> {
        input: &'a mut R,
        error: Option<CodecError>,
    }
    extern "C" fn read<R: std::io::Read>(opaque: *mut c_void, data: *mut u8, size: c_int) -> c_int {
        if size <= 0 {
            return 0;
        }
        // SAFETY: `reader` lives until the native decoder is freed below. FFmpeg calls this
        // synchronously on the invoking thread; its output buffer is valid for `size` bytes.
        let reader = unsafe { &mut *(opaque as *mut Reader<'_, R>) };
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, size as usize) };
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.input.read(bytes)));
        match result {
            Ok(Ok(n)) if n <= bytes.len() => n as c_int,
            result => {
                let detail = match result {
                    Ok(Err(error)) => error.to_string(),
                    Err(_) => "input callback panicked".into(),
                    Ok(Ok(_)) => {
                        "input callback returned more bytes than its supplied buffer".into()
                    }
                };
                reader.error = Some(CodecError::Ffmpeg { op: "read", detail });
                -1
            }
        }
    }
    struct Output<'a, F> {
        callback: &'a mut F,
        source: AudioFormat,
        format: AudioFormat,
        resampler: AudioResampler,
        limits: crate::audio_decoder::DecodeLimits,
        error: Option<CodecError>,
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
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            output.limits.validate(
                output.source,
                Some(output.format.sample_rate),
                samples.len(),
            )?;
            let samples = output.resampler.push_samples(output.source, samples)?;
            if samples.len() > output.limits.max_samples_per_frame {
                return Err(CodecError::DecodeLimit("resampled frame samples"));
            }
            if !samples.is_empty() {
                (output.callback)(output.format, &samples)?;
            }
            Ok(())
        })) {
            Ok(Ok(())) => 0,
            result => {
                output.error = Some(match result {
                    Ok(Err(error)) => error,
                    Err(_) => CodecError::Ffmpeg {
                        op: "output",
                        detail: "output callback panicked".into(),
                    },
                    Ok(Ok(())) => unreachable!(),
                });
                -1
            }
        }
    }
    struct Handle(*mut LkDecoder);
    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: lk_dec_open returned this sole handle and no callback retains it.
            unsafe { lk_dec_free(self.0) };
        }
    }
    let name = format_short_name.map(CString::new).transpose().map_err(|_| CodecError::Ffmpeg {
        op: "container name",
        detail: "contains a NUL byte".into(),
    })?;
    let mut reader = Reader { input, error: None };
    let mut error = ErrBuf::new();
    // SAFETY: the reader and container name outlive the native handle; callbacks do not escape
    // lk_dec_open/lk_dec_run. The opaque pointer is borrowed exclusively on this thread.
    let decoder = unsafe {
        lk_dec_open(
            name.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            out_channels as c_int,
            limits.min_sample_rate as c_int,
            limits.max_sample_rate as c_int,
            limits.max_channels as c_int,
            limits.max_streams as c_int,
            limits.max_samples_per_frame as c_int,
            read::<R>,
            (&mut reader as *mut Reader<'_, R>).cast(),
            error.as_mut_ptr(),
            ERRLEN as c_int,
        )
    };
    if decoder.is_null() {
        return Err(reader.error.unwrap_or_else(|| error.take("open")));
    }
    let decoder = Handle(decoder);
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
    if let Some(error) = reader.error.or(output.error) {
        return Err(error);
    }
    if rc < 0 {
        return Err(error.take("decode"));
    }
    let tail = output.resampler.flush_samples()?;
    if tail.len() > limits.max_samples_per_frame {
        return Err(CodecError::DecodeLimit("resampled frame samples"));
    }
    if !tail.is_empty() {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (output.callback)(format, &tail)))
            .map_err(|_| CodecError::Ffmpeg {
                op: "output",
                detail: "output callback panicked".into(),
            })??;
    }
    Ok(format)
}
