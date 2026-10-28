//! `_decode_loop` (`utils/codecs/decoder.py:404-475`) -- demux, decode, resample to s16.

use super::*;
use crate::workers::{MediaWorker, MediaWorkerClass, MediaWorkers};

#[allow(clippy::disallowed_types, reason = "S0: migrate codec worker state to runtime")]
#[derive(Default)]
struct Shared {
    pcm: Mutex<Vec<i16>>,
    err: Mutex<Option<CodecError>>,
    src: Mutex<Option<AudioFormat>>,
    /// Called from the decode thread when PCM or an error is ready to drain.
    on_output: Mutex<Option<crate::decoder::OnOutput>>,
}

impl Shared {
    fn ready(&self) {
        let on_output = self.on_output.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(on_output) = on_output {
            on_output();
        }
    }
}

/// Blocks in [`StreamBuffer::read`], as a demuxer requires; the C side turns a 0 into `AVERROR_EOF`.
extern "C" fn read_trampoline(opaque: *mut c_void, buf: *mut u8, size: c_int) -> c_int {
    if size <= 0 {
        return 0;
    }
    // SAFETY: `opaque` is the `Arc<StreamBuffer>`'s target, kept alive by the clone the decode
    // thread holds for as long as the C decoder exists. `buf`/`size` are ffmpeg's own buffer.
    let sb = unsafe { &*(opaque as *const StreamBuffer) };
    let slice = unsafe { std::slice::from_raw_parts_mut(buf, size as usize) };
    sb.read(slice) as c_int
}

extern "C" fn pcm_trampoline(opaque: *mut c_void, pcm: *const i16, n: c_int) -> c_int {
    if n <= 0 {
        return 0;
    }
    // SAFETY: same lifetime argument; `pcm`/`n` are the shim's converted output, valid for this
    // call only, which is why it is copied here.
    let shared = unsafe { &*(opaque as *const Shared) };
    let samples = unsafe { std::slice::from_raw_parts(pcm, n as usize) };
    shared.pcm.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(samples);
    shared.ready();
    0
}

pub struct FfmpegDecoder {
    input: Arc<StreamBuffer>,
    shared: Arc<Shared>,
    handle: Option<MediaWorker<()>>,
    workers: MediaWorkers,
    format: Option<CString>,
    out_rate: u32,
    out_channels: u16,
    saw_output: bool,
}

impl FfmpegDecoder {
    /// `format_short_name` is the libav container name (`"mp3"`, `"webm"`), or `None` to probe;
    /// an `out_rate` of 0 keeps the source's rate.
    pub fn new(
        workers: &MediaWorkers,
        format_short_name: Option<&str>,
        out_rate: u32,
        out_channels: u16,
    ) -> Result<Self> {
        Ok(FfmpegDecoder {
            input: Arc::new(StreamBuffer::new()),
            shared: Arc::default(),
            handle: None,
            workers: workers.clone(),
            // Checked at the boundary even though `mime_to_av_format` only returns literals.
            format: match format_short_name {
                Some(s) => Some(CString::new(s).map_err(|_| CodecError::Ffmpeg {
                    op: "container name",
                    detail: "contains a NUL byte".into(),
                })?),
                None => None,
            },
            out_rate,
            out_channels,
            saw_output: false,
        })
    }

    /// ffmpeg's resampler converts to exactly this, unlike the WAV path.
    pub fn output_format(&self) -> Option<AudioFormat> {
        let rate = match self.out_rate {
            0 => self.source_format().map_or(0, |f| f.sample_rate),
            r => r,
        };
        self.saw_output.then(|| AudioFormat::new(rate, self.out_channels))
    }

    /// For logging and metrics only; the output is already resampled.
    pub fn source_format(&self) -> Option<AudioFormat> {
        *self.shared.src.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn start(&mut self) -> Result<()> {
        if self.handle.is_some() {
            return Ok(());
        }
        let input = self.input.clone();
        let shared = self.shared.clone();
        let fmt = self.format.clone();
        let out_rate = self.out_rate as c_int;
        let out_channels = self.out_channels as c_int;

        self.handle = Some(
            self.workers
                .try_spawn(MediaWorkerClass::Decoder, move || {
                    let mut err = ErrBuf::new();
                    let name = fmt.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
                    // SAFETY: `input` outlives the C decoder -- it is dropped at the end of this
                    // closure, after `lk_dec_free`.
                    let d = unsafe {
                        lk_dec_open(
                            name,
                            out_rate,
                            out_channels,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            read_trampoline,
                            Arc::as_ptr(&input) as *mut c_void,
                            err.as_mut_ptr(),
                            ERRLEN as c_int,
                        )
                    };
                    if d.is_null() {
                        *shared.err.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(err.take("open"));
                        shared.ready();
                        return;
                    }
                    let (rate, ch) = unsafe { (lk_dec_src_rate(d), lk_dec_src_channels(d)) };
                    *shared.src.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(AudioFormat::new(rate.max(0) as u32, ch.max(0) as u16));

                    let rc = unsafe {
                        lk_dec_run(
                            d,
                            pcm_trampoline,
                            Arc::as_ptr(&shared) as *mut c_void,
                            err.as_mut_ptr(),
                            ERRLEN as c_int,
                        )
                    };
                    if rc < 0 {
                        let mut slot = shared.err.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(err.take("decode"));
                        }
                        drop(slot);
                        shared.ready();
                    }
                    unsafe { lk_dec_free(d) };
                })
                .map_err(|e| CodecError::Worker(e.to_string()))?,
        );
        Ok(())
    }

    fn drain(&mut self) -> Result<Vec<i16>> {
        if let Some(e) = self.shared.err.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return Err(e);
        }
        let mut g = self.shared.pcm.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return Ok(Vec::new());
        }
        self.saw_output = true;
        Ok(std::mem::take(&mut *g))
    }

    /// Called from the decode thread whenever output is ready, so a caller drains it then
    /// instead of at its next push.
    pub fn set_on_output(&self, on_output: Option<crate::decoder::OnOutput>) {
        *self.shared.on_output.lock().unwrap_or_else(|e| e.into_inner()) = on_output;
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        self.start()?;
        self.input.write(bytes);
        self.drain()
    }

    /// Stop native input immediately and return ownership of its settlement receipt.
    pub fn close(&mut self) -> Option<MediaWorker<()>> {
        self.set_on_output(None);
        self.input.close();
        self.handle.take()
    }

    /// Joins the decode thread, so everything the container held is decoded on return.
    pub fn flush(&mut self) -> Result<Vec<i16>> {
        self.input.end_input();
        if let Some(h) = self.handle.take() {
            h.wait().map_err(|e| CodecError::Worker(e.to_string()))?;
        }
        self.drain()
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        // `aclose()`: close the input so the blocking read returns and the thread ends on its own.
        // Not joined: a drop can run under the host's lock that a pending `on_output` waits on.
        self.close();
    }
}

/// Decode one container through caller-owned I/O. This function creates no thread or queue.
/// The runtime calls it on an admitted blocking worker because FFmpeg's AVIO calls `Read`.
/// Native handles and callback pointers remain local to this invocation.
pub fn decode_stream<R, F>(
    format_short_name: Option<&str>,
    out_rate: u32,
    out_channels: u16,
    limits: crate::decoder::DecodeLimits,
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
        format: AudioFormat,
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
            (output.callback)(output.format, samples)
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
            out_rate as c_int,
            out_channels as c_int,
            limits.min_sample_rate as c_int,
            limits.max_sample_rate as c_int,
            limits.max_channels as c_int,
            limits.max_streams as c_int,
            limits.max_expansion_ratio as c_int,
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
    let mut output = Output { callback: &mut output, format, error: None };
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
    Ok(format)
}
