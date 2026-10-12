//! `_decode_loop` (`utils/codecs/decoder.py:404-475`) -- demux, decode, resample to s16.

use super::*;

#[derive(Default)]
struct Shared {
    pcm: Mutex<Vec<i16>>,
    err: Mutex<Option<CodecError>>,
    src: Mutex<Option<AudioFormat>>,
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

extern "C" fn pcm_trampoline(opaque: *mut c_void, pcm: *const i16, n: c_int) {
    if n <= 0 {
        return;
    }
    // SAFETY: same lifetime argument; `pcm`/`n` are the shim's converted output, valid for this
    // call only, which is why it is copied here.
    let shared = unsafe { &*(opaque as *const Shared) };
    let samples = unsafe { std::slice::from_raw_parts(pcm, n as usize) };
    shared.pcm.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(samples);
}

pub struct FfmpegDecoder {
    input: Arc<StreamBuffer>,
    shared: Arc<Shared>,
    handle: Option<std::thread::JoinHandle<()>>,
    format: Option<CString>,
    out_rate: u32,
    out_channels: u16,
    saw_output: bool,
}

impl FfmpegDecoder {
    /// `format_short_name` is the libav container name (`"mp3"`, `"webm"`), or `None` to probe.
    pub fn new(format_short_name: Option<&str>, out_rate: u32, out_channels: u16) -> Result<Self> {
        Ok(FfmpegDecoder {
            input: Arc::new(StreamBuffer::new()),
            shared: Arc::default(),
            handle: None,
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
        self.saw_output.then(|| AudioFormat::new(self.out_rate, self.out_channels))
    }

    /// For logging and metrics only; the output is already resampled.
    pub fn source_format(&self) -> Option<AudioFormat> {
        *self.shared.src.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn start(&mut self) {
        if self.handle.is_some() {
            return;
        }
        let input = self.input.clone();
        let shared = self.shared.clone();
        let fmt = self.format.clone();
        let out_rate = self.out_rate as c_int;
        let out_channels = self.out_channels as c_int;

        self.handle = Some(
            std::thread::Builder::new()
                .name("lk-agents-media-decode".into())
                .spawn(move || {
                    let mut err = ErrBuf::new();
                    let name = fmt.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
                    // SAFETY: `input` outlives the C decoder -- it is dropped at the end of this
                    // closure, after `lk_dec_free`.
                    let d = unsafe {
                        lk_dec_open(
                            name,
                            out_rate,
                            out_channels,
                            read_trampoline,
                            Arc::as_ptr(&input) as *mut c_void,
                            err.as_mut_ptr(),
                            ERRLEN as c_int,
                        )
                    };
                    if d.is_null() {
                        *shared.err.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(err.take("open"));
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
                    }
                    unsafe { lk_dec_free(d) };
                })
                .expect("spawning the decode thread"),
        );
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

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        self.input.write(bytes);
        self.start();
        self.drain()
    }

    /// Joins the decode thread, so everything the container held is decoded on return.
    pub fn flush(&mut self) -> Result<Vec<i16>> {
        self.input.end_input();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.drain()
    }
}

impl Drop for FfmpegDecoder {
    fn drop(&mut self) {
        // `aclose()`: close the input so the blocking read returns, then join, or a worker leaks a
        // thread per cancelled utterance.
        self.input.close();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
