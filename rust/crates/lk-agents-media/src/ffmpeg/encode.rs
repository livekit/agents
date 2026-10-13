//! ogg/opus encode for `RecorderIO` (`voice/recorder_io/`).

use super::*;

/// libopus through ffmpeg's ogg muxer. Interleaved `i16` in, ogg pages out.
#[allow(clippy::box_collection)]
pub struct OggOpusEncoder {
    enc: *mut LkEncoder,
    /// Boxed so its address survives a move: the C write callback holds a pointer to it.
    out: Box<Vec<u8>>,
    /// 960 at 48 kHz, i.e. 20 ms.
    frame_size: usize,
    channels: usize,
    /// libopus takes exact frame sizes only.
    pending: Vec<i16>,
    finished: bool,
}

// SAFETY: the encoder owns everything it points at and every method takes `&mut self`.
unsafe impl Send for OggOpusEncoder {}

extern "C" fn write_trampoline(opaque: *mut c_void, bytes: *const u8, len: c_int) {
    if len <= 0 {
        return;
    }
    // SAFETY: `opaque` is the boxed `Vec<u8>` above, alive for as long as the encoder is.
    let out = unsafe { &mut *(opaque as *mut Vec<u8>) };
    out.extend_from_slice(unsafe { std::slice::from_raw_parts(bytes, len as usize) });
}

impl OggOpusEncoder {
    /// `sample_rate` must be one libopus accepts: 8000, 12000, 16000, 24000 or 48000.
    pub fn new(sample_rate: u32, channels: u16, bitrate: u32) -> Result<Self> {
        let mut out = Box::new(Vec::<u8>::new());
        let mut err = ErrBuf::new();
        // SAFETY: `out`'s heap allocation outlives `enc` -- `Drop` frees `enc` first.
        let enc = unsafe {
            lk_enc_open(
                sample_rate as c_int,
                channels as c_int,
                bitrate as c_int,
                write_trampoline,
                Box::as_mut(&mut out) as *mut Vec<u8> as *mut c_void,
                err.as_mut_ptr(),
                ERRLEN as c_int,
            )
        };
        if enc.is_null() {
            return Err(err.take("lk_enc_open"));
        }
        let frame_size = unsafe { lk_enc_frame_size(enc) }.max(0) as usize;
        if frame_size == 0 {
            unsafe { lk_enc_free(enc) };
            return Err(CodecError::Ffmpeg {
                op: "lk_enc_open",
                detail: "libopus reported a zero frame size".into(),
            });
        }
        Ok(OggOpusEncoder {
            enc,
            out,
            frame_size,
            channels: channels.max(1) as usize,
            pending: Vec::new(),
            finished: false,
        })
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Routinely empty: a page is several packets.
    pub fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        let block = self.frame_size * self.channels;
        self.pending.extend_from_slice(pcm);
        let mut consumed = 0;
        while consumed + block <= self.pending.len() {
            let mut err = ErrBuf::new();
            // SAFETY: exactly `block` samples, which is what `lk_enc_push` reads.
            let rc = unsafe {
                lk_enc_push(
                    self.enc,
                    self.pending[consumed..].as_ptr(),
                    err.as_mut_ptr(),
                    ERRLEN as c_int,
                )
            };
            if rc < 0 {
                self.pending.drain(..consumed);
                return Err(err.take("lk_enc_push"));
            }
            consumed += block;
        }
        self.pending.drain(..consumed);
        Ok(std::mem::take(&mut *self.out))
    }

    /// By value: an ogg stream with its trailer written cannot be appended to.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        let block = self.frame_size * self.channels;
        if !self.pending.is_empty() {
            // No AV_CODEC_CAP_SMALL_LAST_FRAME: pad the last packet with silence rather than lose
            // up to 20 ms, as ffmpeg's own audio fifo does.
            self.pending.resize(block, 0);
            let mut err = ErrBuf::new();
            let rc = unsafe {
                lk_enc_push(self.enc, self.pending.as_ptr(), err.as_mut_ptr(), ERRLEN as c_int)
            };
            if rc < 0 {
                return Err(err.take("lk_enc_push(tail)"));
            }
            self.pending.clear();
        }
        let mut err = ErrBuf::new();
        let rc = unsafe { lk_enc_finish(self.enc, err.as_mut_ptr(), ERRLEN as c_int) };
        if rc < 0 {
            return Err(err.take("lk_enc_finish"));
        }
        self.finished = true;
        Ok(std::mem::take(&mut *self.out))
    }
}

impl Drop for OggOpusEncoder {
    fn drop(&mut self) {
        if !self.enc.is_null() {
            // No trailer on drop: an unfinished encoder is an abandoned recording.
            unsafe { lk_enc_free(self.enc) };
            self.enc = std::ptr::null_mut();
        }
    }
}
