//! The ffmpeg half, over `csrc/lk_agents_media.c`'s `extern "C"` shim (hand-written ffmpeg struct
//! layouts read wrong memory instead of failing to link). The demuxer needs a blocking read, so
//! decode runs on its own thread against [`StreamBuffer`]: the split across `push` calls varies,
//! but the total over `push`* + `flush` is deterministic.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::{Arc, Mutex};

use crate::AudioFormat;
use crate::error::{CodecError, Result};
use crate::stream_buffer::StreamBuffer;

mod decode;
mod encode;

pub use decode::FfmpegDecoder;
pub use encode::OggOpusEncoder;

#[repr(C)]
struct LkDecoder {
    _private: [u8; 0],
}
#[repr(C)]
struct LkEncoder {
    _private: [u8; 0],
}

type ReadFn = extern "C" fn(*mut c_void, *mut u8, c_int) -> c_int;
type PcmFn = extern "C" fn(*mut c_void, *const i16, c_int);
type WriteFn = extern "C" fn(*mut c_void, *const u8, c_int);

unsafe extern "C" {
    fn lk_dec_open(
        format_short_name: *const c_char,
        out_rate: c_int,
        out_channels: c_int,
        read_cb: ReadFn,
        read_opaque: *mut c_void,
        err: *mut c_char,
        errlen: c_int,
    ) -> *mut LkDecoder;
    fn lk_dec_src_rate(d: *const LkDecoder) -> c_int;
    fn lk_dec_src_channels(d: *const LkDecoder) -> c_int;
    fn lk_dec_run(
        d: *mut LkDecoder,
        on_pcm: PcmFn,
        opaque: *mut c_void,
        err: *mut c_char,
        errlen: c_int,
    ) -> c_int;
    fn lk_dec_free(d: *mut LkDecoder);

    fn lk_enc_open(
        rate: c_int,
        channels: c_int,
        bitrate: c_int,
        write_cb: WriteFn,
        write_opaque: *mut c_void,
        err: *mut c_char,
        errlen: c_int,
    ) -> *mut LkEncoder;
    fn lk_enc_frame_size(e: *const LkEncoder) -> c_int;
    fn lk_enc_push(e: *mut LkEncoder, pcm: *const i16, err: *mut c_char, errlen: c_int) -> c_int;
    fn lk_enc_finish(e: *mut LkEncoder, err: *mut c_char, errlen: c_int) -> c_int;
    fn lk_enc_free(e: *mut LkEncoder);

    fn lk_avcodec_version() -> c_uint;
    fn lk_avformat_version() -> c_uint;
}

const ERRLEN: usize = 256;

struct ErrBuf([c_char; ERRLEN]);

impl ErrBuf {
    fn new() -> Self {
        ErrBuf([0; ERRLEN])
    }
    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr()
    }
    fn take(&self, op: &'static str) -> CodecError {
        let bytes: Vec<u8> = self.0.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
        let detail = String::from_utf8_lossy(&bytes).into_owned();
        CodecError::Ffmpeg {
            op,
            detail: if detail.is_empty() { "no detail".into() } else { detail },
        }
    }
}

/// `(major, minor, micro)` of the linked libavcodec.
pub fn avcodec_version() -> (u32, u32, u32) {
    let v = unsafe { lk_avcodec_version() };
    (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
}

/// `(major, minor, micro)` of the linked libavformat.
pub fn avformat_version() -> (u32, u32, u32) {
    let v = unsafe { lk_avformat_version() };
    (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
}

#[cfg(test)]
mod fixtures_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: linking against a system ffmpeg instead of the pinned one. §10 vendors precisely
    // so behaviour is identical everywhere; an LK_CODECS_FFMPEG_PREFIX pointing at, say, brew's
    // ffmpeg 8 would link, run, and decode *slightly* differently -- which is the failure mode
    // vendoring exists to prevent, and the only one nobody would notice.
    #[test]
    fn the_linked_ffmpeg_is_the_major_version_the_shim_was_written_against() {
        // 7.1 ships libavcodec 61 and libavformat 61.
        assert_eq!(avcodec_version().0, 61, "libavcodec major (ffmpeg 7.1 -> 61)");
        assert_eq!(avformat_version().0, 61, "libavformat major (ffmpeg 7.1 -> 61)");
    }
}
