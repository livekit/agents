//! The WAV fast path, `_WavInlineDecoder` (`utils/codecs/decoder.py:145-335`). A push that begins
//! with `RIFF` while streaming starts a new segment: providers stream several files down one connection.

use crate::AudioFormat;
use crate::error::{CodecError, Result};

/// A push yields more than one when a provider concatenates WAV files of different formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavSegment {
    pub format: AudioFormat,
    pub samples: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    RiffHeader,
    ChunkHeader,
    FmtData,
    SkipChunkData,
    Streaming,
}

const RIFF_MAGIC: &[u8; 4] = b"RIFF";

#[derive(Debug, Clone)]
pub struct WavDecoder {
    state: State,
    hdr: Vec<u8>,
    need: usize,
    skip_remaining: u64,
    chunk_size: u32,
    format: Option<AudioFormat>,
    carry: Option<u8>,
    /// So `push` does not report empty runs.
    started_segment: bool,
}

impl Default for WavDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WavDecoder {
    pub fn new() -> Self {
        WavDecoder {
            state: State::RiffHeader,
            hdr: Vec::with_capacity(16),
            need: 12,
            skip_remaining: 0,
            chunk_size: 0,
            format: None,
            carry: None,
            started_segment: false,
        }
    }

    pub fn format(&self) -> Option<AudioFormat> {
        self.format
    }

    /// A push that only advanced the header returns an empty vector.
    pub fn push(&mut self, data: &[u8]) -> Result<Vec<WavSegment>> {
        let mut out: Vec<WavSegment> = Vec::new();

        // The restart check is at the start of a push only, as in the reference: scanning for
        // `RIFF` mid-stream would eventually corrupt 16-bit PCM containing those bytes.
        if self.state == State::Streaming && data.len() >= 4 && &data[..4] == RIFF_MAGIC {
            self.reset_state();
        }

        let mut pos = 0usize;
        while pos < data.len() {
            match self.state {
                State::RiffHeader => pos = self.consume_riff(data, pos)?,
                State::ChunkHeader => pos = self.consume_chunk_header(data, pos)?,
                State::FmtData => pos = self.consume_fmt_data(data, pos)?,
                State::SkipChunkData => pos = self.consume_skip(data, pos),
                State::Streaming => {
                    self.emit_pcm(&data[pos..], &mut out);
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Release a half sample left by a push boundary; `None` rather than an empty segment.
    pub fn flush(&mut self) -> Option<WavSegment> {
        // A trailing odd byte is a truncated sample; nothing can invent the missing byte.
        self.carry = None;
        None
    }

    fn reset_state(&mut self) {
        self.state = State::RiffHeader;
        self.hdr.clear();
        self.need = 12;
        self.skip_remaining = 0;
        self.chunk_size = 0;
        self.format = None;
        self.carry = None;
        self.started_segment = false;
    }

    fn fill(&mut self, data: &[u8], pos: usize) -> (usize, bool) {
        let take = (self.need - self.hdr.len()).min(data.len() - pos);
        self.hdr.extend_from_slice(&data[pos..pos + take]);
        (pos + take, self.hdr.len() >= self.need)
    }

    fn consume_riff(&mut self, data: &[u8], pos: usize) -> Result<usize> {
        let (pos, done) = self.fill(data, pos);
        if !done {
            return Ok(pos);
        }
        if &self.hdr[..4] != RIFF_MAGIC || &self.hdr[8..12] != b"WAVE" {
            return Err(CodecError::MissingRiffWave(self.hdr.clone()));
        }
        self.hdr.clear();
        self.need = 8;
        self.state = State::ChunkHeader;
        Ok(pos)
    }

    fn consume_chunk_header(&mut self, data: &[u8], pos: usize) -> Result<usize> {
        let (pos, done) = self.fill(data, pos);
        if !done {
            return Ok(pos);
        }
        let id: [u8; 4] = self.hdr[..4].try_into().expect("need == 8");
        let size = u32::from_le_bytes(self.hdr[4..8].try_into().expect("need == 8"));
        self.hdr.clear();
        self.chunk_size = size;

        match &id {
            b"fmt " => {
                if size < 16 {
                    return Err(CodecError::ShortWavFmt(size));
                }
                self.need = size as usize;
                self.state = State::FmtData;
            }
            b"data" => {
                // The declared size is ignored, as in the reference, so a growing file (size 0 or
                // 0xFFFFFFFF) streams.
                if self.format.is_none() {
                    return Err(CodecError::WavDataBeforeFmt);
                }
                self.started_segment = true;
                self.state = State::Streaming;
            }
            _ => {
                // Consume RIFF's even-length pad byte, which the reference's fast path forgets
                // (ffmpeg consumes it); odd `LIST`/`INFO` chunks are common.
                self.skip_remaining = u64::from(size) + u64::from(size % 2);
                self.state = State::SkipChunkData;
            }
        }
        Ok(pos)
    }

    fn consume_fmt_data(&mut self, data: &[u8], pos: usize) -> Result<usize> {
        let (pos, done) = self.fill(data, pos);
        if !done {
            return Ok(pos);
        }
        let fmt = &self.hdr[..self.chunk_size as usize];
        let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
        let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
        let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
        let bits = u16::from_le_bytes([fmt[14], fmt[15]]);

        // Bits before the format tag, as in the reference: a 24-bit file reports the bit depth.
        if bits != 16 {
            return Err(CodecError::UnsupportedWavBits(bits));
        }
        if audio_format != 1 {
            return Err(CodecError::UnsupportedWavFormat(audio_format));
        }

        self.format = Some(AudioFormat { sample_rate: rate, num_channels: channels });
        self.hdr.clear();
        self.need = 8;
        self.state = State::ChunkHeader;
        Ok(pos)
    }

    fn consume_skip(&mut self, data: &[u8], pos: usize) -> usize {
        let take = (self.skip_remaining).min((data.len() - pos) as u64) as usize;
        self.skip_remaining -= take as u64;
        let pos = pos + take;
        if self.skip_remaining == 0 {
            self.hdr.clear();
            self.need = 8;
            self.state = State::ChunkHeader;
        }
        pos
    }

    fn emit_pcm(&mut self, bytes: &[u8], out: &mut Vec<WavSegment>) {
        let format = self.format.expect("STREAMING is only reachable once fmt is parsed");
        let mut samples = Vec::with_capacity(bytes.len().div_ceil(2));
        let mut i = 0usize;
        if let Some(lo) = self.carry.take() {
            if bytes.is_empty() {
                self.carry = Some(lo);
            } else {
                samples.push(i16::from_le_bytes([lo, bytes[0]]));
                i = 1;
            }
        }
        while i + 2 <= bytes.len() {
            samples.push(i16::from_le_bytes([bytes[i], bytes[i + 1]]));
            i += 2;
        }
        if i < bytes.len() {
            self.carry = Some(bytes[i]);
        }
        if !samples.is_empty() {
            out.push(WavSegment { format, samples });
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds a WAV file byte-for-byte. Fixtures are generated, not committed: a checked-in
    /// binary is a fixture nobody can read a diff of, and every property these tests care about
    /// (rate, depth, chunk order, chunk parity) is a parameter here.
    fn wav(
        rate: u32,
        channels: u16,
        bits: u16,
        fmt_tag: u16,
        extra_chunks: &[(&[u8; 4], Vec<u8>)],
        pcm: &[u8],
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&fmt_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        let block_align = channels * bits / 8;
        fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes()); // byte rate
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());

        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");
        for (id, data) in extra_chunks {
            body.extend_from_slice(*id);
            body.extend_from_slice(&(data.len() as u32).to_le_bytes());
            body.extend_from_slice(data);
            if data.len() % 2 == 1 {
                body.push(0); // RIFF word-alignment pad, NOT counted in the declared size
            }
        }
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        body.extend_from_slice(pcm);

        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    /// A canonical 16-bit PCM file; the decoder and ffmpeg tests build their inputs with it too.
    pub(crate) fn wav_file(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        wav(rate, channels, 16, 1, &[], &pcm_bytes(samples))
    }

    fn simple(samples: &[i16]) -> Vec<u8> {
        wav_file(24_000, 1, samples)
    }

    fn decode_all(chunks: &[&[u8]]) -> Result<Vec<WavSegment>> {
        let mut d = WavDecoder::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(d.push(c)?);
        }
        d.flush();
        Ok(out)
    }

    fn merged(segs: &[WavSegment]) -> Vec<i16> {
        segs.iter().flat_map(|s| s.samples.iter().copied()).collect()
    }

    // Catches: the generator and the decoder agreeing with each other and disagreeing with every
    // real WAV. Every other test trusts `wav()`, so the canonical 44-byte header is written out by
    // hand once, decoded, and the generator is required to reproduce it byte for byte.
    #[test]
    fn a_hand_written_canonical_header_decodes_and_the_generator_reproduces_it() {
        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&40u32.to_le_bytes()); // file size - 8
        file.extend_from_slice(b"WAVE");
        file.extend_from_slice(b"fmt ");
        file.extend_from_slice(&16u32.to_le_bytes());
        file.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
        file.extend_from_slice(&1u16.to_le_bytes()); // mono
        file.extend_from_slice(&24_000u32.to_le_bytes());
        file.extend_from_slice(&48_000u32.to_le_bytes()); // byte rate
        file.extend_from_slice(&2u16.to_le_bytes()); // block align
        file.extend_from_slice(&16u16.to_le_bytes()); // bits
        file.extend_from_slice(b"data");
        file.extend_from_slice(&4u32.to_le_bytes());
        file.extend_from_slice(&[0x01, 0x00, 0xFF, 0xFF]); // 1, -1
        assert_eq!(file.len(), 48);

        let segs = decode_all(&[&file]).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].format, AudioFormat::new(24_000, 1));
        assert_eq!(segs[0].samples, vec![1, -1]);
        assert_eq!(simple(&[1, -1]), file, "the generator drifted from the canonical layout");
    }

    // Catches: the header accumulator being reset by a push that does not complete it -- the
    // classic form of this bug reads `RIFF` from the second chunk and rejects the file. Also
    // catches losing the odd carry byte, since a 1-byte cadence splits every single sample.
    #[test]
    fn feeding_the_same_file_one_byte_at_a_time_gives_the_same_samples() {
        let want: Vec<i16> = (0..64).map(|i| (i * 517 - 16000) as i16).collect();
        let file = simple(&want);
        let one_shot = merged(&decode_all(&[&file]).unwrap());

        let mut d = WavDecoder::new();
        let mut got = Vec::new();
        for b in &file {
            for seg in d.push(std::slice::from_ref(b)).unwrap() {
                got.extend(seg.samples);
            }
        }
        d.flush();
        assert_eq!(got, want);
        assert_eq!(got, one_shot);
    }

    // Catches: a split placed inside each individual header field. A `fill` that mis-tracked
    // `need` would accept a short `fmt ` and read the rate out of the wrong four bytes -- audio
    // at the wrong speed, no error.
    #[test]
    fn every_split_point_in_the_file_yields_the_same_samples() {
        let want: Vec<i16> = vec![7, -7, 1000, -1000, 32000, -32000];
        let file = simple(&want);
        for cut in 0..=file.len() {
            let segs = decode_all(&[&file[..cut], &file[cut..]]).unwrap();
            assert_eq!(merged(&segs), want, "split at {cut}");
        }
    }

    // Catches: emitting samples from a header that never arrived. A stream that dies after the
    // first few bytes must produce nothing and must not panic -- an index-out-of-bounds here
    // takes down the whole worker, which is strictly worse than a silent utterance.
    #[test]
    fn a_truncated_header_yields_no_audio_and_no_panic() {
        for n in 0..44usize {
            let file = simple(&[1, 2, 3, 4]);
            let mut d = WavDecoder::new();
            let segs = d.push(&file[..n]).unwrap();
            assert!(merged(&segs).is_empty(), "truncated at {n} produced audio");
        }
    }

    // Catches: guessing instead of refusing. Not RIFF/WAVE (an mp3's ID3 tag, or an AVI's `RIFF`
    // header) framed as PCM is noise nobody traces to the mime type; a non-16-bit file read as
    // 16-bit is static at the wrong speed -- the reference checks the depth BEFORE the format tag,
    // so 32-bit float reports its depth; a `data` chunk before `fmt ` would play at a guessed rate.
    #[test]
    fn a_file_we_cannot_decode_is_refused_by_name() {
        let mut avi = b"RIFF".to_vec();
        avi.extend_from_slice(&100u32.to_le_bytes());
        avi.extend_from_slice(b"AVI ");
        let mut data_first = b"RIFF".to_vec();
        data_first.extend_from_slice(&40u32.to_le_bytes());
        data_first.extend_from_slice(b"WAVEdata");
        data_first.extend_from_slice(&8u32.to_le_bytes());
        data_first.extend_from_slice(&[0u8; 8]);
        let cases: [(Vec<u8>, fn(&CodecError) -> bool); 6] = [
            (b"ID3\x04\x00\x00\x00\x00\x00\x00mp3 data here".to_vec(), |e| matches!(e, CodecError::MissingRiffWave(_))),
            (avi, |e| matches!(e, CodecError::MissingRiffWave(_))),
            (wav(24_000, 1, 24, 1, &[], &[0; 12]), |e| *e == CodecError::UnsupportedWavBits(24)),
            (wav(24_000, 1, 32, 3, &[], &[0; 16]), |e| *e == CodecError::UnsupportedWavBits(32)),
            (wav(24_000, 1, 16, 0xFFFE, &[], &[0; 8]), |e| *e == CodecError::UnsupportedWavFormat(0xFFFE)),
            (data_first, |e| *e == CodecError::WavDataBeforeFmt),
        ];
        for (i, (bytes, want)) in cases.into_iter().enumerate() {
            let err = WavDecoder::new().push(&bytes).unwrap_err();
            assert!(want(&err), "case {i}: {err:?}");
        }
    }

    // Catches: treating an unknown chunk's payload as audio. `LIST`/`INFO` metadata ahead of the
    // data chunk is what most encoders write, and playing it is a burst of noise at the start of
    // every file.
    #[test]
    fn chunks_between_the_header_and_the_data_are_skipped() {
        let want: Vec<i16> = vec![10, 20, 30, 40];
        let f = wav(
            16_000,
            1,
            16,
            1,
            &[(b"LIST", b"INFOISFTLavf58.29.100\0".to_vec()), (b"JUNK", vec![0xAA; 24])],
            &pcm_bytes(&want),
        );
        let segs = decode_all(&[&f]).unwrap();
        assert_eq!(segs[0].format, AudioFormat::new(16_000, 1));
        assert_eq!(segs[0].samples, want);
    }

    // Catches: the reference's own bug, and it is reachable with ordinary files.
    //
    // RIFF pads every chunk to an even length and does NOT count the pad byte in the declared
    // size. `_consume_skip` skips exactly `chunk_size`, so after an odd-sized chunk it is one
    // byte behind: the next 8-byte "chunk header" is the pad plus seven real bytes, so the id
    // reads as b"\0fmt" and the size as garbage, and the garbage skip swallows the rest of the
    // file. Verified against the reference itself, not inferred -- a 19-byte LIST payload makes
    // `_WavInlineDecoder` emit ZERO samples for a perfectly valid file, while the same file with
    // a 20-byte payload decodes fine. `LIST`/`INFO` strings are odd-length routinely, and
    // ffmpeg's own wav demuxer consumes the pad, so the reference's slow path and its fast path
    // disagree on the same bytes -- which is what makes this a bug rather than a choice.
    #[test]
    fn an_odd_sized_chunk_pad_byte_does_not_desynchronise_the_parser() {
        let want: Vec<i16> = vec![-5, 5, -6, 6, -7, 7];
        for payload in [
            b"INFOIART_odd_lengt".to_vec(), // 18, even: works in the reference too
            b"INFOIART_odd_lengt_".to_vec(), // 19, odd: the reference emits nothing at all
        ] {
            let odd = payload.len() % 2 == 1;
            let f = wav(48_000, 2, 16, 1, &[(b"LIST", payload)], &pcm_bytes(&want));
            assert_eq!(f.len() % 2, 0, "the file itself stays word-aligned");
            let segs = decode_all(&[&f]).unwrap();
            assert_eq!(segs.len(), 1, "the parser lost the data chunk (odd payload: {odd})");
            assert_eq!(segs[0].format, AudioFormat::new(48_000, 2));
            assert_eq!(segs[0].samples, want, "odd payload: {odd}");
        }
    }

    // Catches: trusting the declared `data` size. A live writer does not know the length yet and
    // writes 0 (or 0xFFFFFFFF); a decoder that stopped at the declared size would emit silence
    // for every streaming WAV, which is the single most common way a TTS sends WAV.
    #[test]
    fn a_data_chunk_whose_declared_size_is_a_placeholder_still_streams() {
        let want: Vec<i16> = vec![100, -100, 200, -200];
        for declared in [0u32, 0xFFFF_FFFF] {
            let mut f = simple(&want);
            let n = f.len();
            f[n - want.len() * 2 - 4..n - want.len() * 2].copy_from_slice(&declared.to_le_bytes());
            let segs = decode_all(&[&f]).unwrap();
            assert_eq!(merged(&segs), want, "declared size {declared:#x}");
        }
    }

    // Catches: playing the second file's 44-byte header as audio when a provider concatenates
    // whole WAV files down one connection, and catches keeping the first file's sample rate for
    // the second. Both are silent -- a click then wrong-speed audio.
    #[test]
    fn a_new_riff_header_at_a_push_boundary_starts_a_new_segment() {
        let a: Vec<i16> = vec![1, 2, 3, 4];
        let b: Vec<i16> = vec![-1, -2, -3];
        let mut d = WavDecoder::new();
        let mut segs = d.push(&simple(&a)).unwrap();
        segs.extend(d.push(&wav(16_000, 2, 16, 1, &[], &pcm_bytes(&b))).unwrap());

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].format, AudioFormat::new(24_000, 1));
        assert_eq!(segs[0].samples, a);
        assert_eq!(segs[1].format, AudioFormat::new(16_000, 2));
        assert_eq!(segs[1].samples, b);
    }

    // Catches: scanning for `RIFF` anywhere in the buffer instead of only at a push boundary.
    // 16-bit PCM contains the bytes `RIFF` (0x52 0x49, 0x46 0x46 -> samples 18770 and 17990)
    // eventually, and a mid-buffer scan would cut the stream there and then reject the "header".
    // The reference checks the start of a push only; this pins that.
    #[test]
    fn pcm_that_happens_to_contain_the_riff_bytes_is_not_mistaken_for_a_header() {
        let want: Vec<i16> = vec![0, 18770, 17990, 0, 5];
        let file = simple(&want);
        let segs = decode_all(&[&file]).unwrap();
        assert_eq!(merged(&segs), want);
    }
}
