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

/// A canonical 16-bit PCM WAV file around interleaved little-endian `pcm`.
pub fn encode(sample_rate: u32, num_channels: u16, pcm: &[u8]) -> Vec<u8> {
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let block_align = num_channels * 2;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(RIFF_MAGIC);
    out.extend_from_slice(&36u32.saturating_add(data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&sample_rate.saturating_mul(u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[derive(Debug, Clone)]
pub struct WavDecoder {
    state: State,
    hdr: Vec<u8>,
    need: usize,
    skip_remaining: u64,
    chunk_size: u32,
    format: Option<AudioFormat>,
    carry: Option<u8>,
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
        }
    }

    /// A push that only advanced the header returns an empty vector.
    pub fn push(&mut self, data: &[u8]) -> Result<Vec<WavSegment>> {
        let mut out: Vec<WavSegment> = Vec::new();

        // Only a push that begins with `RIFF` restarts a segment: providers stream several files
        // down one connection, and 16-bit PCM eventually contains the bytes `RIFF`.
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

    /// Validate the terminal boundary separately from the nonterminal drift flush.
    pub fn end_input(&self) -> Result<()> {
        if self.state != State::Streaming || self.carry.is_some() {
            return Err(CodecError::TruncatedWav);
        }
        Ok(())
    }

    fn reset_state(&mut self) {
        self.state = State::RiffHeader;
        self.hdr.clear();
        self.need = 12;
        self.skip_remaining = 0;
        self.chunk_size = 0;
        self.format = None;
        self.carry = None;
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
                self.need = 16;
                self.state = State::FmtData;
            }
            b"data" => {
                // The declared size is ignored, as in the reference, so a growing file (size 0 or
                // 0xFFFFFFFF) streams.
                if self.format.is_none() {
                    return Err(CodecError::WavDataBeforeFmt);
                }
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
        let fmt = &self.hdr;
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
        // PCM needs only its fixed fields. Retaining an attacker-declared extension would let
        // a small bounded ingress grow an unbounded header accumulator over many pushes.
        self.skip_remaining = u64::from(self.chunk_size - 16) + u64::from(self.chunk_size % 2);
        self.state =
            if self.skip_remaining == 0 { State::ChunkHeader } else { State::SkipChunkData };
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
#[path = "wav_tests.rs"]
pub(crate) mod tests;
