//! One decode job's synchronous work: what it reads (a file, a finite buffer, raw PCM or bytes
//! streamed in), the codec, channel remixing and resampling to the requested format, and the
//! shaping of its PCM into frames. The caller runs it on a thread that may block in its reader.

use std::io::Read;

use lk_agents_abi::AudioFrame;

use crate::decoder::{DecodeLimits, remix};
use crate::error::CodecError;
use crate::{AudioFormat, Decoder};

/// Encoded or raw input of one decode job.
pub enum Source {
    /// Replayed from its start while `looping` and it produced audio.
    File { path: String, looping: bool },
    /// A complete container held in memory.
    Container { mime: String, data: Vec<u8> },
    /// Interleaved little-endian PCM in `format`.
    Pcm { data: Vec<u8>, format: AudioFormat },
    /// Written as it arrives: `read` blocks for the next write, `None` at the end of input. Each
    /// write is a WAV stream's restart boundary.
    Stream { mime: String, read: Box<dyn FnMut() -> Option<Vec<u8>> + Send> },
}

/// Where decoded PCM goes, already at the requested format. An error stops the job.
pub type Output<'a, E> = &'a mut dyn FnMut(AudioFormat, &[i16]) -> std::result::Result<(), E>;

/// Decodes `source` to `sample_rate` (`None` keeps the source's) and `num_channels`, until its
/// input ends, `stopped` returns true between file loops, or `output` refuses.
pub fn decode<E: From<CodecError>>(
    source: Source,
    sample_rate: Option<u32>,
    num_channels: u16,
    limits: DecodeLimits,
    stopped: &dyn Fn() -> bool,
    output: Output<'_, E>,
) -> std::result::Result<(), E> {
    let format = Target { sample_rate, num_channels, limits };
    match source {
        Source::File { path, looping } => loop {
            if stopped() {
                return Ok(());
            }
            let file = std::fs::File::open(&path).map_err(read_error)?;
            let emitted = format.decode(file_mime(&path), &mut Blocks(file), output)?;
            if !looping || !emitted {
                return Ok(());
            }
        },
        Source::Container { mime, data } => {
            format.decode(&mime, &mut Blocks(std::io::Cursor::new(data)), output).map(drop)
        }
        Source::Pcm { data, format: source } => format.pcm(&data, source, output),
        Source::Stream { mime, mut read } => format.decode(&mime, &mut read, output).map(drop),
    }
}

/// `samples` as frames of at most `chunk_samples` interleaved samples.
pub fn frames(
    format: AudioFormat,
    samples: &[i16],
    chunk_samples: usize,
) -> impl Iterator<Item = AudioFrame> + '_ {
    let channels = usize::from(format.num_channels.max(1));
    samples.chunks((chunk_samples / channels * channels).max(channels)).map(move |chunk| {
        AudioFrame::new(
            crate::pcm_bytes(chunk),
            format.sample_rate,
            format.num_channels,
            u32::try_from(chunk.len() / channels).unwrap_or(u32::MAX),
        )
    })
}

/// Refuses, before any worker is admitted, a streamed container this build cannot decode.
pub fn check_stream(mime: &str) -> crate::Result<()> {
    if cfg!(feature = "ffmpeg") || crate::mime_to_av_format(Some(mime)) == Some("wav") {
        return Ok(());
    }
    Err(CodecError::UnsupportedMime {
        mime: mime.into(),
        reason: "container decoding needs the ffmpeg feature",
    })
}

/// A file decodes by its WAV extension without ffmpeg, else by probing.
fn file_mime(path: &str) -> &'static str {
    let wav =
        std::path::Path::new(path).extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    if wav && !cfg!(feature = "ffmpeg") { "audio/wav" } else { "" }
}

fn read_error(error: std::io::Error) -> CodecError {
    CodecError::Read(error.to_string())
}

struct Target {
    sample_rate: Option<u32>,
    num_channels: u16,
    limits: DecodeLimits,
}

impl Target {
    /// Whether any audio came out.
    fn decode<E: From<CodecError>>(
        &self,
        mime: &str,
        input: &mut dyn Chunks,
        output: Output<'_, E>,
    ) -> std::result::Result<bool, E> {
        let mut emitted = false;
        if crate::mime_to_av_format(Some(mime)) == Some("wav") {
            let mut decoder =
                Decoder::wav(self.sample_rate, self.num_channels, self.limits.whole_pushes())?;
            while let Some(chunk) = input.next_chunk().map_err(read_error)? {
                let pcm = decoder.push(&chunk)?;
                emitted |= wav_output(&decoder, &pcm, output)?;
            }
            let tail = decoder.finish()?;
            emitted |= wav_output(&decoder, &tail, output)?;
            return Ok(emitted);
        }
        #[cfg(feature = "ffmpeg")]
        {
            // FFmpeg's callback carries only codec errors; the caller's is kept aside.
            let mut refused = None;
            let decoded = crate::ffmpeg::decode_stream(
                crate::mime_to_av_format(Some(mime)),
                self.sample_rate.unwrap_or(0),
                self.num_channels,
                self.limits,
                &mut Joined { chunks: input, chunk: Vec::new(), read: 0 },
                |format, samples| {
                    emitted |= !samples.is_empty();
                    output(format, samples).map_err(|error| {
                        refused = Some(error);
                        CodecError::Ffmpeg { op: "PCM delivery", detail: "refused".into() }
                    })
                },
            );
            match (decoded, refused) {
                (_, Some(error)) => Err(error),
                (Err(error), None) => Err(error.into()),
                (Ok(_), None) => Ok(emitted),
            }
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = (input, output);
            Err(CodecError::UnsupportedMime {
                mime: mime.into(),
                reason: "container decoding needs the ffmpeg feature",
            }
            .into())
        }
    }

    fn pcm<E: From<CodecError>>(
        &self,
        bytes: &[u8],
        source: AudioFormat,
        output: Output<'_, E>,
    ) -> std::result::Result<(), E> {
        let rate = self.sample_rate.unwrap_or(source.sample_rate);
        self.limits.validate(source, Some(rate), 0)?;
        let width = usize::from(source.num_channels) * 2;
        if !bytes.len().is_multiple_of(width) {
            return Err(CodecError::DecodeLimit("truncated PCM sample").into());
        }
        let resample_error = |error: crate::ResampleError| CodecError::Resample {
            from: source.sample_rate,
            to: rate,
            detail: error.to_string(),
        };
        let mut resampler = (source.sample_rate != rate)
            .then(|| {
                crate::new_resampler(
                    source.sample_rate,
                    rate,
                    u32::from(self.num_channels),
                    crate::Quality::default(),
                )
            })
            .transpose()
            .map_err(resample_error)?;
        let format = AudioFormat::new(rate, self.num_channels);
        // Ten milliseconds of source PCM bounds conversion and resampling scratch space.
        for chunk in bytes.chunks((source.sample_rate as usize / 100).max(1) * width) {
            let samples =
                remix(&crate::pcm_samples(chunk), source.num_channels, self.num_channels)?;
            match &mut resampler {
                Some(resampler) => {
                    output(format, resampler.push(&samples).map_err(resample_error)?)?
                }
                None => output(format, &samples)?,
            }
        }
        if let Some(resampler) = &mut resampler {
            output(format, resampler.flush().map_err(resample_error)?)?;
        }
        Ok(())
    }
}

/// What a WAV push decoded, once its header gave the format.
fn wav_output<E: From<CodecError>>(
    decoder: &Decoder,
    pcm: &[i16],
    output: Output<'_, E>,
) -> std::result::Result<bool, E> {
    let Some(format) = decoder.output_format() else { return Ok(false) };
    output(format, pcm)?;
    Ok(!pcm.is_empty())
}

/// Encoded input as a decoder reads it. A WAV stream restarts only at a chunk boundary
/// (providers concatenate files, one per write), so the WAV decoder takes chunks whole.
trait Chunks {
    /// The next chunk, `None` at the end of input.
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>>;
}

/// A file or a finite buffer, in 4 KiB chunks.
struct Blocks<R>(R);

impl<R: Read> Chunks for Blocks<R> {
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        let mut chunk = vec![0; 4096];
        let n = self.0.read(&mut chunk)?;
        chunk.truncate(n);
        Ok((n > 0).then_some(chunk))
    }
}

impl Chunks for Box<dyn FnMut() -> Option<Vec<u8>> + Send> {
    fn next_chunk(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        Ok(self())
    }
}

/// Chunks as the byte stream a demuxer reads.
#[cfg(feature = "ffmpeg")]
struct Joined<'a> {
    chunks: &'a mut dyn Chunks,
    chunk: Vec<u8>,
    read: usize,
}

#[cfg(feature = "ffmpeg")]
impl Read for Joined<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        while self.read == self.chunk.len() {
            match self.chunks.next_chunk()? {
                Some(chunk) => (self.chunk, self.read) = (chunk, 0),
                None => return Ok(0),
            }
        }
        let n = bytes.len().min(self.chunk.len() - self.read);
        bytes[..n].copy_from_slice(&self.chunk[self.read..self.read + n]);
        self.read += n;
        Ok(n)
    }
}
