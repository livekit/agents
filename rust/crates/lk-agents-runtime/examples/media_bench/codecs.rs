//! Matched ten-second decode bursts, inline and on admitted runtime workers.

use std::error::Error;
use std::time::{Duration, Instant};

use lk_agents_media::Decoder;
use lk_agents_media::decoder::DecodeLimits;
use lk_agents_runtime::codecs::{AudioDecoder, DecodedAudio, DecoderError, DecoderLimits};
use lk_agents_runtime::media_workers::{MediaWorkerLimits, MediaWorkers};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::bench::{Distribution, heartbeat};

const SAMPLES: usize = 48000 * 10;

pub async fn measure() -> Result<Value, Box<dyn Error>> {
    let mut sources = vec![Source::wav()];
    #[cfg(feature = "media-ffmpeg")]
    sources.push(Source::ogg()?);
    let mut cases = Vec::new();
    for source in sources {
        for sessions in [1, 8, 32] {
            for worker_mode in [false, true] {
                let inputs = (0..sessions).map(|_| source.bytes.clone()).collect::<Vec<_>>();
                let workers = MediaWorkers::new(
                    tokio::runtime::Handle::current(),
                    MediaWorkerLimits {
                        total: sessions + 2,
                        decoders: sessions,
                        recordings: 1,
                        images: 1,
                    },
                )?;
                let limits = DecoderLimits {
                    format: DecodeLimits {
                        min_sample_rate: 8000,
                        max_sample_rate: 192000,
                        max_channels: 8,
                        max_streams: 8,
                        max_expansion_ratio: 24,
                        max_samples_per_frame: 65536,
                    },
                    input_bytes: source.bytes.len(),
                    input_chunks: 1,
                    output_samples: source.expected.len() + 1920,
                    output_chunk_samples: 960,
                };
                let mut decoders = if worker_mode {
                    (0..sessions)
                        .map(|_| AudioDecoder::open(&workers, source.mime, Some(48000), 1, limits))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    Vec::new()
                };
                let mut runtime_pcm = (0..sessions)
                    .map(|_| {
                        Vec::<DecodedAudio>::with_capacity(source.expected.len().div_ceil(960))
                    })
                    .collect::<Vec<_>>();
                let mut inline_pcm =
                    (0..sessions).map(|_| Vec::with_capacity(2)).collect::<Vec<_>>();
                let (stop, stopping) = oneshot::channel();
                let (ready, started) = oneshot::channel();
                let monitor = tokio::spawn(heartbeat(stopping, ready));
                started.await?;
                let mut admission = Vec::new();
                let mut finalize = Vec::new();
                let mut decoded_bytes = 0;
                let mut admission_busy = 0;
                let mut eof_busy = 0;
                let elapsed = Instant::now();
                if worker_mode {
                    for (decoder, mut bytes) in decoders.iter().zip(inputs) {
                        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
                        loop {
                            let start = Instant::now();
                            let result = decoder.try_push(&mut bytes);
                            admission.push(start.elapsed().as_nanos() as u64);
                            match result {
                                Ok(()) => break,
                                Err(DecoderError::Busy)
                                    if tokio::time::Instant::now() < deadline =>
                                {
                                    admission_busy += 1;
                                    tokio::task::yield_now().await;
                                }
                                Err(error) => return Err(error.into()),
                            }
                        }
                        loop {
                            let start = Instant::now();
                            let result = decoder.end_input();
                            finalize.push(start.elapsed().as_nanos() as u64);
                            match result {
                                Ok(()) => break,
                                Err(DecoderError::Busy)
                                    if tokio::time::Instant::now() < deadline =>
                                {
                                    eof_busy += 1;
                                    tokio::task::yield_now().await;
                                }
                                Err(error) => return Err(error.into()),
                            }
                        }
                    }
                    for (decoder, collected) in decoders.iter_mut().zip(&mut runtime_pcm) {
                        while let Some(frame) = decoder.next().await? {
                            decoded_bytes += frame.data.len();
                            collected.push(frame);
                        }
                        decoder.close(Duration::from_secs(1)).await?;
                    }
                } else {
                    for (bytes, collected) in inputs.into_iter().zip(&mut inline_pcm) {
                        let mut decoder = Decoder::new(source.mime, Some(48000), 1)?;
                        let start = Instant::now();
                        let pcm = decoder.push(&bytes)?;
                        admission.push(start.elapsed().as_nanos() as u64);
                        decoded_bytes += pcm.len() * 2;
                        collected.push(pcm);
                        let start = Instant::now();
                        let pcm = decoder.flush()?;
                        finalize.push(start.elapsed().as_nanos() as u64);
                        decoded_bytes += pcm.len() * 2;
                        collected.push(pcm);
                    }
                }
                let elapsed_ms = elapsed.elapsed().as_secs_f64() * 1000.;
                tokio::time::sleep(Duration::from_millis(10)).await;
                let _ = stop.send(());
                let lag = Distribution::from_ns(monitor.await?)?;
                if decoded_bytes != sessions * source.expected.len() * 2 {
                    return Err("decode benchmark lost or duplicated accepted PCM".into());
                }
                // Content validation is outside the heartbeat interval and allocates no second PCM copy.
                if worker_mode {
                    for frames in &runtime_pcm {
                        if frames
                            .iter()
                            .any(|frame| (frame.sample_rate, frame.num_channels) != (48000, 1))
                        {
                            return Err("decoded frame format changed".into());
                        }
                        source.validate(
                            frames
                                .iter()
                                .flat_map(|frame| frame.data.chunks_exact(2))
                                .map(|sample| i16::from_le_bytes([sample[0], sample[1]])),
                        )?;
                    }
                } else {
                    for chunks in &inline_pcm {
                        source.validate(chunks.iter().flatten().copied())?;
                    }
                }
                let gates = [10, 20].map(|cadence| json!({
                    "cadence_ms": cadence, "lag_p999_limit_ms": cadence as f64 * 0.2,
                    "lag_max_exclusive_ms": cadence as f64 * 0.5,
                    "passed": lag.p999_ms <= cadence as f64 * 0.2 && lag.max_ms < cadence as f64 * 0.5,
                }));
                eprintln!(
                    "codec {} {} sessions={sessions}: lag p99.9={:.3}ms max={:.3}ms",
                    source.format,
                    if worker_mode { "worker" } else { "inline" },
                    lag.p999_ms,
                    lag.max_ms
                );
                cases.push(json!({
                    "mode": if worker_mode { "worker" } else { "inline" },
                    "sessions": sessions, "worker_total_limit": sessions + 2,
                    "decoder_class_limit": sessions, "source_seconds": 10, "sample_rate": 48000,
                    "channels": 1, "source_bytes_per_stream": source.bytes.len(),
                    "source_format": source.format, "mime": source.mime, "chunk_count_per_stream": 1,
                    "expected_samples_per_stream": source.expected.len(),
                    "sample_tolerance_pcm_units": source.tolerance,
                    "decoded_bytes": decoded_bytes, "elapsed_ms": elapsed_ms,
                    "admission_busy_refusals": admission_busy, "eof_busy_refusals": eof_busy,
                    "admission_wall": Distribution::from_ns(admission)?,
                    "finalize_wall": Distribution::from_ns(finalize)?,
                    "loop_lag": lag, "gates": gates,
                }));
            }
        }
    }
    Ok(
        json!({"workload": "10s 48kHz mono WAV and Ogg/Opus, whole-response burst per stream; exact decoded count and bounded PCM difference verified outside capture",
        "input_preparation": "outside capture", "gil": "excluded", "cases": cases}),
    )
}

struct Source {
    format: &'static str,
    mime: &'static str,
    bytes: Vec<u8>,
    expected: Vec<i16>,
    tolerance: i32,
}

impl Source {
    fn wav() -> Self {
        Self {
            format: "wav_pcm_s16le",
            mime: "audio/wav",
            bytes: wav(),
            expected: (0..SAMPLES).map(|n| (n % 32768) as i16).collect(),
            tolerance: 0,
        }
    }

    #[cfg(feature = "media-ffmpeg")]
    fn ogg() -> Result<Self, Box<dyn Error>> {
        let samples = (0..SAMPLES)
            .map(|n| (12000. * (n as f64 * 440. * std::f64::consts::TAU / 48000.).sin()) as i16)
            .collect::<Vec<_>>();
        let mut encoder = lk_agents_media::OggOpusEncoder::new(48000, 1, 64000)?;
        let packet_samples = encoder.frame_size();
        let mut bytes = encoder.push(&samples)?;
        bytes.extend(encoder.finish()?);
        let mut decoder = Decoder::new("audio/ogg", Some(48000), 1)?;
        let mut expected = decoder.push(&bytes)?;
        expected.extend(decoder.flush()?);
        if expected.len().abs_diff(SAMPLES) > packet_samples * 2
            || !expected.iter().any(|sample| *sample > 1000)
            || !expected.iter().any(|sample| *sample < -1000)
        {
            return Err("compressed fixture lost duration or signal".into());
        }
        Ok(Self { format: "ogg_opus", mime: "audio/ogg", bytes, expected, tolerance: 2 })
    }

    fn validate(&self, actual: impl Iterator<Item = i16>) -> Result<(), Box<dyn Error>> {
        let mut count = 0;
        for (offset, sample) in actual.enumerate() {
            let expected = *self.expected.get(offset).ok_or("decoded extra PCM samples")?;
            if (i32::from(sample) - i32::from(expected)).abs() > self.tolerance {
                return Err("decoded PCM content changed".into());
            }
            count += 1;
        }
        if count != self.expected.len() {
            return Err("decoded PCM tail was lost".into());
        }
        Ok(())
    }
}

fn wav() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(44 + SAMPLES * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + SAMPLES as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&48000u32.to_le_bytes());
    bytes.extend_from_slice(&96000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(SAMPLES as u32 * 2).to_le_bytes());
    bytes.extend((0..SAMPLES).flat_map(|n| ((n % 32768) as i16).to_le_bytes()));
    bytes
}
