//! Compare the original synchronous recorder with its admitted ordered runtime sink using real
//! Opus encoders and temporary files. Heartbeat includes open, staging, encode/write and finish.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use lk_agents_abi::AudioFrame;
use lk_agents_media::OggOpusEncoder;
use lk_agents_media::recorder::{PcmFrame, SessionRecorder as InlineRecorder};
use lk_agents_runtime::media_workers::{MediaWorkerLimits, MediaWorkers};
use lk_agents_runtime::recording::{RecordingLimits, RecordingSegment, SessionRecorder};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::bench::{Distribution, heartbeat};

const RATE: u32 = 48_000;
const ITERATIONS: usize = 3;
const SEGMENT: RecordingSegment = RecordingSegment { generation: 1, segment: 1 };

pub(crate) async fn measure() -> Result<Value, Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join(format!("media-recording-bench-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let outcome = measure_at(&root).await;
    std::fs::remove_dir_all(root)?;
    outcome
}

async fn measure_at(root: &std::path::Path) -> Result<Value, Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1, 8] {
        for samples in [960, 480_000] {
            let frame = AudioFrame::new(
                lk_agents_media::pcm_bytes(
                    &(0..samples)
                        .map(|sample| ((sample * 37) % 20000) as i16 - 10000)
                        .collect::<Vec<_>>(),
                ),
                RATE,
                1,
                samples as u32,
            );
            for worker in [false, true] {
                let (stop, stopped) = oneshot::channel();
                let (ready, started) = oneshot::channel();
                let monitor = tokio::spawn(heartbeat(stopped, ready));
                started.await?;
                let started = Instant::now();
                let mut staging = Vec::new();
                let mut maximum_retained = 0;
                let mut bytes_written = 0;
                let duration = samples as f64 / f64::from(RATE);
                if worker {
                    let workers = MediaWorkers::new(
                        tokio::runtime::Handle::current(),
                        MediaWorkerLimits { total: 8, decoders: 1, recordings: 8, images: 1 },
                    )?;
                    let mut recorders = Vec::new();
                    for index in 0..count {
                        recorders.push(
                            SessionRecorder::open(
                                &workers,
                                path(root, worker, count, samples, index),
                                RATE,
                                Some(0.0),
                                64_000,
                                RecordingLimits {
                                    queued_commands: 128,
                                    queued_bytes: 8 * 1024 * 1024,
                                    frame_bytes: 2 * 1024 * 1024,
                                    retained_bytes: 16 * 1024 * 1024,
                                    segment_bytes: 2 * 1024 * 1024,
                                    segments: 1,
                                    block_duration: Duration::from_millis(100),
                                },
                            )
                            .await?,
                        );
                    }
                    for iteration in 0..ITERATIONS {
                        let offset = iteration as f64 * duration;
                        for recorder in &recorders {
                            let capture = Instant::now();
                            recorder.try_capture_frame(SEGMENT, offset, frame.clone())?;
                            staging.push(capture.elapsed().as_nanos() as u64);
                            recorder.try_playback_progressed(SEGMENT, offset, offset, duration)?;
                            maximum_retained = maximum_retained.max(recorder.retained_bytes());
                        }
                        for recorder in &recorders {
                            recorder.flush(offset + duration + 1.0).await?;
                        }
                    }
                    for recorder in recorders {
                        let summary = recorder.finish(ITERATIONS as f64 * duration).await?;
                        if summary.dropped_samples != [0, 0] {
                            return Err("recording benchmark dropped samples".into());
                        }
                        bytes_written += summary.bytes_written;
                    }
                } else {
                    let mut recorders = Vec::new();
                    for index in 0..count {
                        recorders.push(InlineRecorder::new(
                            OggOpusEncoder::new(RATE, 2, 64_000)?,
                            RATE,
                            Some(0.0),
                            path(root, worker, count, samples, index),
                        )?);
                    }
                    // Frame extraction used to happen in every tap callback. Include its sample
                    // conversion in the inline ingress measurement; worker ingress retains an Arc.
                    for iteration in 0..ITERATIONS {
                        let offset = iteration as f64 * duration;
                        for recorder in &recorders {
                            let capture = Instant::now();
                            recorder.capture_pcm(offset, &PcmFrame::from_audio_frame(&frame));
                            staging.push(capture.elapsed().as_nanos() as u64);
                            recorder.playback_progressed(offset, offset, duration);
                        }
                        for recorder in &recorders {
                            recorder.flush(offset + duration + 1.0)?;
                        }
                    }
                    for recorder in recorders {
                        recorder.finish(ITERATIONS as f64 * duration)?;
                        bytes_written += recorder.bytes_written();
                    }
                }
                let elapsed = started.elapsed();
                tokio::time::sleep(Duration::from_millis(5)).await;
                let _ = stop.send(());
                let lag = Distribution::from_ns(monitor.await?)?;
                let ingress = Distribution::from_ns(staging)?;
                let expected_frames = ITERATIONS * samples;
                let mut decoded_frames = 0;
                for index in 0..count {
                    let mut input = std::fs::File::open(path(root, worker, count, samples, index))?;
                    let mut frames = 0;
                    let mut right_energy = 0_u128;
                    lk_agents_media::ffmpeg::decode_stream(
                        Some("ogg"),
                        RATE,
                        2,
                        lk_agents_media::decoder::DecodeLimits {
                            min_sample_rate: 8000,
                            max_sample_rate: 192000,
                            max_channels: 2,
                            max_streams: 1,
                            max_expansion_ratio: 24,
                            max_samples_per_frame: 192000,
                        },
                        &mut input,
                        |format, pcm| {
                            if format.num_channels != 2 {
                                return Err(lk_agents_media::CodecError::DecodeLimit(
                                    "recording stereo channels",
                                ));
                            }
                            frames += pcm.len() / 2;
                            right_energy += pcm
                                .chunks_exact(2)
                                .map(|pair| i128::from(pair[1]).unsigned_abs().pow(2))
                                .sum::<u128>();
                            Ok(())
                        },
                    )?;
                    // Opus packet padding is at most one 20ms packet at either boundary.
                    if frames.abs_diff(expected_frames) > 1920 || right_energy == 0 {
                        return Err(format!("recording decoded {frames} frames, expected {expected_frames}, energy={right_energy}").into());
                    }
                    decoded_frames += frames;
                }
                if bytes_written == 0 {
                    return Err("empty recording benchmark output".into());
                }
                eprintln!(
                    "recording worker={worker} count={count} samples={samples}: ingress p99.9={:.6}ms loop p99.9={:.3}ms max={:.3}ms",
                    ingress.p999_ms, lag.p999_ms, lag.max_ms
                );
                cases.push(json!({
                    "worker":worker,"recorders":count,"frame_samples":samples,"iterations":ITERATIONS,
                    "elapsed_ms":elapsed.as_secs_f64()*1000.0,"capture_wall":ingress,"loop_lag":lag,
                    "maximum_observed_retained_bytes":maximum_retained,"bytes_written":bytes_written,
                    "decoded_frames":decoded_frames,"expected_frames_per_recording":expected_frames,
                    "gate_20ms": {"p999_limit_ms":4.0,"max_exclusive_ms":10.0,"passed":lag.p999_ms<=4.0 && lag.max_ms<10.0},
                }));
            }
        }
    }
    Ok(
        json!({"cases":cases,"scope":"Real temporary filesystem and native Opus; ingress, writes and cleanup on current-thread Tokio; Python/GIL excluded; retained bytes sampled at ingress, not a peak-memory profiler."}),
    )
}

fn path(
    root: &std::path::Path,
    worker: bool,
    count: usize,
    samples: usize,
    index: usize,
) -> PathBuf {
    root.join(format!("{worker}-{count}-{samples}-{index}.ogg"))
}
