//! Opt-in measurements through real codec, image and recording adapters. Timings are observations;
//! assertions cover delivered media and bounded admission, independent of the host's CPU speed.
#![cfg(all(feature = "media-images", feature = "media-ffmpeg", unix))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput};
use lk_agents_abi::recorder::{AudioRecorder, SessionRecorder as RecorderApi};
use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_inference::codecs::{AudioDecoder, decoder_limits};
use lk_agents_media::workers::MediaWorkerClass;
use lk_agents_runtime::recording::SessionRecorder;
use serde_json::{Value, json};

fn cpu_ns(clock: libc::clockid_t) -> u64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes the selected CPU clock to a valid timespec.
    assert_eq!(unsafe { libc::clock_gettime(clock, &mut time) }, 0);
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

fn distribution(mut values: Vec<u64>) -> Value {
    values.sort_unstable();
    assert!(!values.is_empty(), "measurement must contain observations");
    json!({
        "count": values.len(),
        "p999_ms": values[(values.len() * 999).div_ceil(1000) - 1] as f64 / 1e6,
        "max_ms": *values.last().unwrap() as f64 / 1e6,
    })
}

struct Measured<F> {
    future: Pin<Box<F>>,
    wall: Vec<u64>,
    cpu: Vec<u64>,
}

impl<F: Future> Future for Measured<F> {
    type Output = (F::Output, Value, Value);

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let wall = Instant::now();
        let cpu = cpu_ns(libc::CLOCK_THREAD_CPUTIME_ID);
        let result = self.future.as_mut().poll(cx);
        self.cpu.push(cpu_ns(libc::CLOCK_THREAD_CPUTIME_ID) - cpu);
        self.wall.push(wall.elapsed().as_nanos() as u64);
        result.map(|result| {
            (
                result,
                distribution(std::mem::take(&mut self.wall)),
                distribution(std::mem::take(&mut self.cpu)),
            )
        })
    }
}

async fn measure<F: Future>(name: &str, workload: Value, future: F) -> F::Output {
    let (stop, mut stopped) = tokio::sync::oneshot::channel::<()>();
    let (ready, started) = tokio::sync::oneshot::channel();
    let heartbeat = tokio::spawn(async move {
        let period = Duration::from_millis(1);
        let mut deadline = tokio::time::Instant::now() + period;
        let mut lag = Vec::new();
        let _ = ready.send(());
        loop {
            tokio::select! {
                _ = &mut stopped => break,
                _ = tokio::time::sleep_until(deadline) => {
                    let now = tokio::time::Instant::now();
                    lag.push(now.saturating_duration_since(deadline).as_nanos() as u64);
                    deadline += period * ((now.saturating_duration_since(deadline).as_millis() + 1) as u32);
                }
            }
        }
        lag
    });
    started.await.unwrap();
    let start = Instant::now();
    let process_cpu = cpu_ns(libc::CLOCK_PROCESS_CPUTIME_ID);
    let (result, wall, cpu) =
        Measured { future: Box::pin(future), wall: Vec::new(), cpu: Vec::new() }.await;
    // Observe a delayed ready heartbeat even when the workload completes in its first poll.
    tokio::time::sleep(Duration::from_millis(2)).await;
    let _ = stop.send(());
    let lag = heartbeat.await.unwrap();
    println!(
        "MEDIA_MEASUREMENT {}",
        json!({
            "name": name, "workload": workload, "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
            "loop_lag": distribution(lag), "poll_wall": wall, "poll_thread_cpu": cpu,
        "process_cpu_ms": (cpu_ns(libc::CLOCK_PROCESS_CPUTIME_ID) - process_cpu) as f64 / 1e6,
        })
    );
    result
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "real native workload; records host-dependent measurements"]
async fn media_never_block_measurements() {
    let manifest: Value =
        serde_json::from_str(include_str!("../../../../scripts/bench/budgets.json")).unwrap();
    println!(
        "MEDIA_PROFILE {}",
        json!({"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
               "runtime": "current_thread", "features": ["media-images", "media-ffmpeg"],
               "budget": manifest["interfaces"]["media"]})
    );
    let workers = lk_agents_runtime::process::workers().unwrap();
    let mut admission = Vec::new();
    let mut ceilings = serde_json::Map::new();
    for class in [MediaWorkerClass::Decoder, MediaWorkerClass::Recording, MediaWorkerClass::Image] {
        let before = admission.len();
        while let Ok(permit) = workers.try_admit(class) {
            admission.push(permit);
        }
        ceilings.insert(format!("{class:?}"), json!(admission.len() - before));
    }
    println!("MEDIA_ADMISSION {}", json!(ceilings));
    drop(admission);
    measure("idle", json!({"heartbeat_ms": 1}), tokio::time::sleep(Duration::from_secs(1))).await;
    for (width, height) in [(1920, 1080), (3840, 2160)] {
        // Pixel construction is fixture setup, outside the adapter timing. The same immutable
        // input is shared by concurrent callers; the production adapter still encodes every call.
        let pixels = bytes::Bytes::from(
            (0..width * height)
                .flat_map(|at| {
                    let (x, y) = (at % width, at / width);
                    [(x / 16) as u8, (y / 16) as u8, ((x + y) / 32) as u8, 255]
                })
                .collect::<Vec<_>>(),
        );
        for format in ["JPEG", "PNG"] {
            measure(
                "image_encode",
                json!({"width": width, "height": height, "format": format, "streams": 4}),
                async {
                    let mut tasks = Vec::new();
                    for _ in 0..4 {
                        let workers = workers.clone();
                        let data = pixels.clone();
                        tasks.push(async move {
                            let image = lk_agents_runtime::images::prepare_on(
                                &workers,
                                std::sync::Arc::new(lk_agents_media::images::Images::default()),
                                ImageInput::Frame(VideoFrame {
                                    width,
                                    height,
                                    buffer_type: 0,
                                    data,
                                }),
                                ImageEncodeOptions { format: format.into(), ..Default::default() },
                            )
                            .await
                            .unwrap();
                            assert!(!image.0.unwrap().is_empty());
                        });
                    }
                    futures_util::future::join_all(tasks).await;
                },
            )
            .await;
        }
    }

    let path =
        std::env::temp_dir().join(format!("lk-media-measurements-{}.ogg", std::process::id()));
    let fixture_path = path.clone();
    workers
        .try_spawn(MediaWorkerClass::Recording, move || {
            let mut encoder = lk_agents_media::OggOpusEncoder::new(48_000, 1, 64_000).unwrap();
            let pcm = (0..480_000).map(|at| ((at % 1024) as i16 - 512) * 16).collect::<Vec<_>>();
            let mut data = encoder.push(&pcm).unwrap();
            data.extend(encoder.finish().unwrap());
            std::fs::write(fixture_path, data).unwrap();
        })
        .unwrap()
        .await
        .unwrap();
    measure(
        "ogg_decode",
        json!({"streams": 32, "audio_seconds_per_stream": 10, "sample_rate": 48000, "channels": 1}),
        async {
            let mut decoders = Vec::new();
            for _ in 0..32 {
                decoders.push(
                    AudioDecoder::open_file(
                        &workers,
                        path.to_string_lossy().into_owned(),
                        false,
                        48_000,
                        1,
                        decoder_limits(48_000, 1),
                    )
                    .unwrap(),
                );
            }
            // Each stream has more PCM than its bounded output queue, so capacity stays owned
            // while consumers are stalled. A 33rd stream is refused before its file is opened.
            assert!(
                AudioDecoder::open_file(
                    &workers,
                    path.to_string_lossy().into_owned(),
                    false,
                    48_000,
                    1,
                    decoder_limits(48_000, 1),
                )
                .is_err()
            );
            let mut tasks = Vec::new();
            for mut decoder in decoders {
                tasks.push(async move {
                    let mut samples = 0;
                    while let Some(frame) = decoder.next().await.unwrap() {
                        samples += frame.samples_per_channel as usize;
                    }
                    decoder.close().await.unwrap();
                    assert_eq!(samples, 480_000, "codec tail and head must preserve duration");
                });
            }
            futures_util::future::join_all(tasks).await;
        },
    )
    .await;
    std::fs::remove_file(&path).unwrap();

    let pcm = lk_agents_media::pcm_bytes(&vec![1000; 480_000]);
    let frame = lk_agents_abi::AudioFrame::new(pcm, 48_000, 1, 480_000);
    for streams in [1, 8, 16] {
        measure("recording_burst", json!({"streams": streams, "audio_seconds_per_stream": 10, "sample_rate": 48000, "input_channels": 1, "output_channels": 2}), async {
            let mut recordings = Vec::new();
            for index in 0..streams {
                let path = path.with_extension(format!("{streams}-{index}.ogg"));
                let recorder = <SessionRecorder as RecorderApi>::new(path.to_string_lossy().into_owned(), 48_000, Some(0.0), 64_000).unwrap();
                recordings.push((recorder, path));
            }
            if streams == 16 {
                assert!(<SessionRecorder as RecorderApi>::new(
                    path.with_extension("refused.ogg").to_string_lossy().into_owned(),
                    48_000, Some(0.0), 64_000).is_err());
            }
            let mut tasks = Vec::new();
            for (recorder, path) in recordings {
                let frame = frame.clone();
                tasks.push(async move {
                    RecorderApi::ready(&recorder).await.unwrap();
                    recorder.on_input_frame(10.0, &frame);
                    let summary = recorder.finish(10.0).await.unwrap();
                    assert!(summary.bytes_written > 0);
                    assert_eq!(summary.dropped_input_samples, 0);
                    tokio::fs::remove_file(path).await.unwrap();
                });
            }
            futures_util::future::join_all(tasks).await;
        }).await;
    }
}
