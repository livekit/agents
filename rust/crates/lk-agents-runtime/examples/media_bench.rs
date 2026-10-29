//! `cargo run -p lk-agents-runtime --features media-ffmpeg,media-resample,media-images
//! --example media_bench -- output.json [codecs|recording|images|control|all]`.
//! Measures the background file decoder, declared recorder implementation and shipped
//! admitted image preparation against an independent anchored 1ms Tokio heartbeat.

#[cfg(not(all(feature = "media-ffmpeg", feature = "media-images")))]
fn main() {
    eprintln!("media_bench requires media-ffmpeg,media-resample,media-images");
    std::process::exit(2);
}

#[cfg(all(feature = "media-ffmpeg", feature = "media-images"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::run()
}

#[cfg(all(feature = "media-ffmpeg", feature = "media-images"))]
mod bench {
    use std::future::Future;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use lk_agents_abi::AudioFrame;
    use lk_agents_abi::media::images::{ImageEncodeOptions, ImageInput};
    use lk_agents_abi::recorder::{AudioRecorder, SessionRecorder as RecorderClass};
    use lk_agents_abi::utils::video::VideoFrame;
    use lk_agents_runtime::recording::SessionRecorder;
    use serde_json::{Value, json};
    use tokio::sync::oneshot;

    type Error = Box<dyn std::error::Error>;
    const OBSERVATIONS: usize = 1_000;
    const RATE: u32 = 48_000;
    const SAMPLES: usize = RATE as usize * 10;

    pub fn run() -> Result<(), Error> {
        let mut arguments = std::env::args().skip(1);
        let output = arguments
            .next()
            .ok_or("usage: media_bench output.json [codecs|recording|images|control|all]")?;
        let selected = arguments.next().unwrap_or_else(|| "all".into());
        if arguments.next().is_some()
            || !["codecs", "recording", "images", "control", "all"].contains(&selected.as_str())
        {
            return Err("invalid workload".into());
        }
        let root = std::env::temp_dir().join(format!("media-bench-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let wav = root.join("source.wav");
        std::fs::write(&wav, wav_bytes())?;
        let ogg = root.join("source.ogg");
        let pcm = (0..SAMPLES)
            .map(|n| {
                ((n as f64 * 440. * std::f64::consts::TAU / f64::from(RATE)).sin() * 12000.) as i16
            })
            .collect::<Vec<_>>();
        let mut encoder = lk_agents_media::OggOpusEncoder::new(RATE, 1, 64_000)?;
        let mut encoded = encoder.push(&pcm)?;
        encoded.extend(encoder.finish()?);
        std::fs::write(&ogg, encoded)?;
        // Starting the production process service is outside steady-state heartbeat capture.
        lk_agents_runtime::process::workers()?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_time().build()?;
        let outcome = runtime.block_on(async {
            let mut cases = Vec::new();
            if ["all", "control"].contains(&selected.as_str()) {
                cases.push(control().await?);
            }
            if ["all", "codecs"].contains(&selected.as_str()) {
                for path in [&wav, &ogg] {
                    for count in [1, 8, 32, 64] {
                        cases.push(decode(path, count).await?);
                    }
                }
            }
            if ["all", "recording"].contains(&selected.as_str()) {
                for count in [1, 8, 32, 64] {
                    for samples in [960, SAMPLES] {
                        cases.push(record(&root, count, samples).await?);
                    }
                }
            }
            if ["all", "images"].contains(&selected.as_str()) {
                for (width, height) in [(1920, 1080), (3840, 2160)] {
                    let rgba = (0..width * height)
                        .flat_map(|n| [(n % 251) as u8, (n % 193) as u8, (n % 137) as u8, 255])
                        .collect::<Vec<_>>();
                    for count in [1, 8] {
                        for resize in [false, true] {
                            cases.push(image(&rgba, width, height, count, resize).await?);
                        }
                    }
                }
            }
            Ok::<_, Error>(json!({
                "heartbeat_ms": 1, "cadence_ms": 20,
                "scope": concat!(
                    "Shipped native entrypoints: background file decoder, ",
                    "ABI recorder implementation, admitted image preparation. Sustained active waves, ",
                    "anchored scheduled deadlines including missed ticks; thread CPU and wall poll ",
                    "timing. Python/GIL/RTC conversion excluded.",
                ),
                "cases": cases
            }))
        });
        std::fs::remove_dir_all(root)?;
        let result = outcome?;
        std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
        Ok(())
    }

    struct HeartbeatSamples {
        lag: Vec<f64>,
        serviced: usize,
    }

    async fn start_heartbeat()
    -> (oneshot::Sender<()>, tokio::task::JoinHandle<HeartbeatSamples>, Arc<AtomicUsize>) {
        let observations = Arc::new(AtomicUsize::new(0));
        let observed = observations.clone();
        let (stop, mut stopped) = oneshot::channel();
        let (ready, started) = oneshot::channel();
        let task = tokio::spawn(async move {
            let interval = Duration::from_millis(1);
            let mut deadline = tokio::time::Instant::now() + interval;
            let mut samples = HeartbeatSamples { lag: Vec::new(), serviced: 0 };
            let _ = ready.send(());
            loop {
                let ending = tokio::select! {
                    _ = &mut stopped => true,
                    _ = tokio::time::sleep_until(deadline) => false,
                };
                let now = tokio::time::Instant::now();
                samples.serviced += usize::from(!ending);
                // Preserve every scheduled deadline, including ticks missed during a stall.
                while deadline <= now {
                    samples.lag.push(now.saturating_duration_since(deadline).as_secs_f64() * 1000.);
                    deadline += interval;
                }
                observed.store(samples.lag.len(), Ordering::Release);
                if ending {
                    return samples;
                }
            }
        });
        started.await.unwrap();
        (stop, task, observations)
    }

    fn distribution(mut samples: Vec<f64>) -> Value {
        assert!(!samples.is_empty(), "measurement collected no observations");
        samples.sort_by(f64::total_cmp);
        let p999 = samples[(samples.len() as f64 * 0.999).ceil() as usize - 1];
        let max = samples[samples.len() - 1];
        let mut result = json!({"observations":samples.len(),"max_ms":max});
        if samples.len() >= OBSERVATIONS {
            result["p999_ms"] = json!(p999);
        }
        result
    }

    async fn stop_heartbeat(
        stop: oneshot::Sender<()>,
        task: tokio::task::JoinHandle<HeartbeatSamples>,
    ) -> Result<Value, Error> {
        let _ = stop.send(());
        let samples = task.await?;
        let mut lag = distribution(samples.lag);
        lag["serviced_wakeups"] = json!(samples.serviced);
        let p999 = lag["p999_ms"].as_f64().unwrap();
        let max = lag["max_ms"].as_f64().unwrap();
        lag["gate_20ms"] =
            json!({"p999_limit_ms":4.,"max_exclusive_ms":10.,"passed":p999<=4. && max<10.});
        lag["observation_kind"] = json!("anchored_1ms_deadlines_including_missed_ticks");
        Ok(lag)
    }

    fn thread_cpu() -> f64 {
        let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: clock_gettime writes to a valid timespec for the calling thread's CPU clock.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
        assert_eq!(status, 0, "thread CPU clock unavailable");
        time.tv_sec as f64 + time.tv_nsec as f64 / 1_000_000_000.
    }

    async fn measure_polls<F: Future>(future: F) -> (F::Output, Vec<f64>, f64) {
        let mut future = std::pin::pin!(future);
        let mut cpu = Vec::new();
        let mut maximum = 0_f64;
        let result = std::future::poll_fn(|cx| {
            let start = Instant::now();
            let cpu_start = thread_cpu();
            let result = future.as_mut().poll(cx);
            cpu.push((thread_cpu() - cpu_start) * 1000.);
            maximum = maximum.max(start.elapsed().as_secs_f64() * 1000.);
            result
        })
        .await;
        (result, cpu, maximum)
    }

    async fn control() -> Result<Value, Error> {
        let (stop, heartbeat, observations) = start_heartbeat().await;
        while observations.load(Ordering::Acquire) < OBSERVATIONS {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let result = json!({
            "kind": "idle_heartbeat_control",
            "loop_lag": stop_heartbeat(stop, heartbeat).await?,
        });
        eprintln!("{result}");
        Ok(result)
    }

    async fn decode(path: &Path, count: usize) -> Result<Value, Error> {
        let workers = lk_agents_runtime::process::workers()?;
        let (stop, heartbeat, observations) = start_heartbeat().await;
        let start = Instant::now();
        let mut samples = 0;
        let mut refused = 0;
        let mut waves = 0;
        let mut poll_cpu = Vec::new();
        let mut poll_max_ms = 0_f64;
        let mut open_max_ms = 0_f64;
        let mut open_cpu = Vec::new();
        while observations.load(Ordering::Acquire) < OBSERVATIONS || poll_cpu.len() < OBSERVATIONS {
            waves += 1;
            let mut jobs = tokio::task::JoinSet::new();
            let mut starts = Vec::new();
            for _ in 0..count {
                let open_started = Instant::now();
                let open_cpu_start = thread_cpu();
                match lk_agents_inference::codecs::AudioDecoder::open_file(
                    &workers,
                    path.to_string_lossy().into_owned(),
                    false,
                    RATE,
                    1,
                    lk_agents_inference::codecs::decoder_limits(RATE, 1),
                ) {
                    Ok(mut decoder) => {
                        let (start, ready) = oneshot::channel();
                        starts.push(start);
                        jobs.spawn(measure_polls(async move {
                            ready.await.map_err(|_| "decoder wave cancelled".to_owned())?;
                            let mut samples = 0;
                            let mut signal = false;
                            while let Some(frame) =
                                decoder.next().await.map_err(|e| e.to_string())?
                            {
                                samples += frame.samples_per_channel as usize;
                                signal |= frame.data.iter().any(|byte| *byte != 0);
                            }
                            decoder.close().await.map_err(|e| e.to_string())?;
                            if samples.abs_diff(SAMPLES) > 1920 || !signal {
                                return Err("decoded duration/signal mismatch".to_owned());
                            }
                            Ok::<_, String>(samples)
                        }));
                    }
                    Err(lk_agents_inference::codecs::DecoderError::WorkerAdmission(_)) => {
                        refused += 1
                    }
                    Err(error) => return Err(error.to_string().into()),
                }
                open_cpu.push((thread_cpu() - open_cpu_start) * 1000.);
                open_max_ms = open_max_ms.max(open_started.elapsed().as_secs_f64() * 1000.);
                tokio::task::yield_now().await;
            }
            for start in starts {
                let _ = start.send(());
            }
            while let Some(job) = jobs.join_next().await {
                let (result, cpu, maximum) = job?;
                samples += result?;
                poll_cpu.extend(cpu);
                poll_max_ms = poll_max_ms.max(maximum);
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let lag = stop_heartbeat(stop, heartbeat).await?;
        let result = json!({
            "kind": "background_file_decoder",
            "format": path.extension().unwrap().to_string_lossy(),
            "requested": count,
            "waves": waves,
            "admitted": count * waves - refused,
            "refused": refused,
            "decoded_samples": samples,
            "poll_cpu": distribution(poll_cpu),
            "poll_max_ms": poll_max_ms,
            "open_max_ms": open_max_ms,
            "open_cpu": distribution(open_cpu),
            "elapsed_ms": elapsed,
            "loop_lag": lag,
        });
        eprintln!("{result}");
        Ok(result)
    }

    async fn record(root: &Path, count: usize, samples: usize) -> Result<Value, Error> {
        let data = lk_agents_media::pcm_bytes(
            &(0..samples).map(|n| ((n * 37) % 20_000) as i16 - 10_000).collect::<Vec<_>>(),
        );
        let frame = AudioFrame::new(data, RATE, 1, samples as u32);
        let (stop, heartbeat, observations) = start_heartbeat().await;
        let start = Instant::now();
        let mut refused = 0;
        let mut waves = 0;
        let mut poll_cpu = Vec::new();
        let mut poll_max_ms = 0_f64;
        let mut open_max_ms = 0_f64;
        let mut open_cpu = Vec::new();
        let mut admitted = 0;
        let mut max_capture_ms = 0_f64;
        let mut paths = Vec::new();
        while observations.load(Ordering::Acquire) < OBSERVATIONS || poll_cpu.len() < OBSERVATIONS {
            waves += 1;
            let mut recorders = Vec::new();
            for index in 0..count {
                let open_started = Instant::now();
                let open_cpu_start = thread_cpu();
                let path = root.join(format!("record-{count}-{samples}-{index}.ogg"));
                match <SessionRecorder as RecorderClass>::new(
                    path.to_string_lossy().into_owned(),
                    RATE,
                    Some(0.),
                    64_000,
                ) {
                    Ok(recorder) => recorders.push((Arc::new(recorder), path)),
                    Err(error) if error.to_string().contains("capacity exhausted") => refused += 1,
                    Err(error) => return Err(error.into()),
                }
                open_cpu.push((thread_cpu() - open_cpu_start) * 1000.);
                open_max_ms = open_max_ms.max(open_started.elapsed().as_secs_f64() * 1000.);
                tokio::task::yield_now().await;
            }
            admitted += recorders.len();
            let mut jobs = tokio::task::JoinSet::new();
            for (recorder, path) in recorders {
                let frame = frame.clone();
                jobs.spawn(measure_polls(async move {
                    recorder.ready().await.map_err(|e| e.to_string())?;
                    let start = Instant::now();
                    AudioRecorder::capture_frame(&*recorder, 0., &frame);
                    recorder.on_playback_progressed(0., 0., samples as f64 / f64::from(RATE));
                    let capture_ms = start.elapsed().as_secs_f64() * 1000.;
                    recorder
                        .flush(samples as f64 / f64::from(RATE))
                        .await
                        .map_err(|e| e.to_string())?;
                    <SessionRecorder as RecorderClass>::finish(
                        &*recorder,
                        Some(samples as f64 / f64::from(RATE)),
                    )
                    .await
                    .map_err(|e| e.to_string())?;
                    Ok::<_, String>((capture_ms, path))
                }));
            }
            paths.clear();
            while let Some(job) = jobs.join_next().await {
                let (result, cpu, maximum) = job?;
                let (capture, path) = result?;
                max_capture_ms = max_capture_ms.max(capture);
                poll_cpu.extend(cpu);
                poll_max_ms = poll_max_ms.max(maximum);
                paths.push(path);
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let lag = stop_heartbeat(stop, heartbeat).await?;
        // Verify final output after capture, outside the filesystem/decoder lag measurement.
        for path in &paths {
            let mut reader = std::fs::File::open(path)?;
            let mut decoded = 0_usize;
            let mut signal = false;
            lk_agents_media::ffmpeg::decode_stream(
                Some("ogg"),
                RATE,
                2,
                lk_agents_media::decoder::DecodeLimits {
                    min_sample_rate: 8000,
                    max_sample_rate: 192000,
                    max_channels: 2,
                    max_streams: 1,
                    max_expansion_ratio: 128,
                    max_samples_per_frame: 192000,
                },
                &mut reader,
                |_, pcm| {
                    decoded += pcm.len() / 2;
                    signal |= pcm.iter().any(|sample| *sample != 0);
                    Ok(())
                },
            )?;
            if decoded.abs_diff(samples) > 1920 || !signal {
                return Err("recorded duration/signal mismatch".into());
            }
        }
        let result = json!({
            "kind": "declared_session_recorder",
            "requested": count,
            "waves": waves,
            "admitted": admitted,
            "refused": refused,
            "poll_cpu": distribution(poll_cpu),
            "poll_max_ms": poll_max_ms,
            "open_max_ms": open_max_ms,
            "open_cpu": distribution(open_cpu),
            "frame_samples": samples,
            "capture_max_ms": max_capture_ms,
            "elapsed_ms": elapsed,
            "loop_lag": lag,
        });
        eprintln!("{result}");
        Ok(result)
    }

    async fn image(
        rgba: &[u8],
        width: u32,
        height: u32,
        count: usize,
        resize: bool,
    ) -> Result<Value, Error> {
        let frame = VideoFrame { width, height, buffer_type: 0, data: rgba.to_vec().into() };
        let options = ImageEncodeOptions {
            format: "JPEG".into(),
            quality: Some(75),
            resize_width: resize.then_some(1024),
            resize_height: resize.then_some(1024),
            resize_strategy: resize.then(|| "center_aspect_fit".into()),
        };
        let workers = lk_agents_runtime::process::workers()?;
        let images: std::sync::Arc<
            dyn lk_agents_abi::llm::provider_format::ImageResolver + Send + Sync,
        > = std::sync::Arc::new(lk_agents_media::images::Images::default());
        let (stop, heartbeat, observations) = start_heartbeat().await;
        let mut encoded = 0;
        let mut waves = 0;
        let mut poll_cpu = Vec::new();
        let mut poll_max_ms = 0_f64;
        let mut max_encode_ms = 0_f64;
        while observations.load(Ordering::Acquire) < OBSERVATIONS {
            waves += 1;
            for _ in 0..count {
                let start = Instant::now();
                let (prepared, cpu, maximum) =
                    measure_polls(lk_agents_runtime::images::prepare_on(
                        &workers,
                        images.clone(),
                        ImageInput::Frame(frame.clone()),
                        options.clone(),
                    ))
                    .await;
                encoded += prepared?.0.ok_or("image preparation returned no bytes")?.len();
                poll_cpu.extend(cpu);
                poll_max_ms = poll_max_ms.max(maximum);
                max_encode_ms = max_encode_ms.max(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        let lag = stop_heartbeat(stop, heartbeat).await?;
        let result = json!({
            "kind": "admitted_image_encoder",
            "width": width,
            "height": height,
            "count": count,
            "waves": waves,
            "resize": resize,
            "encoded_bytes": encoded,
            "encode_max_ms": max_encode_ms,
            "poll_cpu": distribution(poll_cpu),
            "poll_max_ms": poll_max_ms,
            "loop_lag": lag,
        });
        eprintln!("{result}");
        Ok(result)
    }

    fn wav_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(44 + SAMPLES * 2);
        bytes.extend(b"RIFF");
        bytes.extend((36 + SAMPLES as u32 * 2).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16_u32.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(RATE.to_le_bytes());
        bytes.extend((RATE * 2).to_le_bytes());
        bytes.extend(2_u16.to_le_bytes());
        bytes.extend(16_u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend((SAMPLES as u32 * 2).to_le_bytes());
        bytes.extend((0..SAMPLES).flat_map(|n| ((n % 32768) as i16).to_le_bytes()));
        bytes
    }
}
