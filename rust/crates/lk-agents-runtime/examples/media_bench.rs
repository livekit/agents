//! `cargo run -p lk-agents-runtime --features media-ffmpeg,media-resample,media-images
//! --example media_bench -- output.json [codecs|recording|images|all]`.
//! Measures the registered background decoder, declared recorder implementation and shipped
//! synchronous image encoder against an independent 1ms Tokio heartbeat.

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
    use std::path::Path;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use lk_agents_abi::AudioFrame;
    use lk_agents_abi::recorder::{AudioRecorder, SessionRecorder as RecorderClass};
    use lk_agents_media::images::encode_image;
    use lk_agents_runtime::recording::SessionRecorder;
    use serde_json::{Value, json};
    use tokio::sync::oneshot;

    type Error = Box<dyn std::error::Error>;
    const RATE: u32 = 48_000;
    const SAMPLES: usize = RATE as usize * 10;

    pub fn run() -> Result<(), Error> {
        let mut arguments = std::env::args().skip(1);
        let output = arguments
            .next()
            .ok_or("usage: media_bench output.json [codecs|recording|images|all]")?;
        let selected = arguments.next().unwrap_or_else(|| "all".into());
        if arguments.next().is_some()
            || !["codecs", "recording", "images", "all"].contains(&selected.as_str())
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
                    let rgba = (0..width*height).flat_map(|n| [(n%251) as u8, (n%193) as u8, (n%137) as u8, 255]).collect::<Vec<_>>();
                    for count in [1, 8] {
                        for resize in [false, true] {
                            cases.push(image(&rgba, width, height, count, resize).await?);
                        }
                    }
                }
            }
            Ok::<_, Error>(json!({
                "heartbeat_ms": 1, "cadence_ms": 20,
                "scope": "Shipped native entrypoints: registered background file decoder, ABI recorder implementation, synchronous encode_image. Python/GIL/RTC conversion excluded; loaded shared host; no soak or hardware-profile qualification.",
                "cases": cases
            }))
        });
        std::fs::remove_dir_all(root)?;
        let result = outcome?;
        std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
        Ok(())
    }

    async fn start_heartbeat() -> (oneshot::Sender<()>, tokio::task::JoinHandle<Vec<f64>>) {
        let (stop, mut stopped) = oneshot::channel();
        let (ready, started) = oneshot::channel();
        let task = tokio::spawn(async move {
            let interval = Duration::from_millis(1);
            let mut deadline = tokio::time::Instant::now() + interval;
            let mut samples = Vec::new();
            let _ = ready.send(());
            loop {
                tokio::select! {
                    _ = &mut stopped => return samples,
                    _ = tokio::time::sleep_until(deadline) => {
                        let now = tokio::time::Instant::now();
                        samples.push(now.saturating_duration_since(deadline).as_secs_f64()*1000.);
                        deadline = now + interval;
                    }
                }
            }
        });
        started.await.unwrap();
        (stop, task)
    }

    async fn stop_heartbeat(
        stop: oneshot::Sender<()>,
        task: tokio::task::JoinHandle<Vec<f64>>,
    ) -> Result<Value, Error> {
        tokio::time::sleep(Duration::from_millis(5)).await;
        let _ = stop.send(());
        let mut samples = task.await?;
        if samples.is_empty() {
            return Err("heartbeat collected no observations".into());
        }
        samples.sort_by(f64::total_cmp);
        let p999 = samples[(samples.len() as f64 * 0.999).ceil() as usize - 1];
        let max = samples[samples.len() - 1];
        Ok(json!({"observations":samples.len(),"p999_ms":p999,"max_ms":max,
            "gate_20ms":{"p999_limit_ms":4.,"max_exclusive_ms":10.,"passed":p999<=4. && max<10.}}))
    }

    async fn decode(path: &Path, count: usize) -> Result<Value, Error> {
        let (stop, heartbeat) = start_heartbeat().await;
        let start = Instant::now();
        let mut jobs = tokio::task::JoinSet::new();
        let mut refused = 0;
        for _ in 0..count {
            match lk_agents_abi::media::decoder::open(
                lk_agents_abi::media::decoder::FileDecoderOptions {
                    path: path.to_string_lossy().into_owned(),
                    looping: false,
                    sample_rate: RATE,
                    num_channels: 1,
                },
            ) {
                Ok(mut decoder) => {
                    jobs.spawn(async move {
                        let mut samples = 0;
                        let mut signal = false;
                        while let Some(frame) = decoder.next().await? {
                            samples += frame.samples_per_channel as usize;
                            signal |= frame.data.iter().any(|byte| *byte != 0);
                        }
                        decoder.close(Duration::from_secs(5)).await?;
                        if samples.abs_diff(SAMPLES) > 1920 || !signal {
                            return Err("decoded duration/signal mismatch".to_owned());
                        }
                        Ok::<_, String>(samples)
                    });
                }
                Err(error) if error.contains("capacity exhausted") => refused += 1,
                Err(error) => return Err(error.into()),
            }
        }
        let mut samples = 0;
        while let Some(job) = jobs.join_next().await {
            samples += job??;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let lag = stop_heartbeat(stop, heartbeat).await?;
        let result = json!({"kind":"background_file_decoder","format":path.extension().unwrap().to_string_lossy(),"requested":count,"admitted":count-refused,"refused":refused,"decoded_samples":samples,"elapsed_ms":elapsed,"loop_lag":lag});
        eprintln!("{result}");
        Ok(result)
    }

    async fn record(root: &Path, count: usize, samples: usize) -> Result<Value, Error> {
        let data = lk_agents_media::pcm_bytes(
            &(0..samples).map(|n| ((n * 37) % 20_000) as i16 - 10_000).collect::<Vec<_>>(),
        );
        let frame = AudioFrame::new(data, RATE, 1, samples as u32);
        let (stop, heartbeat) = start_heartbeat().await;
        let start = Instant::now();
        let mut recorders = Vec::new();
        let mut refused = 0;
        for index in 0..count {
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
        }
        let mut max_capture_ms = 0_f64;
        let mut jobs = tokio::task::JoinSet::new();
        for (recorder, path) in recorders {
            let frame = frame.clone();
            jobs.spawn(async move {
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
            });
        }
        let mut paths = Vec::new();
        while let Some(job) = jobs.join_next().await {
            let (capture, path) = job??;
            max_capture_ms = max_capture_ms.max(capture);
            paths.push(path);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let lag = stop_heartbeat(stop, heartbeat).await?;
        // Verify actual final output after capture; filesystem/decoder validation is excluded from lag.
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
        let result = json!({"kind":"declared_session_recorder","requested":count,"admitted":paths.len(),"refused":refused,"frame_samples":samples,"capture_max_ms":max_capture_ms,"elapsed_ms":elapsed,"loop_lag":lag});
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
        let (stop, heartbeat) = start_heartbeat().await;
        let mut encoded = 0;
        let mut max_encode_ms = 0_f64;
        for _ in 0..count {
            let start = Instant::now();
            encoded += encode_image(
                rgba,
                width,
                height,
                "JPEG",
                Some(75),
                resize.then_some(1024),
                resize.then_some(1024),
                resize.then_some("center_aspect_fit"),
            )?
            .len();
            max_encode_ms = max_encode_ms.max(start.elapsed().as_secs_f64() * 1000.);
        }
        let lag = stop_heartbeat(stop, heartbeat).await?;
        let result = json!({"kind":"shipped_sync_image_encoder","width":width,"height":height,"count":count,"resize":resize,"encoded_bytes":encoded,"encode_max_ms":max_encode_ms,"loop_lag":lag});
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
