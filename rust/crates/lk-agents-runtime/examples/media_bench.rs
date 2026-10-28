//! Run with `cargo run -p lk-agents-runtime --features media-images,media-ffmpeg,lk-agents-storage/sqlite
//! --example media_bench -- output.json [iterations] [all|images|codecs|recording]`.
//! Encodes identical JPEG frames inline and through
//! admitted media workers while a current-thread Tokio runtime services a 1 ms heartbeat.

#[cfg(not(feature = "media-images"))]
fn main() {
    eprintln!("media_bench requires --features media-images");
    std::process::exit(2);
}

#[cfg(feature = "media-images")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::run()
}

#[cfg(feature = "media-images")]
#[path = "media_bench/codecs.rs"]
mod codecs;

#[cfg(all(feature = "media-images", feature = "media-ffmpeg"))]
#[path = "media_bench/recording.rs"]
mod recording;

#[cfg(feature = "media-images")]
mod bench {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::time::Duration;

    use lk_agents_media::images::{
        EncodeOptions, ResizeOptions, ResizeStrategy, RgbaFrame, encode,
    };
    use lk_agents_runtime::images::{ImageEncoder, ImageLimits, RgbaImage};
    use lk_agents_runtime::media_workers::{MediaWorkerLimits, MediaWorkers};
    use serde::Serialize;
    use tokio::sync::oneshot;
    use tokio::time::Instant;

    const WORKERS: usize = 8;
    const BLOCKING_THREADS: usize = 34;
    const HEARTBEAT: Duration = Duration::from_millis(1);

    #[derive(Serialize)]
    pub(crate) struct Distribution {
        pub(crate) count: usize,
        pub(crate) p99_ms: f64,
        pub(crate) p999_ms: f64,
        pub(crate) max_ms: f64,
        buckets_ms: std::collections::BTreeMap<&'static str, usize>,
    }

    impl Distribution {
        pub(crate) fn from_ns(mut values: Vec<u64>) -> Result<Self, &'static str> {
            if values.is_empty() {
                return Err("empty measurement");
            }
            values.sort_unstable();
            let mut buckets_ms = std::collections::BTreeMap::new();
            for value in &values {
                let label = [
                    (10_000, "0.01"),
                    (100_000, "0.1"),
                    (500_000, "0.5"),
                    (1_000_000, "1"),
                    (2_000_000, "2"),
                    (5_000_000, "5"),
                    (10_000_000, "10"),
                    (20_000_000, "20"),
                ]
                .into_iter()
                .find(|(limit, _)| value <= limit)
                .map_or(">20", |(_, label)| label);
                *buckets_ms.entry(label).or_default() += 1;
            }
            let quantile = |percent: f64| {
                values[(values.len() as f64 * percent).ceil() as usize - 1] as f64 / 1e6
            };
            Ok(Self {
                count: values.len(),
                p99_ms: quantile(0.99),
                p999_ms: quantile(0.999),
                max_ms: values[values.len() - 1] as f64 / 1e6,
                buckets_ms,
            })
        }
    }

    #[derive(Debug, Clone, Copy, Serialize)]
    enum Mode {
        Inline,
        Worker,
    }

    #[derive(Serialize)]
    struct Case {
        mode: Mode,
        width: u32,
        height: u32,
        resize_to: Option<(u32, u32)>,
        concurrent_images: usize,
        completed: usize,
        elapsed_ms: f64,
        encode_wall: Distribution,
        loop_lag: Distribution,
        gates: Vec<Gate>,
    }

    #[derive(Clone)]
    struct ImageWorkload {
        width: u32,
        height: u32,
        resize_to: Option<(u32, u32)>,
        concurrent_images: usize,
        source: Arc<Vec<u8>>,
        options: EncodeOptions,
        expected_bytes: usize,
    }

    #[derive(Serialize)]
    struct Gate {
        cadence_ms: u64,
        lag_p999_limit_ms: f64,
        lag_max_exclusive_ms: f64,
        passed: bool,
    }

    #[derive(Serialize)]
    struct Provenance {
        head_at_capture: String,
        executable_git_blob: String,
        compiled_source_git_blobs: Vec<(&'static str, String)>,
        debug_assertions: bool,
        os: &'static str,
        arch: &'static str,
        available_parallelism: usize,
    }

    #[derive(Serialize)]
    struct Capture {
        provenance: Provenance,
        heartbeat_ms: u64,
        image_worker_limit: usize,
        blocking_thread_limit: usize,
        iterations: usize,
        captured_scope: String,
        workload: &'static str,
        evidence_limits: [&'static str; 4],
        cases: Vec<Case>,
        images_status: &'static str,
        codecs: serde_json::Value,
        recording: serde_json::Value,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let output =
            PathBuf::from(args.next().ok_or("usage: media_bench output.json [iterations]")?);
        let iterations = args
            .next()
            .map(|value| value.to_string_lossy().parse::<usize>())
            .transpose()?
            .unwrap_or(3);
        let selected = args
            .next()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| "all".to_owned());
        if args.next().is_some()
            || !(1..=100).contains(&iterations)
            || !["all", "images", "codecs", "recording"].contains(&selected.as_str())
        {
            return Err("usage: media_bench output.json [iterations in 1..=100] [all|images|codecs|recording]".into());
        }
        #[cfg(not(feature = "media-ffmpeg"))]
        if ["all", "recording"].contains(&selected.as_str()) {
            return Err("recording capture requires --features media-ffmpeg".into());
        }
        let provenance = provenance()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(BLOCKING_THREADS)
            .build()?;
        let cases = if ["all", "images"].contains(&selected.as_str()) {
            runtime.block_on(measure(iterations))?
        } else {
            Vec::new()
        };
        let codecs = if ["all", "codecs"].contains(&selected.as_str()) {
            runtime.block_on(crate::codecs::measure())?
        } else {
            serde_json::json!({ "status": "not_run" })
        };
        #[cfg(feature = "media-ffmpeg")]
        let recording = if ["all", "recording"].contains(&selected.as_str()) {
            runtime.block_on(crate::recording::measure())?
        } else {
            serde_json::json!({ "status": "not_run" })
        };
        #[cfg(not(feature = "media-ffmpeg"))]
        let recording =
            serde_json::json!({ "status": "not_run", "requires_feature": "media-ffmpeg" });
        let capture = Capture {
            provenance,
            heartbeat_ms: 1,
            image_worker_limit: WORKERS,
            blocking_thread_limit: BLOCKING_THREADS,
            iterations,
            captured_scope: selected,
            workload: "nonuniform deterministic RGBA; JPEG quality 75; full size or 1024x1024 aspect-fit canvas; source copy and RGB conversion included",
            evidence_limits: [
                "Tokio current-thread worker boundary; Python bindings, GIL and RTC SDK conversion are excluded.",
                "No native image decoding, encoded cache or production capacity qualification.",
                "Shared host scheduling and CPU contention affect heartbeat maxima; source preparation and warmup are outside capture.",
                "Gate thresholds match bench/budgets.json: p99.9 <=20% and max <50% of each frame cadence.",
            ],
            images_status: if cases.is_empty() { "not_run" } else { "completed" },
            cases,
            codecs,
            recording,
        };
        let temporary = output.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&capture)?)?;
        std::fs::rename(temporary, output)?;
        Ok(())
    }

    async fn measure(iterations: usize) -> Result<Vec<Case>, Box<dyn std::error::Error>> {
        let workers = MediaWorkers::new(
            tokio::runtime::Handle::current(),
            MediaWorkerLimits { total: WORKERS, decoders: 1, recordings: 1, images: WORKERS },
        )?;
        let encoder = ImageEncoder::new(
            workers,
            ImageLimits {
                source_pixels: 3840 * 2160,
                encoded_pixels: 3840 * 2160,
                encoded_bytes: 32 * 1024 * 1024,
            },
        )?;
        let mut cases = Vec::new();
        for (width, height) in [(1920, 1080), (3840, 2160)] {
            let source = Arc::new(source(width, height));
            for resize in [None, Some((1024, 1024))] {
                let options = EncodeOptions {
                    resize_options: resize.map(|(width, height)| ResizeOptions {
                        width,
                        height,
                        strategy: ResizeStrategy::CenterAspectFit,
                    }),
                    ..EncodeOptions::default()
                };
                // Both measured paths must encode the same content before collecting timings.
                let expected = encode(RgbaFrame { width, height, data: &source }, &options)?;
                let prepared = source.clone();
                let actual = encoder
                    .encode(width, height, options.clone(), move || {
                        Ok(RgbaImage { width, height, data: (*prepared).clone() })
                    })
                    .await?;
                if expected != actual {
                    return Err("worker and inline encoded bytes differ".into());
                }
                let encoded_bytes = expected.len();
                for concurrent in [1, WORKERS] {
                    let workload = ImageWorkload {
                        width,
                        height,
                        resize_to: resize,
                        concurrent_images: concurrent,
                        source: source.clone(),
                        options: options.clone(),
                        expected_bytes: encoded_bytes,
                    };
                    for mode in [Mode::Inline, Mode::Worker] {
                        let case =
                            measure_case(mode, iterations, workload.clone(), encoder.clone())
                                .await?;
                        eprintln!(
                            "{mode:?} {width}x{height} resize={resize:?} concurrent={concurrent}: lag p99.9={:.3}ms max={:.3}ms",
                            case.loop_lag.p999_ms, case.loop_lag.max_ms,
                        );
                        cases.push(case);
                    }
                }
            }
        }
        Ok(cases)
    }

    async fn measure_case(
        mode: Mode,
        iterations: usize,
        workload: ImageWorkload,
        encoder: ImageEncoder,
    ) -> Result<Case, Box<dyn std::error::Error>> {
        let ImageWorkload {
            width,
            height,
            resize_to,
            concurrent_images,
            source,
            options,
            expected_bytes,
        } = workload;
        let (stop, stopped) = oneshot::channel();
        let (ready, started) = oneshot::channel();
        let heartbeat = tokio::spawn(heartbeat(stopped, ready));
        started.await?;
        let start = Instant::now();
        let mut durations = Vec::with_capacity(concurrent_images * iterations);
        for _ in 0..iterations {
            let mut jobs = tokio::task::JoinSet::new();
            for _ in 0..concurrent_images {
                let source = source.clone();
                let options = options.clone();
                let encoder = encoder.clone();
                jobs.spawn(async move {
                    let start = Instant::now();
                    let bytes = match mode {
                        Mode::Inline => {
                            let prepared = (*source).clone();
                            encode(RgbaFrame { width, height, data: &prepared }, &options)
                                .map_err(|error| error.to_string())?
                        }
                        Mode::Worker => encoder
                            .encode(width, height, options, move || {
                                Ok(RgbaImage { width, height, data: (*source).clone() })
                            })
                            .await
                            .map_err(|error| error.to_string())?,
                    };
                    if bytes.len() != expected_bytes {
                        return Err("encoded output length changed".to_owned());
                    }
                    Ok(start.elapsed().as_nanos() as u64)
                });
            }
            while let Some(job) = jobs.join_next().await {
                durations.push(job??);
            }
            // Give the heartbeat an observation after the final synchronous encode returns.
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let _ = stop.send(());
        let loop_lag = Distribution::from_ns(heartbeat.await?)?;
        let gates = [10, 20]
            .into_iter()
            .map(|cadence_ms| {
                let lag_p999_limit_ms = cadence_ms as f64 * 0.2;
                let lag_max_exclusive_ms = cadence_ms as f64 * 0.5;
                Gate {
                    cadence_ms,
                    lag_p999_limit_ms,
                    lag_max_exclusive_ms,
                    passed: loop_lag.p999_ms <= lag_p999_limit_ms
                        && loop_lag.max_ms < lag_max_exclusive_ms,
                }
            })
            .collect();
        Ok(Case {
            mode,
            width,
            height,
            resize_to,
            concurrent_images,
            completed: durations.len(),
            elapsed_ms,
            encode_wall: Distribution::from_ns(durations)?,
            loop_lag,
            gates,
        })
    }

    pub(crate) async fn heartbeat(
        mut stop: oneshot::Receiver<()>,
        ready: oneshot::Sender<()>,
    ) -> Vec<u64> {
        let mut deadline = Instant::now() + HEARTBEAT;
        let mut samples = Vec::new();
        let _ = ready.send(());
        loop {
            tokio::select! {
                _ = &mut stop => return samples,
                _ = tokio::time::sleep_until(deadline) => {
                    let now = Instant::now();
                    let late = now.saturating_duration_since(deadline);
                    samples.push(late.as_nanos() as u64);
                    let missed = late.as_nanos() / HEARTBEAT.as_nanos();
                    deadline += HEARTBEAT * (missed as u32 + 1);
                }
            }
        }
    }

    fn source(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    (x ^ y) as u8,
                    (x / 3 + y / 7) as u8,
                    (x * 13 + y * 29) as u8,
                    255,
                ]);
            }
        }
        pixels
    }

    fn provenance() -> Result<Provenance, Box<dyn std::error::Error>> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let head = Command::new("git").current_dir(&repo).args(["rev-parse", "HEAD"]).output()?;
        if !head.status.success() {
            return Err("git rev-parse failed".into());
        }
        let executable = Command::new("git")
            .current_dir(&repo)
            .arg("hash-object")
            .arg(std::env::current_exe()?)
            .output()?;
        if !executable.status.success() {
            return Err("executable hashing failed".into());
        }
        let mut sources = Vec::new();
        for (name, data) in [
            ("runtime/images.rs", include_bytes!("../src/images.rs").as_slice()),
            ("runtime/media_workers.rs", include_bytes!("../src/media_workers.rs").as_slice()),
            (
                "media/images/encode.rs",
                include_bytes!("../../lk-agents-media/src/images/encode.rs").as_slice(),
            ),
            ("runtime/examples/media_bench.rs", include_bytes!("media_bench.rs").as_slice()),
            (
                "runtime/examples/media_bench/codecs.rs",
                include_bytes!("media_bench/codecs.rs").as_slice(),
            ),
            ("runtime/codecs.rs", include_bytes!("../src/codecs.rs").as_slice()),
            ("media/decoder.rs", include_bytes!("../../lk-agents-media/src/decoder.rs").as_slice()),
            ("media/wav.rs", include_bytes!("../../lk-agents-media/src/wav.rs").as_slice()),
            #[cfg(feature = "media-ffmpeg")]
            (
                "runtime/examples/media_bench/recording.rs",
                include_bytes!("media_bench/recording.rs").as_slice(),
            ),
            #[cfg(feature = "media-ffmpeg")]
            ("runtime/recording.rs", include_bytes!("../src/recording.rs").as_slice()),
            #[cfg(feature = "media-ffmpeg")]
            (
                "media/recorder.rs",
                include_bytes!("../../lk-agents-media/src/recorder.rs").as_slice(),
            ),
            #[cfg(feature = "media-ffmpeg")]
            (
                "media/ffmpeg/encode.rs",
                include_bytes!("../../lk-agents-media/src/ffmpeg/encode.rs").as_slice(),
            ),
            #[cfg(feature = "media-ffmpeg")]
            (
                "media/ffmpeg.pin",
                include_bytes!("../../lk-agents-media/vendor/ffmpeg.pin").as_slice(),
            ),
            ("Cargo.lock", include_bytes!("../../../Cargo.lock").as_slice()),
        ] {
            let mut hash = Command::new("git")
                .current_dir(&repo)
                .args(["hash-object", "--stdin"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()?;
            hash.stdin.take().ok_or("hash stdin missing")?.write_all(data)?;
            let output = hash.wait_with_output()?;
            if !output.status.success() {
                return Err("source hashing failed".into());
            }
            sources.push((name, String::from_utf8(output.stdout)?.trim().to_owned()));
        }
        Ok(Provenance {
            head_at_capture: String::from_utf8(head.stdout)?.trim().to_owned(),
            executable_git_blob: String::from_utf8(executable.stdout)?.trim().to_owned(),
            compiled_source_git_blobs: sources,
            debug_assertions: cfg!(debug_assertions),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            available_parallelism: std::thread::available_parallelism()?.get(),
        })
    }
}
