use super::*;
use lk_agents_media::workers::MediaWorkerLimits;

fn workers() -> MediaWorkers {
    MediaWorkers::new(MediaWorkerLimits {
        total: 3,
        decoders: 1,
        recordings: 1,
        images: 1,
        formatting: 1,
        eot_models: 1,
        vad_models: 1,
    })
    .unwrap()
}

fn limits() -> DecoderLimits {
    DecoderLimits {
        format: DecodeLimits {
            min_sample_rate: 8000,
            max_sample_rate: 192000,
            max_channels: 8,
            max_streams: 8,
            max_expansion_ratio: 24,
            max_samples_per_frame: 65536,
        },
        output_samples: 16 * 1024,
        output_chunk_samples: 128,
    }
}

fn wav(channels: u16, samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + samples.len() as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&24000u32.to_le_bytes());
    bytes.extend_from_slice(&(48000 * u32::from(channels)).to_le_bytes());
    bytes.extend_from_slice(&(2 * channels).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
    bytes.extend(lk_agents_media::pcm_bytes(samples));
    bytes
}

struct AudioFile(std::path::PathBuf);

impl AudioFile {
    fn wav(name: &str, pcm: &[i16]) -> Self {
        let path =
            std::env::temp_dir().join(format!("lk-decoder-{name}-{}.wav", std::process::id()));
        std::fs::write(&path, wav(1, pcm)).unwrap();
        Self(path)
    }

    fn path(&self) -> String {
        self.0.to_str().unwrap().to_owned()
    }
}

impl Drop for AudioFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[tokio::test]
async fn decoded_stereo_is_normalized_before_delivery_to_a_mono_input() {
    let file = AudioFile::wav("stereo-normalization", &[]);
    std::fs::write(&file.0, wav(2, &[100, 200].repeat(480))).unwrap();
    let mut decoder =
        AudioDecoder::open_file(&workers(), file.path(), false, 24_000, 1, limits()).unwrap();
    let mut samples = Vec::new();
    while let Some(frame) = decoder.next().await.unwrap() {
        assert_eq!((frame.sample_rate, frame.num_channels), (24_000, 1));
        samples.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
    }
    assert_eq!(samples, vec![150; 480]);
    decoder.close().await.unwrap();
}

fn byte_options(data: Vec<u8>) -> ByteDecoderOptions {
    ByteDecoderOptions {
        mime: "audio/pcm".into(),
        data,
        pcm_format: Some((24_000, 1)),
        sample_rate: 24_000,
        num_channels: 1,
        max_input_bytes: 2 * 1024 * 1024,
        max_duration: Duration::from_secs(10),
    }
}

#[tokio::test]
async fn buffered_pcm_is_normalized_on_the_admitted_decoder_worker() {
    for (source_rate, target_rate) in
        [(24_000, 24_000), (48_000, 24_000), (24_000, 96_000), (24_000, 192_000)]
    {
        let options = ByteDecoderOptions {
            data: lk_agents_media::pcm_bytes(&[100, 200].repeat(source_rate / 10)),
            pcm_format: Some((source_rate as u32, 2)),
            sample_rate: target_rate,
            ..byte_options(Vec::new())
        };
        let mut decoder = AudioDecoder::open_bytes(&workers(), options, limits()).unwrap();
        let mut samples = Vec::new();
        while let Some(frame) = decoder.next().await.unwrap() {
            assert_eq!((frame.sample_rate, frame.num_channels), (target_rate, 1));
            samples.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        }
        assert_eq!(
            samples.len(),
            target_rate as usize / 10,
            "normalization must preserve the head's duration"
        );
        if source_rate as u32 == target_rate {
            assert_eq!(samples, vec![150; 2400]);
        }
        decoder.close().await.unwrap();
    }
}

#[tokio::test]
async fn buffered_decoder_refuses_encoded_and_decoded_excess_without_delivering_it() {
    let options = ByteDecoderOptions { max_input_bytes: 4, ..byte_options(vec![0; 6]) };
    let workers = workers();
    assert!(matches!(
        AudioDecoder::open_bytes(&workers, options, limits()),
        Err(DecoderError::Overflow("encoded bytes"))
    ));
    let options = ByteDecoderOptions {
        max_duration: Duration::from_millis(10),
        ..byte_options(lk_agents_media::pcm_bytes(&vec![100; 480]))
    };
    let mut decoder = AudioDecoder::open_bytes(&workers, options, limits()).unwrap();
    let mut samples = 0;
    loop {
        match decoder.next().await {
            Ok(Some(frame)) => samples += frame.samples_per_channel,
            Err(DecoderError::Overflow("decoded duration")) => break,
            result => panic!("expected an explicit duration overflow, got {result:?}"),
        }
    }
    assert!(samples <= 240, "excess decoded audio must never enter the output queue");
    decoder.close().await.unwrap();
    let mut valid =
        AudioDecoder::open_bytes(&workers, byte_options(vec![0; 480]), limits()).unwrap();
    assert!(valid.next().await.unwrap().is_some(), "refused heads must release decoder admission");
    valid.close().await.unwrap();
}

#[cfg(feature = "media-ffmpeg")]
#[tokio::test]
async fn compressed_heads_stop_before_retaining_audio_beyond_the_duration_limit() {
    let mut encoder = lk_agents_media::ffmpeg::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
    let mut data = encoder.push(&vec![100; 4800]).unwrap();
    data.extend(encoder.finish().unwrap());
    let options = ByteDecoderOptions {
        mime: "audio/ogg;codecs=opus".into(),
        data,
        pcm_format: None,
        max_duration: Duration::from_millis(20),
        ..byte_options(Vec::new())
    };
    let mut decoder = AudioDecoder::open_bytes(&workers(), options, limits()).unwrap();
    let mut samples = 0;
    loop {
        match decoder.next().await {
            Ok(Some(frame)) => samples += frame.samples_per_channel,
            Err(DecoderError::Overflow("decoded duration")) => break,
            result => panic!("compressed duration excess must be explicit: {result:?}"),
        }
    }
    assert!(samples <= 480);
    decoder.close().await.unwrap();
}

#[tokio::test]
async fn background_file_service_preserves_audio_and_reports_file_and_codec_failures() {
    let pcm = (0..48_000).map(|sample| (sample % 16_000) as i16).collect::<Vec<_>>();
    let file = AudioFile::wav("service", &pcm);
    let mut decoder =
        lk_agents_abi::media::decoder::open(lk_agents_abi::media::decoder::FileDecoderOptions {
            path: file.path(),
            looping: false,
            sample_rate: 24_000,
            num_channels: 1,
        })
        .unwrap();
    let mut actual = Vec::new();
    while let Some(frame) = decoder.next().await.unwrap() {
        assert_eq!((frame.sample_rate, frame.num_channels), (24_000, 1));
        actual.extend(
            frame.data.chunks_exact(2).map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
        );
    }
    assert_eq!(actual, pcm);
    decoder.close(Duration::from_secs(1)).await.unwrap();
    let mut missing =
        lk_agents_abi::media::decoder::open(lk_agents_abi::media::decoder::FileDecoderOptions {
            path: format!("{}.missing", file.path()),
            looping: false,
            sample_rate: 24_000,
            num_channels: 1,
        })
        .unwrap();
    assert!(missing.next().await.unwrap_err().contains("No such file"));
    missing.close(Duration::from_secs(1)).await.unwrap();

    let garbage = AudioFile::wav("garbage", &[]);
    std::fs::write(&garbage.0, b"not audio at all").unwrap();
    let mut unreadable =
        lk_agents_abi::media::decoder::open(lk_agents_abi::media::decoder::FileDecoderOptions {
            path: garbage.path(),
            looping: false,
            sample_rate: 24_000,
            num_channels: 1,
        })
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), unreadable.next())
        .await
        .expect("garbage input must settle its pull")
        .unwrap_err();
    #[cfg(feature = "media-ffmpeg")]
    assert!(error.starts_with("ffmpeg:"), "expected codec failure, got {error}");
    #[cfg(not(feature = "media-ffmpeg"))]
    assert!(error.starts_with("invalid WAV file:"), "expected codec failure, got {error}");
    unreadable.close(Duration::from_secs(1)).await.unwrap();
}

#[tokio::test]
async fn a_stream_keeps_its_slot_until_cancellation_settles_then_refuses_input() {
    let workers = workers();
    let open = || AudioDecoder::open_stream(&workers, "audio/wav", Some(24_000), 1, limits());
    let (mut first, input) = open().unwrap();
    // A header alone leaves the native reader waiting for more input.
    input.push(&wav(1, &[1; 480])[..44]).unwrap();
    assert!(matches!(
        open(),
        Err(DecoderError::WorkerAdmission(MediaWorkerError::Exhausted(MediaWorkerClass::Decoder)))
    ));
    first.close().await.unwrap();
    assert!(matches!(input.push(&[0; 4]), Err(DecoderError::InputClosed)));

    let (mut second, input) = open().unwrap();
    let encoded = wav(1, &[7; 480]);
    for chunk in encoded.chunks(37) {
        input.push(chunk).unwrap();
    }
    input.end_input();
    let mut samples = Vec::new();
    while let Some(frame) = second.next().await.unwrap() {
        samples.extend(frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
    }
    assert_eq!(samples, vec![7; 480]);
    assert!(matches!(input.push(&[0; 4]), Err(DecoderError::InputClosed)));
    second.close().await.unwrap();
}

#[tokio::test]
async fn looping_file_backpressure_keeps_its_slot_until_cancelled_decode_settles() {
    let pcm = (0..48_000).map(|sample| (sample % 16_000) as i16).collect::<Vec<_>>();
    let file = AudioFile::wav("looping", &pcm);
    let workers = workers();
    let limits = DecoderLimits { output_samples: 4096, ..limits() };
    let mut decoder =
        AudioDecoder::open_file(&workers, file.path(), true, 24_000, 1, limits).unwrap();
    let mut actual = Vec::new();
    while actual.len() < pcm.len() + 4096 {
        let frame = decoder.next().await.unwrap().unwrap();
        actual.extend(
            frame.data.chunks_exact(2).map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
        );
    }
    assert_eq!(&actual[..pcm.len()], pcm);
    assert_eq!(&actual[pcm.len()..], &pcm[..actual.len() - pcm.len()]);
    assert!(matches!(
        AudioDecoder::open_file(&workers, file.path(), false, 24_000, 1, limits),
        Err(DecoderError::WorkerAdmission(MediaWorkerError::Exhausted(MediaWorkerClass::Decoder)))
    ));
    decoder.close().await.unwrap();
    let mut replacement =
        AudioDecoder::open_file(&workers, file.path(), false, 24_000, 1, limits).unwrap();
    replacement.close().await.unwrap();
}

#[cfg(unix)]
fn thread_cpu() -> Duration {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes to a valid timespec for this thread's CPU clock.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    assert_eq!(status, 0, "thread CPU clock unavailable");
    Duration::new(time.tv_sec as u64, time.tv_nsec as u32)
}

#[cfg(unix)]
#[tokio::test]
async fn ready_file_pcm_yields_to_control_without_dropping_audio() {
    let file = AudioFile::wav("fairness", &[1; 16_000]);
    let limits = DecoderLimits { output_chunk_samples: 1, ..limits() };
    let mut decoder =
        AudioDecoder::open_file(&workers(), file.path(), false, 24_000, 1, limits).unwrap();
    loop {
        let ready = decoder.shared.ready.notified();
        if decoder.shared.try_state().map(|state| state.done).unwrap_or(false) {
            break;
        }
        ready.await;
    }
    let (ready, mut control) = tokio::sync::oneshot::channel();
    let started = thread_cpu();
    let control_task = tokio::spawn(async move {
        let _ = ready.send(thread_cpu() - started);
    });
    let mut delivered = 0;
    let control_cpu = loop {
        assert!(decoder.next().await.unwrap().is_some(), "control must run before EOF");
        delivered += 1;
        if let Ok(elapsed) = control.try_recv() {
            break elapsed;
        }
    };
    assert!(
        control_cpu <= Duration::from_millis(2),
        "ready control exceeded the 2ms synchronous CPU budget: {control_cpu:?}",
    );
    while decoder.next().await.unwrap().is_some() {
        delivered += 1;
    }
    assert_eq!(delivered, 16_000);
    decoder.close().await.unwrap();
    control_task.await.unwrap();
}
