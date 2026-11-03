use std::sync::{Arc, Weak};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::media::emitter::EmitterError;
use lk_agents_abi::media::emitter::{AudioEmitter as _, EmitterOutput};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{AudioFrame, TranscriptChunk};
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;

use super::{AudioEmitter, EmitterEnv};

/// One `SynthesizedAudio` as the host channel receives it.
struct Sent {
    samples: u32,
    channels: u16,
    pcm: Vec<i16>,
    is_final: bool,
}

#[derive(Default)]
struct Output {
    sent: Mutex<Vec<Sent>>,
    ids: Mutex<Vec<Vec<String>>>,
    /// Called back from inside `send`, as a consumer's callback chain can.
    reentry: Mutex<Option<Weak<AudioEmitter>>>,
}

impl EmitterOutput for Output {
    fn send(
        &self,
        frame: AudioFrame,
        _request_id: String,
        _segment_id: String,
        is_final: bool,
        _timed_transcript: Vec<TranscriptChunk>,
        _usage: Option<TokenUsage>,
        _characters_count: Option<u64>,
    ) {
        self.sent.lock().push(Sent {
            samples: frame.samples_per_channel,
            channels: frame.num_channels,
            pcm: lk_agents_media::pcm_samples(&frame.data),
            is_final,
        });
        let emitter = self.reentry.lock().as_ref().and_then(Weak::upgrade);
        if let Some(emitter) = emitter {
            emitter.num_segments();
            emitter.flush().unwrap();
        }
    }

    fn provider_request_ids(&self, ids: Vec<String>) {
        self.ids.lock().push(ids);
    }

    fn closed(&self) -> Result<bool, HostError> {
        Ok(false)
    }
}

impl Output {
    fn real_and_finals(&self) -> (u32, usize) {
        let sent = self.sent.lock();
        let real = sent.iter().filter(|s| !s.is_final).map(|s| s.samples).sum();
        (real, sent.iter().filter(|s| s.is_final).count())
    }

    /// Every sample sent, final frame included, and whether the final came last and only once.
    fn total_and_final_last(&self) -> (u32, bool) {
        let sent = self.sent.lock();
        let finals = sent.iter().filter(|s| s.is_final).count();
        (sent.iter().map(|s| s.samples).sum(), finals == 1 && sent.last().unwrap().is_final)
    }
}

fn emitter(output: &Arc<Output>) -> AudioEmitter {
    let env = EmitterEnv {
        spawner: Arc::new(TokioSpawner::current()),
        clock: Arc::new(RealClock::new()),
    };
    AudioEmitter::with_env("test".into(), output.clone(), env)
}

fn initialize(emitter: &AudioEmitter, mime: &str, sample_rate: u32, stream: bool) {
    emitter.initialize("req".into(), sample_rate, 1, mime.into(), 100, stream).unwrap();
}

// Catches: a lock held while effects reach the host, which deadlocks a consumer that calls back
// into the emitter, or a re-entrant command dropped or applied twice.
#[tokio::test]
async fn a_consumer_re_entering_from_send_is_queued_behind_the_delivery() {
    let output = Arc::new(Output::default());
    let emitter = Arc::new(emitter(&output));
    *output.reentry.lock() = Some(Arc::downgrade(&emitter));
    initialize(&emitter, "audio/pcm", 1000, true);
    emitter.start_segment("s1".into()).unwrap();
    emitter.push(&[0xff; 600]).unwrap();
    emitter.end_segment().unwrap();
    assert_eq!(output.real_and_finals(), (300, 1), "every sample once, one final marker");
}

// Catches: a machine error discarding the effects its call already produced, or the provider id
// noted before a rejected `start_segment` never reaching the span.
#[tokio::test]
async fn effects_before_a_machine_error_are_delivered_and_the_error_follows() {
    let output = Arc::new(Output::default());
    let emitter = emitter(&output);
    initialize(&emitter, "audio/pcm", 1000, true);
    emitter.start_segment("p1".into()).unwrap();
    emitter.push(&[0xff; 600]).unwrap();
    let error = emitter.start_segment("p2".into()).unwrap_err();
    assert_eq!(error, EmitterError::SegmentAlreadyOpen);
    assert!(output.real_and_finals().0 > 0, "the frames from the push reached the host");
    assert_eq!(
        *output.ids.lock(),
        vec![vec!["p1".to_owned()], vec!["p1".to_owned(), "p2".to_owned()]],
        "each new id is written once, including the one noted before the rejection"
    );
}

// Catches: accepting a mime type nothing can decode, so container bytes are framed as noise.
#[tokio::test]
async fn a_mime_type_no_decoder_understands_is_refused() {
    let emitter = emitter(&Arc::new(Output::default()));
    let error = emitter
        .initialize("req".into(), 1000, 1, "audio/x-not-a-real-container".into(), 100, false)
        .unwrap_err();
    assert!(matches!(error, EmitterError::UnsupportedMime(_)), "{error:?}");
}

// Catches: a closed emitter re-initialized, or native audio framed after close.
#[tokio::test]
async fn a_closed_synthesis_stays_closed() {
    let output = Arc::new(Output::default());
    let emitter = emitter(&output);
    initialize(&emitter, "audio/pcm", 1000, false);
    emitter.aclose().await.unwrap();
    let error =
        emitter.initialize("again".into(), 1000, 1, "audio/pcm".into(), 100, false).unwrap_err();
    assert_eq!(error, EmitterError::AlreadyStarted);
    emitter.push(&[0; 200]).unwrap();
    assert_eq!(output.real_and_finals(), (0, 0));
}

// Catches: a WAV segment ended without its resampler's tail (the last milliseconds of every
// sentence cut), or the tail framed after the final marker; and a header split across pushes
// played as audio.
#[tokio::test]
async fn wav_is_resampled_and_its_tail_framed_before_the_segment_ends() {
    let output = Arc::new(Output::default());
    let emitter = emitter(&output);
    initialize(&emitter, "audio/wav", 24_000, true);
    let samples: Vec<i16> = (0..1600).map(|i| ((i % 64) * 256) as i16).collect();
    let file = lk_agents_media::wav::encode(16_000, 1, &lk_agents_media::pcm_bytes(&samples));
    emitter.start_segment("s".into()).unwrap();
    for chunk in file.chunks(7) {
        emitter.push(chunk).unwrap();
    }
    emitter.end_segment().unwrap();
    let (total, final_last) = output.total_and_final_last();
    assert!(final_last, "one final, after all the audio");
    assert!(total.abs_diff(2400) <= 10, "100 ms at 24 kHz, the resampler drained: {total}");
}

// Catches: a WAV framed in its own channel count rather than the emitter's: a stereo file into a
// mono emitter played every other sample as the next, twice as long at half the pitch.
#[tokio::test]
async fn a_wav_is_framed_in_the_emitters_channel_count() {
    let mono: Vec<i16> = (0..2400).map(|i| ((i % 64) * 256) as i16).collect();
    let stereo: Vec<i16> = mono.iter().flat_map(|s| [s + 100, s - 100]).collect();
    let doubled: Vec<i16> = mono.iter().flat_map(|s| [*s, *s]).collect();
    for (file, file_channels, channels, want) in [(&stereo, 2, 1, &mono), (&mono, 1, 2, &doubled)] {
        let output = Arc::new(Output::default());
        let emitter = emitter(&output);
        emitter.initialize("req".into(), 24_000, channels, "audio/wav".into(), 100, true).unwrap();
        let wav =
            lk_agents_media::wav::encode(24_000, file_channels, &lk_agents_media::pcm_bytes(file));
        emitter.start_segment("s".into()).unwrap();
        emitter.push(&wav).unwrap();
        emitter.end_segment().unwrap();
        let sent = output.sent.lock();
        assert!(sent.iter().all(|s| s.channels == channels), "{file_channels} -> {channels}");
        let samples: u32 = sent.iter().map(|s| s.samples).sum();
        assert_eq!(samples, 2400, "{file_channels} -> {channels}: 100 ms at 24 kHz");
        let pcm: Vec<i16> = sent.iter().flat_map(|s| s.pcm.iter().copied()).collect();
        assert_eq!(&pcm, want, "{file_channels} -> {channels}");
    }
}

// Catches: bytes that are not a WAV file framed as PCM noise instead of failing the push.
#[tokio::test]
async fn a_push_that_is_not_wav_fails_instead_of_playing_noise() {
    let output = Arc::new(Output::default());
    let emitter = emitter(&output);
    initialize(&emitter, "audio/wav", 24_000, false);
    let error = emitter.push(b"ID3\x04\x00\x00\x00\x00\x00\x00 not a riff file").unwrap_err();
    assert!(matches!(error, EmitterError::Decode(_)), "{error:?}");
    assert_eq!(output.real_and_finals(), (0, 0));
}

#[cfg(feature = "ffmpeg")]
fn opus(samples: usize) -> Vec<u8> {
    let mut encoder = lk_agents_media::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
    let mut encoded = encoder.push(&vec![100; samples]).unwrap();
    encoded.extend(encoder.finish().unwrap());
    encoded
}

// Catches: close returning while a decoder still runs, or audio framed after close.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn close_settles_a_decoding_segment_and_frames_nothing_after() {
    use lk_agents_media::pool::{MediaPool, MediaPoolLimits};
    let media_pool = MediaPool::new(MediaPoolLimits { decoders: 1, ..Default::default() }).unwrap();
    let output = Arc::new(Output::default());
    let env = EmitterEnv {
        spawner: Arc::new(TokioSpawner::current()),
        clock: Arc::new(RealClock::new()),
    };
    let mut emitter = AudioEmitter::with_env("test".into(), output.clone(), env);
    Arc::get_mut(&mut emitter.0).unwrap().media_pool = media_pool.clone();
    initialize(&emitter, "audio/ogg", 48_000, false);
    // A partial container leaves its decoder waiting for input.
    emitter.push(&opus(4800)[..64]).unwrap();
    emitter.end_input().unwrap();
    emitter.aclose().await.unwrap();
    let after = output.sent.lock().len();
    let _slot = media_pool
        .try_admit(lk_agents_media::pool::JobClass::Decoder)
        .expect("close releases the decoder's admission");
    tokio::task::yield_now().await;
    assert_eq!(output.sent.lock().len(), after);
}
