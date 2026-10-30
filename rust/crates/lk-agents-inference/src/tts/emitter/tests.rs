use std::sync::{Arc, Mutex, Weak};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::media::emitter::EmitterError;
use lk_agents_abi::media::emitter::{AudioEmitter as _, EmitterOutput};
use lk_agents_abi::{AudioFrame, TimedString};
use lk_agents_media::pool::MediaPool;
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;

use super::{AudioEmitter, EmitterEnv};

/// One `SynthesizedAudio` as the host channel receives it.
struct Sent {
    samples: u32,
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
        _timed_transcript: Vec<TimedString>,
    ) {
        self.sent.lock().unwrap().push(Sent { samples: frame.samples_per_channel, is_final });
        let emitter = self.reentry.lock().unwrap().as_ref().and_then(Weak::upgrade);
        if let Some(emitter) = emitter {
            emitter.num_segments();
            emitter.flush().unwrap();
        }
    }

    fn span_attribute(&self, key: String, values: Vec<String>) {
        assert_eq!(key, lk_agents_abi::telemetry::attrs::PROVIDER_REQUEST_IDS);
        self.ids.lock().unwrap().push(values);
    }

    fn closed(&self) -> Result<bool, HostError> {
        Ok(false)
    }
}

impl Output {
    fn real_and_finals(&self) -> (u32, usize) {
        let sent = self.sent.lock().unwrap();
        let real = sent.iter().filter(|s| !s.is_final).map(|s| s.samples).sum();
        (real, sent.iter().filter(|s| s.is_final).count())
    }
}

fn emitter(output: &Arc<Output>) -> AudioEmitter {
    let env = EmitterEnv {
        spawner: Arc::new(TokioSpawner::current()),
        clock: Arc::new(RealClock::new()),
        media_pool: MediaPool::new(Default::default()).unwrap(),
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
    *output.reentry.lock().unwrap() = Some(Arc::downgrade(&emitter));
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
        *output.ids.lock().unwrap(),
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

#[cfg(feature = "ffmpeg")]
fn opus(samples: usize) -> Vec<u8> {
    let mut encoder = lk_agents_media::OggOpusEncoder::new(48_000, 1, 24_000).unwrap();
    let mut encoded = encoder.push(&vec![100; samples]).unwrap();
    encoded.extend(encoder.finish().unwrap());
    encoded
}

// Catches: a segment ended before its container finished decoding: the final marker went out
// first and the segment's audio was then refused outside any segment.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn each_compressed_segment_ends_after_its_decoded_audio() {
    let output = Arc::new(Output::default());
    let emitter = emitter(&output);
    initialize(&emitter, "audio/ogg", 48_000, true);
    let encoded = opus(4800);
    for segment in ["a", "b"] {
        emitter.start_segment(segment.into()).unwrap();
        for chunk in encoded.chunks(97) {
            emitter.push(chunk).unwrap();
        }
        emitter.end_segment().unwrap();
    }
    emitter.end_input().unwrap();
    emitter.join().await.unwrap();
    let sent = output.sent.lock().unwrap();
    let finals: Vec<usize> = (0..sent.len()).filter(|&i| sent[i].is_final).collect();
    assert_eq!(finals.len(), 2);
    let first: u32 = sent[..finals[0]].iter().map(|s| s.samples).sum();
    let second: u32 = sent[finals[0] + 1..finals[1]].iter().map(|s| s.samples).sum();
    assert!(first >= 4000, "the first segment's audio precedes its final marker: {first}");
    assert_eq!(first, second, "each segment decodes its own container");
    drop(sent);
    emitter.aclose().await.unwrap();
}

// Catches: close returning while a decoder still runs, or audio framed after close.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn close_settles_a_decoding_segment_and_frames_nothing_after() {
    let media_pool = MediaPool::new(lk_agents_media::pool::MediaPoolLimits {
        decoders: 1,
        ..Default::default()
    })
    .unwrap();
    let output = Arc::new(Output::default());
    let env = EmitterEnv {
        spawner: Arc::new(TokioSpawner::current()),
        clock: Arc::new(RealClock::new()),
        media_pool: media_pool.clone(),
    };
    let emitter = AudioEmitter::with_env("test".into(), output.clone(), env);
    initialize(&emitter, "audio/ogg", 48_000, false);
    // A partial container leaves its decoder waiting for input.
    emitter.push(&opus(4800)[..64]).unwrap();
    emitter.end_input().unwrap();
    emitter.aclose().await.unwrap();
    let after = output.sent.lock().unwrap().len();
    let _slot = media_pool
        .try_admit(lk_agents_media::pool::JobClass::Decoder)
        .expect("close releases the decoder's admission");
    tokio::task::yield_now().await;
    assert_eq!(output.sent.lock().unwrap().len(), after);
}
