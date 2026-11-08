use super::*;
use crate::ffmpeg::audio_decoder::tests::{decode, sine, tone_strength};

// libopus has no AV_CODEC_CAP_SMALL_LAST_FRAME, so a partial final packet must be padded and
// encoded; comparing against the same audio without the partial frame isolates it.
#[test]
fn finish_encodes_the_partial_last_packet_instead_of_dropping_it() {
    fn encode(samples: usize) -> Vec<u8> {
        let mut e = OggOpusEncoder::new(48_000, 1, 32_000).unwrap();
        let mut b = e.push(&sine(48_000, 440.0, samples)).unwrap();
        b.extend(e.finish().unwrap());
        b
    }
    let frame = OggOpusEncoder::new(48_000, 1, 32_000).unwrap().frame_size();
    assert_eq!(frame, 960, "libopus at 48 kHz should give 20 ms packets");

    let whole = encode(frame * 4);
    let with_tail = encode(frame * 4 + frame / 2);
    assert!(
        with_tail.len() > whole.len(),
        "the trailing half packet produced no extra bytes: {} vs {}",
        with_tail.len(),
        whole.len()
    );

    let only_tail = encode(frame / 2);
    assert_eq!(&only_tail[..4], b"OggS");
    assert!(only_tail.windows(8).any(|w| w == b"OpusHead"));
}

// A wrong channel layout, sample format or byte order still produces a valid ogg file, so only
// decoding it back finds them.
#[test]
fn pcm_survives_an_encode_and_decode_round_trip() {
    let src = sine(48_000, 440.0, 48_000 / 2); // 0.5 s mono
    let mut e = OggOpusEncoder::new(48_000, 1, 64_000).unwrap();
    let mut ogg = e.push(&src).unwrap();
    ogg.extend(e.finish().unwrap());

    let (back, _) = decode("audio/ogg", Some(48_000), 1, &ogg, ogg.len()).unwrap();

    // Opus is lossy and adds ~6.5 ms of algorithmic delay, so the length is close, not equal.
    let ratio = back.len() as f64 / src.len() as f64;
    assert!((0.9..1.3).contains(&ratio), "length ratio {ratio}");
    assert!(tone_strength(&back, 48_000, 440.0) > 0.8, "the tone did not survive");
    assert!(tone_strength(&back, 48_000, 1000.0) < 0.2, "energy appeared where there was none");
}

// Five seconds, because ffmpeg's ogg muxer batches a page per 1 s `page_duration`, so a
// one-second test cannot tell streaming from buffered.
#[test]
fn the_encoder_emits_pages_as_it_goes_rather_than_only_at_finish() {
    let mut e = OggOpusEncoder::new(48_000, 1, 64_000).unwrap();
    let mut during = 0usize;
    for _ in 0..250 {
        during += e.push(&sine(48_000, 440.0, 960)).unwrap().len();
    }
    let at_end = e.finish().unwrap().len();
    assert!(during > 0, "nothing was emitted until finish");
    // One page (the last second) is expected at `finish`; 3:1 leaves room for the page boundary.
    assert!(
        during > 3 * at_end,
        "{during} bytes during vs {at_end} at finish -- the muxer is buffering the whole stream"
    );
}
