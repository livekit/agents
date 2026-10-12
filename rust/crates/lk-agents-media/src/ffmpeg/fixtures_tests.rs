//! Tests for the ffmpeg half. `--features ffmpeg` only.
//!
//! # Fixtures
//!
//! No binary files are committed. The two container fixtures are ~800-byte hex string constants
//! below, and the command that produced each is written above it, so they are reproducible and a
//! diff of them is at least readable. Everything else -- WAV, PCM -- is generated in code.
//!
//! They are here rather than generated at test time because this build has no mp3 or matroska
//! *muxer* (§10's list is decode-only for those), so there is nothing to generate them with. That
//! is the right trade: adding two muxers to the shipped binary to make a test convenient would be
//! shipping code for a test's benefit.

use super::*;
use crate::wav::tests::wav_file;
use crate::{AudioFormat, Decoder};

/// 802 bytes. Produced by:
///   ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
///          -ac 1 -c:a libopus -b:a 12k -f webm t.webm
/// This is the container the JS SDK sends for pre-connect audio
/// (`lk-agents-io/src/pre_connect.rs:203`).
const TONE_WEBM_OPUS_HEX: &str = concat!(
    "1a45dfa39f4286810142f7810142f2810442f381084282847765626d42878104428581021853806701000000000002f2",
    "114d9b74ba4dbb8b53ab841549a96653ac81a14dbb8b53ab841654ae6b53ac81d64dbb8c53ab841254c36753ac820140",
    "4dbb8c53ab841c53bb6b53ac8202dcec0100000000000059000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000001549a966b02ad7b1830f42404d808c4c61766636322e332e31303057418c4c",
    "61766636322e332e313030448988405b0000000000001654ae6be5ae010000000000005cd7810173c5887669dc3c0045",
    "094a9c810022b59c83756e648881008686415f4f50555356aa83632ea056bb8404c4b400838102e1919f8101b58840e7",
    "7000000000006264811063a2934f707573486561640101380180bb00000000001254c367fc73739f63c08067c89945a3",
    "87454e434f44455244878c4c61766636322e332e3130307373d763c08b63c5887669dc3c0045094a67c8a245a387454e",
    "434f4445524487954c61766336322e31312e313030206c69626f70757367c8a145a3884455524154494f4e4487933030",
    "3a30303a30302e313038303030303030001f43b6754115e78100a3b68100008048822eb76c56b7f40001e5cd9e014650",
    "8597b73c93c91759c3119a41df009ef2d73ab6238bdfdfec00435d2d70969a7f5664a3a98100158048a48857ac988503",
    "5c2609933127f13a49ec37cca0bdc3e91b9338391aa5c8a4fee9756be0a3aa81002980489c1b52514500ace2d4b4ab98",
    "5e49e54ecb8d67d32730b037701347018b2a9c1c43b6440ff6a3af81003d80489c1b5256ce1fea10cab91c549950b976",
    "4cb91a7c0fb220b2b0650fd8420e29a8681cd42ed479408c90aaa3ae81005180489c1b9f759cfc4933bf4688b68be337",
    "0c9aec6cdd564f9337b56861b3e735ddfdbfd568dcb0ad900f80a0a0a197810065004805eac5f011e788229d7587793b",
    "0a0501ba1d75a28400cdfe601c53bb6b91bb8fb38100b78af78101f18201c1f08103",
);

/// 740 bytes. Produced by:
///   ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
///          -ac 1 -ar 24000 -c:a libmp3lame -b:a 24k t.mp3
/// The container `AudioEmitter` refuses today.
const TONE_MP3_HEX: &str = concat!(
    "49443304000000000022545353450000000e0000034c61766636322e332e3130300000000000000000000000fff384c0",
    "000000000000000000496e666f0000000f00000007000002b80061616161616161616161616161617b7b7b7b7b7b7b7b",
    "7b7b7b7b7b7b9696969696969696969696969696b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0cbcbcbcbcbcbcbcbcbcbcbcbcb",
    "cbe5e5e5e5e5e5e5e5e5e5e5e5e5e5ffffffffffffffffffffffffffff000000004c61766336322e3131000000000000",
    "00000000000024042000000000000002b82bf941ca0000000000000000000000000000000000000000000000fff334c4",
    "000d0842a45f4918021ae5b6d000468d1a3468db485000c0180304c364ed83e181002018489c1f0ff04383efca1c94f0",
    "ff071da03fc0873407f94777f42a4050140b753efff334c4060ea8b680019b5000e988b11ea1a0a1b9f085014d0f70d4",
    "cb4480c04e6df6cf78034012015fc4105f05d10ff8f455221f0fbe5415111efac15111efe0ad709fa87c0800fff334c4",
    "060e30562c01de0000e5ad01006981e85918540f91827068984a8891e4ebdc19ef0879852855182a827182d813180000",
    "79644bd2adce3488efffffa502e7a6037490dbb0fff334c4080e48522c54e7b240a660802186482717dc84244c10446c",
    "ea936d4c5f4244fc80cb14bc0a9d9c3971b961ffd49e9fb7defd4bff208b3ff555f42a7fe2f9c71ac0580607fff334c4",
    "090d685658015d180004061906460b0846448de6edf9064616863b05c6070049ea5db54f03cdcbed83e0f83e0841f838",
    "7386394770c55108204101e12347a1a8eff851b1fff334c40e10c0f284519a7800c6f2cfa55fe4610389dfa5ff1542f0",
    "7fb0c497fe97461d6e75d673ffea851bf6457ffffffff1de3fbbc7ff8902e09b7f28171516174c414d45332efff334c4",
    "060000034801c00000313030555555555555555555555555555555555555555555555555555555555555555555555555",
    "5555555555555555555555555555555555555555",
);

fn unhex(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0);
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// Goertzel: how much energy sits at `freq`, normalised against the signal's own RMS. A decoded
/// tone scores near 1; noise, silence or a wrong-rate playback scores far below.
fn tone_strength(pcm: &[i16], rate: u32, freq: f64) -> f64 {
    if pcm.len() < 64 {
        return 0.0;
    }
    let n = pcm.len();
    let k = (n as f64 * freq / f64::from(rate)).round();
    let w = 2.0 * std::f64::consts::PI * k / n as f64;
    let coeff = 2.0 * w.cos();
    let (mut s0, mut s1, mut s2);
    (s1, s2) = (0.0f64, 0.0f64);
    let mut energy = 0.0f64;
    for &x in pcm {
        let x = f64::from(x);
        energy += x * x;
        s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let mag2 = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    if energy <= 0.0 {
        return 0.0;
    }
    // 2/N normalises a single bin's power against the total.
    (mag2 * 2.0 / (energy * n as f64)).sqrt().min(1.0)
}

fn sine(rate: u32, freq: f64, samples: usize) -> Vec<i16> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (12000.0 * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16
        })
        .collect()
}

/// push in `chunk` sized pieces, then flush; returns everything.
fn decode_chunked(d: &mut FfmpegDecoder, bytes: &[u8], chunk: usize) -> Vec<i16> {
    let mut out = Vec::new();
    for c in bytes.chunks(chunk) {
        out.extend(d.push(c).unwrap());
    }
    out.extend(d.flush().unwrap());
    out
}

// Catches: the ffmpeg path and the WAV fast path disagreeing on the same bytes. They are two
// implementations of one thing, and the fast path is only safe to prefer if they agree; this is
// the test that makes "prefer the fast path" a decision rather than a hope. Exact equality is
// available here because no resampling happens.
#[test]
fn the_ffmpeg_path_and_the_wav_fast_path_decode_the_same_file_identically() {
    let want = sine(24_000, 440.0, 2400);
    let file = wav_file(24_000, 1, &want);

    let mut ff = FfmpegDecoder::new(Some("wav"), 24_000, 1).unwrap();
    let via_ffmpeg = decode_chunked(&mut ff, &file, 4096);

    let mut fast = Decoder::new("audio/wav", 24_000, 1).unwrap();
    let mut via_fast = fast.push(&file).unwrap();
    via_fast.extend(fast.flush().unwrap());

    assert_eq!(via_ffmpeg, want);
    assert_eq!(via_fast, want);
}

// Catches: the exact failure `AudioEmitter::initialize` refuses to risk today -- framing an ID3
// tag as PCM. A decoder that returned the container bytes reinterpreted would score near zero
// here and be a burst of noise in production. Also catches decoding at the wrong rate, which
// moves the tone off 440 Hz.
#[test]
fn an_mp3_decodes_to_the_tone_it_holds_rather_than_to_its_own_bytes() {
    let mp3 = unhex(TONE_MP3_HEX);
    let mut d = Decoder::new("audio/mpeg", 24_000, 1).unwrap();
    let mut pcm = d.push(&mp3).unwrap();
    pcm.extend(d.flush().unwrap());

    assert!(pcm.len() > 1_500, "0.1 s at 24 kHz is ~2400 samples, got {}", pcm.len());
    assert!(pcm.len() < 4_000, "far more audio than the file holds: {}", pcm.len());
    // 24 kbit/s mp3 attenuates hard, so the bar is dominance rather than an absolute: 440 Hz
    // must hold most of the energy and every other bin must be near nothing. Container bytes
    // played as PCM score ~0 at 440 and spread across everything else.
    let at440 = tone_strength(&pcm, 24_000, 440.0);
    assert!(at440 > 0.7, "not a 440 Hz tone: {at440:.3}");
    for f in [110.0, 220.0, 660.0, 880.0, 1760.0] {
        let other = tone_strength(&pcm, 24_000, f);
        assert!(other * 10.0 < at440, "energy at {f} Hz: {other:.3} vs {at440:.3} at 440");
    }
    assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 1)));
}

// Catches: the pre-connect path staying broken. `audio/webm;codecs=opus` is the literal mime type
// the JS SDK sends, parameter and all, and `lk-agents-io` refuses it today. This is end to end
// through the public `Decoder`, mime string included, because the mime string is half the bug.
#[test]
fn the_exact_mime_type_the_js_sdk_sends_for_pre_connect_audio_decodes() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let mut d = Decoder::new("audio/webm;codecs=opus", 48_000, 1).unwrap();
    let mut pcm = d.push(&webm).unwrap();
    pcm.extend(d.flush().unwrap());

    assert!(pcm.len() > 3_000, "0.1 s at 48 kHz is ~4800 samples, got {}", pcm.len());
    assert!(tone_strength(&pcm, 48_000, 440.0) > 0.8, "not a 440 Hz tone");
    assert_eq!(d.output_format(), Some(AudioFormat::new(48_000, 1)));
}

// Catches: a demuxer that is fed in small pieces losing or duplicating audio at the boundaries.
// A network TTS delivers a container in whatever sizes the socket gives, never whole; the split
// sizes here are deliberately not powers of two and not aligned to any frame.
#[test]
fn a_container_delivered_in_ragged_chunks_decodes_to_the_same_audio_as_one_push() {
    let mp3 = unhex(TONE_MP3_HEX);
    let mut whole = FfmpegDecoder::new(Some("mp3"), 24_000, 1).unwrap();
    let baseline = decode_chunked(&mut whole, &mp3, mp3.len());

    for chunk in [1usize, 7, 37, 101, 499] {
        let mut d = FfmpegDecoder::new(Some("mp3"), 24_000, 1).unwrap();
        let got = decode_chunked(&mut d, &mp3, chunk);
        assert_eq!(got, baseline, "chunk size {chunk}");
    }
}

// Catches: resampling silently not happening. A 48 kHz source asked for at 24 kHz must come out
// half as long AND still be a 440 Hz tone -- decimating without filtering would keep the length
// right and move the pitch, which is why both are asserted.
#[test]
fn the_requested_output_rate_is_honoured_rather_than_the_sources() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let mut at48 = Decoder::new("audio/webm", 48_000, 1).unwrap();
    let mut a = at48.push(&webm).unwrap();
    a.extend(at48.flush().unwrap());

    let mut at24 = Decoder::new("audio/webm", 24_000, 1).unwrap();
    let mut b = at24.push(&webm).unwrap();
    b.extend(at24.flush().unwrap());

    let ratio = b.len() as f64 / a.len() as f64;
    assert!((ratio - 0.5).abs() < 0.05, "24 kHz output should be half as long, ratio {ratio}");
    assert!(tone_strength(&b, 24_000, 440.0) > 0.8, "resampling moved the pitch");
}

// Catches: requesting stereo and getting mono (or interleaving one channel twice out of step).
// `av.AudioResampler(layout=...)` does the upmix; the WAV fast path deliberately does not, so
// this pins the difference on the path that does.
#[test]
fn the_requested_channel_count_is_honoured_on_the_ffmpeg_path() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let mut d = Decoder::new("audio/webm", 48_000, 2).unwrap();
    let mut pcm = d.push(&webm).unwrap();
    pcm.extend(d.flush().unwrap());
    assert_eq!(pcm.len() % 2, 0);
    assert_eq!(d.output_format(), Some(AudioFormat::new(48_000, 2)));
    let left: Vec<i16> = pcm.iter().step_by(2).copied().collect();
    let right: Vec<i16> = pcm.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(left, right, "a mono source upmixed to stereo must give identical channels");
    assert!(tone_strength(&left, 48_000, 440.0) > 0.8);
}

// Catches: reporting success on bytes that are not the container they claim to be. The honest
// answer is an error naming the failure; returning empty PCM reads downstream as "the provider
// sent silence" and gets blamed on the provider.
#[test]
fn bytes_that_are_not_the_declared_container_fail_with_a_named_error() {
    let mut d = Decoder::new("audio/mpeg", 24_000, 1).unwrap();
    let _ = d.push(b"this is not an mp3 file, not even slightly, no frame sync anywhere at all");
    let err = d.flush().unwrap_err();
    assert!(matches!(err, CodecError::Ffmpeg { .. }), "{err:?}");
}

// Catches: an ogg stream with no header, or with the header written after the audio. Any player
// rejects that, and it is invisible until someone tries to open the recording -- days later, when
// the call is over.
#[test]
fn the_encoder_writes_a_real_ogg_opus_stream() {
    let mut e = OggOpusEncoder::new(48_000, 2, 64_000).unwrap();
    let pcm = sine(48_000, 440.0, 48_000 * 2); // 1 s stereo, interleaved
    let mut bytes = e.push(&pcm).unwrap();
    bytes.extend(e.finish().unwrap());

    assert_eq!(&bytes[..4], b"OggS", "not an ogg stream");
    assert!(
        bytes.windows(8).any(|w| w == b"OpusHead"),
        "no OpusHead: the identification header is missing"
    );
    assert!(
        bytes.windows(8).any(|w| w == b"OpusTags"),
        "no OpusTags: the comment header is missing, which most players require"
    );
    // 1 s at 64 kbit/s is ~8 kB. An order of magnitude either way means it encoded silence or
    // never encoded at all.
    assert!(bytes.len() > 3_000 && bytes.len() < 30_000, "{} bytes", bytes.len());
}

// Catches: the tail being dropped. libopus has no AV_CODEC_CAP_SMALL_LAST_FRAME, so a partial
// final packet has to be padded and encoded; an implementation that simply discarded `pending`
// would lose up to 20 ms off the end of every recording and no assertion on the file's validity
// would notice. Comparing against the same audio WITHOUT the partial frame is what isolates it.
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

    // And a stream that is nothing BUT a partial packet still yields a playable file.
    let only_tail = encode(frame / 2);
    assert_eq!(&only_tail[..4], b"OggS");
    assert!(only_tail.windows(8).any(|w| w == b"OpusHead"));
}

// Catches: the round trip losing the signal -- a wrong channel layout, a wrong sample format, or
// PCM written to the encoder in the wrong byte order. Each of those still produces a valid ogg
// file, so only decoding it back finds them. This is also the `RecorderIO` path end to end.
#[test]
fn pcm_survives_an_encode_and_decode_round_trip() {
    let src = sine(48_000, 440.0, 48_000 / 2); // 0.5 s mono
    let mut e = OggOpusEncoder::new(48_000, 1, 64_000).unwrap();
    let mut ogg = e.push(&src).unwrap();
    ogg.extend(e.finish().unwrap());

    let mut d = Decoder::new("audio/ogg", 48_000, 1).unwrap();
    let mut back = d.push(&ogg).unwrap();
    back.extend(d.flush().unwrap());

    // Opus is lossy and adds ~6.5 ms of algorithmic delay, so the length is close, not equal.
    let ratio = back.len() as f64 / src.len() as f64;
    assert!((0.9..1.3).contains(&ratio), "length ratio {ratio}");
    assert!(tone_strength(&back, 48_000, 440.0) > 0.8, "the tone did not survive");
    assert!(tone_strength(&back, 48_000, 1000.0) < 0.2, "energy appeared where there was none");
}

// Catches: `push` holding every byte until `finish`, which is what a naive implementation does
// and which defeats the recorder's 2.5 s writer loop -- the whole call would sit in memory and be
// written in one burst at the end. Five seconds is used rather than one because ffmpeg's ogg
// muxer batches a page per `page_duration`, which defaults to 1 s (PyAV produces the same), so a
// one-second test cannot tell "streaming" from "buffered" at all.
#[test]
fn the_encoder_emits_pages_as_it_goes_rather_than_only_at_finish() {
    let mut e = OggOpusEncoder::new(48_000, 1, 64_000).unwrap();
    let mut during = 0usize;
    for _ in 0..250 {
        during += e.push(&sine(48_000, 440.0, 960)).unwrap().len();
    }
    let at_end = e.finish().unwrap().len();
    assert!(during > 0, "nothing was emitted until finish");
    // Exactly one page's worth is expected at `finish` -- the last second, still open when the
    // trailer is written. Anything much beyond that means the stream was buffered whole. At five
    // seconds that puts the bar at 4:1; 3:1 leaves room for the page boundary landing either way.
    assert!(
        during > 3 * at_end,
        "{during} bytes during vs {at_end} at finish -- the muxer is buffering the whole stream"
    );
}
