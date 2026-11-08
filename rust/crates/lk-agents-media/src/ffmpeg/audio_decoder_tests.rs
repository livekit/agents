use super::*;
use crate::wav::tests::wav_file;
use crate::{AudioDecoder, AudioFormat, Container, mime_container};

// The fixtures are hex constants because this build has no mp3 or matroska muxer to generate
// them with; each is reproducible from the command above it.

/// 802 bytes. Produced by:
///   ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.1" \
///          -ac 1 -c:a libopus -b:a 12k -f webm t.webm
/// The container the JS SDK sends for pre-connect audio.
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
pub(crate) fn tone_strength(pcm: &[i16], rate: u32, freq: f64) -> f64 {
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

pub(crate) fn sine(rate: u32, freq: f64, samples: usize) -> Vec<i16> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (12000.0 * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16
        })
        .collect()
}

/// `bytes` as `mime`, read at most `chunk` bytes at a time as a network stream delivers them.
pub(crate) fn decode(
    mime: &str,
    rate: Option<u32>,
    channels: u16,
    bytes: &[u8],
    chunk: usize,
) -> Result<(Vec<i16>, AudioFormat)> {
    struct Chunked<'a>(&'a [u8], usize);
    impl std::io::Read for Chunked<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.0.len().min(self.1).min(out.len());
            out[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }
    let mut pcm = Vec::new();
    let format = decode_stream(
        mime_container(mime),
        rate.unwrap_or(0),
        channels,
        decode_limits(),
        &mut Chunked(bytes, chunk),
        |_, samples| {
            pcm.extend_from_slice(samples);
            Ok(())
        },
    )?;
    Ok((pcm, format))
}

// The WAV fast path is only safe to prefer if it agrees exactly with the ffmpeg path.
#[test]
fn the_ffmpeg_path_and_the_wav_fast_path_decode_the_same_file_identically() {
    let want = sine(24_000, 440.0, 2400);
    let file = wav_file(24_000, 1, &want);

    let (via_ffmpeg, _) = decode("audio/wav", Some(24_000), 1, &file, 4096).unwrap();

    let mut fast = AudioDecoder::wav(Some(24_000), 1, decode_limits()).unwrap();
    let mut via_fast = fast.push(&file).unwrap();
    via_fast.extend(fast.flush().unwrap());

    assert_eq!(via_ffmpeg, want);
    assert_eq!(via_fast, want);
}

// Container bytes read as PCM would score near zero; a wrong rate moves the tone off 440 Hz.
#[test]
fn an_mp3_decodes_to_the_tone_it_holds_rather_than_to_its_own_bytes() {
    let mp3 = unhex(TONE_MP3_HEX);
    let (pcm, format) = decode("audio/mpeg", Some(24_000), 1, &mp3, mp3.len()).unwrap();

    assert!(pcm.len() > 1_500, "0.1 s at 24 kHz is ~2400 samples, got {}", pcm.len());
    assert!(pcm.len() < 4_000, "far more audio than the file holds: {}", pcm.len());
    // 24 kbit/s mp3 attenuates hard, so the bar is dominance rather than an absolute level.
    let at440 = tone_strength(&pcm, 24_000, 440.0);
    assert!(at440 > 0.7, "not a 440 Hz tone: {at440:.3}");
    for f in [110.0, 220.0, 660.0, 880.0, 1760.0] {
        let other = tone_strength(&pcm, 24_000, f);
        assert!(other * 10.0 < at440, "energy at {f} Hz: {other:.3} vs {at440:.3} at 440");
    }
    assert_eq!(format, AudioFormat::new(24_000, 1));
}

// `audio/webm;codecs=opus` is the literal mime type the JS SDK sends, parameter and all.
#[test]
fn the_exact_mime_type_the_js_sdk_sends_for_pre_connect_audio_decodes() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let (pcm, format) =
        decode("audio/webm;codecs=opus", Some(48_000), 1, &webm, webm.len()).unwrap();

    assert!(pcm.len() > 3_000, "0.1 s at 48 kHz is ~4800 samples, got {}", pcm.len());
    assert!(tone_strength(&pcm, 48_000, 440.0) > 0.8, "not a 440 Hz tone");
    assert_eq!(format, AudioFormat::new(48_000, 1));
}

// A network TTS delivers a container in arbitrary sizes; these splits align to nothing.
#[test]
fn a_container_delivered_in_ragged_chunks_decodes_to_the_same_audio_as_one_push() {
    let mp3 = unhex(TONE_MP3_HEX);
    let (baseline, _) = decode("audio/mpeg", Some(24_000), 1, &mp3, mp3.len()).unwrap();

    for chunk in [1usize, 7, 37, 101, 499] {
        let (got, _) = decode("audio/mpeg", Some(24_000), 1, &mp3, chunk).unwrap();
        assert_eq!(got, baseline, "chunk size {chunk}");
    }
}

// Decimating without filtering keeps the length right but moves the pitch, so both are asserted.
#[test]
fn the_requested_output_rate_is_honoured_rather_than_the_sources() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let (a, _) = decode("audio/webm", Some(48_000), 1, &webm, webm.len()).unwrap();
    let (b, _) = decode("audio/webm", Some(24_000), 1, &webm, webm.len()).unwrap();

    let ratio = b.len() as f64 / a.len() as f64;
    assert!((ratio - 0.5).abs() < 0.05, "24 kHz output should be half as long, ratio {ratio}");
    assert!(tone_strength(&b, 24_000, 440.0) > 0.8, "resampling moved the pitch");
}

#[test]
fn the_requested_channel_count_is_honoured_on_the_ffmpeg_path() {
    let webm = unhex(TONE_WEBM_OPUS_HEX);
    let (pcm, format) = decode("audio/webm", Some(48_000), 2, &webm, webm.len()).unwrap();
    assert_eq!(pcm.len() % 2, 0);
    assert_eq!(format, AudioFormat::new(48_000, 2));
    let left: Vec<i16> = pcm.iter().step_by(2).copied().collect();
    let right: Vec<i16> = pcm.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(left, right, "a mono source upmixed to stereo must give identical channels");
    assert!(tone_strength(&left, 48_000, 440.0) > 0.8);
}

// Empty PCM would read downstream as "the provider sent silence".
#[test]
fn bytes_that_are_not_the_declared_container_fail_with_a_named_error() {
    let garbage = b"this is not an mp3 file, not even slightly, no frame sync anywhere at all";
    let err = decode("audio/mpeg", Some(24_000), 1, garbage, garbage.len()).unwrap_err();
    assert!(matches!(err, CodecError::Ffmpeg { .. }), "{err:?}");
}

#[test]
fn no_requested_rate_keeps_the_containers_rate() {
    let mp3 = unhex(TONE_MP3_HEX);
    let (pcm, format) = decode("audio/mpeg", None, 1, &mp3, mp3.len()).unwrap();
    assert!(pcm.len() > 1_500 && pcm.len() < 4_000, "0.1 s at 24 kHz, got {}", pcm.len());
    assert!(tone_strength(&pcm, 24_000, 440.0) > 0.7, "not a 440 Hz tone at the source rate");
    assert_eq!(format, AudioFormat::new(24_000, 1));
}

#[test]
fn synchronous_decode_propagates_a_reader_failure_instead_of_reporting_eof() {
    struct FailingRead(std::io::Cursor<Vec<u8>>);
    impl std::io::Read for FailingRead {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            match std::io::Read::read(&mut self.0, bytes)? {
                0 => Err(std::io::Error::other("source read failed")),
                n => Ok(n),
            }
        }
    }
    let mut input = FailingRead(std::io::Cursor::new(wav_file(24000, 1, &[1; 2400])));
    let error =
        decode_stream(Some(Container::Wav), 24000, 1, decode_limits(), &mut input, |_, _| Ok(()))
            .unwrap_err();
    assert!(
        matches!(error, CodecError::Ffmpeg { op: "read", ref detail } if detail == "source read failed")
    );
}

#[test]
fn synchronous_decode_stops_after_pcm_delivery_is_refused() {
    let mut input = std::io::Cursor::new(wav_file(24000, 1, &[1; 24000]));
    let refused = CodecError::Ffmpeg { op: "PCM delivery", detail: "receiver closed".into() };
    let mut callbacks = 0;
    let result =
        decode_stream(Some(Container::Wav), 24000, 1, decode_limits(), &mut input, |_, _| {
            callbacks += 1;
            Err(refused.clone())
        });
    assert_eq!(result, Err(refused));
    assert_eq!(callbacks, 1, "native decode must stop at the first refused output");
}

fn decode_limits() -> crate::audio_decoder::DecodeLimits {
    crate::audio_decoder::DecodeLimits {
        min_sample_rate: 8000,
        max_sample_rate: 192000,
        max_channels: 8,
        max_streams: 8,
        max_expansion_ratio: 24,
        max_samples_per_frame: 65536,
    }
}

#[test]
fn a_panicking_pcm_callback_becomes_a_decode_error_without_crossing_the_c_abi() {
    // The short resampled stream is delivered only while draining the final soxr tail.
    for (rate, samples) in [(24_000, 2400), (48_000, 24)] {
        let mut input = std::io::Cursor::new(wav_file(24_000, 1, &vec![1; samples]));
        let error =
            decode_stream(Some(Container::Wav), rate, 1, decode_limits(), &mut input, |_, _| {
                panic!("callback failed");
            })
            .unwrap_err();
        assert!(
            matches!(error, CodecError::Ffmpeg { op: "output", ref detail } if detail == "output callback panicked")
        );
    }
}

#[test]
fn native_pcm_expansion_is_rejected_before_allocating_an_oversized_output() {
    let mut input = std::io::Cursor::new(wav_file(8000, 1, &[1; 2400]));
    let limits =
        crate::audio_decoder::DecodeLimits { max_samples_per_frame: 128, ..decode_limits() };
    assert!(
        decode_stream(Some(Container::Wav), 192000, 1, limits, &mut input, |_, _| Ok(())).is_err()
    );
}

#[test]
fn a_reader_that_overreports_its_buffer_is_refused_at_the_ffi_boundary() {
    struct OverreportedRead;
    impl std::io::Read for OverreportedRead {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            Ok(output.len() + 1)
        }
    }
    let error = decode_stream(
        Some(Container::Wav),
        24000,
        1,
        decode_limits(),
        &mut OverreportedRead,
        |_, _| Ok(()),
    )
    .unwrap_err();
    assert!(matches!(error, CodecError::Ffmpeg { op: "read", ref detail }
        if detail == "input callback returned more bytes than its supplied buffer"));
}

// A short stream can sit entirely in soxr's filter. Both decoder paths must release its
// tail, at the same duration and with the same filter response (allowing int16 dithering).
#[test]
fn container_rate_conversion_matches_the_shared_resampler_including_short_tails() {
    for (from, to, channels) in [(48_000, 16_000, 1), (16_000, 48_000, 2), (44_100, 24_000, 2)] {
        let samples: Vec<i16> = (0..from / 100)
            .flat_map(|i| {
                let value = ((f64::from(i) * 440.0 * std::f64::consts::TAU / f64::from(from)).sin()
                    * 10_000.0) as i16;
                std::iter::repeat_n(value, channels as usize)
            })
            .collect();
        let wav = wav_file(from, channels, &samples);
        let mut decoded = Vec::new();
        let format = decode_stream(
            Some(Container::Wav),
            to,
            channels,
            decode_limits(),
            &mut std::io::Cursor::new(wav),
            |_, pcm| {
                decoded.extend_from_slice(pcm);
                Ok(())
            },
        )
        .unwrap();
        let mut direct =
            crate::Resampler::new(from, to, u32::from(channels), crate::Quality::Medium).unwrap();
        let mut expected = direct.push(&samples).unwrap().to_vec();
        expected.extend_from_slice(direct.flush().unwrap());
        assert_eq!(format, AudioFormat::new(to, channels));
        assert!(!decoded.is_empty());
        assert_eq!(decoded.len(), expected.len(), "{from} -> {to} x{channels}");
        assert!(
            decoded.iter().zip(&expected).all(|(&a, &b)| (i32::from(a) - i32::from(b)).abs() <= 2),
            "{from} -> {to} x{channels}: the decoder must use the shared soxr filter"
        );
    }
}
