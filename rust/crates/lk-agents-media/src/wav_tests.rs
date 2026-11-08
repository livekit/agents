use super::*;

/// Builds a WAV file byte-for-byte, with rate, depth, chunk order and chunk parity as parameters.
fn wav(
    rate: u32,
    channels: u16,
    bits: u16,
    fmt_tag: u16,
    extra_chunks: &[(&[u8; 4], Vec<u8>)],
    pcm: &[u8],
) -> Vec<u8> {
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&fmt_tag.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    let block_align = channels * bits / 8;
    fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes()); // byte rate
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());

    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    for (id, data) in extra_chunks {
        body.extend_from_slice(*id);
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(data);
        if data.len() % 2 == 1 {
            body.push(0); // RIFF word-alignment pad, NOT counted in the declared size
        }
    }
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    body.extend_from_slice(pcm);

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// A canonical 16-bit PCM file; the decoder and ffmpeg tests build their inputs with it too.
pub(crate) fn wav_file(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    encode(rate, channels, &crate::pcm_bytes(samples))
}

fn simple(samples: &[i16]) -> Vec<u8> {
    wav_file(24_000, 1, samples)
}

fn decode_all(chunks: &[&[u8]]) -> Result<Vec<WavSegment>> {
    let mut d = WavDecoder::new();
    let mut out = Vec::new();
    for c in chunks {
        out.extend(d.push(c)?);
    }
    d.flush();
    Ok(out)
}

fn merged(segs: &[WavSegment]) -> Vec<i16> {
    segs.iter().flat_map(|s| s.samples.iter().copied()).collect()
}

#[test]
fn invalid_fmt_is_rejected_before_reading_its_declared_extension() {
    let mut prefix = wav_file(24000, 1, &[1, 2]);
    prefix[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    prefix[34..36].copy_from_slice(&24u16.to_le_bytes());
    prefix.truncate(36);
    assert_eq!(WavDecoder::new().push(&prefix), Err(CodecError::UnsupportedWavBits(24)));
}

#[test]
fn odd_fmt_extensions_and_their_pad_are_skipped_before_pcm() {
    let samples = [5, -5, 8, -8];
    let mut file = wav_file(24000, 1, &samples);
    file[16..20].copy_from_slice(&17u32.to_le_bytes());
    file.splice(36..36, [0xAA, 0]);
    let mut decoder = WavDecoder::new();
    let mut pcm = Vec::new();
    for chunk in file.chunks(7) {
        for segment in decoder.push(chunk).unwrap() {
            pcm.extend(segment.samples);
        }
    }
    assert_eq!(pcm, samples);
}

// Every other test trusts `wav()`, so the canonical 44-byte header is written out by hand once.
#[test]
fn a_hand_written_canonical_header_decodes_and_the_generator_reproduces_it() {
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&40u32.to_le_bytes()); // file size - 8
    file.extend_from_slice(b"WAVE");
    file.extend_from_slice(b"fmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    file.extend_from_slice(&1u16.to_le_bytes()); // mono
    file.extend_from_slice(&24_000u32.to_le_bytes());
    file.extend_from_slice(&48_000u32.to_le_bytes()); // byte rate
    file.extend_from_slice(&2u16.to_le_bytes()); // block align
    file.extend_from_slice(&16u16.to_le_bytes()); // bits
    file.extend_from_slice(b"data");
    file.extend_from_slice(&4u32.to_le_bytes());
    file.extend_from_slice(&[0x01, 0x00, 0xFF, 0xFF]); // 1, -1
    assert_eq!(file.len(), 48);

    let segs = decode_all(&[&file]).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].format, AudioFormat::new(24_000, 1));
    assert_eq!(segs[0].samples, vec![1, -1]);
    assert_eq!(simple(&[1, -1]), file, "the generator drifted from the canonical layout");
}

// A header split across pushes must accumulate, and a 1-byte cadence must keep the odd carry byte.
#[test]
fn feeding_the_same_file_one_byte_at_a_time_gives_the_same_samples() {
    let want: Vec<i16> = (0..64).map(|i| (i * 517 - 16000) as i16).collect();
    let file = simple(&want);
    let one_shot = merged(&decode_all(&[&file]).unwrap());

    let mut d = WavDecoder::new();
    let mut got = Vec::new();
    for b in &file {
        for seg in d.push(std::slice::from_ref(b)).unwrap() {
            got.extend(seg.samples);
        }
    }
    d.flush();
    assert_eq!(got, want);
    assert_eq!(got, one_shot);
}

// A split inside each header field; a mis-tracked `need` reads the rate from the wrong bytes.
#[test]
fn every_split_point_in_the_file_yields_the_same_samples() {
    let want: Vec<i16> = vec![7, -7, 1000, -1000, 32000, -32000];
    let file = simple(&want);
    for cut in 0..=file.len() {
        let segs = decode_all(&[&file[..cut], &file[cut..]]).unwrap();
        assert_eq!(merged(&segs), want, "split at {cut}");
    }
}

// A stream that dies mid-header produces nothing and must not panic.
#[test]
fn a_truncated_header_yields_no_audio_and_no_panic() {
    for n in 0..44usize {
        let file = simple(&[1, 2, 3, 4]);
        let mut d = WavDecoder::new();
        let segs = d.push(&file[..n]).unwrap();
        assert!(merged(&segs).is_empty(), "truncated at {n} produced audio");
    }
}

// Refuse rather than guess: non-RIFF/WAVE input, a non-16-bit file (depth is checked before the
// format tag, as the reference does) and a `data` chunk before `fmt `.
#[test]
fn a_file_we_cannot_decode_is_refused_by_name() {
    let mut avi = b"RIFF".to_vec();
    avi.extend_from_slice(&100u32.to_le_bytes());
    avi.extend_from_slice(b"AVI ");
    let mut data_first = b"RIFF".to_vec();
    data_first.extend_from_slice(&40u32.to_le_bytes());
    data_first.extend_from_slice(b"WAVEdata");
    data_first.extend_from_slice(&8u32.to_le_bytes());
    data_first.extend_from_slice(&[0u8; 8]);
    type ErrorPredicate = fn(&CodecError) -> bool;
    let cases: [(Vec<u8>, ErrorPredicate); 6] = [
        (b"ID3\x04\x00\x00\x00\x00\x00\x00mp3 data here".to_vec(), |e| {
            matches!(e, CodecError::MissingRiffWave(_))
        }),
        (avi, |e| matches!(e, CodecError::MissingRiffWave(_))),
        (wav(24_000, 1, 24, 1, &[], &[0; 12]), |e| *e == CodecError::UnsupportedWavBits(24)),
        (wav(24_000, 1, 32, 3, &[], &[0; 16]), |e| *e == CodecError::UnsupportedWavBits(32)),
        (wav(24_000, 1, 16, 0xFFFE, &[], &[0; 8]), |e| {
            *e == CodecError::UnsupportedWavFormat(0xFFFE)
        }),
        (data_first, |e| *e == CodecError::WavDataBeforeFmt),
    ];
    for (i, (bytes, want)) in cases.into_iter().enumerate() {
        let err = WavDecoder::new().push(&bytes).unwrap_err();
        assert!(want(&err), "case {i}: {err:?}");
    }
}

// `LIST`/`INFO` metadata ahead of the data chunk is skipped, not played.
#[test]
fn chunks_between_the_header_and_the_data_are_skipped() {
    let want: Vec<i16> = vec![10, 20, 30, 40];
    let f = wav(
        16_000,
        1,
        16,
        1,
        &[(b"LIST", b"INFOISFTLavf58.29.100\0".to_vec()), (b"JUNK", vec![0xAA; 24])],
        &crate::pcm_bytes(&want),
    );
    let segs = decode_all(&[&f]).unwrap();
    assert_eq!(segs[0].format, AudioFormat::new(16_000, 1));
    assert_eq!(segs[0].samples, want);
}

// RIFF pads odd chunks to an even length without counting the pad. The reference fast path
// skips exactly `chunk_size` and loses the rest of a file with a 19-byte LIST payload; ffmpeg
// consumes the pad.
#[test]
fn an_odd_sized_chunk_pad_byte_does_not_desynchronise_the_parser() {
    let want: Vec<i16> = vec![-5, 5, -6, 6, -7, 7];
    for payload in [
        b"INFOIART_odd_lengt".to_vec(), // 18, even: works in the reference too
        b"INFOIART_odd_lengt_".to_vec(), // 19, odd: the reference emits nothing at all
    ] {
        let odd = payload.len() % 2 == 1;
        let f = wav(48_000, 2, 16, 1, &[(b"LIST", payload)], &crate::pcm_bytes(&want));
        assert_eq!(f.len() % 2, 0, "the file itself stays word-aligned");
        let segs = decode_all(&[&f]).unwrap();
        assert_eq!(segs.len(), 1, "the parser lost the data chunk (odd payload: {odd})");
        assert_eq!(segs[0].format, AudioFormat::new(48_000, 2));
        assert_eq!(segs[0].samples, want, "odd payload: {odd}");
    }
}

// A live writer declares a `data` size of 0 or 0xFFFFFFFF, so the declared size is ignored.
#[test]
fn a_data_chunk_whose_declared_size_is_a_placeholder_still_streams() {
    let want: Vec<i16> = vec![100, -100, 200, -200];
    for declared in [0u32, 0xFFFF_FFFF] {
        let mut f = simple(&want);
        let n = f.len();
        f[n - want.len() * 2 - 4..n - want.len() * 2].copy_from_slice(&declared.to_le_bytes());
        let segs = decode_all(&[&f]).unwrap();
        assert_eq!(merged(&segs), want, "declared size {declared:#x}");
    }
}

// Concatenated WAV files: the second header is not audio, and its sample rate applies.
#[test]
fn a_new_riff_header_at_a_push_boundary_starts_a_new_segment() {
    let a: Vec<i16> = vec![1, 2, 3, 4];
    let b: Vec<i16> = vec![-1, -2, -3];
    let mut d = WavDecoder::new();
    let mut segs = d.push(&simple(&a)).unwrap();
    segs.extend(d.push(&wav(16_000, 2, 16, 1, &[], &crate::pcm_bytes(&b))).unwrap());

    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].format, AudioFormat::new(24_000, 1));
    assert_eq!(segs[0].samples, a);
    assert_eq!(segs[1].format, AudioFormat::new(16_000, 2));
    assert_eq!(segs[1].samples, b);
}

// `RIFF` is recognised only at a push boundary; 16-bit PCM eventually contains those bytes.
#[test]
fn pcm_that_happens_to_contain_the_riff_bytes_is_not_mistaken_for_a_header() {
    let want: Vec<i16> = vec![0, 18770, 17990, 0, 5];
    let file = simple(&want);
    let segs = decode_all(&[&file]).unwrap();
    assert_eq!(merged(&segs), want);
}
