use lk_agents_abi::media::codecs::Container;

/// `None` means "let ffmpeg probe". Media-type parameters are stripped first, so
/// `audio/webm;codecs=opus` (what the JS SDK sends) is not left to a too-small probe.
pub fn mime_container(mime: &str) -> Option<Container> {
    let base = mime.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    Some(match base.as_str() {
        "audio/mpeg" | "audio/mp3" | "audio/x-mpeg" => Container::Mp3,
        "audio/aac" | "audio/x-aac" => Container::Aac,
        "audio/flac" | "audio/x-flac" => Container::Flac,
        "audio/wav" | "audio/wave" | "audio/x-wav" => Container::Wav,
        // Raw opus arrives in ogg in every producer; the reference maps it to `ogg` too.
        "audio/opus" | "audio/ogg" => Container::Ogg,
        "audio/webm" => Container::Webm,
        // Safari's MediaRecorder records audio only as `video/mp4` (client-sdk-js pre-connect).
        "audio/mp4" | "video/mp4" => Container::Mp4,
        _ => return None,
    })
}

/// What an `AudioEmitter` does with the mime type a TTS provider declares for its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioDecoding {
    /// `audio/pcm` or `audio/raw`: no decoder is built.
    Passthrough,
    /// G.711 bytes, expanded sample by sample to PCM.
    G711(G711),
    /// The container the decoder reads.
    Decode(Container),
}

/// The G.711 companding laws (ITU-T G.711): one byte per sample, at the declared rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G711 {
    MuLaw,
    ALaw,
}

impl G711 {
    /// The law a media type names, parameters stripped.
    pub fn of(mime: &str) -> Option<G711> {
        let base = mime.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
        match base.as_str() {
            "audio/basic" | "audio/x-mulaw" | "audio/pcmu" => Some(G711::MuLaw),
            "audio/pcma" | "audio/x-alaw" => Some(G711::ALaw),
            _ => None,
        }
    }

    /// Little-endian PCM bytes, two per input byte.
    pub fn decode(self, bytes: &[u8]) -> Vec<u8> {
        let expand = match self {
            G711::MuLaw => mulaw,
            G711::ALaw => alaw,
        };
        bytes.iter().flat_map(|&b| expand(b).to_le_bytes()).collect()
    }
}

fn mulaw(byte: u8) -> i16 {
    const BIAS: i16 = 0x84;
    let u = !byte;
    let magnitude = ((i16::from(u & 0x0F) << 3) + BIAS) << ((u >> 4) & 0x07);
    if u & 0x80 != 0 { BIAS - magnitude } else { magnitude - BIAS }
}

fn alaw(byte: u8) -> i16 {
    let a = byte ^ 0x55;
    let mut t = i16::from(a & 0x0F) << 4;
    match (a & 0x70) >> 4 {
        0 => t += 8,
        1 => t += 0x108,
        segment => t = (t + 0x108) << (segment - 1),
    }
    if a & 0x80 != 0 { t } else { -t }
}

/// `Display` is the refusal's sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownMime(pub String);

impl std::fmt::Display for UnknownMime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mime type {:?} is not one this build recognises. audio/pcm and audio/raw need no \
             decoder; G.711 (audio/pcmu, audio/pcma), mp3, aac, flac, wav, ogg/opus, webm and mp4 \
             are decoded.",
            self.0
        )
    }
}

impl std::error::Error for UnknownMime {}

/// `AudioEmitter.initialize`'s branch. Unknown types are refused here because ffmpeg would
/// otherwise probe them, and an ID3 tag framed as PCM is noise, not an error. The PCM test is a
/// prefix match, as the reference's is.
pub fn audio_decoding(mime_type: &str) -> Result<AudioDecoding, UnknownMime> {
    let mt = mime_type.trim().to_ascii_lowercase();
    // before the PCM prefix test, which `audio/pcmu` and `audio/pcma` would match
    if let Some(law) = G711::of(&mt) {
        return Ok(AudioDecoding::G711(law));
    }
    if mt.starts_with("audio/pcm") || mt.starts_with("audio/raw") {
        return Ok(AudioDecoding::Passthrough);
    }
    mime_container(&mt).map(AudioDecoding::Decode).ok_or_else(|| UnknownMime(mime_type.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // `audio/pcm`, which nearly every plugin emits, builds no decoder and is not refused.
    #[test]
    fn pcm_and_raw_are_passed_through_whatever_their_parameters_or_case() {
        for mime in
            ["audio/pcm", "audio/raw", "AUDIO/PCM", " audio/pcm;rate=24000 ", "audio/raw;bits=16"]
        {
            assert_eq!(audio_decoding(mime), Ok(AudioDecoding::Passthrough), "{mime}");
        }
    }

    // A mime type nothing can decode is refused, with a sentence a plugin author can act on.
    #[test]
    fn a_known_container_is_decoded_and_an_unknown_one_is_refused() {
        assert_eq!(audio_decoding(" Audio/MPEG "), Ok(AudioDecoding::Decode(Container::Mp3)));
        assert_eq!(
            audio_decoding("audio/webm;codecs=opus"),
            Ok(AudioDecoding::Decode(Container::Webm)),
            "the table strips the parameter on lookup"
        );
        let err = audio_decoding("audio/x-not-a-real-container").unwrap_err();
        assert_eq!(
            err.0, "audio/x-not-a-real-container",
            "the provider's own spelling is reported"
        );
        assert!(audio_decoding("").is_err(), "an empty declaration is not PCM");
    }

    // Catches: `audio/pcmu` taken for raw PCM by the prefix test, or a wrong expansion table
    // (silence must decode to 0 and the extremes to the G.711 peaks).
    #[test]
    fn g711_is_decoded_not_passed_through_as_pcm() {
        for mime in ["audio/pcmu", "audio/basic", "AUDIO/X-MULAW", "audio/pcmu;rate=8000"] {
            assert_eq!(audio_decoding(mime), Ok(AudioDecoding::G711(G711::MuLaw)), "{mime}");
        }
        for mime in ["audio/pcma", "audio/x-alaw"] {
            assert_eq!(audio_decoding(mime), Ok(AudioDecoding::G711(G711::ALaw)), "{mime}");
        }
        let pcm = |law: G711, b: u8| i16::from_le_bytes(law.decode(&[b]).try_into().unwrap());
        assert_eq!(pcm(G711::MuLaw, 0xFF), 0);
        assert_eq!(pcm(G711::MuLaw, 0x7F), 0);
        assert_eq!(pcm(G711::MuLaw, 0x80), 32124);
        assert_eq!(pcm(G711::MuLaw, 0x00), -32124);
        assert_eq!(pcm(G711::ALaw, 0xD5), 8);
        assert_eq!(pcm(G711::ALaw, 0x55), -8);
        assert_eq!(pcm(G711::ALaw, 0xAA), 32256);
        assert_eq!(pcm(G711::ALaw, 0x2A), -32256);
        assert_eq!(G711::MuLaw.decode(&[0xFF, 0x80]).len(), 4, "two bytes per sample");
    }

    // Every alias row, also upper-cased, with media-type parameters stripped; an unknown type
    // means "probe" rather than a guess, since guessing mp3 frames an ID3 tag as audio.
    #[test]
    fn a_mime_type_maps_to_the_reference_container_or_asks_for_probing() {
        let table = [
            ("audio/mpeg", Some("mp3")),
            ("audio/mp3", Some("mp3")),
            ("audio/x-mpeg", Some("mp3")),
            ("audio/aac", Some("aac")),
            ("audio/x-aac", Some("aac")),
            ("audio/flac", Some("flac")),
            ("audio/x-flac", Some("flac")),
            ("audio/wav", Some("wav")),
            ("audio/wave", Some("wav")),
            ("audio/x-wav", Some("wav")),
            ("audio/opus", Some("ogg")),
            ("audio/ogg", Some("ogg")),
            ("audio/webm", Some("webm")),
            ("audio/mp4", Some("mp4")),
            ("audio/webm;codecs=opus", Some("webm")),
            ("audio/webm; codecs=opus", Some("webm")),
            ("audio/ogg; codecs=\"opus\"", Some("ogg")),
            ("audio/wav; rate=24000", Some("wav")),
            ("", None),
            ("   ", None),
            ("audio/pcm", None),
            ("audio/l16", None),
            ("application/octet-stream", None),
        ];
        for (mime, want) in table {
            assert_eq!(mime_container(mime).map(Container::as_str), want, "{mime}");
            let upper = mime_container(&mime.to_ascii_uppercase());
            assert_eq!(upper.map(Container::as_str), want, "{mime}");
        }
    }
}
