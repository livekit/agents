//! `_mime_to_av_format` (`utils/codecs/decoder.py:35-62`).

/// `None` means "let ffmpeg probe". Divergence: media-type parameters are stripped first, so
/// `audio/webm;codecs=opus` (what the JS SDK sends) is not left to a too-small probe.
pub fn mime_to_av_format(mime: Option<&str>) -> Option<&'static str> {
    let mime = mime?;
    // `if not mime`: empty or all-whitespace is falsy.
    let base = mime.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    if base.is_empty() {
        return None;
    }
    Some(match base.as_str() {
        "audio/mpeg" | "audio/mp3" | "audio/x-mpeg" => "mp3",
        "audio/aac" | "audio/x-aac" => "aac",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/wav" | "audio/wave" | "audio/x-wav" => "wav",
        // Raw opus arrives in ogg in every producer; the reference maps it to `ogg` too.
        "audio/opus" | "audio/ogg" => "ogg",
        "audio/webm" => "webm",
        "audio/mp4" => "mp4",
        _ => return None,
    })
}

/// What an `AudioEmitter` does with the mime type a TTS provider declares for its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodePlan {
    /// `audio/pcm` or `audio/raw`: no decoder is built.
    Passthrough,
    /// Carries the normalised (trimmed, lower-cased) mime `Decoder::new` takes.
    Decode(String),
}

/// `Display` is the refusal's sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownMime(pub String);

impl std::fmt::Display for UnknownMime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mime type {:?} is not one this build recognises. audio/pcm and audio/raw need no \
             decoder; mp3, aac, flac, wav, ogg/opus, webm and mp4 are decoded.",
            self.0
        )
    }
}

impl std::error::Error for UnknownMime {}

/// `AudioEmitter.initialize`'s branch. Unknown types are refused here because ffmpeg would
/// otherwise probe them, and an ID3 tag framed as PCM is noise, not an error. The PCM test is a
/// prefix match, as the reference's is.
pub fn decode_plan(mime_type: &str) -> Result<DecodePlan, UnknownMime> {
    let mt = mime_type.trim().to_ascii_lowercase();
    if mt.starts_with("audio/pcm") || mt.starts_with("audio/raw") {
        return Ok(DecodePlan::Passthrough);
    }
    if mime_to_av_format(Some(&mt)).is_none() {
        return Err(UnknownMime(mime_type.to_owned()));
    }
    Ok(DecodePlan::Decode(mt))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: building a decoder for the majority path, or refusing it. `audio/pcm` is what
    // nearly every plugin emits; a decoder there puts a copy and a resampler in front of every
    // frame of every session, and a refusal breaks every plugin at once.
    #[test]
    fn pcm_and_raw_are_passed_through_whatever_their_parameters_or_case() {
        for mime in ["audio/pcm", "audio/raw", "AUDIO/PCM", " audio/pcm;rate=24000 ", "audio/raw;bits=16"] {
            assert_eq!(decode_plan(mime), Ok(DecodePlan::Passthrough), "{mime}");
        }
    }

    // Catches: accepting a mime type nothing can decode, which is the obvious over-correction to
    // "we decode containers now" -- and the sentence, which a plugin author reads in a traceback.
    #[test]
    fn a_known_container_is_decoded_under_its_normalised_name_and_an_unknown_one_is_refused() {
        assert_eq!(decode_plan(" Audio/MPEG "), Ok(DecodePlan::Decode("audio/mpeg".to_owned())));
        assert_eq!(
            decode_plan("audio/webm;codecs=opus"),
            Ok(DecodePlan::Decode("audio/webm;codecs=opus".to_owned())),
            "the parameter survives normalisation; the table strips it on lookup"
        );
        let err = decode_plan("audio/x-not-a-real-container").unwrap_err();
        assert_eq!(err.0, "audio/x-not-a-real-container", "the provider's own spelling is reported");
        assert!(decode_plan("").is_err(), "an empty declaration is not PCM");
    }

    // Catches: a typo or dropped row in the alias table (`audio/x-wav` taking the ffmpeg path
    // instead of the WAV fast path is silent), a media-type parameter defeating the lookup
    // (`audio/webm;codecs=opus` is what the JS SDK sends for pre-connect audio, and falling
    // through to probing at `probesize=32` fails), or a guess for an unknown type -- `None` means
    // "probe", and guessing mp3 is how an ID3 tag gets framed as audio. The reference lower-cases
    // first, so every row is also checked upper-cased.
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
            assert_eq!(mime_to_av_format(Some(mime)), want, "{mime}");
            assert_eq!(mime_to_av_format(Some(&mime.to_ascii_uppercase())), want, "{mime}");
        }
        assert_eq!(mime_to_av_format(None), None);
    }
}
