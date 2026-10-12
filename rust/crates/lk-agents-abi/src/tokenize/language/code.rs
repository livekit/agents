//! Every function but [`normalize`] expects a normalised string.

use std::fmt;

use super::data;

/// Normalise a language code or name to BCP-47.
// In the reference's order: English names (`"english"`), then ISO 639-3 (`"eng"`), then compound
// tags (case fixed, `_` -> `-`, primary subtag left alone so `cmn-Hans-CN` round-trips to Google
// STT), else lowercased passthrough (`"multi"`). Bare `"cmn"` does become `"zh"`.
#[crate::host(export(name = "normalize_language", pid = false))]
pub fn normalize(code: &str) -> String {
    let lowered = code.trim().to_lowercase();

    if let Some(mapped) = data::name_to_code(&lowered) {
        return mapped.to_string();
    }

    if let Some(mapped) = data::iso_639_3_to_1(&lowered) {
        return mapped.unwrap_or(&lowered).to_string();
    }

    let dashed = lowered.replace('_', "-");
    if !dashed.contains('-') {
        return lowered;
    }

    let mut parts = dashed.split('-');
    let mut out = String::with_capacity(dashed.len());
    out.push_str(parts.next().unwrap_or_default());
    for part in parts {
        out.push('-');
        // Four characters (not bytes) is a script (`Hans`), else a region (`US`, `419`).
        if part.chars().count() == 4 {
            let mut chars = part.chars();
            if let Some(first) = chars.next() {
                out.extend(first.to_uppercase());
                out.push_str(chars.as_str());
            }
        } else {
            out.push_str(&part.to_uppercase());
        }
    }
    out
}

/// Base language, ISO 639-1 where one exists: `"en"` from `"en-US"`, `"zh"` from `"cmn-Hans-CN"`.
#[crate::host(export(pid = false))]
pub fn base_language(code: &str) -> &str {
    let base = code.split('-').next().unwrap_or(code);
    match data::iso_639_3_to_1(base) {
        Some(Some(mapped)) => mapped,
        _ => base,
    }
}

/// Region subtag: the first two-character subtag after the primary (`None` for `"es-419"`).
#[crate::host(export(name = "language_region", pid = false))]
pub fn region(code: &str) -> Option<&str> {
    code.split('-').skip(1).find(|part| part.chars().count() == 2)
}

/// The 639-1 base plus the region: `"zh-CN"` from `"cmn-Hans-CN"` (`turn_detector/base.py:241`).
#[crate::host(export(name = "language_iso", pid = false))]
pub fn iso(code: &str) -> String {
    let base = base_language(code);
    match region(code) {
        Some(region) => format!("{base}-{region}"),
        None => base.to_string(),
    }
}

/// Lowercase English (NLTK punkt) name of the base language, or `None` if unknown.
#[crate::host(export(pid = false))]
pub fn language_name(code: &str) -> Option<&'static str> {
    data::code_to_name(base_language(code))
}

/// A normalised BCP-47 language identifier: the Rust half of `livekit.agents.LanguageCode`. Only
/// [`LanguageCode::new`] constructs it, and it normalises. Crosses to a host as its string.
#[crate::host(string)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LanguageCode(String);

impl LanguageCode {
    pub fn new(code: &str) -> Self {
        LanguageCode(normalize(code))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn language(&self) -> &str {
        base_language(&self.0)
    }

    pub fn iso(&self) -> String {
        iso(&self.0)
    }

    pub fn region(&self) -> Option<&str> {
        region(&self.0)
    }

    pub fn language_name(&self) -> Option<&'static str> {
        language_name(&self.0)
    }
}

impl fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Infallible, like the host's `LanguageCode(str)`.
impl std::str::FromStr for LanguageCode {
    type Err = std::convert::Infallible;

    fn from_str(code: &str) -> Result<Self, Self::Err> {
        Ok(LanguageCode::new(code))
    }
}

impl AsRef<str> for LanguageCode {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for LanguageCode {
    fn from(code: &str) -> Self {
        LanguageCode::new(code)
    }
}

impl PartialEq<str> for LanguageCode {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for LanguageCode {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: dropping the name or 639-3 lookup (#4926), mapping a compound tag's 639-3 primary
    // subtag, upper- or title-casing every subtag, forgetting `_` or the trim, an unknown code
    // treated as an error, byte length in the script test, or the struct skipping normalisation.
    #[test]
    fn normalize_follows_the_reference() {
        for (raw, want) in [
            ("english", "en"),
            ("eng", "en"),
            ("en", "en"),
            ("chinese", "zh"),
            ("cmn", "zh"),
            ("cmn-Hans-CN", "cmn-Hans-CN"),
            ("cmn-CN", "cmn-CN"),
            ("zh-hant-tw", "zh-Hant-TW"),
            ("EN-us", "en-US"),
            ("es-419", "es-419"),
            ("en_us", "en-US"),
            ("  EN_US \n", "en-US"),
            ("multi", "multi"),
            ("MULTI", "multi"),
            ("xx", "xx"),
            ("", ""),
            ("en-éàèù", "en-Éàèù"),
        ] {
            assert_eq!(normalize(raw), want, "{raw:?}");
            assert_eq!(LanguageCode::new(raw).as_str(), want);
        }
    }

    // Catches: `.language` returning the raw primary subtag ("cmn" instead of "zh").
    #[test]
    fn base_language_maps_the_primary_subtag_through_639_3() {
        assert_eq!(LanguageCode::new("cmn-Hans-CN").language(), "zh");
        assert_eq!(LanguageCode::new("en-US").language(), "en");
        assert_eq!(LanguageCode::new("yue").language(), "yue", "no 639-1 equivalent: unchanged");
        assert_eq!(LanguageCode::new("multi").language(), "multi");
    }

    // Catches: `.iso` taking parts[1] as the region ("zh-Hans").
    #[test]
    fn iso_skips_the_script_subtag() {
        assert_eq!(LanguageCode::new("cmn-Hans-CN").iso(), "zh-CN");
        assert_eq!(LanguageCode::new("zh-Hans-CN").iso(), "zh-CN");
        assert_eq!(LanguageCode::new("en-US").iso(), "en-US");
        assert_eq!(LanguageCode::new("cmn").iso(), "zh", "no region: base only, no trailing dash");
    }

    // Catches: `.region` reporting the script, or `""` instead of None.
    #[test]
    fn region_is_the_first_two_character_subtag_or_none() {
        assert_eq!(LanguageCode::new("zh-Hans-CN").region(), Some("CN"));
        assert_eq!(LanguageCode::new("en-US").region(), Some("US"));
        assert_eq!(LanguageCode::new("en").region(), None);
        assert_eq!(LanguageCode::new("es-419").region(), None, "M.49 regions are not two chars");
    }

    // Catches: looking the name up by the full tag instead of the base.
    #[test]
    fn language_name_resolves_through_the_base_language() {
        assert_eq!(LanguageCode::new("en-US").language_name(), Some("english"));
        assert_eq!(LanguageCode::new("cmn-Hans-CN").language_name(), Some("chinese"));
        assert_eq!(LanguageCode::new("multi").language_name(), None);
    }

}
