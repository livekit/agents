//! Per-language EOT thresholds (`inference/eot/languages.py`), rekeyed through
//! [`LanguageCode::language`] so `"en-US"`, `"eng"` and `"english"` share a row.

use std::collections::BTreeMap;

use super::code::LanguageCode;

/// Shipped with the local mini model (#4722), on its own probability scale. Sorted, for binary
/// search.
pub const LOCAL_LANGUAGES: &[(&str, f64)] = &[
    ("ar", 0.3500),
    ("de", 0.2450),
    ("en", 0.3600),
    ("es", 0.3500),
    ("fr", 0.2850),
    ("hi", 0.3050),
    ("id", 0.3450),
    ("it", 0.2300),
    ("ja", 0.2950),
    ("ko", 0.4000),
    ("nl", 0.2000),
    ("pt", 0.3200),
    ("tr", 0.2550),
    ("zh", 0.3550),
];

fn local_threshold(language: &str) -> Option<f64> {
    LOCAL_LANGUAGES.binary_search_by_key(&language, |(k, _)| k).ok().map(|i| LOCAL_LANGUAGES[i].1)
}

fn local_layer() -> Layer {
    Layer {
        thresholds: LOCAL_LANGUAGES.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
        // `LOCAL_LANGUAGES["en"]`: "en" must exist, as `lookup(None)` assumes.
        default: local_threshold("en").expect("en is in LOCAL_LANGUAGES"),
    }
}

/// The binding raises `APIError(..., retryable=False)`: a retry would fail the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ThresholdError {
    #[error("turn detector session created without usable default thresholds")]
    UnusableDefaults,
}

#[crate::host(record(module = "turn_events", emit, extract))]
#[crate::host(wire_strs(msg = format!(
    "unknown turn detector model {s:?}: expected turn-detector-v1 or turn-detector-v1-mini"
)))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorModel {
    /// Cloud. Thresholds arrive with `SessionCreated`.
    #[host(rename = "turn-detector-v1")]
    V1,
    /// Local. Produces no backchannel probability.
    #[host(rename = "turn-detector-v1-mini")]
    V1Mini,
}

/// One threshold for every language, or thresholds per language merged over the defaults.
// `None` (no override) is `NOT_GIVEN`.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "turn_events", emit, extract, transparent))]
pub enum ThresholdOverride {
    All(f64),
    PerLanguage(BTreeMap<String, f64>),
}

fn normalize_overrides(overrides: Option<ThresholdOverride>) -> Option<ThresholdOverride> {
    match overrides {
        Some(ThresholdOverride::PerLanguage(map)) => {
            Some(ThresholdOverride::PerLanguage(rekey(map.iter().map(|(k, v)| (k.as_str(), *v)))))
        }
        other => other,
    }
}

fn rekey<'a>(entries: impl Iterator<Item = (&'a str, f64)>) -> BTreeMap<String, f64> {
    entries.map(|(k, v)| (LanguageCode::new(k).language().to_owned(), v)).collect()
}

/// `round(value, 4)`, but half away from zero (only exact binary ties differ).
fn round4(value: f64) -> f64 {
    (value * 1e4).round() / 1e4
}

#[derive(Debug, Clone)]
struct Layer {
    thresholds: BTreeMap<String, f64>,
    default: f64,
}

/// Shared by the detector and its stream (`eot/base.py:137`), so a fallback in one is seen by the
/// other.
#[derive(Debug, Clone)]
pub struct ThresholdOptions {
    model: TurnDetectorModel,
    overrides: Option<ThresholdOverride>,
    bc_overrides: Option<ThresholdOverride>,
    /// `None` for [`TurnDetectorModel::V1`] until `SessionCreated`.
    server: Option<Layer>,
    /// Cloud-only, and absent unless both map and default are set.
    server_bc: Option<Layer>,
    thresholds: BTreeMap<String, f64>,
    default: Option<f64>,
    bc_thresholds: BTreeMap<String, f64>,
    bc_default: Option<f64>,
}

impl ThresholdOptions {
    pub fn new(
        model: TurnDetectorModel,
        overrides: Option<ThresholdOverride>,
        backchannel_overrides: Option<ThresholdOverride>,
    ) -> Self {
        let mut this = ThresholdOptions {
            model,
            overrides: normalize_overrides(overrides),
            bc_overrides: normalize_overrides(backchannel_overrides),
            // The mini model's thresholds ship with it; the cloud model's do not exist yet.
            server: match model {
                TurnDetectorModel::V1Mini => Some(local_layer()),
                TurnDetectorModel::V1 => None,
            },
            server_bc: None,
            thresholds: BTreeMap::new(),
            default: None,
            bc_thresholds: BTreeMap::new(),
            bc_default: None,
        };
        this.resolve();
        this
    }

    pub fn model(&self) -> TurnDetectorModel {
        self.model
    }

    pub fn overrides(&self) -> Option<&ThresholdOverride> {
        self.overrides.as_ref()
    }

    pub fn backchannel_overrides(&self) -> Option<&ThresholdOverride> {
        self.bc_overrides.as_ref()
    }

    pub fn thresholds(&self) -> &BTreeMap<String, f64> {
        &self.thresholds
    }

    pub fn default_threshold(&self) -> Option<f64> {
        self.default
    }

    /// `None` (no language from the STT) is scored as English, not as the global default.
    pub fn lookup(&self, language: Option<&LanguageCode>) -> Option<f64> {
        let key = language.map_or("en", LanguageCode::language);
        self.thresholds.get(key).copied().or(self.default)
    }

    /// `None` disables backchannel detection: no backchannel layer, or a threshold of zero.
    pub fn lookup_backchannel(&self, language: Option<&LanguageCode>) -> Option<f64> {
        if self.bc_thresholds.is_empty() && self.bc_default.unwrap_or(0.0) == 0.0 {
            return None;
        }
        let key = language.map_or("en", LanguageCode::language);
        self.bc_thresholds.get(key).copied().or(self.bc_default).filter(|t| *t > 0.0)
    }

    /// Before `SessionCreated` the cloud model claims every language. A tautology, kept in the
    /// reference's shape (`eot/base.py:88`).
    pub fn supports(&self, language: Option<&LanguageCode>) -> bool {
        let pending = self.model == TurnDetectorModel::V1 && self.server.is_none();
        pending || self.lookup(language).is_some()
    }

    pub fn update_overrides(&mut self, overrides: Option<ThresholdOverride>) {
        self.overrides = normalize_overrides(overrides);
        self.resolve();
    }

    pub fn update_backchannel_overrides(&mut self, overrides: Option<ThresholdOverride>) {
        self.bc_overrides = normalize_overrides(overrides);
        self.resolve();
    }

    /// An empty map or non-positive default is rejected; backchannel defaults are optional.
    pub fn update_defaults(
        &mut self,
        server_thresholds: &BTreeMap<String, f64>,
        server_default: f64,
        backchannel_thresholds: Option<&BTreeMap<String, f64>>,
        backchannel_default: f64,
    ) -> Result<(), ThresholdError> {
        if server_thresholds.is_empty() || server_default <= 0.0 {
            return Err(ThresholdError::UnusableDefaults);
        }

        self.server = Some(Layer {
            thresholds: rekey(server_thresholds.iter().map(|(k, v)| (k.as_str(), round4(*v)))),
            default: round4(server_default),
        });

        let bc_map = backchannel_thresholds
            .filter(|m| !m.is_empty())
            .map(|m| rekey(m.iter().map(|(k, v)| (k.as_str(), round4(*v)))));
        let bc_default = (backchannel_default > 0.0).then(|| round4(backchannel_default));
        self.server_bc = match (bc_map, bc_default) {
            (Some(thresholds), Some(default)) => Some(Layer { thresholds, default }),
            _ => None,
        };

        self.resolve();
        Ok(())
    }

    /// Rescales each effective threshold as a ratio of its server one so overrides survive. One-way
    /// (`eot/base.py:137`); languages without a local or non-zero server threshold are dropped.
    pub fn to_local_fallback(&mut self) {
        if self.model == TurnDetectorModel::V1Mini {
            return;
        }

        let rescaled = self.server.as_ref().filter(|s| !s.thresholds.is_empty()).map(|server| {
            server
                .thresholds
                .iter()
                .filter_map(|(language, server_t)| {
                    let effective = self.lookup(Some(&LanguageCode::new(language)))?;
                    let local = local_threshold(language)?;
                    if *server_t == 0.0 {
                        return None;
                    }
                    Some((language.clone(), local * (effective / server_t)))
                })
                .collect::<BTreeMap<String, f64>>()
        });

        self.model = TurnDetectorModel::V1Mini;
        self.server = Some(local_layer());
        self.server_bc = None;
        self.resolve();

        if let Some(rescaled) = rescaled {
            // `lookup` reads the rescaled map, so `en`'s rescaled value becomes the default.
            self.thresholds = rescaled;
            self.default = self.lookup(Some(&LanguageCode::new("en")));
        }
    }

    fn resolve(&mut self) {
        (self.thresholds, self.default) =
            Self::resolve_layer(self.server.as_ref(), self.overrides.as_ref());
        (self.bc_thresholds, self.bc_default) =
            Self::resolve_layer(self.server_bc.as_ref(), self.bc_overrides.as_ref());
    }

    /// Before the defaults arrive only a scalar resolves; a map override is re-applied when they do.
    fn resolve_layer(
        server: Option<&Layer>,
        overrides: Option<&ThresholdOverride>,
    ) -> (BTreeMap<String, f64>, Option<f64>) {
        let scalar = match overrides {
            Some(ThresholdOverride::All(value)) => Some(*value),
            _ => None,
        };
        let Some(server) = server else {
            return (BTreeMap::new(), scalar);
        };
        match overrides {
            None => (server.thresholds.clone(), Some(server.default)),
            Some(ThresholdOverride::All(value)) => (BTreeMap::new(), Some(*value)),
            Some(ThresholdOverride::PerLanguage(map)) => {
                let mut merged = server.thresholds.clone();
                merged.extend(map.iter().map(|(k, v)| (k.clone(), *v)));
                (merged, Some(server.default))
            }
        }
    }
}

#[cfg(test)]
#[path = "eot_thresholds_tests.rs"]
mod tests;
