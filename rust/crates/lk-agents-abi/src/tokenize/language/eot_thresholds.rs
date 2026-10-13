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

/// `None` is `NOT_GIVEN`. `All` replaces the per-language map, `PerLanguage` merges over it.
#[derive(Debug, Clone, PartialEq)]
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
mod tests {
    use super::*;

    fn map(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    }

    fn lang(code: &str) -> LanguageCode {
        LanguageCode::new(code)
    }

    // Catches: an out-of-order row making the binary search miss a language.
    #[test]
    fn local_languages_is_sorted() {
        assert!(LOCAL_LANGUAGES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    // Catches: the cloud model refusing every language until `SessionCreated`.
    #[test]
    fn the_cloud_model_claims_every_language_before_its_defaults_arrive() {
        let cloud = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
        assert_eq!(cloud.lookup(Some(&lang("de"))), None, "nothing to look up yet");
        assert!(cloud.supports(Some(&lang("de"))));
        assert!(cloud.supports(None));

        let mini = ThresholdOptions::new(TurnDetectorModel::V1Mini, None, None);
        assert_eq!(mini.lookup(Some(&lang("de"))), Some(0.2450), "shipped with the model");
        assert_eq!(mini.default_threshold(), Some(0.3600), "the English row is the default");
    }

    // Catches: keying the resolved map on the raw override string.
    #[test]
    fn override_keys_go_through_language_normalisation() {
        let opts = ThresholdOptions::new(
            TurnDetectorModel::V1Mini,
            Some(ThresholdOverride::PerLanguage(map(&[("en-US", 0.9), ("German", 0.1)]))),
            None,
        );
        assert_eq!(opts.lookup(Some(&lang("en-GB"))), Some(0.9), "keyed on en, not en-US");
        assert_eq!(opts.lookup(Some(&lang("deu"))), Some(0.1));
    }

    // Catches: merging a scalar override instead of replacing the map.
    #[test]
    fn a_scalar_override_replaces_the_whole_map() {
        let opts = ThresholdOptions::new(
            TurnDetectorModel::V1Mini,
            Some(ThresholdOverride::All(0.7)),
            None,
        );
        assert!(opts.thresholds().is_empty());
        assert_eq!(opts.lookup(Some(&lang("de"))), Some(0.7));
        assert_eq!(opts.lookup(Some(&lang("en"))), Some(0.7));
        assert_eq!(opts.lookup(None), Some(0.7));
    }

    // Catches: a map override replacing the server map wholesale.
    #[test]
    fn a_map_override_merges_over_the_server_map() {
        let opts = ThresholdOptions::new(
            TurnDetectorModel::V1Mini,
            Some(ThresholdOverride::PerLanguage(map(&[("en", 0.9)]))),
            None,
        );
        assert_eq!(opts.lookup(Some(&lang("en"))), Some(0.9));
        assert_eq!(opts.lookup(Some(&lang("de"))), Some(0.2450), "untouched");
        assert_eq!(opts.default_threshold(), Some(0.3600), "still the server default");
    }

    // Catches: a scalar override not resolving before the server answers (a map must not).
    #[test]
    fn only_a_scalar_override_resolves_before_the_server_answers() {
        let scalar =
            ThresholdOptions::new(TurnDetectorModel::V1, Some(ThresholdOverride::All(0.8)), None);
        assert_eq!(scalar.lookup(Some(&lang("de"))), Some(0.8));

        let mut per_language = ThresholdOptions::new(
            TurnDetectorModel::V1,
            Some(ThresholdOverride::PerLanguage(map(&[("de", 0.8)]))),
            None,
        );
        assert_eq!(per_language.lookup(Some(&lang("de"))), None);
        per_language.update_defaults(&map(&[("de", 0.4)]), 0.5, None, 0.0).unwrap();
        assert_eq!(per_language.lookup(Some(&lang("de"))), Some(0.8), "re-applied on arrival");
    }

    // Catches: accepting a malformed `SessionCreated`, or it disturbing the previous state.
    #[test]
    fn unusable_server_defaults_are_rejected_without_mutating_state() {
        let mut opts = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
        assert_eq!(
            opts.update_defaults(&map(&[]), 0.5, None, 0.0),
            Err(ThresholdError::UnusableDefaults)
        );
        assert_eq!(
            opts.update_defaults(&map(&[("en", 0.5)]), 0.0, None, 0.0),
            Err(ThresholdError::UnusableDefaults)
        );
        assert!(opts.supports(None), "still pending, not poisoned");
        assert_eq!(opts.default_threshold(), None);

        opts.update_defaults(&map(&[("en", 0.512_345_6)]), 0.5, None, 0.0).unwrap();
        assert_eq!(opts.lookup(Some(&lang("en"))), Some(0.5123), "rounded to 4dp");
    }

    // Catches: enabling backchannel without a server map, or treating zero as "always".
    #[test]
    fn backchannel_stays_off_unless_the_server_sends_both_halves() {
        let mini = ThresholdOptions::new(TurnDetectorModel::V1Mini, None, None);
        assert_eq!(mini.lookup_backchannel(Some(&lang("en"))), None);

        let mut opts = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
        opts.update_defaults(&map(&[("en", 0.5)]), 0.5, Some(&map(&[("en", 0.4)])), 0.0).unwrap();
        assert_eq!(opts.lookup_backchannel(Some(&lang("en"))), None, "map but no default");

        opts.update_defaults(&map(&[("en", 0.5)]), 0.5, None, 0.3).unwrap();
        assert_eq!(opts.lookup_backchannel(Some(&lang("en"))), None, "default but no map");

        opts.update_defaults(&map(&[("en", 0.5)]), 0.5, Some(&map(&[("en", 0.4)])), 0.3).unwrap();
        assert_eq!(opts.lookup_backchannel(Some(&lang("en"))), Some(0.4));
        assert_eq!(opts.lookup_backchannel(Some(&lang("de"))), Some(0.3), "falls to the default");

        // A user can still enable it before any server map, via a scalar override.
        let scalar =
            ThresholdOptions::new(TurnDetectorModel::V1, None, Some(ThresholdOverride::All(0.25)));
        assert_eq!(scalar.lookup_backchannel(None), Some(0.25));
    }

    // Catches: no STT language falling back to the global default instead of English.
    #[test]
    fn an_absent_language_is_scored_as_english() {
        let opts = ThresholdOptions::new(
            TurnDetectorModel::V1Mini,
            Some(ThresholdOverride::PerLanguage(map(&[("en", 0.9)]))),
            None,
        );
        assert_eq!(opts.lookup(None), Some(0.9), "en, not the 0.36 default");
    }

    // Catches: fallback installing the shipped table and discarding overrides.
    #[test]
    fn the_local_fallback_rescales_the_users_tuning() {
        let mut opts = ThresholdOptions::new(
            TurnDetectorModel::V1,
            Some(ThresholdOverride::PerLanguage(map(&[("en", 0.25)]))),
            None,
        );
        opts.update_defaults(&map(&[("en", 0.5), ("de", 0.4), ("xx", 0.9)]), 0.5, None, 0.0)
            .unwrap();
        opts.to_local_fallback();

        assert_eq!(opts.model(), TurnDetectorModel::V1Mini);
        assert_eq!(opts.lookup(Some(&lang("en"))), Some(0.18), "0.36 * (0.25 / 0.5)");
        assert_eq!(opts.lookup(Some(&lang("de"))), Some(0.2450), "0.245 * (0.4 / 0.4)");
        assert_eq!(opts.default_threshold(), Some(0.18), "the rescaled en, not the stock 0.36");
        assert_eq!(
            opts.lookup(Some(&lang("xx"))),
            Some(0.18),
            "no local row, so it resolves through the default"
        );
    }

    // Catches: a second cloud failure moving the thresholds (repeated fallback must be a fixpoint;
    // only visible in the resolved map).
    #[test]
    fn the_local_fallback_is_a_fixpoint() {
        let mut opts = ThresholdOptions::new(
            TurnDetectorModel::V1,
            Some(ThresholdOverride::PerLanguage(map(&[("en", 0.25)]))),
            None,
        );
        opts.update_defaults(&map(&[("en", 0.5), ("de", 0.4)]), 0.5, None, 0.0).unwrap();
        opts.to_local_fallback();

        let once = (opts.thresholds().clone(), opts.default_threshold(), opts.model());
        assert_eq!(once.0, map(&[("en", 0.18), ("de", 0.2450)]));
        opts.to_local_fallback();
        assert_eq!((opts.thresholds().clone(), opts.default_threshold(), opts.model()), once);
    }

    // Catches: rescaling against a server map that never arrived.
    #[test]
    fn falling_back_before_any_server_defaults_installs_the_plain_local_table() {
        let mut opts = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
        opts.to_local_fallback();
        assert_eq!(opts.lookup(Some(&lang("en"))), Some(0.3600));
        assert_eq!(opts.lookup(Some(&lang("de"))), Some(0.2450));
    }

    // Catches: carrying cloud backchannel thresholds into the local model.
    #[test]
    fn the_local_fallback_drops_the_backchannel_layer() {
        let mut opts = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
        opts.update_defaults(&map(&[("en", 0.5)]), 0.5, Some(&map(&[("en", 0.4)])), 0.3).unwrap();
        assert_eq!(opts.lookup_backchannel(Some(&lang("en"))), Some(0.4));
        opts.to_local_fallback();
        assert_eq!(opts.lookup_backchannel(Some(&lang("en"))), None);
    }
}
