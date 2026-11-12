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
    let opts =
        ThresholdOptions::new(TurnDetectorModel::V1Mini, Some(ThresholdOverride::All(0.7)), None);
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
    opts.update_defaults(&map(&[("en", 0.5), ("de", 0.4), ("xx", 0.9)]), 0.5, None, 0.0).unwrap();
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

/// ISO 639-1, ISO 639-3 and regional tags of one language resolve to the same threshold.
/// https://github.com/livekit/agents/pull/4601 https://github.com/livekit/agents/pull/1195
#[test]
fn compatibility_normalizes_iso_tags_before_lookup() {
    let mini = ThresholdOptions::new(TurnDetectorModel::V1Mini, None, None);
    for (want, tags) in [(0.3600, ["en", "eng", "en-US"]), (0.2450, ["de", "deu", "de-DE"])] {
        for tag in tags {
            assert_eq!(mini.lookup(Some(&lang(tag))), Some(want), "{tag}");
            assert!(mini.supports(Some(&lang(tag))), "{tag}");
        }
    }

    let mut cloud = ThresholdOptions::new(TurnDetectorModel::V1, None, None);
    cloud.update_defaults(&map(&[("eng", 0.5), ("ja", 0.3)]), 0.7, None, 0.0).unwrap();
    assert_eq!(cloud.lookup(Some(&lang("en"))), Some(0.5), "a server key in ISO 639-3 matches");
    assert_eq!(cloud.lookup(Some(&lang("jpn"))), Some(0.3), "a lookup in ISO 639-3 matches");
    assert_eq!(cloud.lookup(Some(&lang("xx"))), Some(0.7), "an unknown language takes the default");
}
