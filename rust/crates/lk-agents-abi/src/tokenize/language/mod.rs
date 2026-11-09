mod code;
mod data;
mod eot_thresholds;

pub use code::{
    LanguageCode, LanguageFunctions, base_language, iso, language_name, normalize, region,
};
pub use eot_thresholds::{
    LOCAL_LANGUAGES, ThresholdError, ThresholdOptions, ThresholdOverride, TurnDetectorModel,
};

// One table for every SDK: Python's `LanguageCode` stays a `str` subclass calling these functions.
