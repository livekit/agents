//! `SpeechSteeringOptions`: a sparse opt-out, a key is set only to take an option away.

use crate::tts::markup::provider::{
    NonverbalField, Provider, ProviderExt, nonverbal_prosody_labels, nonverbal_sound_labels,
    sound_usage_hint,
};

pub use lk_agents_abi::tts::markup::steering::{
    NonverbalOptions, NonverbalSounds, Pace, SpeechSteering,
};

fn enabled(options: &NonverbalOptions, field: NonverbalField) -> bool {
    match field {
        NonverbalField::Laughing => options.laughing,
        NonverbalField::Breathing => options.breathing,
        NonverbalField::Sighing => options.sighing,
        NonverbalField::Crying => options.crying,
        NonverbalField::Vocalizing => options.vocalizing,
        NonverbalField::MouthSounds => options.mouth_sounds,
        NonverbalField::ReflexSounds => options.reflex_sounds,
    }
    .unwrap_or(true)
}

fn steering_removed(
    table: &'static [(NonverbalField, &'static [&'static str])],
    steering: Option<&SpeechSteering>,
) -> Vec<&'static str> {
    let nonverbals = steering.and_then(|s| s.nonverbal_sounds.as_ref());
    // `labels is None`: the provider has no table, so there is nothing to remove.
    if table.is_empty() {
        return Vec::new();
    }
    match nonverbals {
        None | Some(NonverbalSounds::Enabled(true)) => Vec::new(),
        Some(NonverbalSounds::Enabled(false)) => table.iter().flat_map(|(_, l)| l.iter().copied()).collect(),
        Some(NonverbalSounds::Per(flags)) => table
            .iter()
            .filter(|(f, _)| !enabled(flags, *f))
            .flat_map(|(_, l)| l.iter().copied())
            .collect(),
    }
}

pub fn allowed_sounds(provider: Provider, steering: Option<&SpeechSteering>) -> Vec<&'static str> {
    let removed = steering_removed(nonverbal_sound_labels(provider), steering);
    provider.sounds().iter().copied().filter(|s| !removed.contains(s)).collect()
}

/// Only vocal-style labels are governed, so the result is never empty.
pub fn allowed_prosody(provider: Provider, steering: Option<&SpeechSteering>) -> Vec<&'static str> {
    let removed = steering_removed(nonverbal_prosody_labels(provider), steering);
    provider.prosody().iter().copied().filter(|p| !removed.contains(p)).collect()
}

fn sound_guidance(sounds: &[&str]) -> String {
    let mut hints: Vec<&'static str> = Vec::new();
    for sound in sounds {
        if let Some(hint) = sound_usage_hint(sound)
            && !hints.contains(&hint)
        {
            hints.push(hint);
        }
    }
    let mut line =
        String::from("Non-verbal sounds: use one only where the moment genuinely earns it");
    if !hints.is_empty() {
        line.push_str(" — ");
        line.push_str(&hints.join(", "));
    }
    line.push_str(". Most turns have none; never repeat the same sound twice in a row.");
    line
}

/// `""` when nothing changes the default (the caller tests truthiness).
pub fn steering_instructions(provider: Option<Provider>, steering: &SpeechSteering) -> String {
    let mut lines: Vec<String> = Vec::new();

    // Only when steering removes part of the vocabulary and something remains: both are container
    // truthiness in the reference, so `True`, `{}` and `False` all render nothing.
    if let Some(provider) = provider
        && !steering_removed(nonverbal_sound_labels(provider), Some(steering)).is_empty()
    {
        let allowed = allowed_sounds(provider, Some(steering));
        if !allowed.is_empty() {
            lines.push(sound_guidance(&allowed));
        }
    }

    if let Some(disfluencies) = steering.disfluencies {
        lines.push(
            if disfluencies {
                "Sprinkle in natural fillers (um, uh) and openers (oh, well, so), zero to two per \
                 turn, never mechanical."
            } else {
                "No fillers (um, uh). Sound composed and fluent."
            }
            .to_string(),
        );
    }

    if let Some(pace) = steering.pace
        && pace != Pace::Normal
    {
        lines.push(format!("Keep a {} overall speaking pace.", pace.as_str()));
    }

    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::from("Delivery guidelines:\n");
    out.push_str(&lines.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `FromStr` and `as_str` disagreeing, or an unknown pace read as one of the three.
    #[test]
    fn pace_round_trips_its_three_words_and_refuses_the_rest() {
        for p in Pace::ALL {
            assert_eq!(p.as_str().parse::<Pace>(), Ok(p));
        }
        assert_eq!("brisk".parse::<Pace>().unwrap_err().to_string(), "unknown pace \"brisk\"");
    }
}
