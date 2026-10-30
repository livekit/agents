//! Labels a steering disables are omitted along with every example that demonstrates them.

use crate::tts::markup::prompts::*;
use crate::tts::markup::provider::{
    FISHAUDIO_EMOTIONS, FISHAUDIO_TONES, GEMINI_SOUNDS, Provider, XAI_INLINE, XAI_WRAPPING,
};
use crate::tts::markup::steering::{
    SpeechSteering, allowed_prosody, allowed_sounds, sound_guidance,
};

/// Drop example lines that demonstrate a *vocabulary* label not in `allowed`.
fn sound_examples(examples: &[&str], allowed: &[&str], vocabulary: &[&str]) -> Vec<String> {
    let removed: Vec<&str> = vocabulary.iter().copied().filter(|s| !allowed.contains(s)).collect();
    examples
        .iter()
        .filter(|ex| !removed.iter().any(|s| ex.contains(&format!("label=\"{s}\""))))
        .map(|ex| (*ex).to_string())
        .collect()
}

fn numbered_sections(sections: &[String]) -> String {
    sections
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn examples_block(examples: &[String]) -> String {
    let body = examples.iter().map(|ex| format!("  {ex}")).collect::<Vec<_>>().join("\n");
    format!("Examples:\n{body}")
}

fn inworld(sounds: &[&str]) -> String {
    let mut sections = vec![INWORLD_DELIVERY.to_string()];
    if !sounds.is_empty() {
        let fits = if sounds.contains(&"clear throat") { INWORLD_CLEAR_THROAT_FIT } else { "" };
        let mut section = fill(INWORLD_SOUNDS_TMPL, &[sounds[0], &sounds.join(", "), fits]);
        if sounds.contains(&"breathe") {
            section.push_str(INWORLD_BREATHE_NOTE);
        }
        sections.push(section);
    }
    sections.push(INWORLD_PAUSES.to_string());

    let mut parts = vec![
        EXPR_PREAMBLE.to_string(),
        numbered_sections(&sections),
        INWORLD_NO_PROSODY.to_string(),
        INWORLD_WRITE_FOR_EAR.to_string(),
        INWORLD_LABELS_IN_ENGLISH.to_string(),
    ];
    if sounds.contains(&"laugh") {
        parts.push(INWORLD_LAUGHTER.to_string());
    }
    let examples =
        sound_examples(&INWORLD_EXAMPLES, sounds, &crate::tts::markup::provider::INWORLD_SOUNDS);
    if !examples.is_empty() {
        parts.push(examples_block(&examples));
    }
    parts.join("\n\n")
}

fn xai(sounds: &[&str], prosody: &[&str]) -> String {
    let mut sections: Vec<String> = Vec::new();
    if !sounds.is_empty() {
        sections.push(fill(XAI_SOUNDS_TMPL, &[sounds[0], &sounds.join(", ")]));
    }
    sections.push(XAI_PAUSES.to_string());
    let tones: Vec<&str> = prosody.iter().copied().filter(|p| *p != "emphasis").collect();
    sections.push(fill(XAI_PROSODY_TMPL, &[&tones.join(", ")]));
    sections.push(XAI_EMPHASIS.to_string());

    let no_freeform = format!(
        "{XAI_NO_FREEFORM_HEAD}{}{XAI_NO_FREEFORM_TAIL}",
        if sounds.is_empty() { XAI_NO_FREEFORM_NOSOUNDS } else { XAI_NO_FREEFORM_SOUNDS }
    );
    let mut parts = vec![
        EXPR_PREAMBLE.to_string(),
        numbered_sections(&sections),
        no_freeform,
        XAI_WRITE_FOR_EAR.to_string(),
        XAI_LABELS_IN_ENGLISH.to_string(),
        XAI_KEY_DETAILS.to_string(),
    ];

    // Register guidance mentioning only concepts this steering leaves enabled.
    let mut register = vec![XAI_REGISTER_TONES];
    if ["laugh", "chuckle", "giggle"].iter().any(|s| sounds.contains(s)) {
        register.push(XAI_REGISTER_LAUGHTER);
    }
    parts.push(register.join(" "));

    let allowed: Vec<&str> = sounds.iter().chain(prosody.iter()).copied().collect();
    let vocabulary: Vec<&str> = XAI_INLINE.iter().chain(XAI_WRAPPING.iter()).copied().collect();
    let examples = sound_examples(&XAI_EXAMPLES, &allowed, &vocabulary);
    if !examples.is_empty() {
        parts.push(examples_block(&examples));
    }
    parts.join("\n\n")
}

fn fishaudio(sounds: &[&str], disfluencies: bool) -> String {
    let mut sections = vec![fill(FISH_EMOTION_TMPL, &[&FISHAUDIO_EMOTIONS.join(", ")])];
    if !sounds.is_empty() {
        sections.push(fill(FISH_SOUNDS_TMPL, &[sounds[0], &sounds.join(", ")]));
    }
    sections.push(FISH_PAUSES.to_string());
    sections.push(fill(FISH_TONE_TMPL, &[&FISHAUDIO_TONES.join(", ")]));
    sections.push(FISH_EMPHASIS.to_string());

    let mut parts = vec![
        EXPR_PREAMBLE.to_string(),
        numbered_sections(&sections),
        FISH_WRITE_FOR_EAR.to_string(),
        FISH_LABELS_IN_ENGLISH.to_string(),
    ];

    // Register guidance mentioning only concepts this steering leaves enabled.
    let mut register = vec![FISH_REGISTER_HEAVY, FISH_REGISTER_TONES];
    if ["laughing", "chuckling"].iter().any(|s| sounds.contains(s)) {
        register.push(FISH_REGISTER_LAUGHTER);
    }
    if disfluencies {
        register.push(FISH_REGISTER_FILLERS);
    }
    parts.push(register.join(" "));

    let mut pool: Vec<&str> = FISH_EXAMPLES.to_vec();
    if disfluencies {
        pool.extend_from_slice(&FISH_DISFLUENT_EXAMPLES);
    }
    let examples = sound_examples(&pool, sounds, &crate::tts::markup::provider::FISHAUDIO_SOUNDS);
    if !examples.is_empty() {
        parts.push(examples_block(&examples));
    }
    parts.join("\n\n")
}

/// The `expression` label becomes the part's `speech_metadata.style`, so its label space is
/// Gemini's own natural-language style prompt, not a vocabulary of ours.
fn gemini(sounds: &[&str]) -> String {
    let mut sections = vec![GEMINI_DELIVERY.to_string(), GEMINI_ACCENT.to_string()];
    if let Some(first) = sounds.first() {
        let all = sounds.join(", ");
        let guidance = sound_guidance(sounds);
        sections.push(fill(GEMINI_SOUNDS_TMPL, &[first, &all, first, first, &guidance]));
    }
    sections.push(GEMINI_PAUSES.to_string());
    sections.push(GEMINI_EMPHASIS.to_string());
    let examples = sound_examples(&GEMINI_EXAMPLES, sounds, &GEMINI_SOUNDS);
    [
        EXPR_PREAMBLE.to_string(),
        numbered_sections(&sections),
        GEMINI_NO_OTHER_MARKERS.to_string(),
        GEMINI_ONE_LINE.to_string(),
        GEMINI_WRITE_FOR_EAR.to_string(),
        GEMINI_LABELS_IN_ENGLISH.to_string(),
        examples_block(&examples),
    ]
    .join("\n\n")
}

/// `None` for a provider with no markup dialect.
pub(crate) fn llm_instructions(
    provider: Option<Provider>,
    steering: Option<&SpeechSteering>,
) -> Option<String> {
    let provider = provider?;
    Some(match provider {
        Provider::Cartesia => format!("{EXPR_PREAMBLE}{CARTESIA_BODY}"),
        Provider::Inworld => inworld(&allowed_sounds(provider, steering)),
        Provider::Xai => {
            xai(&allowed_sounds(provider, steering), &allowed_prosody(provider, steering))
        }
        Provider::FishAudio => fishaudio(
            &allowed_sounds(provider, steering),
            // `steering.get("disfluencies", True)` — absent means on
            steering.and_then(|s| s.disfluencies).unwrap_or(true),
        ),
        Provider::Gemini => gemini(&allowed_sounds(provider, steering)),
    })
}

/// `tmpl` with its `{}` holes filled by `args`, in order: the prompts contain no other brace.
fn fill(tmpl: &str, args: &[&str]) -> String {
    let mut pieces = tmpl.split("{}");
    let mut out = pieces.next().unwrap_or_default().to_owned();
    for (piece, arg) in pieces.zip(args) {
        out.push_str(arg);
        out.push_str(piece);
    }
    debug_assert_eq!(tmpl.matches("{}").count(), args.len(), "template holes");
    out
}
