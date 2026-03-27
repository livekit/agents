from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RIME_DEFAULT_SPEAKER_BY_LANG: dict[str, str] = {
    "ar": "sakina",
    "de": "lorelei",
    "en": "astra",
    "es": "seraphina",
    "fr": "destin",
}

RIME_ALLOWED_SPEAKERS_BY_LANG: dict[str, set[str]] = {
    "ar": {"batin", "layla", "qadir", "sakina"},
    "de": {"alfhild", "baldur", "kumara", "liesel", "lorelei", "runa"},
    "en": {
        "ahmed_mohamed",
        "albion",
        "andersen_johan",
        "anderson_emily",
        "anderson_jake",
        "anderson_james",
        "anderson_kevin",
        "andromeda",
        "arcade",
        "astra",
        "atrium",
        "bauer_felix",
        "bennett_emily",
        "bennett_ryan",
        "biondi_paul",
        "bond",
        "brooks_jordan",
        "brown_alex",
        "brown_joshua",
        "brown_madison",
        "brown_matthew",
        "brown_steven",
        "bruno_katie",
        "carter_colin",
        "celeste",
        "chatterjee_rini",
        "chen_david",
        "chen_mei",
        "clark_tyler",
        "cohen_emily",
        "cohen_jared",
        "collins_emily",
        "cooper_logan",
        "cupola",
        "das_sourav",
        "davies_james",
        "dela_cristina",
        "diallo_amara",
        "dubois_emma",
        "duncan_colin",
        "duval_pierre",
        "eliphas",
        "estelle",
        "esther",
        "eucalyptus",
        "evans_jason",
        "fern",
        "fernandez_carlos",
        "goldberg_ryan",
        "gomez_daniela",
        "gomez_diego",
        "gomez_isabel",
        "gomez_isabella",
        "gomez_javon",
        "gonzalez_maya",
        "gonzalez_michael",
        "gonzalez_ryan",
        "grayson_avery",
        "hanson_ryan",
        "harris_luke",
        "harris_lynette",
        "harrison_brianna",
        "harrison_joey",
        "harrison_mary",
        "hassan_omar",
        "henderson_brittney",
        "hernandez_juanita",
        "holliday_jewel",
        "iyer_arun",
        "jensen_mikkel",
        "johnny_jackson",
        "johnson_angela",
        "johnson_asha",
        "johnson_avery",
        "johnson_brianna",
        "johnson_cynthia",
        "johnson_elijah",
        "johnson_james",
        "johnson_joshua",
        "johnson_latisha",
        "johnson_lisa",
        "johnson_madison",
        "johnson_malachi",
        "johnson_marcel",
        "johnson_mary",
        "johnson_matthew",
        "johnson_melissa",
        "johnson_monique",
        "johnson_nia",
        "johnson_tasha",
        "johnson_tia",
        "johnson_walter",
        "kelly_aoife",
        "kelly_jennifer",
        "kelly_john",
        "kelly_maureen",
        "khan_fatima",
        "khan_umar",
        "kim_ashley",
        "kim_daniel",
        "kim_sunny",
        "kima",
        "lee_sarah",
        "levi_david",
        "levine_emily",
        "levine_joshua",
        "levy_hannah",
        "li_xiao",
        "lintel",
        "luna",
        "lyra",
        "maguire_jason",
        "malik_ahmad",
        "marinelli_giulia",
        "marlu",
        "martinez_amber",
        "martinez_ana",
        "martinez_dylan",
        "martinez_jaime",
        "martinez_leticia",
        "martinez_rosa",
        "martinez_ryan",
        "masonry",
        "mbunda_james",
        "mccarthy_james",
        "mccarthy_teresa",
        "mcdowell_peter",
        "mckinley_robert",
        "mendoza_alonzo",
        "mendoza_jesus",
        "mendoza_luz",
        "merritt_jimmy",
        "miller_cameron",
        "miller_judy",
        "miller_kelsey",
        "miller_lisa",
        "miller_logan",
        "miyamoto_akari",
        "montgomery_elise",
        "montgomery_emily",
        "morgan_brianna",
        "morgan_charles",
        "morris_colin",
        "morris_james",
        "morris_leticia",
        "morris_melvin",
        "morton_daine",
        "moss",
        "moyo_david",
        "murphy_colin",
        "murphy_emily",
        "murphy_grace",
        "murphy_hannah",
        "murphy_liam",
        "murphy_nolan",
        "neal_colin",
        "novak_emily",
        "nowak_joanna",
        "nowak_michal",
        "oculus",
        "olsson_erik",
        "orion",
        "parapet",
        "park_minseo",
        "park_sumin",
        "patel_amit",
        "patel_asha",
        "pham_daniel",
        "pilaster",
        "pola",
        "ramirez_maya",
        "ramos_raul",
        "reddy_arjun",
        "reddy_sunil",
        "ricci_giulia",
        "ricci_lorenzo",
        "rodrigues_miguel",
        "rodriguez_carla",
        "rodriguez_carlos",
        "rodriguez_eduardo",
        "rodriguez_isabela",
        "rodriguez_miguel",
        "rossi_matteo",
        "santos_angelica",
        "schmidt_joshua",
        "schmidt_julia",
        "schmidt_sophie",
        "schneider_eric",
        "schneider_jack",
        "sharma_amit",
        "silva_ana",
        "singh_anjali",
        "sirius",
        "smith_heather",
        "smith_lisa",
        "smith_michael",
        "smith_mike",
        "stucco",
        "tauro",
        "thalassa",
        "thomas_sarah",
        "thompson_kevin",
        "torres_miguel",
        "tran_david",
        "tran_jessica",
        "tran_tu",
        "transom",
        "truss",
        "tupou_leilani",
        "ursa",
        "vashti",
        "vespera",
        "walnut",
        "wang_mei",
        "watson_emily",
        "williams_anna",
        "williams_brian",
        "williams_darnell",
        "williams_jennifer",
        "williams_jordan",
        "williams_ryan",
        "williams_terence",
        "williams_tiffany",
        "wilson_emma",
        "wong_kenny",
        "wright_cooper",
        "wright_jason",
        "wright_julianne",
        "wright_michael",
        "zhang_mei",
    },
    "es": {"lark", "nova", "pola", "seraphina", "sirius", "ursa"},
    "fr": {"destin", "morel_marianne", "solstice", "serrin_joseph"},
}

AURA_DEFAULT_VOICE_BY_VARIANT: dict[str, str] = {
    "2": "aura-2-thalia-en",
    "2-en": "aura-2-thalia-en",
    "2-es": "aura-2-celeste-es",
}


def normalize_region_override(
    region_override: str | list[str] | None,
) -> str | None:
    if region_override is None:
        return None

    if isinstance(region_override, str):
        raw_values = region_override.split(",")
    else:
        raw_values = [str(value) for value in region_override]

    values = [value.strip().lower() for value in raw_values if value.strip()]
    if not values:
        return None

    return ", ".join(values)


@dataclass(frozen=True)
class ModelRef:
    raw: str
    provider: str
    model: str
    variant: str | None
    route_provider: str
    route_model: str


def parse_model_ref(model: str) -> ModelRef:
    raw = (model or "").strip()
    if not raw:
        raise ValueError("model must not be empty")

    if ":" in raw:
        model_path, variant = raw.rsplit(":", 1)
        if not variant:
            raise ValueError("model variant must not be empty")
    else:
        model_path, variant = raw, None

    parts = [p for p in model_path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"invalid model '{raw}'; expected '<provider>/<model>' or 'slng/<provider>/<model>'"
        )

    provider = parts[0]
    model_name = "/".join(parts[1:])

    if provider == "slng":
        if len(parts) < 3:
            raise ValueError(f"invalid model '{raw}'; expected 'slng/<provider>/<model>'")
        route_provider = parts[1]
        route_model = "/".join(parts[2:])
    else:
        route_provider = provider
        route_model = model_name

    if not route_provider or not route_model:
        raise ValueError(f"invalid model '{raw}'; provider and model must both be present")

    return ModelRef(
        raw=raw,
        provider=provider,
        model=model_name,
        variant=variant,
        route_provider=route_provider,
        route_model=route_model,
    )


def _rime_lang_from_variant(variant: str | None) -> str | None:
    """Extract language code from Rime Arcana variant strings.

    Handles both plain variants ("en", "es") and versioned variants ("3-en", "3-es").
    """
    if not variant:
        return None
    # Plain language code (e.g., "en", "es", "fr")
    if variant in RIME_DEFAULT_SPEAKER_BY_LANG:
        return variant
    # Versioned variant (e.g., "3-en", "3-es") — extract suffix after first hyphen
    if "-" in variant:
        lang = variant.split("-", 1)[1]
        if lang in RIME_DEFAULT_SPEAKER_BY_LANG:
            return lang
    return None


def is_deepgram_aura_model(model: str) -> bool:
    ref = parse_model_ref(model)
    return ref.route_provider == "deepgram" and ref.route_model == "aura"


def is_rime_arcana_model(model: str) -> bool:
    ref = parse_model_ref(model)
    return ref.route_provider == "rime" and ref.route_model == "arcana"


def normalize_tts_voice(model: str, voice: str) -> str:
    cleaned = (voice or "").strip()
    ref = parse_model_ref(model)

    if is_rime_arcana_model(model):
        if cleaned and cleaned != "default":
            return cleaned
        lang = _rime_lang_from_variant(ref.variant)
        if lang:
            return RIME_DEFAULT_SPEAKER_BY_LANG[lang]
        return RIME_DEFAULT_SPEAKER_BY_LANG["en"]

    if is_deepgram_aura_model(model):
        if cleaned and cleaned != "default":
            return cleaned
        if ref.variant and ref.variant in AURA_DEFAULT_VOICE_BY_VARIANT:
            return AURA_DEFAULT_VOICE_BY_VARIANT[ref.variant]
        return AURA_DEFAULT_VOICE_BY_VARIANT["2"]

    return cleaned


def validate_tts_voice(model: str, voice: str) -> list[str]:
    errors: list[str] = []
    cleaned = (voice or "").strip()
    ref = parse_model_ref(model)

    if is_deepgram_aura_model(model):
        if not cleaned:
            errors.append(
                f"tts_voice is required for {model}; expected an aura-2 voice like "
                "'aura-2-thalia-en' or 'aura-2-celeste-es'"
            )
            return errors

        if not cleaned.startswith("aura-2-"):
            errors.append(
                f"tts_voice '{cleaned}' is invalid for {model}; expected an aura-2 model id"
            )
            return errors

        if ref.variant == "2-en" and not cleaned.endswith("-en"):
            errors.append(
                f"tts_voice '{cleaned}' is invalid for {model}; expected an English '-en' voice"
            )
        if ref.variant == "2-es" and not cleaned.endswith("-es"):
            errors.append(
                f"tts_voice '{cleaned}' is invalid for {model}; expected a Spanish '-es' voice"
            )
        if ref.variant in {"2", None} and not (cleaned.endswith("-en") or cleaned.endswith("-es")):
            errors.append(
                f"tts_voice '{cleaned}' is invalid for {model}; expected an '-en' or '-es' voice"
            )

    if is_rime_arcana_model(model):
        lang = _rime_lang_from_variant(ref.variant)
        if not cleaned:
            errors.append(f"tts_voice is required for {model}; expected a valid speaker")
            return errors
        if lang and lang in RIME_ALLOWED_SPEAKERS_BY_LANG:
            allowed = RIME_ALLOWED_SPEAKERS_BY_LANG[lang]
            if cleaned not in allowed:
                allowed_speakers = ", ".join(sorted(allowed))
                errors.append(
                    f"tts_voice '{cleaned}' is not valid for {model}; "
                    f"allowed speakers: {allowed_speakers}"
                )

    # Generic check for all other models: warn if voice is empty
    if (
        not errors
        and not cleaned
        and not is_deepgram_aura_model(model)
        and not is_rime_arcana_model(model)
    ):
        errors.append(f"tts_voice is empty for {model}; a voice identifier should be provided")

    return errors


def resolve_deepgram_stt_model(model: str | None) -> str | None:
    if not model:
        return None

    ref = parse_model_ref(model)
    if ref.route_provider != "deepgram" or ref.route_model != "nova":
        return None

    variant = (ref.variant or "").lower()
    if variant.startswith("3-medical"):
        return "nova-3-medical"
    if variant.startswith("3"):
        return "nova-3"
    if variant.startswith("2"):
        return "nova-2"
    return None


def build_tts_init_payload(
    *,
    model: str,
    voice: str,
    language: str,
    sample_rate: int,
    encoding: str,
    speed: float,
    model_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(model_options or {})
    config: dict[str, Any] = {
        "language": language,
        "encoding": encoding,
        "sample_rate": sample_rate,
        "speed": speed,
    }
    payload: dict[str, Any] = {
        "type": "init",
        "model": model,
        "voice": voice,
        "language": language,
        "config": config,
    }

    if is_deepgram_aura_model(model):
        payload["model"] = voice

    if is_rime_arcana_model(model):
        config["modelId"] = options.get("modelId", "arcana")
        config["segment"] = options.get("segment", "bySentence")
        for key in (
            "speakingStyle",
            "addBreathing",
            "addDisfluencies",
            "phonemizeBetweenBrackets",
            "translateTo",
        ):
            if key in options:
                config[key] = options[key]
        payload["speaker"] = voice

    return payload


def build_stt_init_payload(
    *,
    model: str | None,
    language: str,
    sample_rate: int,
    encoding: str,
    vad_threshold: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    enable_diarization: bool,
    enable_partial_transcripts: bool,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model is not None:
        parse_model_ref(model)

    config: dict[str, Any] = {
        "language": language,
        "sample_rate": sample_rate,
        "encoding": "linear16" if encoding == "pcm_s16le" else encoding,
        "vad_threshold": vad_threshold,
        "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
        "vad_speech_pad_ms": vad_speech_pad_ms,
        "enable_diarization": enable_diarization,
        "enable_partials": enable_partial_transcripts,
        "enable_partial_transcripts": enable_partial_transcripts,
    }

    if min_speakers is not None:
        config["min_speakers"] = min_speakers
    if max_speakers is not None:
        config["max_speakers"] = max_speakers

    if model_options:
        config.update(model_options)

    partials_value = config.get(
        "enable_partials",
        config.get("enable_partial_transcripts", enable_partial_transcripts),
    )
    config["enable_partials"] = partials_value
    config["enable_partial_transcripts"] = partials_value

    payload: dict[str, Any] = {"type": "init", "config": config}

    deepgram_model = resolve_deepgram_stt_model(model)
    if deepgram_model:
        payload["model"] = deepgram_model

    return payload
