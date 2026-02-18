from dataclasses import dataclass


@dataclass
class PersonaConfig:
    """Configuration for Keyframe Labs avatar persona.

    Provide exactly one of persona_id or persona_slug.
    """

    persona_id: str | None = None
    persona_slug: str | None = None

    def __post_init__(self) -> None:
        has_id = bool(self.persona_id and self.persona_id.strip())
        has_slug = bool(self.persona_slug and self.persona_slug.strip())
        if has_id == has_slug:
            raise ValueError("Provide exactly one of persona_id or persona_slug")
