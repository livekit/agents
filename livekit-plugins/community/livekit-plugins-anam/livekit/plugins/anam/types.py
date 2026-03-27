from dataclasses import dataclass


@dataclass
class PersonaConfig:
    """Configuration for Anam avatar persona"""

    name: str
    avatarId: str
