from dataclasses import dataclass


@dataclass
class ConfigOption:
    name: str
    value: str
