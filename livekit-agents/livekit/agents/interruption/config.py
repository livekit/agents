"""
Configuration file and quick start script for LiveKit Interruption Handler
"""

import os
from dataclasses import dataclass
from typing import list

import yaml


@dataclass
class InterruptionConfig:
    """Configuration for interruption handling"""

    # Filler words configuration
    english_fillers: list[str]
    hindi_fillers: list[str]
    custom_fillers: list[str]

    # Behavior settings
    confidence_threshold: float
    min_word_duration: float
    log_all_events: bool
    allow_runtime_updates: bool

    # Performance settings
    max_history_size: int
    enable_statistics: bool

    @classmethod
    def from_yaml(cls, filepath: str) -> "InterruptionConfig":
        """Load configuration from YAML file"""
        with open(filepath) as f:
            data = yaml.safe_load(f)

        return cls(
            english_fillers=data.get("english_fillers", []),
            hindi_fillers=data.get("hindi_fillers", []),
            custom_fillers=data.get("custom_fillers", []),
            confidence_threshold=data.get("confidence_threshold", 0.6),
            min_word_duration=data.get("min_word_duration", 0.2),
            log_all_events=data.get("log_all_events", True),
            allow_runtime_updates=data.get("allow_runtime_updates", True),
            max_history_size=data.get("max_history_size", 1000),
            enable_statistics=data.get("enable_statistics", True),
        )

    @classmethod
    def from_env(cls) -> "InterruptionConfig":
        """Load configuration from environment variables"""
        return cls(
            english_fillers=os.getenv(
                "INTERRUPTION_ENGLISH_FILLERS", "uh,um,umm,hmm,ah,er,mm"
            ).split(","),
            hindi_fillers=os.getenv("INTERRUPTION_HINDI_FILLERS", "haan,han,ha,achha,theek").split(
                ","
            ),
            custom_fillers=os.getenv("INTERRUPTION_CUSTOM_FILLERS", "").split(","),
            confidence_threshold=float(os.getenv("INTERRUPTION_CONFIDENCE_THRESHOLD", "0.6")),
            min_word_duration=float(os.getenv("INTERRUPTION_MIN_WORD_DURATION", "0.2")),
            log_all_events=os.getenv("INTERRUPTION_LOG_ALL", "true").lower() == "true",
            allow_runtime_updates=os.getenv("INTERRUPTION_ALLOW_UPDATES", "true").lower() == "true",
            max_history_size=int(os.getenv("INTERRUPTION_MAX_HISTORY", "1000")),
            enable_statistics=os.getenv("INTERRUPTION_ENABLE_STATS", "true").lower() == "true",
        )

    def get_all_ignored_words(self) -> list[str]:
        """Get combined list of all ignored words"""
        all_words = []
        all_words.extend(self.english_fillers)
        all_words.extend(self.hindi_fillers)
        all_words.extend([w for w in self.custom_fillers if w])
        return list(set(all_words))  # Remove duplicates

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        data = {
            "english_fillers": self.english_fillers,
            "hindi_fillers": self.hindi_fillers,
            "custom_fillers": self.custom_fillers,
            "confidence_threshold": self.confidence_threshold,
            "min_word_duration": self.min_word_duration,
            "log_all_events": self.log_all_events,
            "allow_runtime_updates": self.allow_runtime_updates,
            "max_history_size": self.max_history_size,
            "enable_statistics": self.enable_statistics,
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Default configuration YAML template
DEFAULT_CONFIG_YAML = """
# LiveKit Interruption Handler Configuration

# English filler words to ignore during agent speech
english_fillers:
  - uh
  - um
  - umm
  - hmm
  - hm
  - mm
  - mhmm
  - ah
  - er
  - eh

# Hindi filler words
hindi_fillers:
  - haan
  - han
  - ha
  - achha
  - theek
  - ji

# Custom filler words for your use case
custom_fillers:
  - yeah
  - yep
  - okay  # Only if you want to ignore casual acknowledgments

# Minimum confidence score to process (0.0 - 1.0)
# Higher = stricter (ignores more), Lower = lenient (ignores less)
confidence_threshold: 0.6

# Minimum word duration in seconds
min_word_duration: 0.2

# Log all interruption events (not just valid ones)
log_all_events: true

# Allow updating ignored words at runtime
allow_runtime_updates: true

# Maximum interruption history to keep in memory
max_history_size: 1000

# Enable statistics tracking
enable_statistics: true
"""


def create_default_config(filepath: str = "interruption_config.yaml"):
    """Create a default configuration file"""
    with open(filepath, "w") as f:
        f.write(DEFAULT_CONFIG_YAML)
    print(f"Created default configuration at: {filepath}")


# Quick start script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LiveKit Interruption Handler Setup")
    parser.add_argument(
        "--create-config", action="store_true", help="Create default configuration file"
    )
    parser.add_argument(
        "--config-file", default="interruption_config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--test", action="store_true", help="Run quick configuration test")

    args = parser.parse_args()

    if args.create_config:
        create_default_config(args.config_file)

    elif args.test:
        print("Testing configuration loading...")

        # Test YAML loading
        try:
            config = InterruptionConfig.from_yaml(args.config_file)
            print(f"✅ Loaded from YAML: {args.config_file}")
            print(f"   - Ignored words: {len(config.get_all_ignored_words())}")
            print(f"   - Confidence threshold: {config.confidence_threshold}")
        except FileNotFoundError:
            print(f"❌ Config file not found: {args.config_file}")
            print("   Run with --create-config to create default")

        # Test environment loading
        print("\n Testing environment variable loading...")
        config_env = InterruptionConfig.from_env()
        print("✅ Loaded from environment")
        print(f"   - Ignored words: {len(config_env.get_all_ignored_words())}")
        print(f"   - Confidence threshold: {config_env.confidence_threshold}")

        print("\n✅ Configuration test complete!")

    else:
        parser.print_help()


# Example usage in code
"""
# Load from YAML
config = InterruptionConfig.from_yaml('interruption_config.yaml')
handler = IntelligentInterruptionHandler(
    ignored_words=config.get_all_ignored_words(),
    confidence_threshold=config.confidence_threshold,
    log_all_events=config.log_all_events,
    allow_runtime_updates=config.allow_runtime_updates
)

# Or load from environment
config = InterruptionConfig.from_env()
handler = IntelligentInterruptionHandler(
    ignored_words=config.get_all_ignored_words(),
    confidence_threshold=config.confidence_threshold
)
"""
