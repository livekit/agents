
"""
Test script to demonstrate soft-ack configuration modularity.

This script shows how soft-acks can be configured via environment variables.
"""

import os
import sys

def test_default_softacks():
    """Test default soft-ack configuration."""
    print("\n=== Test 1: Default Configuration ===")
    # Reset env var
    if "LIVEKIT_SOFT_ACKS" in os.environ:
        del os.environ["LIVEKIT_SOFT_ACKS"]
    
    # Reload module to pick up clean env
    if "livekit.agents.voice.softack_config" in sys.modules:
        del sys.modules["livekit.agents.voice.softack_config"]
    
    from livekit.agents.voice.softack_config import SOFT_ACK_SET
    print(f"Default soft-acks: {sorted(SOFT_ACK_SET)}")
    assert SOFT_ACK_SET == {"okay", "yeah", "uhhuh", "ok", "hmm", "right"}
    print("✓ Default configuration works correctly")


def test_custom_softacks():
    """Test custom soft-ack configuration via environment variable."""
    print("\n=== Test 2: Custom Configuration via ENV ===")
    
    # Set custom soft-acks
    os.environ["LIVEKIT_SOFT_ACKS"] = "yes,nope,cool,gotcha"
    
    # Reload module to pick up new env
    if "livekit.agents.voice.softack_config" in sys.modules:
        del sys.modules["livekit.agents.voice.softack_config"]
    
    from livekit.agents.voice.softack_config import SOFT_ACK_SET
    print(f"Custom soft-acks: {sorted(SOFT_ACK_SET)}")
    assert SOFT_ACK_SET == {"yes", "nope", "cool", "gotcha"}
    print("✓ Custom configuration works correctly")


def test_softack_detection():
    """Test soft-ack detection function."""
    print("\n=== Test 3: Soft-ack Detection ===")
    
    os.environ["LIVEKIT_SOFT_ACKS"] = "yeah,okay,sure"
    
    # Reload modules
    if "livekit.agents.voice.softack_config" in sys.modules:
        del sys.modules["livekit.agents.voice.softack_config"]
    
    from livekit.agents.voice.softack_config import is_soft_ack
    
    test_cases = [
        ("yeah", True),
        ("Yeah", True),
        ("YEAH.", True),
        ("okay", True),
        ("Okay?", True),
        ("sure!", True),
        ("no", False),
        ("maybe", False),
        ("", False),
    ]
    
    for text, expected in test_cases:
        result = is_soft_ack(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} is_soft_ack('{text}') = {result} (expected {expected})")
        assert result == expected, f"Failed for '{text}'"
    
    print("✓ All soft-ack detection tests passed")


def test_env_file_format():
    """Test proper .env file format."""
    print("\n=== Test 4: .env File Format Documentation ===")
    print("\nExample .env configuration:")
    print('LIVEKIT_SOFT_ACKS="okay,yeah,uh-huh,ok,hmm,right"')
    print("\nSupported formats:")
    print("- Comma-separated values")
    print("- Spaces around values are trimmed automatically")
    print("- Case-insensitive (automatically lowercased)")
    print("- Punctuation is removed during detection")
    print("✓ .env format documentation complete")


if __name__ == "__main__":
    print("=" * 60)
    print("Soft-Ack Configuration Modularity Tests")
    print("=" * 60)
    
    try:
        test_default_softacks()
        test_custom_softacks()
        test_softack_detection()
        test_env_file_format()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
