#!/usr/bin/env python3
"""
Verify LiveKit Voice Agent Dependencies Installation
This script checks if all required packages are installed and properly configured.
"""

import sys
import subprocess
from typing import Tuple, List

def check_python_version() -> bool:
    """Check if Python version is compatible (3.9+)"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} - FAILED (requires 3.9+)")
        return False

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"

def check_core_packages() -> bool:
    """Check core LiveKit packages"""
    print("\n=== Core LiveKit Packages ===")
    core_packages = [
        ("livekit", "livekit"),
        ("livekit-agents", "livekit.agents"),
        ("livekit-api", "livekit_api"),
        ("livekit-protocol", "livekit.protocol"),
    ]
    
    all_ok = True
    for package, import_name in core_packages:
        is_installed, version = check_package(package, import_name)
        status = "✓" if is_installed else "✗"
        print(f"{status} {package}: {version}")
        all_ok = all_ok and is_installed
    
    return all_ok

def check_plugin_packages() -> bool:
    """Check required plugin packages (STT, VAD, TTS)"""
    print("\n=== Required Plugin Packages ===")
    plugins = [
        ("livekit-plugins-deepgram", "livekit.plugins.deepgram"),
        ("livekit-plugins-silero", "livekit.plugins.silero"),
        ("livekit-plugins-cartesia", "livekit.plugins.cartesia"),
    ]
    
    all_ok = True
    for package, import_name in plugins:
        is_installed, version = check_package(package, import_name)
        status = "✓" if is_installed else "✗"
        print(f"{status} {package}: {version}")
        all_ok = all_ok and is_installed
    
    return all_ok

def check_optional_plugins() -> bool:
    """Check optional plugin packages"""
    print("\n=== Optional Plugin Packages ===")
    plugins = [
        ("livekit-plugins-openai", "livekit.plugins.openai"),
        ("livekit-plugins-google", "livekit.plugins.google"),
        ("livekit-plugins-anthropic", "livekit.plugins.anthropic"),
        ("livekit-plugins-azure", "livekit.plugins.azure"),
        ("livekit-plugins-groq", "livekit.plugins.groq"),
    ]
    
    for package, import_name in plugins:
        is_installed, version = check_package(package, import_name)
        status = "✓" if is_installed else "○"  # ○ = optional
        print(f"{status} {package}: {version}")
    
    return True  # Optional, so always return True

def check_dependencies() -> bool:
    """Check core dependencies"""
    print("\n=== Core Dependencies ===")
    dependencies = [
        ("aiohttp", None),
        ("numpy", None),
        ("pydantic", None),
        ("click", None),
        ("typer", None),
        ("sounddevice", None),
        ("onnxruntime", None),
        ("av", None),
        ("opentelemetry.api", "opentelemetry"),
        ("prometheus_client", None),
        ("dotenv", "dotenv"),
    ]
    
    all_ok = True
    for package, import_name in dependencies:
        is_installed, version = check_package(package, import_name or package)
        status = "✓" if is_installed else "✗"
        print(f"{status} {package}: {version}")
        all_ok = all_ok and is_installed
    
    return all_ok

def check_environment_config() -> bool:
    """Check environment configuration"""
    print("\n=== Environment Configuration ===")
    import os
    from pathlib import Path
    
    env_files = [
        Path(".env"),
        Path("examples/voice_agents/.env"),
        Path(".env.example"),
    ]
    
    env_found = False
    for env_file in env_files:
        if env_file.exists():
            print(f"✓ Found .env file: {env_file}")
            env_found = True
            
            # Check for required keys
            with open(env_file) as f:
                content = f.read().upper()
                if "LIVEKIT_URL" in content:
                    print("  ✓ LIVEKIT_URL configured")
                if "DEEPGRAM_API_KEY" in content:
                    print("  ✓ DEEPGRAM_API_KEY configured")
                if "CARTESIA_API_KEY" in content:
                    print("  ✓ CARTESIA_API_KEY configured")
                if "LIVEKIT_SOFT_ACKS" in content:
                    print("  ✓ LIVEKIT_SOFT_ACKS configured")
    
    if not env_found:
        print("○ No .env file found (optional, can use environment variables)")
    
    return True

def check_audio_devices() -> bool:
    """Check audio device availability"""
    print("\n=== Audio Devices ===")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if isinstance(devices, dict):
            print(f"✓ Audio devices available: {len(devices)} device(s)")
            print(f"  Default input: {sd.default.device[0]}")
            print(f"  Default output: {sd.default.device[1]}")
        else:
            print(f"✓ Audio devices available: {len(devices)} device(s)")
        return True
    except Exception as e:
        print(f"✗ Audio device check failed: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("LiveKit Voice Agent Dependencies Verification")
    print("=" * 60)
    
    results = []
    
    # Check Python version
    results.append(("Python Version", check_python_version()))
    
    # Check core packages
    results.append(("Core Packages", check_core_packages()))
    
    # Check plugin packages
    results.append(("Plugin Packages (STT/VAD/TTS)", check_plugin_packages()))
    
    # Check optional plugins
    results.append(("Optional Plugins", check_optional_plugins()))
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Check environment configuration
    results.append(("Environment Config", check_environment_config()))
    
    # Check audio devices
    results.append(("Audio Devices", check_audio_devices()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    critical_checks = [
        "Python Version",
        "Core Packages",
        "Plugin Packages (STT/VAD/TTS)",
        "Dependencies",
    ]
    
    critical_ok = all(result for check, result in results if check in critical_checks)
    overall_ok = all(result for _, result in results)
    
    for check, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        critical = " [CRITICAL]" if check in critical_checks and not result else ""
        print(f"{status}: {check}{critical}")
    
    print("\n" + "=" * 60)
    if critical_ok:
        print("✓ ALL CRITICAL CHECKS PASSED - Ready to run!")
        print("\nYou can start your voice agent with:")
        print("  python examples/voice_agents/your_agent.py")
    else:
        print("✗ CRITICAL CHECKS FAILED - Please install missing packages:")
        print("\n  pip install -r requirements.txt")
    
    if not overall_ok:
        print("\nNote: Some optional checks failed. See above for details.")
    
    print("=" * 60)
    
    return 0 if critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())
