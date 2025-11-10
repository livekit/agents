"""
Verify that the installation is complete and working correctly.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required = [
        "livekit",
        "dotenv",
        "aiohttp",
    ]
    
    all_installed = True
    for package in required:
        try:
            if package == "dotenv":
                __import__("dotenv")
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_files():
    """Check if all required files exist."""
    print("\n📁 Checking project files...")
    
    required_files = [
        "interruption_handler.py",
        "agent.py",
        "config.py",
        "test_scenarios.py",
        "demo.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "QUICKSTART.md",
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (missing)")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if core modules can be imported."""
    print("\n🔧 Checking module imports...")
    
    try:
        from interruption_handler import InterruptionHandler, InterruptionConfig
        print("   ✅ interruption_handler")
    except Exception as e:
        print(f"   ❌ interruption_handler ({e})")
        return False
    
    try:
        from config import AgentConfig
        print("   ✅ config")
    except Exception as e:
        print(f"   ❌ config ({e})")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists."""
    print("\n⚙️  Checking environment configuration...")
    
    if os.path.exists(".env"):
        print("   ✅ .env file exists")
        return True
    else:
        print("   ⚠️  .env file not found (copy from .env.example)")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Running quick functionality test...")
    
    try:
        from interruption_handler import InterruptionHandler, InterruptionConfig
        
        config = InterruptionConfig.from_word_list(["uh", "umm"])
        handler = InterruptionHandler(config)
        
        # Test filler detection
        assert handler.should_ignore_speech("uh") == True
        assert handler.should_ignore_speech("hello") == False
        
        print("   ✅ Filler detection works")
        
        # Test confidence filtering
        assert handler.should_ignore_speech("hello", confidence=0.3) == True
        assert handler.should_ignore_speech("hello", confidence=0.7) == False
        
        print("   ✅ Confidence filtering works")
        
        return True
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("  LiveKit Voice Interruption Handler - Installation Verification")
    print("="*60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Project Files", check_files()))
    results.append(("Module Imports", check_imports()))
    results.append(("Environment Config", check_env_file()))
    results.append(("Functionality Test", run_quick_test()))
    
    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ All checks passed! Installation is complete.")
        print("\nNext steps:")
        print("  1. Run: python demo.py")
        print("  2. Run: python test_scenarios.py")
        print("  3. Configure .env with your credentials")
        print("  4. Run: python agent.py dev")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Make sure you're in the salescode directory")
        print("  3. Copy .env.example to .env")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

