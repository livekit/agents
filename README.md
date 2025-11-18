# Filler Word Suppression for LiveKit Agents

**Author:** Niranjani Sharma  
**Implementation Date:** November 19, 2025  
**Feature:** Advanced filler word suppression with multi-algorithm detection

---

## Overview

This implementation adds intelligent filler word suppression to LiveKit Agents, preventing speech disfluencies (like "umm", "hmm", "uh") from causing unwanted interruptions during agent responses. The system uses a 6-algorithm fusion approach with multi-language support and configurable detection modes.

### Key Features

- **6-Algorithm Detection**: Combines confidence thresholding, pattern matching, filler ratio analysis, and acknowledgment detection
- **Multi-Language Support**: Built-in patterns for 10 languages (English, Hindi, Spanish, French, German, Japanese, Chinese, Portuguese, Italian, Korean)
- **Configurable Modes**: Strict (100%), Balanced (≥70%), Lenient (≥80%) detection sensitivity
- **Smart Acknowledgments**: Recognizes meaningful single-word responses like "okay", "yeah", "yes"
- **File-Based Configuration**: Dynamic reload without restarting the agent
- **Production-Ready**: 100% generalization on unseen cases, 88.6% robustness on adversarial tests

---

## How It Works

### Suppression Pipeline

1. **User speaks** while agent is responding
2. **STT transcribes** the speech with confidence score
3. **6-Algorithm Analysis**:
   - Algorithm 1: Confidence thresholding (< 0.5 = suppress)
   - Algorithm 2: Empty text detection
   - Algorithm 3: Pattern matching with filler-only check
   - Algorithm 4: Filler ratio analysis (mode-based thresholds)
   - Algorithm 5: Short utterance special handling
   - Algorithm 6: Acknowledgment word detection
4. **Decision**: Suppress (ignore) or Allow (process as interruption)

### Detection Modes

| Mode | Threshold | Use Case |
|------|-----------|----------|
| **Strict** | 100% fillers | Only suppress pure filler sequences |
| **Balanced** | ≥70% fillers | Default - good for most scenarios |
| **Lenient** | ≥80% fillers | Allow more potential fillers through |

---

## Installation & Usage

### Basic Configuration

```python
from livekit.agents.voice.suppression_config import SuppressionConfig

# Create configuration
config = SuppressionConfig(
    detection_mode="balanced",  # or "strict", "lenient"
    min_confidence=0.5,
    enable_patterns=True,
    min_filler_ratio=0.7
)

# Use in suppression logic
should_suppress, reason = config.should_suppress_advanced(
    text="uh um hello",
    confidence=0.92,
    language="en"
)

if should_suppress:
    print(f"Suppressed: {reason}")
else:
    print(f"Allowed: {reason}")
```

### File-Based Configuration

Create `filler_suppression_config.json`:

```json
{
  "suppression_words": ["custom", "words"],
  "min_confidence": 0.5,
  "enable_patterns": true,
  "detection_mode": "balanced",
  "min_filler_ratio": 0.7
}
```

Load configuration:

```python
config = SuppressionConfig()
config.load_from_file("filler_suppression_config.json")
```

The configuration file is monitored and automatically reloaded on changes (2-second polling interval).

---

## Supported Languages

The system includes built-in patterns for 10 languages:

| Language | Code | Example Fillers |
|----------|------|----------------|
| English | en | uh, um, er, hmm, ah, oh |
| Hindi | hi | haan, arey, yaar, bas, accha |
| Spanish | es | eh, este, pues, bueno, claro |
| French | fr | euh, ben, alors, bon, hein |
| German | de | äh, ähm, also, halt, eben |
| Japanese | ja | ano, eto, ma, ne, sa |
| Chinese | zh | en, 嗯, 啊, 哦, 那个 |
| Portuguese | pt | eh, né, então, tipo |
| Italian | it | ehm, allora, cioè, quindi |
| Korean | ko | uh, um, 그, 저, 음 |

---

## Algorithm Details

### 1. Confidence Thresholding
Suppresses low-confidence transcriptions (default < 0.5) regardless of content.

### 2. Empty Text Detection
Handles empty strings and whitespace-only inputs.

### 3. Pattern Matching
Uses regex patterns to detect filler words across 10 languages, then verifies the text contains only fillers.

### 4. Filler Ratio Analysis
Calculates the percentage of filler words:
- **Strict mode**: Suppress only if 100% fillers
- **Balanced mode**: Suppress if ≥ configured ratio (default 70%)
- **Lenient mode**: Suppress if ≥ 80% fillers

### 5. Short Utterance Handling
Special logic for 1-2 word phrases to avoid false positives.

### 6. Acknowledgment Detection
Whitelists 20 common acknowledgment words that should not be suppressed when used alone:
- Positive: okay, ok, yeah, yes, yep, sure, right, correct, alright, fine, good, great, perfect, excellent
- Negative: no, nope, nah, never
- Emphatic: absolutely, definitely

---

## Testing & Validation

### Test Coverage

The implementation includes comprehensive testing with 12 test suites covering:

1. **Basic Suppression** - Core word matching
2. **Regex Pattern Matching** - Multi-language patterns
3. **Confidence Thresholds** - Dynamic threshold updates
4. **Dynamic Updates** - Add/remove suppression words
5. **Multi-Language Support** - All 10 languages
6. **Case Insensitivity** - Upper/lower/mixed case
7. **Empty/Whitespace** - Edge cases
8. **Advanced Logic** - 6-algorithm integration
9. **Detection Modes** - Strict/balanced/lenient
10. **Real-World Scenarios** - Natural conversation patterns
11. **Acknowledgment Words** - Single-word responses
12. **Filler Ratio** - Mathematical calculations

### Validation Results

- **Unit Tests**: 12/12 passed (100%)
- **Comprehensive Test**: 100/100 samples (100%)
- **Generalization Test**: 49/49 unseen cases (100%)
- **Adversarial Test**: 39/44 corner cases (88.6%)
- **Total Coverage**: ~200 test cases

See `OVERFITTING_ANALYSIS.md` for detailed validation methodology and results.

---

## Examples

### Example 1: Pure Fillers (Suppressed)
```python
config = SuppressionConfig(detection_mode="balanced")
should_suppress, reason = config.should_suppress_advanced("uh um er", 0.85, "en")
# Result: should_suppress = True
# Reason: "Pattern match - only filler words"
```

### Example 2: Question with Filler (Allowed)
```python
should_suppress, reason = config.should_suppress_advanced("uh can you help me", 0.92, "en")
# Result: should_suppress = False
# Reason: "Contains 4 non-filler words (ratio: 0.20)"
```

### Example 3: Acknowledgment (Allowed)
```python
should_suppress, reason = config.should_suppress_advanced("okay", 0.88, "en")
# Result: should_suppress = False
# Reason: "Single acknowledgment word 'okay' - allowing"
```

### Example 4: Low Confidence (Suppressed)
```python
should_suppress, reason = config.should_suppress_advanced("hello", 0.42, "en")
# Result: should_suppress = True
# Reason: "Low confidence (0.42 < 0.50)"
```

---

## File Structure

```
livekit-agents/
├── livekit/
│   └── agents/
│       └── voice/
│           ├── suppression_config.py   # Main implementation (366 lines)
│           ├── config_watcher.py       # File monitoring (115 lines)
│           └── agent_activity.py       # Integration point
tests/
└── test_suppression.py                 # Comprehensive tests (321 lines)
examples/
└── filler_suppression_config.json      # Example configuration
```

---

## Performance Characteristics

- **Pattern Compilation**: One-time at initialization
- **Regex Matching**: O(n) where n = text length
- **Filler Ratio**: O(w) where w = word count
- **Config Reload**: 2-second polling interval
- **Memory**: Minimal (~10KB for patterns)

---

## Production Deployment

### Recommended Settings

**High-Accuracy Scenario** (customer service, medical):
```python
config = SuppressionConfig(
    detection_mode="strict",
    min_confidence=0.7,
    min_filler_ratio=1.0
)
```

**Balanced Scenario** (general chatbots):
```python
config = SuppressionConfig(
    detection_mode="balanced",
    min_confidence=0.5,
    min_filler_ratio=0.7
)
```

**Permissive Scenario** (casual conversation):
```python
config = SuppressionConfig(
    detection_mode="lenient",
    min_confidence=0.4,
    min_filler_ratio=0.8
)
```

---

## Running Tests

```bash
# Navigate to tests directory
cd tests

# Run unit tests
python test_suppression.py

# Expected output:
# ======================================================================
# Running Filler Suppression Tests
# ======================================================================
# 
# ✓ Basic suppression test passed
# ✓ Regex pattern matching test passed
# ✓ Confidence threshold test passed
# ✓ Dynamic updates test passed
# ✓ Multi-language support test passed
# ✓ Case insensitivity test passed
# ✓ Empty/whitespace test passed
# ✓ Advanced suppression logic test passed
# ✓ Detection modes test passed
# ✓ Real-world scenarios test passed
# ✓ Acknowledgment words test passed
# ✓ Filler ratio calculation test passed
# 
# ======================================================================
# ✅ All 12 tests passed successfully!
# ======================================================================
```

---

## License

This implementation follows the LiveKit Agents license (Apache 2.0). See LICENSE file for details.

---

## References

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Overfitting Analysis](OVERFITTING_ANALYSIS.md) - Detailed validation methodology
- Feature implementation based on LiveKit Agents Intern Assessment Task

---

**Contact**: Niranjani Sharma  
**Repository**: https://github.com/Niranjani-sharma/livekit_agents  
**Branch**: feature/filler-suppression-niranjani
