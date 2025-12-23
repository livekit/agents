# Quick Testing Instructions

## 1. File-Based Testing (Easiest)

Process audio files to verify noise reduction:

```bash
# Install dependencies
pip install soundfile matplotlib scipy

# Set model path
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef

# Process an audio file
python test_audio_filtering.py noisy_input.wav clean_output.wav --visualize
```

**What it does:**
- Loads your audio file
- Processes through Krisp filter
- Saves cleaned audio
- Shows before/after spectrograms

**Then:** Listen to both files to hear the difference!

---


## Quick Example

```bash
# 1. Set model path
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef

# 2. Test with a file
python test_audio_filtering.py input.wav output.wav --visualize

# 3. Listen to results
# Compare input.wav and output.wav

---

## Expected Results

✅ **Good results:**
- Background noise reduced
- Voice clarity preserved
- Natural sound
- Spectrograms show cleaner frequency content

❌ **Issues to watch for:**
- No noise reduction → Check model path, verify audio has noise
- Artifacts/distortion → Try different frame duration

---

## Parameters to Adjust

```bash
# Less aggressive (preserves more ambient sound)
python test_audio_filtering.py input.wav output.wav --level 70

# Different frame duration
python test_audio_filtering.py input.wav output.wav --frame-duration 20

# Both
python test_audio_filtering.py input.wav output.wav --level 80 --frame-duration 20
```