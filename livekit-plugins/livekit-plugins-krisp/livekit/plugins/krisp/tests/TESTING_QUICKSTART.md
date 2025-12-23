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

## 2. Real-Time Microphone Test

Test with live audio (use headphones!):

```bash
# Install dependencies
pip install sounddevice

# Run test
python test_realtime_microphone.py
```

**What it does:**
- Captures from your microphone
- Filters in real-time
- Plays back through speakers
- You hear the noise reduction live!

**Tips:**
- Speak into the mic
- Make background noise (keyboard, fan)
- Notice how noise is reduced while voice stays clear

---

## Quick Example

```bash
# 1. Set model path
export KRISP_VIVA_FILTER_MODEL_PATH=/path/to/model.kef

# 2. Test with a file
python test_audio_filtering.py input.wav output.wav --visualize

# 3. Listen to results
# Compare input.wav and output.wav

# 4. Check the spectrogram
# Open krisp_comparison.png to see visual proof
```

---

## Expected Results

✅ **Good results:**
- Background noise reduced
- Voice clarity preserved
- Natural sound
- Spectrograms show cleaner frequency content

❌ **Issues to watch for:**
- Voice sounds muffled → Try lower suppression level (`--level 80`)
- No noise reduction → Check model path, verify audio has noise
- Artifacts/distortion → Try different frame duration

---

## Parameters to Adjust

```bash
# Less aggressive (preserves more ambient sound)
python test_audio_filtering.py input.wav output.wav --level 70

# Lower CPU usage
python test_audio_filtering.py input.wav output.wav --frame-duration 20

# Both
python test_audio_filtering.py input.wav output.wav --level 80 --frame-duration 20
```

---

## Need Test Audio?

Record yourself with background noise:
```bash
# Mac (if ffmpeg installed)
ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 test.wav

# Or just use any WAV file with speech + noise
```

---

See [TESTING.md](TESTING.md) for comprehensive testing guide.

