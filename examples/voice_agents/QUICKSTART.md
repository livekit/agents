# Quick Start Guide - Filler Interruption Handler

## 🚀 5-Minute Setup

### Step 1: Environment Setup (1 min)

```bash
# Navigate to voice_agents directory
cd /Users/satyamkumar/Desktop/salescode_ai2/agents1/examples/voice_agents

# Activate virtual environment (if not already)
source ../../../venv/bin/activate

# Copy environment template
cp .env.filler_example .env
```

### Step 2: Configure API Keys (2 min)

Edit `.env` file with your keys:

```bash
# Minimum required:
LIVEKIT_URL=wss://your-server.com
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
OPENAI_API_KEY=your_openai_key

# Optional (use defaults if not set):
ASSEMBLYAI_API_KEY=your_key
CARTESIA_API_KEY=your_key
```

### Step 3: Test Handler Logic (30 sec)

```bash
# Test the core logic without LiveKit connection
python test_filler_handler.py
```

Expected output:
```
✅ PASS | Test 1: Filler 'umm' during agent speech
✅ PASS | Test 2: Multiple fillers 'uh hmm yeah' during agent speech
...
✅ ALL TESTS PASSED!
```

### Step 4: Run the Agent (1 min)

```bash
# Start the filler-aware agent
python filler_aware_agent.py start
```

### Step 5: Test with Real Voice (30 sec)

1. Connect to the LiveKit room (URL in terminal)
2. Wait for agent to speak
3. Say "umm" → Agent continues
4. Say "wait" → Agent stops
5. Check terminal logs for confirmation

---

## 🎯 Testing Scenarios

### Scenario 1: Filler Ignored
```
YOU: [while agent speaking] "umm"
EXPECTED: Agent continues speaking
LOG: 🚫 FILLER DETECTED | Transcript: 'umm'
```

### Scenario 2: Real Interruption
```
YOU: [while agent speaking] "wait a second"
EXPECTED: Agent stops immediately
LOG: ✅ VALID SPEECH | Transcript: 'wait a second'
```

### Scenario 3: Mixed Input
```
YOU: [while agent speaking] "umm okay stop"
EXPECTED: Agent stops (contains real words)
LOG: ✅ VALID SPEECH | Real words detected: ['okay', 'stop']
```

### Scenario 4: Filler When Quiet
```
YOU: [agent is quiet] "umm hello"
EXPECTED: Processed normally
LOG: ✅ VALID SPEECH | Reason: agent_not_speaking
```

---

## 🔧 Customization

### Change Filler Words

Edit `.env`:
```bash
# Add your language-specific fillers
IGNORED_FILLER_WORDS=uh,um,umm,hmm,haan,achha,theek
```

### Adjust Sensitivity

Edit `.env`:
```bash
# Longer timeout = more tolerant of brief pauses
FALSE_INTERRUPTION_TIMEOUT=1.5
```

---

## 📊 Monitoring

### Watch Logs in Real-Time

```bash
# In another terminal
tail -f /path/to/agent/logs.txt | grep "FILLER\|VALID"
```

### Check Statistics

Look for this in logs when session ends:
```
=== Filler Handler Statistics ===
Total transcripts: 45
Ignored fillers: 12
Valid interruptions: 8
Processed while idle: 25
```

---

## 🐛 Troubleshooting

### Agent Not Starting?
```bash
# Check if all dependencies installed
pip list | grep livekit

# Verify API keys are set
echo $OPENAI_API_KEY
```

### Fillers Not Being Ignored?
```bash
# Check your filler word list
grep IGNORED_FILLER_WORDS .env

# Enable debug logging
export LOG_LEVEL=DEBUG
python filler_aware_agent.py start
```

### Can't Connect to Room?
```bash
# Verify LiveKit credentials
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY

# Test basic agent first
python basic_agent.py start
```

---

## 📝 Checklist

Before testing:
- [ ] Virtual environment activated
- [ ] `.env` file configured with API keys
- [ ] Dependencies installed (`pip list | grep livekit`)
- [ ] Test script passes (`python test_filler_handler.py`)

During testing:
- [ ] Agent starts without errors
- [ ] Can connect to LiveKit room
- [ ] Fillers are logged as ignored
- [ ] Real speech causes interruption
- [ ] Logs show state transitions

---

## 🎓 Understanding the Logs

### Log Format
```
[LEVEL] MESSAGE | Details
```

### Key Log Messages

**Agent State Changes:**
```
INFO | Agent state: idle -> speaking
INFO | Agent state: speaking -> listening
```

**Filler Detected:**
```
INFO | 🚫 FILLER DETECTED | Transcript: 'umm' | Reason: filler_only_during_agent_speech | Agent: speaking
```

**Valid Interruption:**
```
INFO | ✅ VALID SPEECH | Transcript: 'wait' | Reason: real_speech_detected | Agent: speaking
```

**Normal Processing:**
```
INFO | ✅ VALID SPEECH | Transcript: 'hello' | Reason: agent_not_speaking | Agent: idle
```

---

## 🚀 Next Steps

After basic testing works:

1. **Tune for your use case:**
   - Adjust filler word list
   - Modify timeout values
   - Add language-specific words

2. **Test edge cases:**
   - Very fast speech
   - Background noise
   - Accented speech
   - Multiple languages

3. **Monitor performance:**
   - Check statistics
   - Review ignored vs valid ratio
   - Look for false positives

4. **Extend functionality:**
   - Add confidence threshold filtering
   - Implement custom logic
   - Create language-specific handlers

---

## 📚 Reference

- **Full Documentation:** `README_FILLER_HANDLER.md`
- **Handler Code:** `filler_interrupt_handler.py`
- **Example Agent:** `filler_aware_agent.py`
- **Tests:** `test_filler_handler.py`

---

## ✅ Success Criteria

You know it's working when:
- ✅ Test script passes all tests
- ✅ Agent starts without errors
- ✅ Logs show filler detection
- ✅ Real interruptions work
- ✅ Agent continues on fillers
- ✅ Statistics are logged on exit

---

**Ready to go!** 🎉

If you encounter issues, check the full README or examine the logs for details.
