# LiveKit Voice Interruption Handling 

## Overview
This project extends the LiveKit Agent to intelligently handle user speech by ignoring filler words and accurately detecting genuine interruptions in real time.  
The goal was to enhance the speech processing system so that unnecessary filler sounds do not trigger unwanted actions during a conversation.

## Objective
The main objective of this assignment was to design and integrate a **Filler Handler** into the LiveKit framework that:
- Detects filler words such as “uh”, “umm”, “hmm”, and “haan”.
- Differentiates them from real speech or interruption commands.
- Improves the reliability of live speech recognition and interaction flow.

## What Changed
- Added a new file: `filler_handler.py`  
  - Introduces the `FillerHandler` class.  
  - Uses regular expressions to match and filter filler words.  
  - Employs a configurable confidence threshold to ensure accurate recognition.
- Integrated the handler into `audio_recognition.py`  
  - Imported the new class and instantiated it for use within the audio recognition loop.  
  - Ensures real-time detection of fillers versus meaningful speech.
- Created a new test file: `test_simple_filler.py`  
  - Validates that fillers are ignored and real interruptions are detected.
- Updated all dependencies and ensured compatibility with the LiveKit Agents repository.

## Implementation Details
- **FillerHandler Class:**  
  - Uses Python’s `re` module for filler detection.  
  - Handles low-confidence transcriptions by ignoring them.  
  - Provides logging messages to track actions (ignore, register, or interrupt).  

- **Integration with Audio Recognition:**  
  - The `FillerHandler` is imported at the top of `audio_recognition.py`:
    ```python
    from livekit.agents.filler_handler import FillerHandler
    filler_handler = FillerHandler()
    ```
  - Ensures smooth coordination between the ASR (Automatic Speech Recognition) and VAD (Voice Activity Detection) modules.

## What Works
- Ignores filler words (“uh”, “umm”, “hmm”, “haan”) during ongoing speech.  
- Detects real interruptions such as “stop” and “wait”.  
- Logs each event distinctly for easy debugging and monitoring.  
- Maintains a balance between accuracy and responsiveness through confidence-based filtering.

## Result

The implementation successfully distinguishes filler words from meaningful speech. This improves the user interaction quality by ensuring that the system only responds to intentional commands and ignores unnecessary pauses or hesitations.
