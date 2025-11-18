

from typing import List

# 1. Configurable list of Irrelevant Fillers
IGNORED_WORDS: List[str] = [
    'uh', 
    'umm', 
    'hmm', 
    'haan', 
    'like', 
    'you know',
    'Okay',
    'ok',
    'Yeah',
    'Uh-huh',
    'Right',
    'Yup',
    'Sure',
    'Wow',
]

# 2. Defined list of Real Interruption Commands
INTERRUPTION_COMMANDS: List[str] = [
    'wait', 
    'stop', 
    'hold on', 
    'excuse me',
    'no', 
    'not that one'
]

# ASR Confidence Threshold
LOW_CONFIDENCE_THRESHOLD = 0.6