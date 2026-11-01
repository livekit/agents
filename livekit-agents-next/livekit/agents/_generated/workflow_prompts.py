# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/workflows/prompts.rs

from __future__ import annotations

from typing import Final

END_CALL_DESCRIPTION: Final = "\nEnds the current call and disconnects immediately.\n\nCall when:\n- The user clearly indicates they are done (e.g., “that’s all, bye”).\n\nDo not call when:\n- The user asks to pause, hold, or transfer.\n- Intent is unclear.\n\nThis is the final action the agent can take.\nOnce called, no further interaction is possible with the user.\nDon't generate any other text or response when the tool is called.\n"
"""`EndCallTool`'s description, before its `extra_description`."""
DTMF_RECORDER_INSTRUCTIONS: Final = "You are a single step in a broader system, responsible solely for gathering digits input from the user. You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or user will directly say the digits to you. You should be able to handle both cases. If user provides the digits through voice and it is valid, call `record_inputs` with the inputs."
DTMF_CONFIRMER_INSTRUCTIONS: Final = "You are a single step in a broader system, responsible solely for gathering digits input from the user. You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or user will directly say the digits to you. You should be able to handle both cases. Once user has confirmed the digits (by verbally spoken or entered manually), call `confirm_inputs` with the inputs."
EMAIL_COLLECTOR_INSTRUCTIONS: Final = "You only capture the caller's email address, one step in a broader system.\nTreat what you hear as a noisy transcription of an address said aloud: turn 'dot', 'underscore', 'dash', 'plus' and 'at' into `.`, `_`, `-`, `+` and `@`, join a word spelled after it ('john j o h n' is john), and drop filler words. Never invent an address. When only the part before the '@' was said, ask for the domain; when it stays unclear, ask for it in parts. Call confirm_email_address with the address as soon as you have one; its read-back asks the caller to confirm, and a correction means calling it again with the corrected address. No greetings, no markdown, nothing off topic."
ASK_DIGITS_INSTRUCTIONS: Final = "Ask for the digits."
ASK_EMAIL_INSTRUCTIONS: Final = "Ask for their email address, or confirm one they already gave."
EMAIL_READ_BACK: Final = "Just to confirm, your email is {email_address}, right?"
