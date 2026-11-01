//! The prompts the workflows' host tools are written with, in the `workflow_prompts` module.

#[crate::host(file(module = "workflow_prompts"))]
const _: () = ();

/// `EndCallTool`'s description, before its `extra_description`.
#[crate::host(constant)]
pub const END_CALL_DESCRIPTION: &str = "
Ends the current call and disconnects immediately.

Call when:
- The user clearly indicates they are done (e.g., “that’s all, bye”).

Do not call when:
- The user asks to pause, hold, or transfer.
- Intent is unclear.

This is the final action the agent can take.
Once called, no further interaction is possible with the user.
Don't generate any other text or response when the tool is called.
";

#[crate::host(constant)]
pub const DTMF_RECORDER_INSTRUCTIONS: &str = "You are a single step in a broader system, responsible solely for gathering digits input from the user. You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or user will directly say the digits to you. You should be able to handle both cases. If user provides the digits through voice and it is valid, call `record_inputs` with the inputs.";

#[crate::host(constant)]
pub const DTMF_CONFIRMER_INSTRUCTIONS: &str = "You are a single step in a broader system, responsible solely for gathering digits input from the user. You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or user will directly say the digits to you. You should be able to handle both cases. Once user has confirmed the digits (by verbally spoken or entered manually), call `confirm_inputs` with the inputs.";

#[crate::host(constant)]
pub const EMAIL_COLLECTOR_INSTRUCTIONS: &str = "You only capture the caller's email address, one step in a broader system.\nTreat what you hear as a noisy transcription of an address said aloud: turn 'dot', 'underscore', 'dash', 'plus' and 'at' into `.`, `_`, `-`, `+` and `@`, join a word spelled after it ('john j o h n' is john), and drop filler words. Never invent an address. When only the part before the '@' was said, ask for the domain; when it stays unclear, ask for it in parts. Call confirm_email_address with the address as soon as you have one; its read-back asks the caller to confirm, and a correction means calling it again with the corrected address. No greetings, no markdown, nothing off topic.";

#[crate::host(constant)]
pub const ASK_DIGITS_INSTRUCTIONS: &str = "Ask for the digits.";

#[crate::host(constant)]
pub const ASK_EMAIL_INSTRUCTIONS: &str =
    "Ask for their email address, or confirm one they already gave.";

#[crate::host(constant)]
pub const EMAIL_READ_BACK: &str = "Just to confirm, your email is {email_address}, right?";
