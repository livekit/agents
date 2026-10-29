//! The prompts the workflows speak with: the DTMF read-back and the warm transfer's.

pub mod dtmf {
    //! DTMF reception and read-back.

    /// `{digits}` is the space-separated `format_dtmf` rendering.
    pub const CONFIRM_PROMPT: &str = "User has entered the following valid digits on the telephone keypad:\n<dtmf_inputs>{digits}</dtmf_inputs>\nPlease confirm it with the user by saying the digits one by one with space in between (.e.g. 'one two three four five six seven eight nine ten'). Once you are sure, call `confirm_inputs` with the inputs.";

    pub const NOT_FULLY_RECEIVED: &str =
        "Digits input not fully received. Expect {expected} digits, got {got}";
}

pub mod warm_transfer {
    //! `warm_transfer.PERSONA`.

    pub const PERSONA: &str = r#"# Identity

You are an agent that is reaching out to a human agent for help. There has been a previous conversation
between you and a caller, the conversation history is included below.

# Goal

Your main goal is to give the human agent sufficient context about why the caller had called in,
so that the human agent could gain sufficient knowledge to help the caller directly."#;

    pub const INSTRUCTIONS_TEMPLATE: &str = r#"{persona}

# Context

In the conversation, user refers to the human agent, caller refers to the person who's transcript is included.
Remember, you are not speaking to the caller right now, you are speaking to the human agent.

## Conversation history with caller
{_conversation_history}
## End of conversation history with caller

Once the human agent has confirmed, you should call the tool `connect_to_caller` to connect them to the caller.

You are talking to the human agent now, start by giving them a summary of the conversation so far, and answer any questions they might have.

{extra}
"#;

    pub const CONNECT_TOOL: &str = "connect_to_caller";
    pub const CONNECT_DESC: &str = "Called when the human agent wants to connect to the caller.";

    pub const DECLINE_TOOL: &str = "decline_transfer";
    pub const DECLINE_DESC: &str =
        "Handles the case when the human agent explicitly declines to connect to the caller.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the human agent declined to connect to the caller";

    pub const VOICEMAIL_TOOL: &str = "voicemail_detected";
    pub const VOICEMAIL_DESC: &str = "Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting";
}
