use lk_agents_abi::voice::workflows::{
    CapturePrompts, CardPrompts, EmailPrompts, PhonePrompts, TransferPrompts, WorkflowPrompts,
};

/// The module-level constants the reference's `beta/workflows/` publish, as the ABI record a host
/// reads them from, so the Python modules hold no second copy.
#[lk_agents_abi::host(export(pid = false))]
pub fn workflow_prompts() -> WorkflowPrompts {
    WorkflowPrompts {
        email_address: EmailPrompts {
            persona: email::PERSONA.to_owned(),
            audio_specific: email::AUDIO_SPECIFIC.to_owned(),
            text_specific: email::TEXT_SPECIFIC.to_owned(),
            confirmation_instruction: email::CONFIRMATION_INSTRUCTION.to_owned(),
            instructions_template: email::INSTRUCTIONS_TEMPLATE.to_owned(),
            email_regex: EMAIL_REGEX_REFERENCE.to_owned(),
        },
        address: CapturePrompts {
            persona: address::PERSONA.to_owned(),
            audio_specific: address::AUDIO_SPECIFIC.to_owned(),
            text_specific: address::TEXT_SPECIFIC.to_owned(),
            confirmation_instruction: address::CONFIRMATION_INSTRUCTION.to_owned(),
            instructions_template: address::INSTRUCTIONS_TEMPLATE.to_owned(),
        },
        phone_number: PhonePrompts { phone_regex: PHONE_REGEX_REFERENCE.to_owned() },
        warm_transfer: TransferPrompts {
            persona: warm_transfer::PERSONA.to_owned(),
            instructions_template: warm_transfer::INSTRUCTIONS_TEMPLATE.to_owned(),
        },
        credit_card: CardPrompts {
            card_issuers_lookup: crate::workflows::CARD_ISSUERS
                .iter()
                .map(|(digit, name)| (digit.to_string(), (*name).to_owned()))
                .collect(),
        },
    }
}

/// `email_address.EMAIL_REGEX`.
pub const EMAIL_REGEX_REFERENCE: &str =
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$";

/// `phone_number.PHONE_REGEX`.
pub const PHONE_REGEX_REFERENCE: &str = r"^\+?[1-9]\d{6,14}$";

pub mod email {
    //! `email_address.PERSONA`.

    pub const PERSONA: &str = "You are only a single step in a broader system, responsible solely for capturing an email address.";

    pub const AUDIO_SPECIFIC: &str = r#"Handle input as noisy voice transcription. Expect that users will say emails aloud with formats like:
- 'john dot doe at gmail dot com'
- 'susan underscore smith at yahoo dot co dot uk'
- 'dave dash b at protonmail dot com'
- 'jane at example' (partial—prompt for the domain)
- 'theo t h e o at livekit dot io' (name followed by spelling)
Normalize common spoken patterns silently:
- Convert words like 'dot', 'underscore', 'dash', 'plus' into symbols: `.`, `_`, `-`, `+`.
- Convert 'at' to `@`.
- Recognize patterns where users speak their name or a word, followed by spelling: e.g., 'john j o h n'.
- Filter out filler words or hesitations.
- Assume some spelling if contextually obvious (e.g. 'mike b two two' → mikeb22).
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently."#;

    pub const TEXT_SPECIFIC: &str = r#"Handle input as typed text. Expect users to type their email address directly in standard format.
If the address looks almost correct but has minor typos (e.g. missing '@' or domain), prompt for clarification."#;

    pub const CONFIRMATION_INSTRUCTION: &str =
        "Call `confirm_email_address` after the user confirmed the email address is correct.";

    pub const INSTRUCTIONS_TEMPLATE: &str = r#"{persona}

{_modality_specific}

Call `update_email_address` at the first opportunity whenever you form a new hypothesis about the email. (before asking any questions or providing any answers.)
Don't invent new email addresses, stick strictly to what the user said.
{_confirmation}
If the email is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the part before the '@', then the domain—only if needed.

Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.

{extra}
"#;
}

pub mod address {
    //! `address.PERSONA`.

    pub const PERSONA: &str = "You are only a single step in a broader system, responsible solely for capturing an address.";

    pub const AUDIO_SPECIFIC: &str = r#"You will be handling addresses from any country.
Expect that users will say address in different formats with fields filled like:
- 'street_address': '450 SOUTH MAIN ST', 'unit_number': 'FLOOR 2', 'locality': 'SALT LAKE CITY UT 84101', 'country': 'UNITED STATES',
- 'street_address': '123 MAPLE STREET', 'unit_number': 'APARTMENT 10', 'locality': 'OTTAWA ON K1A 0B1', 'country': 'CANADA',
- 'street_address': 'GUOMAO JIE 3 HAO, CHAOYANG QU', 'unit_number': 'GUOMAO DA SHA 18 LOU 101 SHI', 'locality': 'BEIJING SHI 100000', 'country': 'CHINA',
- 'street_address': '5 RUE DE L’ANCIENNE COMÉDIE', 'unit_number': 'APP C4', 'locality': '75006 PARIS', 'country': 'FRANCE',
- 'street_address': 'PLOT 10, NEHRU ROAD', 'unit_number': 'OFFICE 403, 4TH FLOOR', 'locality': 'VILE PARLE (E), MUMBAI MAHARASHTRA 400099', 'country': 'INDIA',
Normalize common spoken patterns silently:
- Convert words like 'dash' and 'apostrophe' into symbols: `-`, `'`.
- Convert spelled out numbers like 'six' and 'seven' into numerals: `6`, `7`.
- Recognize patterns where users speak their address field followed by spelling: e.g., 'guomao g u o m a o'.
- Filter out filler words or hesitations.
- Recognize when there may be accents on certain letters if explicitly said or common in the location specified. Be sure to verify the correct accents if existent.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
When reading a numerical ordinal suffix (st, nd, rd, th), the number must be verbally expanded into its full, correctly pronounced word form.
Do not read the number and the suffix letters separately.
Confirm postal codes by reading them out digit-by-digit as a sequence of single numbers. Do not read them as cardinal numbers.
For example, read 90210 as 'nine zero two one zero.'
Avoid using bullet points and parenthese in any responses.
Spell out the address letter-by-letter when applicable, such as street names and provinces, especially when the user spells it out initially."#;

    pub const TEXT_SPECIFIC: &str = r#"You will be handling addresses from any country.
Expect users to type their address directly.
If the address looks almost correct but has minor issues (e.g. missing country or postal code), prompt for clarification."#;

    pub const CONFIRMATION_INSTRUCTION: &str =
        "Call `confirm_address` after the user confirmed the address is correct.";

    pub const INSTRUCTIONS_TEMPLATE: &str = r#"{persona}

{_modality_specific}

Call `update_address` at the first opportunity whenever you form a new hypothesis about the address. (before asking any questions or providing any answers.)
Don't invent new addresses, stick strictly to what the user said.
{_confirmation}
If the address is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts in this order: street address, unit number if applicable, locality, and country.

Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.

{extra}
"#;
}

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
