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
            card_issuers_lookup: crate::workflows::builtins::credit_card::CARD_ISSUERS
                .iter()
                .map(|(digit, name)| (digit.to_string(), (*name).to_owned()))
                .collect(),
        },
    }
}

/// `email_address.EMAIL_REGEX`, as documentation of what
/// [`crate::workflows::builtins::email_address::is_valid_email`] must accept.
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

    pub const ON_ENTER: &str = "Ask the user to provide an email address.";

    // --- tools ---
    pub const UPDATE_TOOL: &str = "update_email_address";
    pub const UPDATE_DESC: &str = "Update the email address provided by the user.";
    pub const UPDATE_ARG_EMAIL: &str = "The email address provided by the user";

    pub const CONFIRM_TOOL: &str = "confirm_email_address";
    pub const CONFIRM_DESC: &str = "Call after the user confirms the email address is correct.";

    pub const DECLINE_TOOL: &str = "decline_email_capture";
    pub const DECLINE_DESC: &str =
        "Handles the case when the user explicitly declines to provide an email address.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide the email address";

    // --- tool replies ---
    pub const INVALID: &str = "Invalid email address provided: {email}";
    pub const UPDATED: &str = "The email has been updated to {email}\nRepeat the email character by character: {separated} if needed\nPrompt the user for confirmation, do not call `confirm_email_address` directly";
    pub const STALE: &str = "The email has changed since confirmation was requested, ask the user to confirm the updated email.";
    pub const DECLINED: &str = "couldn't get the email address: {reason}";
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

    pub const ON_ENTER: &str = "Ask the user to provide their address.";

    pub const UPDATE_TOOL: &str = "update_address";
    pub const UPDATE_DESC: &str = "Update the address provided by the user.";
    pub const UPDATE_ARG_STREET: &str = "Dependent on country, may include fields like house number, street name, block, or district";
    pub const UPDATE_ARG_UNIT: &str = "The unit number, for example Floor 1 or Apartment 12. If there is no unit number, return ''";
    pub const UPDATE_ARG_LOCALITY: &str =
        "Dependent on country, may include fields like city, zip code, or province";
    pub const UPDATE_ARG_COUNTRY: &str = "The country the user lives in spelled out fully";

    pub const CONFIRM_TOOL: &str = "confirm_address";
    pub const CONFIRM_DESC: &str = "Call after the user confirms the address is correct.";

    pub const DECLINE_TOOL: &str = "decline_address_capture";
    pub const DECLINE_DESC: &str =
        "Handles the case when the user explicitly declines to provide an address.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide the address";

    pub const UPDATED: &str = "The address has been updated to {address}\nRepeat the address field by field: {fields} if needed\nPrompt the user for confirmation, do not call `confirm_address` directly";
    pub const STALE: &str = "The address has changed since confirmation was requested, ask the user to confirm the updated address.";
    pub const DECLINED: &str = "couldn't get the address: {reason}";
}

pub mod phone {
    //! `phone_number._BASE_INSTRUCTIONS`, `.decline_phone_number_capture`.

    pub const BASE_INSTRUCTIONS: &str = r#"
You are only a single step in a broader system, responsible solely for capturing a phone number.
{modality_specific}
Call `update_phone_number` at the first opportunity whenever you form a new hypothesis about the phone number. (before asking any questions or providing any answers.)
Don't invent phone numbers, stick strictly to what the user said.
{confirmation_instructions}
If the number is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the area code, then the remaining digits.
Never repeat the phone number back to the user as a single block of digits. Read it back in groups.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Avoid verbosity by not sharing example phone numbers or formats unless prompted to do so. Do not deviate from the goal of collecting the user's phone number.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.{extra_instructions}
"#;

    pub const AUDIO_SPECIFIC: &str = r#"
Handle input as noisy voice transcription. Expect that users will say phone numbers aloud with formats like:
- '555 123 4567'
- 'five five five, one two three, four five six seven'
- '+1 555 123 4567'
- 'area code 555, 123 4567'
- '555-123-4567'
Normalize common spoken patterns silently:
- Convert spoken digits to their numeric form: 'five' → 5, 'zero' → 0, 'oh' → 0.
- Remove filler words, pauses, and hesitations.
- Strip dashes, spaces, parentheses, and dots from the number.
- Recognize 'plus' at the start as the international prefix `+`.
- Recognize 'area code' as a prefix for the area code digits.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
"#;

    pub const TEXT_SPECIFIC: &str = r#"
Handle input as typed text. Expect users to type their phone number directly.
Strip dashes, spaces, parentheses, and dots from the number.
If the number looks almost correct but has minor formatting issues, clean it up silently.
"#;

    pub const CONFIRMATION_INSTRUCTION: &str =
        "Call `confirm_phone_number` after the user confirmed the phone number is correct.";

    pub const ON_ENTER: &str = "Ask the user to provide their phone number.";

    pub const UPDATE_TOOL: &str = "update_phone_number";
    pub const UPDATE_DESC: &str = "Update the phone number provided by the user.";
    pub const UPDATE_ARG_PHONE: &str =
        "The phone number provided by the user, digits only with optional leading +";

    pub const CONFIRM_TOOL: &str = "confirm_phone_number";
    pub const CONFIRM_DESC: &str = "Call after the user confirms the phone number is correct.";

    pub const DECLINE_TOOL: &str = "decline_phone_number_capture";
    pub const DECLINE_DESC: &str =
        "Handles the case when the user explicitly declines to provide a phone number.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide the phone number";

    pub const INVALID: &str = "Invalid phone number provided: {phone_number}";
    pub const UPDATED: &str = "The phone number has been updated to {cleaned}\nRead the number back to the user in groups.\nPrompt the user for confirmation, do not call `confirm_phone_number` directly";
    pub const STALE: &str = "The phone number has changed since confirmation was requested, ask the user to confirm the updated number.";
    pub const DECLINED: &str = "couldn't get the phone number: {reason}";
}

pub mod name {
    //! `name._BASE_INSTRUCTIONS`, `.on_enter`.

    pub const BASE_INSTRUCTIONS: &str = r#"
You are only a single step in a broader system, responsible solely for capturing the user's name.
You need to naturally collect the name parts in this order: {name_format}.
{modality_specific}
{spelling_instructions}Call `update_name` at the first opportunity whenever you form a new hypothesis about the name. (before asking any questions or providing any answers.)
Don't invent names, stick strictly to what the user said.
{confirmation_instructions}
If the name is unclear or it takes too much back-and-forth, prompt for each name part separately.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Avoid verbosity by not sharing example names or spellings unless prompted to do so. Do not deviate from the goal of collecting the user's name.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.{extra_instructions}
"#;

    pub const AUDIO_SPECIFIC: &str = r#"
Handle input as noisy voice transcription. Expect that users will say names aloud and may:
- Say their name followed by spelling: e.g., 'Michael m i c h a e l'
- Use phonetic alphabet: e.g., 'Mike as in Mike India Charlie Hotel Alpha Echo Lima'
- Have names with special characters or hyphens: e.g., 'Mary-Jane' or 'O'Brien'
- Have names from various cultural backgrounds with different pronunciation patterns
Normalize common spoken patterns silently:
- Convert 'dash' or 'hyphen' to `-`.
- Convert 'apostrophe' to `'`.
- Recognize when users spell out their name letter by letter.
- Filter out filler words or hesitations.
- Capitalize the first letter of each name part appropriately.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
"#;

    pub const TEXT_SPECIFIC: &str = r#"
Handle input as typed text. Expect users to type their name directly.
Capitalize the first letter of each name part appropriately.
If the name contains special characters or hyphens (e.g., 'Mary-Jane' or 'O'Brien'), preserve them as typed.
"#;

    pub const SPELLING_INSTRUCTIONS: &str = "After receiving the name, always verify the spelling by asking the user to confirm or spell out the name letter by letter. When confirming, spell out each name part letter by letter to the user. ";

    pub const CONFIRMATION_INSTRUCTION: &str =
        "Call `confirm_name` after the user confirmed the name is correct.";

    pub const ON_ENTER: &str = "Get the user's name (follow this order '{name_format}' but do not mention the format). First scan the conversation - if a name was already given earlier, ask a short confirmation question rather than asking from scratch. If context about what the name is FOR was provided (a role like 'cardholder', 'guest', 'emergency contact'), anchor your confirmation question to that role so the user knows which name you mean - don't ask abstractly. When pointing at where an existing name came from, reference the source in the conversation (the earlier step, the booking they mentioned), not a presumption about how the name appears in the destination. Only ask fresh when the conversation has no name yet.";

    pub const UPDATE_TOOL: &str = "update_name";
    pub const UPDATE_DESC: &str = "Update the name provided by the user.";
    pub const UPDATE_ARG_FIRST: &str = "The user's first name.";
    pub const UPDATE_ARG_MIDDLE: &str = "The user's middle name, if collected.";
    pub const UPDATE_ARG_LAST: &str = "The user's last name, if collected.";

    pub const CONFIRM_TOOL: &str = "confirm_name";
    pub const CONFIRM_DESC: &str = "Call after the user confirms the name is correct.";

    pub const DECLINE_TOOL: &str = "decline_name_capture";
    pub const DECLINE_DESC: &str =
        "Handles the case when the user explicitly declines to provide their name.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide their name";

    pub const INCOMPLETE: &str = "Incomplete name: {errors}";
    pub const ERR_MISSING_FIRST: &str = "first name is required but was not provided";
    pub const ERR_MISSING_MIDDLE: &str = "middle name is required but was not provided";
    pub const ERR_MISSING_LAST: &str = "last name is required but was not provided";
    /// `{label}` is `first`/`middle`/`last`; `{value}` is the *Python `repr`* of the value.
    pub const ERR_NO_LETTERS: &str =
        "{label} name {value} contains no letters - that doesn't look like a name";

    pub const UPDATED_SPELL: &str = "The name has been updated to {full_name}\nSpell out the name letter by letter for verification: {full_name}\nPrompt the user for confirmation, do not call `confirm_name` directly";
    pub const UPDATED: &str = "The name has been updated to {full_name}\nRepeat the name back to the user and prompt for confirmation, do not call `confirm_name` directly";
    pub const STALE: &str = "The name has changed since confirmation was requested, ask the user to confirm the updated name.";
    pub const DECLINED: &str = "couldn't get the name: {reason}";
}

pub mod dob {
    //! `dob._BASE_INSTRUCTIONS`, `.on_enter`.

    pub const BASE_INSTRUCTIONS: &str = r#"
You are only a single step in a broader system, responsible solely for capturing a date of birth.
{modality_specific}
{time_instructions}Call `update_dob` at the first opportunity whenever you form a new hypothesis about the date of birth. (before asking any questions or providing any answers.)
Don't invent dates, stick strictly to what the user said.
{confirmation_instructions}
When reading back dates, use a natural spoken format like 'January fifteenth, nineteen ninety'.
If the date is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts: first the month, then the day, then the year.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Avoid verbosity by not sharing example dates or formats unless prompted to do so. Do not deviate from the goal of collecting the user's birthday.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.{extra_instructions}
"#;

    pub const AUDIO_SPECIFIC: &str = r#"
Handle input as noisy voice transcription. Expect that users will say dates aloud with formats like:
- 'January 15th 1990'
- 'the fifteenth of January nineteen ninety'
- '01 15 1990' or 'one fifteen ninety'
- 'Jan 15 90'
- '15th January 1990'
Normalize common spoken patterns silently:
- Convert spoken numbers and ordinals to their numeric form: 'fifteenth' → 15, 'ninety' → 1990.
- Recognize month names in various forms: 'Jan', 'January', etc.
- Handle two-digit years appropriately: '90' likely means 1990, '05' likely means 2005.
- Filter out filler words or hesitations.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
"#;

    pub const TEXT_SPECIFIC: &str = r#"
Handle input as typed text. Expect users to type their date of birth directly.
Accept common date formats like 'MM/DD/YYYY', 'January 15, 1990', or '1990-01-15'.
Handle two-digit years appropriately: '90' likely means 1990, '05' likely means 2005.
"#;

    pub const TIME_INSTRUCTIONS: &str = "Also ask for and capture the time of birth if the user knows it. The time is optional - if the user doesn't know it, proceed without it.\n";

    pub const CONFIRMATION_INSTRUCTION: &str =
        "Call `confirm_dob` after the user confirmed the date of birth is correct.";

    pub const ON_ENTER: &str = "Ask the user to provide their date of birth.";
    pub const ON_ENTER_WITH_TIME: &str =
        "Ask the user to provide their date of birth and, if they know it, their time of birth.";

    pub const UPDATE_TOOL: &str = "update_dob";
    pub const UPDATE_DESC: &str = "Update the date of birth provided by the user. Given a spoken month and year (e.g., 'July 2030'), return its numerical representation (7/2030).";
    pub const UPDATE_ARG_YEAR: &str = "The birth year (e.g., 1990)";
    pub const UPDATE_ARG_MONTH: &str = "The birth month (1-12)";
    pub const UPDATE_ARG_DAY: &str = "The birth day (1-31)";

    pub const UPDATE_TIME_TOOL: &str = "update_time";
    pub const UPDATE_TIME_DESC: &str = "Update the time of birth provided by the user.";
    pub const UPDATE_TIME_ARG_HOUR: &str = "The birth hour (0-23)";
    pub const UPDATE_TIME_ARG_MINUTE: &str = "The birth minute (0-59)";

    pub const CONFIRM_TOOL: &str = "confirm_dob";
    pub const CONFIRM_DESC: &str = "Call after the user confirms the date of birth is correct.";

    pub const DECLINE_TOOL: &str = "decline_dob_capture";
    pub const DECLINE_DESC: &str =
        "Handles the case when the user explicitly declines to provide a date of birth.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide the date of birth";

    pub const DOB_UPDATED: &str = "The date of birth has been updated to {formatted_date}";
    pub const DOB_UPDATED_AT_TIME: &str = " at {formatted_time}";
    pub const TIME_UPDATED: &str = "The time of birth has been updated to {formatted_time}";
    pub const DATE_AND_TIME_UPDATED: &str =
        "The date and time of birth has been updated to {formatted_date} at {formatted_time}";
    pub const INVALID_DATE: &str = "Invalid date: {error}";
    pub const INVALID_TIME: &str = "Invalid time: {error}";
    pub const FUTURE: &str = "Invalid date of birth: {formatted} is in the future. Date of birth cannot be a future date.";
    pub const CONFIRM_TAIL: &str = "\nRepeat the date back to the user in a natural spoken format.\nPrompt the user for confirmation, do not call `confirm_dob` directly";
    pub const CONFIRM_TIME_TAIL: &str = "\nRepeat the time back to the user in a natural spoken format.\nPrompt the user for confirmation, do not call `confirm_dob` directly";
    pub const NO_DATE_YET_TAIL: &str =
        "\nThe date of birth has not been provided yet, ask the user to provide it.";
    pub const STALE: &str = "The date of birth has changed since confirmation was requested, ask the user to confirm the updated date.";
    pub const NO_DATE_YET: &str = "No date of birth was provided yet, ask the user to provide it.";
    pub const DECLINED: &str = "couldn't get the date of birth: {reason}";
}

pub mod credit_card {
    //! `credit_card._CARD_NUMBER_BASE_INSTRUCTIONS`, `.GetExpirationDateResult`.

    pub const CARD_NUMBER_BASE: &str = r#"
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the credit card number.
{modality_specific}
If the user refuses to provide a credit card number, call decline_card_capture().
If the user wishes to start over the credit card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's credit card number, back to the user.
{confirmation_instructions}
"#;

    pub const CARD_NUMBER_AUDIO: &str = r#"
Handle input as noisy voice transcription. Expect users to read the card number digit by digit.
Normalize spoken digits silently: 'four' → 4, 'zero' / 'oh' → 0.
Filter out filler words or hesitations.
"#;

    pub const CARD_NUMBER_TEXT: &str = r#"
Handle input as typed text. Users may type the number with or without spaces or dashes (e.g. '4152 6374 8901 2345').
"#;

    pub const SECURITY_CODE_BASE: &str = r#"
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the user's card's security code.
{modality_specific}
If the user refuses to provide a code, call decline_card_capture().
If the user wishes to start over the card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's security code, back to the user.
{confirmation_instructions}
"#;

    pub const SECURITY_CODE_AUDIO: &str = r#"
Handle input as noisy voice transcription. Expect users to read the security code digit by digit.
Normalize spoken digits silently: 'four' → 4, 'zero' / 'oh' → 0.
Filter out filler words or hesitations.
"#;

    pub const SECURITY_CODE_TEXT: &str = r#"
Handle input as typed text. Users will type the security code directly.
"#;

    pub const EXPIRATION_BASE: &str = r#"
You are a single step in a broader process of collecting credit card information.
You are solely responsible for collecting the user's card's expiration date.
{modality_specific}
If the user refuses to provide a date, call decline_card_capture().
If the user wishes to start over the card collection process, call restart_card_collection().
Avoid listing out questions with bullet points or numbers, use a natural conversational tone.
Never repeat any sensitive information, such as the user's expiration date, back to the user.
{confirmation_instructions}
"#;

    pub const EXPIRATION_AUDIO: &str = r#"
Handle input as noisy voice transcription. Expect users to say the expiration date in formats like 'April twenty five', 'oh four twenty five', 'four slash twenty five', or 'April 2025'.
Normalize spoken months and digits silently.
Filter out filler words or hesitations.
"#;

    pub const EXPIRATION_TEXT: &str = r#"
Handle input as typed text. Expect users to type the expiration date in formats like '04/25', '04/2025', or 'April 2025'.
"#;

    pub const CARD_NUMBER_CONFIRMATION: &str =
        "Call `confirm_card_number` once the user has repeated their card number.";
    pub const SECURITY_CODE_CONFIRMATION: &str =
        "Call `confirm_security_code` once the user has repeated their security code.";
    pub const EXPIRATION_CONFIRMATION: &str =
        "Call `confirm_expiration_date` once the user has repeated their expiration date.";

    pub const CARD_NUMBER_ON_ENTER: &str = "Get the user's credit card number. First scan the conversation - if a credit card number was already given (e.g. the user volunteered it before the task started), use it via update_card_number rather than re-asking. Only ask fresh when no credit card number is in the conversation yet.";
    pub const SECURITY_CODE_ON_ENTER: &str = "Get the user's card security code. First scan the conversation - if a code was already given, use it via update_security_code rather than re-asking. Only ask fresh when no code is in the conversation yet.";
    pub const EXPIRATION_ON_ENTER: &str = "Get the user's card expiration date. First scan the conversation - if an expiration date was already given, use it via update_expiration_date rather than re-asking. Only ask fresh when no date is in the conversation yet.";

    pub const DECLINE_TOOL: &str = "decline_card_capture";
    pub const DECLINE_DESC: &str = "Handles the case when the user explicitly declines to provide a detail for their card information.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the user declined to provide card information";
    pub const DECLINED: &str = "couldn't get the card details: {reason}";

    pub const RESTART_TOOL: &str = "restart_card_collection";
    pub const RESTART_DESC: &str = "Handles the case when the user wishes to start over the card information collection process and validate a new card.";
    pub const RESTART_ARG_REASON: &str = "A short explanation of why the user wishes to start over";
    pub const RESTARTING: &str = "starting over: {reason}";

    pub const UPDATE_CARD_NUMBER_TOOL: &str = "update_card_number";
    pub const UPDATE_CARD_NUMBER_DESC: &str = "Call to record the user's card number. Only call once the entire number has been given, do not call in increments.";
    pub const UPDATE_CARD_NUMBER_ARG: &str =
        "The credit card number as a string with no dashes or spaces";
    pub const CONFIRM_CARD_NUMBER_TOOL: &str = "confirm_card_number";
    pub const CONFIRM_CARD_NUMBER_DESC: &str =
        "Call after the user repeats their card number for confirmation.";
    pub const CONFIRM_CARD_NUMBER_ARG: &str = "The card number repeated by the user as a string";

    pub const CARD_NUMBER_BAD_LENGTH: &str =
        "The length of the card number is invalid, ask the user to repeat their card number.";
    pub const CARD_NUMBER_INVALID: &str = "The card number is not valid, ask the user if they made a mistake or to provide another card.";
    pub const CARD_NUMBER_UPDATED: &str = "The card number has been updated.\nAsk them to repeat the number, do not repeat the number back to them.\n";
    pub const CARD_NUMBER_MISMATCH: &str =
        "The repeated card number does not match, ask the user to try again.";

    pub const UPDATE_SECURITY_CODE_TOOL: &str = "update_security_code";
    pub const UPDATE_SECURITY_CODE_DESC: &str = "Call to update the card's security code.";
    pub const UPDATE_SECURITY_CODE_ARG: &str =
        "The card's security code (3-4 digits, may have leading zeros).";
    pub const CONFIRM_SECURITY_CODE_TOOL: &str = "confirm_security_code";
    pub const CONFIRM_SECURITY_CODE_DESC: &str =
        "Call after the user repeats their security code for confirmation.";
    pub const CONFIRM_SECURITY_CODE_ARG: &str = "The security code repeated by the user";

    pub const SECURITY_CODE_BAD_LENGTH: &str = "The security code's length is invalid, ask the user to repeat or to provide a new card and start over.";
    pub const SECURITY_CODE_UPDATED: &str = "The security code has been updated.\nDo not repeat the security code back to the user, ask them to repeat themselves.\nCall `confirm_security_code` once the user confirms, do not call it preemptively.\n";
    pub const SECURITY_CODE_MISMATCH: &str =
        "The repeated security code does not match, ask the user to try again.";

    pub const UPDATE_EXPIRATION_TOOL: &str = "update_expiration_date";
    pub const UPDATE_EXPIRATION_DESC: &str =
        "Call to update the card's expiration date. Collect both the numerical month and year.";
    pub const UPDATE_EXPIRATION_ARG_MONTH: &str =
        "The numerical expiration month of the card, example: '04' for April";
    pub const UPDATE_EXPIRATION_ARG_YEAR: &str = "The numerical expiration year of the card shortened to the last two digits, for example, '35' for 2035";
    pub const CONFIRM_EXPIRATION_TOOL: &str = "confirm_expiration_date";
    pub const CONFIRM_EXPIRATION_DESC: &str =
        "Call after the user repeats their expiration date for confirmation.";
    pub const CONFIRM_EXPIRATION_ARG_MONTH: &str = "The expiration month repeated by the user";
    pub const CONFIRM_EXPIRATION_ARG_YEAR: &str = "The expiration year repeated by the user";

    pub const EXPIRATION_BAD_MONTH: &str =
        "The expiration month is invalid, ask the user to repeat the expiration month.";
    pub const EXPIRATION_BAD_YEAR: &str =
        "The expiration year is invalid, ask the user to repeat the expiration year.";
    pub const EXPIRATION_EXPIRED: &str = "The expiration date is in the past, the card is expired. Ask the user to provide another card.";
    pub const EXPIRATION_UPDATED: &str = "The expiration date has been updated.\nDo not repeat the expiration date back to the user, ask them to repeat themselves.\nCall `confirm_expiration_date` once the user confirms, do not call it preemptively.\n";
    pub const EXPIRATION_MISMATCH: &str =
        "The repeated expiration date does not match, ask the user to try again.";
}

pub mod dtmf {
    //! `dtmf_inputs.GetDtmfTask.__init__`, `._generate_dtmf_reply`.

    pub const BASE: &str = "You are a single step in a broader system, responsible solely for gathering digits input from the user. You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or user will directly say the digits to you. You should be able to handle both cases. ";

    pub const WITH_CONFIRMATION: &str = "Once user has confirmed the digits (by verbally spoken or entered manually), call `confirm_inputs` with the inputs.";
    pub const WITHOUT_CONFIRMATION: &str = "If user provides the digits through voice and it is valid, call `record_inputs` with the inputs.";

    pub const CONFIRM_TOOL: &str = "confirm_inputs";
    pub const CONFIRM_DESC: &str = r#"Finalize the collected digit inputs after explicit user confirmation.

Use this ONLY after the confirmation. You should confirm by verbally reading out the digits one by one and, once the
user confirms they are correct, call this tool with the inputs.

Do not use this tool to capture the initial digits."#;

    pub const RECORD_TOOL: &str = "record_inputs";
    pub const RECORD_DESC: &str = r#"Record the collected digit inputs without additional confirmation.

Call this tool as soon as a valid sequence of digits has been provided by the user (via DTMF or spoken)."#;

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

    // ------------------------------------------------------------------ tools
    pub const CONNECT_TOOL: &str = "connect_to_caller";
    pub const CONNECT_DESC: &str = "Called when the human agent wants to connect to the caller.";

    pub const DECLINE_TOOL: &str = "decline_transfer";
    pub const DECLINE_DESC: &str =
        "Handles the case when the human agent explicitly declines to connect to the caller.";
    pub const DECLINE_ARG_REASON: &str =
        "A short explanation of why the human agent declined to connect to the caller";

    pub const VOICEMAIL_TOOL: &str = "voicemail_detected";
    pub const VOICEMAIL_DESC: &str = "Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting";

    // ------------------------------------------------------------------ errors
    /// `warm_transfer.WarmTransferTask.decline_transfer`.
    pub const DECLINED: &str = "human agent declined to connect: {reason}";
    /// `warm_transfer.WarmTransferTask.voicemail_detected`.
    pub const VOICEMAIL: &str = "voicemail detected";
}

pub mod task_group {
    //! `task_group.TaskGroup._build_out_of_scope_tool`. One description, assembled from three literals plus the task table.

    pub const OUT_OF_SCOPE_TOOL: &str = "out_of_scope";

    pub const OUT_OF_SCOPE_DESC: &str = r#"Call to regress to other tasks according to what the user requested to modify, return the corresponding task ids. For example, if the user wants to change their email and there is a task with id "email_task" with a description of "Collect the user's email", return the id ("get_email_task").If the user requests to regress to multiple tasks, such as changing their phone number and email, return both task ids in the order they were requested.The following are the IDs and their corresponding task description. {task_repr}"#;

    pub const OUT_OF_SCOPE_ARG: &str = "The IDs of the tasks requested";

    /// `task_group.out_of_scope`.
    pub const INVALID_TASK_ID: &str = "unable to regress, invalid task id {task_id}";
}
