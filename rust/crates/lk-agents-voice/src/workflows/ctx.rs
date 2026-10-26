#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warning {
    /// "`extra_instructions` will be ignored when `instructions` is provided".
    ExtraInstructionsIgnored,
    /// "`target_phone_number` is deprecated, use `sip_call_to` instead".
    TargetPhoneNumberDeprecated,
}

impl Warning {
    /// The name a host logs it under; the host owns the wording.
    pub fn as_str(self) -> &'static str {
        match self {
            Warning::ExtraInstructionsIgnored => "extra_instructions_ignored",
            Warning::TargetPhoneNumberDeprecated => "target_phone_number_deprecated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ToolError {
    /// The whole `str(exc)` the reference produces, already formatted.
    pub message: String,
    pub kind: ToolErrorKind,
}

/// The subclass identity of a [`ToolError`]. Open (`class_name` string): workflows declare their
/// own subclasses and `except` on them, and a user agent may too.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolErrorKind {
    #[default]
    Plain,
    /// `class_name` is `type(exc).__name__`; `reason` is the subclass's kept argument.
    Subclass { class_name: String, reason: String },
}

impl ToolError {
    pub fn plain(message: impl Into<String>) -> Self {
        ToolError { message: message.into(), kind: ToolErrorKind::Plain }
    }

    /// A subclass; `message` is the formatted `str(exc)`.
    pub fn subclass(
        class_name: impl Into<String>,
        message: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        ToolError {
            message: message.into(),
            kind: ToolErrorKind::Subclass { class_name: class_name.into(), reason: reason.into() },
        }
    }

    /// `(message, class_name, reason)` for a host to raise the right class.
    pub fn into_parts(self) -> (String, Option<String>, Option<String>) {
        match self.kind {
            ToolErrorKind::Plain => (self.message, None, None),
            ToolErrorKind::Subclass { class_name, reason } => {
                (self.message, Some(class_name), Some(reason))
            }
        }
    }
}
