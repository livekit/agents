pub use lk_agents_abi::voice::workflows::GetNameResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GetNameConfigParts {
    pub first_name: bool,
    pub last_name: bool,
    pub middle_name: bool,
}

impl Default for GetNameConfigParts {
    /// `first_name: bool = True, last_name: bool = False, middle_name: bool = False`.
    fn default() -> Self {
        GetNameConfigParts { first_name: true, last_name: false, middle_name: false }
    }
}

pub fn default_name_format(parts: &GetNameConfigParts) -> String {
    let mut out: Vec<&str> = Vec::new();
    if parts.first_name {
        out.push("{first_name}");
    }
    if parts.middle_name {
        out.push("{middle_name}");
    }
    if parts.last_name {
        out.push("{last_name}");
    }
    out.join(" ")
}

pub fn clean_name_arg(value: Option<&str>) -> Option<String> {
    let value = value?;
    let value = value.trim().trim_matches(|c| c == '\'' || c == '"');
    if value.is_empty() {
        return None;
    }
    const SENTINELS: [&str; 6] = ["null", "none", "nil", "n/a", "unknown", "unspecified"];
    if SENTINELS.contains(&value.to_lowercase().as_str()) {
        return None;
    }
    Some(value.to_owned())
}
