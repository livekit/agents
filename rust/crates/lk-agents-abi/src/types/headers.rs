//! Ordered HTTP headers shared by clients and the host boundary.

#[crate::host(codec(
    extract,
    emit,
    host = "list[tuple[str, str]]",
    via = Vec<(String, String)>
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Headers(Vec<(String, String)>);

/// One value per name; names compare case-insensitively, as in `http::HeaderMap`.
impl Headers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `name` to `value`, in the name's place when it is already set.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let (name, value) = (name.into(), value.into());
        match self.0.iter_mut().find(|(key, _)| key.eq_ignore_ascii_case(&name)) {
            Some(slot) => *slot = (name, value),
            None => self.0.push((name, value)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    pub fn remove(&mut self, name: &str) {
        self.0.retain(|(key, _)| !key.eq_ignore_ascii_case(name));
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (String, String)> {
        self.0.iter()
    }
}

/// The host's pairs: a repeated name keeps its last value, as a Python `dict` of them would.
impl FromIterator<(String, String)> for Headers {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(pairs: I) -> Self {
        let mut headers = Self::new();
        for (name, value) in pairs {
            headers.insert(name, value);
        }
        headers
    }
}

impl IntoIterator for Headers {
    type Item = (String, String);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<Vec<(String, String)>> for Headers {
    fn from(pairs: Vec<(String, String)>) -> Self {
        pairs.into_iter().collect()
    }
}

impl From<Headers> for Vec<(String, String)> {
    fn from(headers: Headers) -> Self {
        headers.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(headers: &Headers) -> Vec<(&str, &str)> {
        headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }

    #[test]
    fn host_pairs_preserve_order_and_replace_repeated_names() {
        let mut headers = Headers::from(vec![
            ("Authorization".into(), "old".into()),
            ("X-Job".into(), "job".into()),
            ("Authorization".into(), "new".into()),
        ]);
        assert_eq!(headers.get("authorization"), Some("new"));
        headers.remove("AUTHORIZATION");
        assert_eq!(Vec::from(headers), [("X-Job".into(), "job".into())]);
    }

    // Catches: an insert that matches its name by case and so leaves a second `Authorization`
    // beside the one it meant to replace.
    #[test]
    fn insert_replaces_a_name_whatever_its_case() {
        let mut headers = Headers::from(vec![
            ("authorization".into(), "Bearer a".into()),
            ("X-Job".into(), "job".into()),
        ]);
        headers.insert("Authorization", "Bearer c");
        assert_eq!(pairs(&headers), [("Authorization", "Bearer c"), ("X-Job", "job")]);
    }
}
