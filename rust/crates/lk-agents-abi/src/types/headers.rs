//! Ordered HTTP headers shared by clients and the host boundary.

#[crate::host(codec(
    extract,
    emit,
    host = "list[tuple[str, str]]",
    via = Vec<(String, String)>
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Headers(Vec<(String, String)>);

impl Headers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = name.into();
        let value = value.into();
        match self.0.iter_mut().find(|(key, _)| *key == name) {
            Some(slot) => slot.1 = value,
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

impl FromIterator<(String, String)> for Headers {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(pairs: I) -> Self {
        let mut headers = Self::new();
        for (name, value) in pairs {
            headers.set(name, value);
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
}
