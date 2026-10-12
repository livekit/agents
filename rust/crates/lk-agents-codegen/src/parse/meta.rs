//! The attribute grammar both walks read: every `host`/`python`/`node`/`binding` attribute,
//! flattened to [`Meta`] entries, and the lookups over them.

use super::Res;

/// One flattened entry (`python.module`), as `lk_agents_bindgen::meta` reads it.
#[derive(Debug, Clone)]
pub(super) struct Meta {
    pub(super) key: String,
    pub(super) value: Option<String>,
    pub(super) quoted: bool,
}

impl Meta {
    fn from_bindgen(m: lk_agents_bindgen::Meta, prefix: &str) -> Meta {
        let (value, quoted) = match m.value {
            lk_agents_bindgen::Value::Word => (None, false),
            lk_agents_bindgen::Value::Text(s) => (Some(s), true),
            lk_agents_bindgen::Value::Rust(r) => (Some(r), false),
        };
        Meta { key: format!("{prefix}{}", m.key), value, quoted }
    }

    /// The value as Rust source (`defaults(..)`, `stream(sink = ..)`).
    pub(super) fn rust(&self) -> Option<String> {
        let v = self.value.clone()?;
        if self.quoted { lk_agents_bindgen::meta::rust_of(&lk_agents_bindgen::Value::Text(v)) } else { Some(v) }
    }
}

/// Every `host`/`python`/`node`/`binding` attribute, flattened: `host` keys unprefixed, others
/// under their name. A malformed attribute already failed the expansion; here it reads as empty.
pub(super) fn metas(attrs: &[syn::Attribute]) -> Vec<Meta> {
    let mut out: Vec<Meta> =
        lk_agents_bindgen::Metas::of(attrs).unwrap_or_default().0.into_iter().map(|m| Meta::from_bindgen(m, "")).collect();
    for outer in ["python", "node", "binding"] {
        let flat = lk_agents_bindgen::Metas::named(attrs, outer).unwrap_or_default();
        out.extend(flat.0.into_iter().map(|m| Meta::from_bindgen(m, &format!("{outer}."))));
    }
    out
}

pub(super) fn has_host_derive(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if !a.path().is_ident("derive") {
            return false;
        }
        let mut found = false;
        let _ = a.parse_nested_meta(|m| {
            if m.path.is_ident("Host") {
                found = true;
            }
            Ok(())
        });
        found
    })
}

/// A key directly inside `#[host(..)]`, for every host. Only `rename` (e.g. the keyword `type`).
pub(super) fn unprefixed(attrs: &[syn::Attribute], key: &str) -> Option<String> {
    metas(attrs).into_iter().rfind(|m| m.key == key).and_then(|m| m.value)
}

pub(super) fn str_arg(attrs: &[syn::Attribute], outer: &str, key: &str) -> Res<Option<String>> {
    let want = format!("{outer}.{key}");
    Ok(metas(attrs).into_iter().rfind(|m| m.key == want).and_then(|m| m.value))
}

pub(super) fn rust_arg(attrs: &[syn::Attribute], outer: &str, key: &str) -> Option<String> {
    let want = format!("{outer}.{key}");
    metas(attrs).into_iter().rfind(|m| m.key == want).and_then(|m| m.rust())
}

/// Every `#[<outer>(<key> = "..")]`, in order: the import key is repeatable.
pub(super) fn str_args(attrs: &[syn::Attribute], outer: &str, key: &str) -> Res<Vec<String>> {
    let want = format!("{outer}.{key}");
    Ok(metas(attrs).into_iter().filter(|m| m.key == want).filter_map(|m| m.value).collect())
}

pub(super) fn flag(attrs: &[syn::Attribute], outer: &str, word: &str) -> bool {
    let want = format!("{outer}.{word}");
    metas(attrs).into_iter().any(|m| m.key == want && m.value.is_none())
}

pub(super) fn head_of(m: &[Meta]) -> String {
    m.first().map(|x| x.key.split('.').next().unwrap_or("").to_owned()).unwrap_or_default()
}

pub(super) fn host_attr_strict(label: &str, attrs: &[syn::Attribute]) -> Res<Option<Vec<Meta>>> {
    Ok(host_attrs_strict(label, attrs)?.into_iter().next())
}

/// A type may carry two `stream(..)`s.
pub(super) fn host_attrs_strict(label: &str, attrs: &[syn::Attribute]) -> Res<Vec<Vec<Meta>>> {
    let each = lk_agents_bindgen::Metas::each(attrs).map_err(|e| format!("{label}: #[host(..)]: {e}"))?;
    Ok(each.into_iter().map(|m| m.0.into_iter().map(|m| Meta::from_bindgen(m, "")).collect()).collect())
}

pub(super) fn want_head(at: &str, m: &[Meta], want: &str) -> Res<()> {
    match m.first().map(|x| x.key.split('.').next().unwrap_or("")) {
        Some(h) if h == want => Ok(()),
        other => Err(format!("{at}: expected #[host({want}..)], found #[host({})]", other.unwrap_or(""))),
    }
}

pub(super) fn meta_str(m: &[Meta], key: &str) -> Option<String> {
    m.iter().rfind(|x| x.key == key).and_then(|x| x.value.clone())
}

pub(super) fn meta_rust(m: &[Meta], key: &str) -> Option<String> {
    m.iter().rfind(|x| x.key == key).and_then(Meta::rust)
}
