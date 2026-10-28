//! Static sections inserted into the built-in workflow templates.

use lk_agents_voice::generation::{Rendered, safe_render};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// `str.format` over a built-in template, dropping `missing` ([`render_checked`] proves there are none).
pub fn render(template: &str, data: &[(&str, &str)]) -> String {
    render_checked(template, data).text
}

pub fn render_checked(template: &str, data: &[(&str, &str)]) -> Rendered {
    let map: BTreeMap<&str, &str> = data.iter().copied().collect();
    let obj: Map<String, Value> =
        map.into_iter().map(|(k, v)| (k.to_owned(), Value::String(v.to_owned()))).collect();
    safe_render(template, &Value::Object(obj))
}
