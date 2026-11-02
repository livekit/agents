//! XML rendering for the host chat-context utility. Content and attributes are not escaped.

/// Render an element with owned attribute pairs, retaining their supplied order.
pub(super) fn to_xml_of(
    tag_name: &str,
    content: Option<&str>,
    attrs: Option<Vec<(String, String)>>,
) -> String {
    let attrs = attrs.unwrap_or_default();
    to_xml(
        tag_name,
        content,
        &attrs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>(),
    )
}

/// An absent or empty body renders a self-closing element.
fn to_xml(tag_name: &str, content: Option<&str>, attrs: &[(&str, &str)]) -> String {
    let attrs_str = if attrs.is_empty() {
        None
    } else {
        Some(attrs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect::<Vec<_>>().join(" "))
    };

    match content.filter(|c| !c.is_empty()) {
        Some(content) => {
            let open = match &attrs_str {
                Some(a) => format!("<{tag_name} {a}>"),
                None => format!("<{tag_name}>"),
            };
            format!("{open}\n{content}\n</{tag_name}>")
        }
        None => match &attrs_str {
            Some(a) => format!("<{tag_name} {a} />"),
            None => format!("<{tag_name} />"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_content_is_self_closing() {
        assert_eq!(to_xml("t", Some(""), &[]), "<t />");
        assert_eq!(to_xml("t", None, &[("k", "v")]), r#"<t k="v" />"#);
    }
}
