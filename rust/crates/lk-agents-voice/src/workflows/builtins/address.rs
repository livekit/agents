pub fn python_str_repr(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\");
    if s.contains('\'') && !s.contains('"') {
        format!("\"{escaped}\"")
    } else {
        format!("'{}'", escaped.replace('\'', "\\'"))
    }
}
