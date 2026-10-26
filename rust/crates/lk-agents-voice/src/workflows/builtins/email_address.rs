pub fn is_valid_email(email: &str) -> bool {
    let Some((local, domain)) = email.split_once('@') else { return false };
    if domain.contains('@') {
        return false;
    }

    let mut chars = local.chars();
    if !chars.next().is_some_and(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')) {
        return false;
    }

    let labels: Vec<&str> = domain.split('.').collect();
    // `(?:label\.)+` requires at least one label before the TLD.
    if labels.len() < 2 {
        return false;
    }
    let (tld, rest) = labels.split_last().expect("len >= 2");
    if tld.len() < 2 || !tld.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    rest.iter().all(|label| is_domain_label(label))
}

fn is_domain_label(label: &str) -> bool {
    let bytes = label.as_bytes();
    match bytes {
        [] => false,
        [only] => only.is_ascii_alphanumeric(),
        [first, middle @ .., last] => {
            first.is_ascii_alphanumeric()
                && last.is_ascii_alphanumeric()
                && middle.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'-')
        }
    }
}
