pub fn clean_phone_number(phone_number: &str) -> String {
    phone_number
        .trim()
        .chars()
        .filter(|c| !(c.is_whitespace() || matches!(c, '-' | '(' | ')' | '.')))
        .collect()
}

pub fn is_valid_phone_number(cleaned: &str) -> bool {
    let digits = cleaned.strip_prefix('+').unwrap_or(cleaned);
    let mut chars = digits.chars();
    if !chars.next().is_some_and(|c| ('1'..='9').contains(&c)) {
        return false;
    }
    // `\d{6,14}` after the leading non-zero digit: 7 to 15 digits in total.
    let rest: Vec<char> = chars.collect();
    (6..=14).contains(&rest.len()) && rest.iter().all(char::is_ascii_digit)
}
