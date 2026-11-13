//! Token helpers for [`crate::sources`]; nothing here checks a rule. Tokens skip comments and keep string literals whole, and macro bodies stay
//! visible, which is where `select!` and attribute-generating macros live.

use proc_macro2::{Delimiter, Group, Spacing, TokenStream, TokenTree};

/// Calls `visit` with the tokens of every nesting level, outermost first.
pub fn levels(stream: TokenStream, visit: &mut impl FnMut(&[TokenTree])) {
    let level: Vec<TokenTree> = stream.into_iter().collect();
    visit(&level);
    for token in &level {
        if let TokenTree::Group(group) = token {
            levels(group.stream(), visit);
        }
    }
}

pub fn line(token: &TokenTree) -> usize {
    token.span().start().line
}

pub fn ident(token: Option<&TokenTree>) -> Option<String> {
    match token {
        Some(TokenTree::Ident(ident)) => Some(ident.to_string()),
        _ => None,
    }
}

pub fn punct(token: Option<&TokenTree>, ch: char) -> bool {
    matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == ch)
}

pub fn group(token: Option<&TokenTree>, delimiter: Delimiter) -> Option<&Group> {
    match token {
        Some(TokenTree::Group(group)) if group.delimiter() == delimiter => Some(group),
        _ => None,
    }
}

/// `a :: b`: the identifier after a path separator at `at`.
pub fn path_segment(level: &[TokenTree], at: usize) -> Option<String> {
    let joint = matches!(level.get(at), Some(TokenTree::Punct(p)) if p.as_char() == ':' && p.spacing() == Spacing::Joint);
    (joint && punct(level.get(at + 1), ':')).then(|| ident(level.get(at + 2))).flatten()
}

/// An attribute's path: `crate::host(..)` -> `["crate", "host"]`.
pub fn attribute(group: &Group) -> Vec<String> {
    let tokens: Vec<TokenTree> = group.stream().into_iter().collect();
    let mut path: Vec<String> = ident(tokens.first()).into_iter().collect();
    while let Some(next) = path.last().and_then(|_| path_segment(&tokens, 3 * path.len() - 2)) {
        path.push(next);
    }
    path
}

/// The future expression of each `select!` branch, in order (`else` and guards excluded).
pub fn select_futures(body: &[TokenTree]) -> Vec<Vec<TokenTree>> {
    let mut futures = Vec::new();
    let mut at = if ident(body.first()).as_deref() == Some("biased") { 2 } else { 0 };
    while at < body.len() {
        let is_else = ident(body.get(at)).as_deref() == Some("else");
        if !is_else {
            while at < body.len() && !single_eq(body, at) {
                at += 1;
            }
            at += 1;
            let start = at;
            while at < body.len() && !fat_arrow(body, at) && !guard(body, at) {
                at += 1;
            }
            futures.push(body[start..at.min(body.len())].to_vec());
        }
        while at < body.len() && !fat_arrow(body, at) {
            at += 1;
        }
        at += 2;
        if group(body.get(at), Delimiter::Brace).is_some() {
            at += 1;
        } else {
            while at < body.len() && !punct(body.get(at), ',') {
                at += 1;
            }
        }
        if punct(body.get(at), ',') {
            at += 1;
        }
    }
    futures
}

/// The last method a future expression calls: `rx.recv()` -> `recv`.
pub fn last_call(future: &[TokenTree]) -> Option<String> {
    (1..future.len())
        .rev()
        .find(|&at| {
            group(future.get(at), Delimiter::Parenthesis).is_some()
                && ident(future.get(at - 1)).is_some()
        })
        .and_then(|at| ident(future.get(at - 1)))
}

pub fn single_eq(level: &[TokenTree], at: usize) -> bool {
    let after_joint =
        at > 0 && matches!(&level[at - 1], TokenTree::Punct(p) if p.spacing() == Spacing::Joint);
    matches!(&level[at], TokenTree::Punct(p) if p.as_char() == '=' && p.spacing() == Spacing::Alone)
        && !after_joint
}

pub fn fat_arrow(level: &[TokenTree], at: usize) -> bool {
    matches!(&level[at], TokenTree::Punct(p) if p.as_char() == '=' && p.spacing() == Spacing::Joint)
        && punct(level.get(at + 1), '>')
}

fn guard(level: &[TokenTree], at: usize) -> bool {
    punct(level.get(at), ',') && ident(level.get(at + 1)).as_deref() == Some("if")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calls(source: &str) -> Vec<Option<String>> {
        let body: Vec<TokenTree> =
            source.parse::<TokenStream>().expect("tokens").into_iter().collect();
        select_futures(&body).iter().map(|f| last_call(f)).collect()
    }

    #[test]
    fn branches_split_with_blocks_guards_and_else() {
        let found = calls(
            "biased; Some(x) = rx.recv(), if open => { a(x) } \
             r = shell.run() => r, () = &mut sleep => break, else => return",
        );
        assert_eq!(found, [Some("recv".into()), Some("run".into()), None]);
    }

    #[test]
    fn comparisons_in_patterns_and_handlers_do_not_split() {
        let found =
            calls("v = self.next_event() => if v == 1 { x() } else { y() }, _ = t.closed() => ()");
        assert_eq!(found, [Some("next_event".into()), Some("closed".into())]);
    }
}
