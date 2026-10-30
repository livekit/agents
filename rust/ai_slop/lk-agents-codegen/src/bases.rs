//! The plugin base classes (the Python classes plugins subclass for STT, TTS, LLM, VAD and
//! realtime, and their streams): hand-written Python beside the proxy trait whose host class each
//! is, `#[host(proxy(host = "..", base = "tts.py"))]`. A file is written into the generated module
//! of its name after the module's records: its imports merged into the module's, the rest
//! verbatim. Every call a proxy declares must be a method of its host class there, so a trait
//! method the base class does not answer is a codegen error rather than a runtime one.

use crate::decl::{Declarations, Proxy, ProxyCall};
use std::path::Path;

/// What one module gains: its classes, the imports they need, and their public names.
pub struct Bases {
    pub module: String,
    pub sources: Vec<String>,
    pub body: String,
    /// `(module, name)`; `module` empty is `import name`, `?module` is under `TYPE_CHECKING`.
    pub imports: Vec<(String, String)>,
    /// `(class, public module)`: the `__module__` fixups.
    pub paths: Vec<(String, String)>,
}

/// Every base file a declared proxy names, in first-seen order.
pub fn read(root: &Path, declared: &Declarations) -> Result<Vec<Bases>, String> {
    let mut files: Vec<(String, Vec<&Proxy>)> = Vec::new();
    for p in &declared.proxies {
        let Some(base) = &p.base else { continue };
        let dir = p.source.rsplit_once('/').map_or("", |(dir, _)| dir);
        let path = format!("{dir}/{base}");
        match files.iter_mut().find(|(f, _)| *f == path) {
            Some((_, proxies)) => proxies.push(p),
            None => files.push((path, vec![p])),
        }
    }
    files
        .iter()
        .map(|(path, proxies)| {
            let text = std::fs::read_to_string(root.join(path))
                .map_err(|e| format!("{path}: {e} (named by `base = \"..\"`)"))?;
            parse(path, &text, proxies)
        })
        .collect()
}

/// One base file: the imports it opens with, then the classes.
pub fn parse(path: &str, text: &str, proxies: &[&Proxy]) -> Result<Bases, String> {
    let module = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{path}: a base file is `<module>.py`"))?
        .to_owned();
    let (imports, body) = split(path, text, &module)?;
    let public = proxies.iter().map(|p| p.host.rsplit_once('.').map_or("", |(m, _)| m));
    let public = public.fold(None, |one: Option<&str>, m| match one {
        Some(o) if o != m => Some(""),
        _ => Some(m),
    });
    let public = public
        .filter(|m| !m.is_empty())
        .ok_or_else(|| format!("{path}: its proxies' host classes share one public module"))?;
    for p in proxies {
        let class = p.host.rsplit('.').next().unwrap_or(&p.host);
        let block = class_block(body, class)
            .ok_or_else(|| format!("{path}: no `class {class}`, {}'s host class", p.rust_name))?;
        if p.generic && !header(block).contains("Generic[") {
            return Err(format!("{path}: `{class}` is `generic` but not `Generic[..]`"));
        }
        for m in p.methods.iter().filter(|m| m.kind != ProxyCall::Bind) {
            let def = |kw: &str| format!("\n    {kw} {}(", m.name);
            if !block.contains(&def("def")) && !block.contains(&def("async def")) {
                return Err(format!(
                    "{path}: `{class}` answers no `{}::{}` (`{}`)",
                    p.rust_name, m.rust_name, m.name
                ));
            }
        }
    }
    let paths = body
        .lines()
        .filter_map(|l| l.strip_prefix("class "))
        .map(|l| l.split(['(', ':']).next().unwrap_or(l).to_owned())
        .filter(|c| !c.starts_with('_'))
        .map(|c| (c, public.to_owned()))
        .collect();
    Ok(Bases { module, sources: vec![path.to_owned()], body: body.to_owned(), imports, paths })
}

/// The imports the file opens with, and the rest. An import from the module itself is the file's
/// alone: the generated module defines those records above the classes.
fn split<'t>(
    path: &str,
    text: &'t str,
    module: &str,
) -> Result<(Vec<(String, String)>, &'t str), String> {
    let own = format!(".{module}");
    let mut imports: Vec<(String, String)> = Vec::new();
    let mut lines = text.split_inclusive('\n');
    let mut at = 0;
    let mut type_checking = false;
    while let Some(line) = lines.next() {
        let l = line.trim_end();
        let indented = l.starts_with(' ');
        let stmt = l.trim_start();
        if stmt.is_empty() || stmt.starts_with('#') || l == "from __future__ import annotations" {
        } else if l == "if TYPE_CHECKING:" {
            type_checking = true;
        } else if let Some(name) = l.strip_prefix("import ") {
            imports.push((String::new(), name.to_owned()));
        } else if let Some(rest) = stmt.strip_prefix("from ")
            && (!indented || type_checking)
        {
            let (m, first) =
                rest.split_once(" import ").ok_or_else(|| format!("{path}: `{l}`: no import"))?;
            let mut names = first.to_owned();
            while names.starts_with('(') && !names.contains(')') {
                let next = lines.next().ok_or_else(|| format!("{path}: `{l}` is not closed"))?;
                at += next.len();
                names.push_str(next);
            }
            let prefix = if indented { "?" } else { "" };
            for n in names
                .split(',')
                .map(|n| n.trim_matches(|c: char| c.is_whitespace() || "()".contains(c)))
            {
                if !n.is_empty() && m != own && !(m == "typing" && n == "TYPE_CHECKING") {
                    imports.push((format!("{prefix}{m}"), n.to_owned()));
                }
            }
        } else if indented {
            return Err(format!("{path}: `{l}`: only imports go under `if TYPE_CHECKING:`"));
        } else {
            return Ok((imports, &text[at..]));
        }
        at += line.len();
    }
    Err(format!("{path}: holds no class"))
}

/// `class <name>` to the next top-level statement.
fn class_block<'t>(body: &'t str, name: &str) -> Option<&'t str> {
    let start = ["(", ":"].iter().find_map(|end| {
        body.find(&format!("class {name}{end}")).filter(|&i| i == 0 || body[..i].ends_with('\n'))
    })?;
    let rest = &body[start..];
    let end = rest
        .match_indices('\n')
        .map(|(i, _)| i + 1)
        .find(|&i| rest[i..].starts_with(|c: char| c.is_alphabetic() || c == '_' || c == '@'))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// A class block's header: `class X(..):`, over its lines when wrapped.
fn header(block: &str) -> &str {
    block.find(":\n").map_or(block, |i| &block[..i])
}
