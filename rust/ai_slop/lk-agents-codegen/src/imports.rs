//! Merging one generated module's top-level imports into another's header, as ruff's isort
//! (`combine-as-imports`, order by type) would write them: a module imported by both keeps one
//! statement with the union of its names; any other statement joins its section in order.

/// Merge generated import headers, retaining the scope of typing-only imports.
pub fn merge(header: &str, extra: &str) -> Result<String, String> {
    let Some((extra_runtime, added)) = extra.split_once("if TYPE_CHECKING:\n") else {
        return merge_scope(header, extra);
    };
    let imports = merge_scope(header, extra_runtime)?;
    let (runtime, existing) = imports.split_once("if TYPE_CHECKING:\n").unwrap_or((&imports, ""));
    // A name the module already imports at runtime needs no type-checking import.
    let lines = |text: &str| text.lines().map(str::to_owned).collect::<Vec<_>>();
    let imported = statements(&lines(runtime))?;
    let added = added
        .lines()
        .map(|line| line.strip_prefix("    ").unwrap_or(line).to_owned())
        .collect::<Vec<_>>();
    let added = statements(&added)?
        .into_iter()
        .filter_map(|mut st| {
            st.names.retain(|n| {
                !imported.iter().any(|i| i.from && i.module == st.module && i.names.contains(n))
            });
            match (st.from, st.names.is_empty()) {
                (true, true) => None,
                (true, false) => Some(render(&st.module, &st.names).join("\n")),
                (false, _) => Some(format!("import {}", st.module)),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let dedent = |block: &str| {
        block
            .lines()
            .map(|line| line.strip_prefix("    ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let checking = merge_scope(&dedent(existing), &dedent(&added))?;
    if checking.trim().is_empty() {
        return Ok(runtime.to_owned());
    }
    let checking = checking
        .trim()
        .lines()
        .map(|line| if line.is_empty() { String::new() } else { format!("    {line}") })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(format!("{}\n\nif TYPE_CHECKING:\n{checking}\n", runtime.trim_end()))
}

/// Merge imports within one scope, leaving any other header text in place.
fn merge_scope(header: &str, extra: &str) -> Result<String, String> {
    let mut lines: Vec<String> = header.lines().map(str::to_owned).collect();
    for statement in statements(&extra.lines().map(str::to_owned).collect::<Vec<_>>())? {
        let ours = statements(&lines)?;
        match ours.iter().find(|s| s.from && s.module == statement.module) {
            Some(existing) if statement.from => {
                let mut names = existing.names.clone();
                names.extend(statement.names.iter().cloned());
                names.sort_by_key(|n| order(n));
                names.dedup();
                lines.splice(existing.lines.clone(), render(&statement.module, &names));
            }
            _ if ours
                .iter()
                .any(|s| !s.from && !statement.from && s.module == statement.module) => {}
            _ => {
                let key = statement.key();
                let mut rendered = match statement.from {
                    true => render(&statement.module, &statement.names),
                    false => vec![format!("import {}", statement.module)],
                };
                let mut section = ours.iter().filter(|s| s.key().0 == key.0);
                let at = match (section.clone().find(|s| s.key() > key), section.next_back()) {
                    (Some(after), _) => after.lines.start,
                    (None, Some(last)) => last.lines.end,
                    // A section of its own: before the next one, else after the last.
                    (None, None) => match ours.iter().find(|s| s.key().0 > key.0) {
                        Some(next) => {
                            rendered.push(String::new());
                            next.lines.start
                        }
                        None => {
                            rendered.insert(0, String::new());
                            match ours.last() {
                                Some(last) => last.lines.end,
                                None => future_end(&lines),
                            }
                        }
                    },
                };
                lines.splice(at..at, rendered);
            }
        }
    }
    Ok(lines.join("\n") + "\n")
}

/// Past `from __future__ import annotations`, else the top.
fn future_end(lines: &[String]) -> usize {
    lines.iter().position(|l| l.starts_with("from __future__ ")).map_or(0, |i| i + 1)
}

struct Statement {
    from: bool,
    module: String,
    names: Vec<String>,
    lines: std::ops::Range<usize>,
}

impl Statement {
    /// isort's position: section, then plain imports before `from` imports, then the module.
    fn key(&self) -> (u8, bool, String) {
        let section = match self.module.as_str() {
            m if m.starts_with('.') => 3,
            m if m == "livekit" || m.starts_with("livekit.") => 2,
            m if STDLIB.contains(&m.split('.').next().unwrap_or(m)) => 0,
            _ => 1,
        };
        let module = if self.module.starts_with('.') {
            let parent = self.module.bytes().take_while(|ch| *ch == b'.').count();
            format!("{:04}_{}", 999usize.saturating_sub(parent), &self.module[parent..])
        } else {
            self.module.clone()
        };
        (section, self.from, module)
    }
}

/// The standard-library modules generated code imports.
const STDLIB: &[&str] = &[
    "__future__",
    "abc",
    "builtins",
    "asyncio",
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "json",
    "inspect",
    "time",
    "types",
    "typing",
];

/// The top-level import statements of `lines` (a `TYPE_CHECKING` block's are its own).
fn statements(lines: &[String]) -> Result<Vec<Statement>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].as_str();
        let start = i;
        i += 1;
        if let Some(module) = line.strip_prefix("import ") {
            out.push(Statement {
                from: false,
                module: module.trim().to_owned(),
                names: Vec::new(),
                lines: start..i,
            });
            continue;
        }
        let Some(rest) = line.strip_prefix("from ") else { continue };
        let (module, names) =
            rest.split_once(" import ").ok_or_else(|| format!("not an import: `{line}`"))?;
        let mut names = names.trim().to_owned();
        if names == "(" {
            names.clear();
            while i < lines.len() && lines[i].trim() != ")" {
                names.push_str(lines[i].trim());
                i += 1;
            }
            i += 1;
        }
        if module == "__future__" {
            continue;
        }
        out.push(Statement {
            from: true,
            module: module.to_owned(),
            names: names
                .split(',')
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(str::to_owned)
                .collect(),
            lines: start..i,
        });
    }
    Ok(out)
}

/// ruff's order by type: CONSTANTS, then Classes, then the rest.
fn order(name: &str) -> (u8, String) {
    let kind = if name.chars().all(|c| c.is_uppercase() || c == '_') {
        0
    } else if name.starts_with(char::is_uppercase) {
        1
    } else {
        2
    };
    (kind, name.to_lowercase())
}

/// `from module import names`, wrapped as ruff formats it at 100 columns.
pub(crate) fn render(module: &str, names: &[String]) -> Vec<String> {
    let flat = format!("from {module} import {}", names.join(", "));
    if flat.len() <= 100 {
        return vec![flat];
    }
    std::iter::once(format!("from {module} import ("))
        .chain(names.iter().map(|n| format!("    {n},")))
        .chain(std::iter::once(")".to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    // Catches: a declared record's import repeated beside the legacy one (ruff F811), or a new
    // statement placed outside its isort section and order (ruff I001).
    #[test]
    fn imports_merge_into_one_statement_per_module_in_isort_order() {
        let legacy = "from __future__ import annotations\n\nimport time\nfrom enum import Enum, unique\nfrom typing import TYPE_CHECKING, Annotated, Any, Literal\n\nfrom pydantic import BaseModel\n\nfrom .. import _bindings\n\nif TYPE_CHECKING:\n    from ..voice.speech_handle import SpeechHandle\n";
        let declared = "from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Literal, TypeAlias\n";
        assert_eq!(
            super::merge(legacy, declared).unwrap(),
            "from __future__ import annotations\n\nimport time\nfrom dataclasses import dataclass\nfrom enum import Enum, unique\nfrom typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias\n\nfrom pydantic import BaseModel\n\nfrom .. import _bindings\n\nif TYPE_CHECKING:\n    from ..voice.speech_handle import SpeechHandle\n"
        );
    }

    #[test]
    fn typing_imports_merge_in_their_own_scope() {
        let header = "from typing import TYPE_CHECKING\n\nfrom ..types import NOT_GIVEN\n\nif TYPE_CHECKING:\n    from ..types import Existing\n";
        let extra = "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from ..types import NotGivenOr\n";
        assert_eq!(
            super::merge(header, extra).unwrap(),
            "from typing import TYPE_CHECKING\n\nfrom ..types import NOT_GIVEN\n\nif TYPE_CHECKING:\n    from ..types import Existing, NotGivenOr\n"
        );
    }
}
