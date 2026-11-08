//! SDK numbers and SDK events.
//!
//! A `wire_enum` an SDK holds as its number crosses as that number wherever it is declared so:
//! an `sdk` proxy's returns, a `#[host(sdk)]` function parameter and an `sdk` record's fields. An
//! unknown number reads as the enum's `Default`.
//!
//! `proxy(.., events(..))` declares the SDK events the host object's emitter delivers, each into
//! the native entry of the same name in the declared functions trait. Codegen writes the
//! subscription (`_generated/<module>.py`): one handler per event, its callback arguments
//! converted by the entry's parameter types.

use crate::decl::vocabulary::{self as voc, Entry, Value};
use crate::decl::{Declarations, Field, Form, HostForm, Param, Record, Ty};
use crate::python::Out;

/// A `wire_enum` (or an optional one) that can cross as its SDK number.
pub fn sdk_number(ty: &Ty) -> bool {
    enum_of(ty).is_some()
}

/// Its annotation as an SDK number.
pub fn annotation(ty: &Ty) -> &'static str {
    match ty {
        Ty::Optional(_) => "int | None",
        _ => "int",
    }
}

/// `(enum's Rust path, optional)`.
fn enum_of(ty: &Ty) -> Option<(&str, bool)> {
    fn plain(t: &Ty) -> Option<&str> {
        match t {
            Ty::Record(Record { host: HostForm::Wire, rust, .. }) => Some(rust),
            _ => None,
        }
    }
    match ty {
        Ty::Optional(inner) => plain(inner).map(|rust| (rust, true)),
        other => plain(other).map(|rust| (rust, false)),
    }
}

/// `PyResult<T>` of the object `o` holding the SDK number of `rust`.
fn number(o: &str, rust: &str) -> String {
    format!(
        "{o}.extract::<i32>().map(|v| <{rust} as ::std::convert::TryFrom<i32>>::try_from(v).unwrap_or_default())"
    )
}

/// `PyResult<T>` of `v`, `None` read as `None` when optional.
fn read_number(v: &str, ty: &Ty) -> String {
    let (rust, optional) = enum_of(ty).expect("an SDK number");
    match optional {
        true => {
            format!("if {v}.is_none() {{ Ok(None) }} else {{ ({}).map(Some) }}", number(v, rust))
        }
        false => number(v, rust),
    }
}

/// A `#[host(sdk)]` parameter's adapter type and conversion.
pub fn input(p: &Param) -> Result<(String, Option<String>), String> {
    let n = &p.name;
    Ok(match enum_of(&p.ty) {
        Some((rust, true)) => (
            "Option<Bound<'_, PyAny>>".into(),
            Some(format!("{n}.map(|v| {}).transpose()?", number("v", rust))),
        ),
        Some((rust, false)) => ("Bound<'_, PyAny>".into(), Some(format!("{}?", number(n, rust)))),
        None => return Err(format!("{n}: sdk applies to a wire enum, optional or not")),
    })
}

/// An `sdk` record's field, read as its SDK number.
pub fn field_read(owner: &str, f: &Field, obj: &str) -> String {
    format!(
        "crate::py_values::attr_with({obj}, {owner:?}, {:?}, |v| {})?",
        f.host_name(),
        read_number("v", &f.ty)
    )
}

/// An `sdk` record's field, emitted as its SDK number.
pub fn field_set(f: &Field) -> String {
    let (key, name) = (f.host_name(), &f.rust_name);
    match enum_of(&f.ty) {
        Some((_, true)) => format!(
            "        kwargs.set_item({key:?}, {name}.map(::std::convert::Into::<i32>::into))?;\n"
        ),
        _ => format!(
            "        kwargs.set_item({key:?}, ::std::convert::Into::<i32>::into({name}))?;\n"
        ),
    }
}

/// `proxy(.., events(module = "..", on = "a.b", entries = Trait, pass(p = "self.x"), event(args(..), p = "arg.x", when(arg = "module.Class")), ..))`.
#[derive(Debug, Clone)]
pub struct Events {
    /// The generated module (`_generated/<module>.py`) holding the subscription.
    pub module: String,
    /// The emitter: the attribute path from the host object whose `on` subscribes.
    pub on: Vec<String>,
    /// The `#[host(functions)]` trait holding each event's entry, by the event's name.
    pub entries: String,
    /// Entry parameters every event passes, read once from the host object (`self.x`).
    pub pass: Vec<(String, Vec<String>)>,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone)]
pub struct Event {
    /// The SDK event, and its entry's Rust name.
    pub name: String,
    /// The callback's positional arguments, by name. An entry parameter of the same name takes it.
    pub args: Vec<String>,
    /// Entry parameters read along an attribute path from an argument (`event.digit`); an
    /// optional one is `None` once a step is.
    pub paths: Vec<(String, Vec<String>)>,
    /// `(argument, "module.Class")`: the event is delivered only when the argument is one.
    pub when: Vec<(String, String)>,
}

const SELF: &str = "self";

pub fn read(at: &str, events: &Entry) -> Result<Events, String> {
    let at = format!("{at}: events");
    let path = |text: &str| -> Result<Vec<String>, String> {
        let steps: Vec<String> = text.split('.').map(str::to_owned).collect();
        let word = |s: &String| {
            s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        match steps.iter().all(word) {
            true => Ok(steps),
            false => Err(format!("{at}: `{text}` is an attribute path")),
        }
    };
    let module = voc::text(&at, &events.nested, "module")?
        .ok_or_else(|| format!("{at}: module = \"..\" names the generated module"))?;
    let on = path(&voc::text(&at, &events.nested, "on")?.unwrap_or_default())?;
    let entries = match voc::rust(&at, &events.nested, "entries")? {
        Some(ty) => quote::quote!(#ty).to_string().replace(' ', ""),
        None => return Err(format!("{at}: entries = Trait names the entries' functions trait")),
    };
    let pairs = |entries: &[Entry], root: &dyn Fn(&str) -> bool| {
        entries
            .iter()
            .map(|e| match &e.value {
                Some(Value::Text(t)) => {
                    let steps = path(t)?;
                    match root(&steps[0]) {
                        true => Ok((e.key.clone(), steps)),
                        false => Err(format!("{at}: `{t}` starts at an argument")),
                    }
                }
                _ => Err(format!("{at}: `{}` takes an attribute path, quoted", e.key)),
            })
            .collect::<Result<Vec<_>, String>>()
    };
    let mut pass = Vec::new();
    let mut list = Vec::new();
    for e in &events.nested {
        match e.key.as_str() {
            "module" | "on" | "entries" => {}
            "pass" => pass = pairs(&e.nested, &|root| root == SELF)?,
            name => {
                let event = format!("{at}({name})");
                let args = match voc::find(&e.nested, "args") {
                    Some(args) => args.nested.iter().map(|a| a.key.clone()).collect(),
                    None => Vec::new(),
                };
                let when = match voc::find(&e.nested, "when") {
                    Some(when) => when
                        .nested
                        .iter()
                        .map(|w| match (&w.value, args.contains(&w.key)) {
                            (Some(Value::Text(class)), true) if class.contains('.') => {
                                Ok((w.key.clone(), class.clone()))
                            }
                            _ => Err(format!("{event}: when(arg = \"module.Class\")")),
                        })
                        .collect::<Result<_, String>>()?,
                    None => Vec::new(),
                };
                let rest: Vec<Entry> = e
                    .nested
                    .iter()
                    .filter(|n| n.key != "args" && n.key != "when")
                    .cloned()
                    .collect();
                let paths = pairs(&rest, &|root| args.iter().any(|a| a == root))?;
                list.push(Event { name: name.to_owned(), args, paths, when });
            }
        }
    }
    Ok(Events { module, on, entries, pass, events: list })
}

/// The subscription modules of every proxy declaring `events(..)`.
pub fn python(decls: &Declarations) -> Result<Vec<Out>, String> {
    decls
        .proxies
        .iter()
        .filter_map(|p| p.events.as_ref().map(|e| (p, e)))
        .map(|(p, events)| subscription(decls, p, events))
        .collect()
}

fn subscription(
    decls: &Declarations,
    p: &crate::decl::Proxy,
    events: &Events,
) -> Result<Out, String> {
    let at = format!("{}: events", p.trait_path);
    let entries = decls
        .classes
        .iter()
        .find(|c| c.form == Form::Functions && c.rust_name == events.entries)
        .ok_or_else(|| format!("{at}: `{}` is not a functions trait", events.entries))?;
    let host = crate::decl::resolve::snake(&p.rust_name);
    // The object the calls address: the view over the host object, if any.
    let (host_module, host_class) = match &p.view {
        Some((module, class)) => (module.as_str(), class.as_str()),
        None => p.host.rsplit_once('.').expect("dotted host class"),
    };
    // Modules by alias (`livekit.rtc` as `_rtc`): the adapters' imported in the call, as the
    // adapters' module imports this one.
    let mut adapters: Vec<String> = Vec::new();
    let mut modules: Vec<String> = Vec::new();
    let named = |class: &str, into: &mut Vec<String>| -> String {
        let (module, name) = class.rsplit_once('.').expect("dotted class");
        if !into.iter().any(|m| m == module) {
            into.push(module.to_owned());
        }
        format!("_{}.{name}", module.rsplit('.').next().expect("a module"))
    };
    let mut body = String::new();
    let attrs = |root: &str, steps: &[String]| -> String {
        std::iter::once(root.to_owned()).chain(steps.iter().cloned()).collect::<Vec<_>>().join(".")
    };
    let mut pass = String::new();
    for (name, steps) in &events.pass {
        pass.push_str(&format!("    {name} = {}\n", attrs(&host, &steps[1..])));
    }
    for event in &events.events {
        let at = format!("{at}({})", event.name);
        let entry =
            entries.methods.iter().find(|m| m.rust_name == event.name).ok_or_else(|| {
                format!("{at}: `{}` has no entry `{}`", events.entries, event.name)
            })?;
        let mut lines = String::new();
        for (arg, class) in &event.when {
            let name = named(class, &mut modules);
            lines.push_str(&format!(
                "        if not isinstance({arg}, {name}):\n            return\n"
            ));
        }
        let mut values = Vec::new();
        for param in entry.params.iter().filter(|q| !q.ty.hidden()) {
            let value = if events.pass.iter().any(|(n, _)| *n == param.name) {
                param.name.clone()
            } else if event.args.contains(&param.name) {
                param.name.clone()
            } else if let Some((_, steps)) = event.paths.iter().find(|(n, _)| *n == param.name) {
                match (&param.ty, steps.split_last()) {
                    // An optional value is `None` when its owner is.
                    (Ty::Optional(_), Some((last, owner))) if steps.len() > 1 => {
                        let parent = format!("{}_owner", param.name);
                        lines.push_str(&format!("        {parent} = {}\n", owner.join(".")));
                        format!("None if {parent} is None else {parent}.{last}")
                    }
                    _ => steps.join("."),
                }
            } else {
                return Err(format!("{at}: entry parameter `{}` has no source", param.name));
            };
            if sdk_number(&param.ty) && !param.sdk {
                return Err(format!(
                    "{at}({}): an SDK number is a #[host(sdk)] parameter",
                    param.name
                ));
            }
            values.push(match &param.ty {
                // The host object's adapter of another SDK object: `Class(host, value)`.
                Ty::Proxy { host: adapter, .. } => {
                    format!("{}({host}, {value})", named(adapter, &mut adapters))
                }
                _ => value,
            });
        }
        let used = |arg: &String| {
            entry.params.iter().any(|q| q.name == *arg)
                || event.paths.iter().any(|(_, s)| s[0] == *arg)
                || event.when.iter().any(|(a, _)| a == arg)
        };
        let args: Vec<String> = event
            .args
            .iter()
            .map(|a| if used(a) { format!("{a}: Any") } else { format!("_{a}: Any") })
            .collect();
        body.push_str(&format!(
            "\n    def {name}({}) -> None:\n{lines}        _core.{}({})\n\n    on({name:?}, {name})\n",
            args.join(", "),
            entry.name,
            values.join(", "),
            name = event.name,
        ));
    }
    let import = |module: &String| match module.rsplit_once('.') {
        Some((parent, leaf)) => format!("from {parent} import {leaf} as _{leaf}\n"),
        None => format!("import {module} as _{module}\n"),
    };
    let mut imports = String::from("from typing import TYPE_CHECKING, Any\n\n");
    imports.extend(modules.iter().map(import));
    imports.push_str("\nfrom .. import _core\n");
    imports.push_str(&format!("\nif TYPE_CHECKING:\n    from {host_module} import {host_class}\n"));
    let local = adapters.iter().map(|m| format!("    {}", import(m))).collect::<String>();
    let local = if local.is_empty() { String::new() } else { format!("{local}\n") };
    let text = format!(
        "# GENERATED FILE -- DO NOT EDIT.\n#\n# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `events(..)`\n# of `{trait_path}` in {source}.\n\nfrom __future__ import annotations\n\n{imports}\n\ndef subscribe({host}: {host_class}) -> None:\n    \"\"\"Deliver each SDK event `{host}.{on}` emits to its native entry.\"\"\"\n{local}    on = {host}.{on}.on\n{pass}{body}",
        trait_path = p.trait_path,
        source = p.source,
        on = events.on.join("."),
    );
    Ok(Out { path: format!("{}.py", events.module), text })
}

#[cfg(test)]
#[path = "proxy_sdk_tests.rs"]
mod tests;
