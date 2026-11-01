//! Exception constructors and presentation resolve alongside the native error declarations.
//! Emission consumes typed metadata; no emitter discovers or interprets source declarations.
use crate::decl::{
    Declarations, Literal,
    vocabulary::{self as voc, Entry, Value},
};
use crate::python::Out;

#[derive(Debug, Clone)]
pub struct Family {
    module: String,
    path: String,
    classes: Vec<Class>,
}
#[derive(Debug, Clone)]
struct Class {
    name: String,
    base: String,
    doc: Option<String>,
    fields: Option<Vec<Field>>,
    base_args: Vec<String>,
    normalizers: Vec<Normalize>,
    display: Option<String>,
    repr: Option<Vec<String>>,
}
#[derive(Debug, Clone)]
struct Field {
    name: String,
    annotation: String,
    default: Option<Literal>,
    class_default: Option<Literal>,
    positional: bool,
    inherited: bool,
    hidden: bool,
    readonly: bool,
}
#[derive(Debug, Clone)]
struct Normalize {
    field: String,
    hook: String,
    args: Vec<String>,
}

fn required(at: &str, entries: &[Entry], key: &str) -> Result<String, String> {
    voc::text(at, entries, key)?.ok_or_else(|| format!("{at}: missing `{key}`"))
}
fn literal(value: &Value) -> Result<Literal, String> {
    match value {
        Value::Text(value) => Ok(Literal::Str(value.clone())),
        Value::Literal(value) => Ok(value.clone()),
        _ => Err("an exception default must be a literal".into()),
    }
}
fn default(entries: &[Entry], name: &str) -> Result<Option<Literal>, String> {
    voc::find(entries, name).and_then(|entry| entry.value.as_ref()).map(literal).transpose()
}

pub fn resolve(schema: &Entry) -> Result<Family, String> {
    let keys = &schema.nested;
    voc::only("exceptions", keys, &["module", "path", "classes"])?;
    let module = required("exceptions", keys, "module")?;
    let path = required("exceptions", keys, "path")?;
    let classes = voc::find(keys, "classes").ok_or("exceptions: missing classes")?;
    let mut resolved = Vec::new();
    for class in &classes.nested {
        let name = &class.key;
        let keys = &class.nested;
        voc::only(name, keys, &["base", "doc", "fields", "base_args", "normalize", "str", "repr"])?;
        let mut out = Class {
            name: name.clone(),
            base: required(name, keys, "base")?,
            doc: voc::text(name, keys, "doc")?,
            fields: None,
            base_args: Vec::new(),
            normalizers: Vec::new(),
            display: voc::text(name, keys, "str")?,
            repr: voc::find(keys, "repr")
                .map(|entry| entry.nested.iter().map(|entry| entry.key.clone()).collect()),
        };
        if let Some(fields) = voc::find(keys, "fields") {
            let mut resolved = Vec::new();
            let mut keyword = false;
            for field in &fields.nested {
                let keys = &field.nested;
                voc::only(
                    name,
                    keys,
                    &[
                        "ty",
                        "default",
                        "positional",
                        "inherited",
                        "hidden",
                        "class_default",
                        "readonly",
                    ],
                )?;
                let field = Field {
                    name: field.key.clone(),
                    annotation: required(name, keys, "ty")?,
                    default: default(keys, "default")?,
                    class_default: default(keys, "class_default")?,
                    positional: voc::find(keys, "positional").is_some(),
                    inherited: voc::find(keys, "inherited").is_some(),
                    hidden: voc::find(keys, "hidden").is_some(),
                    readonly: voc::find(keys, "readonly").is_some(),
                };
                if field.hidden && field.default.is_none() {
                    return Err(format!(
                        "{name}.{}: a hidden argument needs a default",
                        field.name
                    ));
                }
                if field.readonly
                    && (field.hidden || field.inherited || field.class_default.is_some())
                {
                    return Err(format!("{name}: a readonly field owns its constructor value"));
                }
                if !field.hidden {
                    if keyword && field.positional {
                        return Err(format!(
                            "{name}: positional argument follows keyword argument"
                        ));
                    }
                    keyword |= !field.positional;
                }
                resolved.push(field);
            }
            let args =
                voc::find(keys, "base_args").ok_or_else(|| format!("{name}: missing base_args"))?;
            for arg in &args.nested {
                if !resolved.iter().any(|field: &Field| field.name == arg.key) {
                    return Err(format!("{name}: unknown base argument {}", arg.key));
                }
                out.base_args.push(arg.key.clone());
            }
            if let Some(normalize) = voc::find(keys, "normalize") {
                for target in &normalize.nested {
                    voc::only(name, &target.nested, &["call", "args"])?;
                    let args =
                        voc::find(&target.nested, "args").ok_or("normalization needs args")?;
                    for arg in
                        std::iter::once(&target.key).chain(args.nested.iter().map(|arg| &arg.key))
                    {
                        if !resolved.iter().any(|field| field.name == *arg) {
                            return Err(format!("{name}: normalization names unknown field {arg}"));
                        }
                    }
                    out.normalizers.push(Normalize {
                        field: target.key.clone(),
                        hook: required(name, &target.nested, "call")?,
                        args: args.nested.iter().map(|arg| arg.key.clone()).collect(),
                    });
                }
            }
            out.fields = Some(resolved);
        } else if voc::find(keys, "base_args").is_some() || voc::find(keys, "normalize").is_some() {
            return Err(format!("{name}: constructor metadata needs fields"));
        }
        resolved.push(out);
    }
    Ok(Family { module, path, classes: resolved })
}

pub fn python(decls: &Declarations) -> Vec<Out> {
    decls.exception_families.iter().map(emit).collect()
}

fn emit(family: &Family) -> Out {
    let path = &family.path;
    let mut imports = crate::class_stub::Imports::default();
    let bases =
        family.classes.iter().map(|class| imports.annotation(&class.base, "")).collect::<Vec<_>>();
    let mut text = String::from(
        "# GENERATED by lk-agents-codegen from ABI exception declarations. DO NOT EDIT.\n\nfrom __future__ import annotations\n\nfrom typing import Literal\n",
    );
    text = crate::imports::merge(&text, &imports.render(&[])).expect("resolved exception bases");
    for (class, base) in family.classes.iter().zip(bases) {
        let name = &class.name;
        text.push_str(&format!("\n\nclass {name}({base}):\n"));
        if let Some(doc) = &class.doc {
            text.push_str(&format!("    {doc:?}\n"));
        }
        text.push_str(&format!("    __module__ = {path:?}\n"));
        if let Some(fields) = &class.fields {
            let mut params = Vec::new();
            let mut keyword = false;
            for field in fields {
                if !field.inherited {
                    text.push_str(&format!(
                        "    {}{}: {}",
                        if field.readonly { "_" } else { "" },
                        field.name,
                        field.annotation
                    ));
                    if let Some(value) = &field.class_default {
                        text.push_str(&format!(" = {}", crate::class_stub::py_literal(value)));
                    }
                    text.push('\n');
                }
                if field.hidden {
                    continue;
                }
                if !field.positional && !keyword {
                    params.push("*".into());
                    keyword = true;
                }
                let default = field
                    .default
                    .as_ref()
                    .map(|value| format!(" = {}", crate::class_stub::py_literal(value)))
                    .unwrap_or_default();
                params.push(format!("{}: {}{default}", field.name, field.annotation));
            }
            let params =
                if params.is_empty() { String::new() } else { format!(", {}", params.join(", ")) };
            text.push_str(&format!("\n    def __init__(self{params}) -> None:\n"));
            for normalize in &class.normalizers {
                text.push_str(&format!(
                    "        from {path} import {}\n\n        {} = {}({})\n",
                    normalize.hook,
                    normalize.field,
                    normalize.hook,
                    normalize.args.join(", ")
                ));
            }
            let args = class
                .base_args
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let field = fields
                        .iter()
                        .find(|field| field.name == *name)
                        .expect("resolved base argument");
                    let value = if field.hidden {
                        crate::class_stub::py_literal(
                            field.default.as_ref().expect("resolved hidden default"),
                        )
                    } else {
                        name.clone()
                    };
                    if i == 0 { value } else { format!("{name}={value}") }
                })
                .collect::<Vec<_>>();
            text.push_str(&format!("        super().__init__({})\n", args.join(", ")));
            for field in fields.iter().filter(|field| !field.inherited) {
                let value = if field.hidden {
                    crate::class_stub::py_literal(
                        field.default.as_ref().expect("resolved hidden default"),
                    )
                } else {
                    field.name.clone()
                };
                text.push_str(&format!(
                    "        self.{}{} = {value}\n",
                    if field.readonly { "_" } else { "" },
                    field.name
                ));
            }
            for field in fields.iter().filter(|field| field.readonly) {
                text.push_str(&format!(
                    "\n    @property\n    def {}(self) -> {}:\n        return self._{}\n",
                    field.name, field.annotation, field.name
                ));
            }
        }
        if let Some(display) = &class.display {
            text.push_str("\n    def __str__(self) -> str:\n");
            if display == "message" {
                text.push_str("        return self.message\n");
            } else {
                text.push_str(&format!(
                    "        from {path} import {display}\n\n        return {display}(self)\n"
                ));
            }
        }
        if let Some(repr) = &class.repr {
            let parts = repr
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if i == 0 {
                        format!("{{self.{name}!r}}")
                    } else {
                        format!("{name}={{self.{name}!r}}")
                    }
                })
                .collect::<Vec<_>>();
            text.push_str(&format!("\n    def __repr__(self) -> str:\n        return f\"{{type(self).__name__}}({})\"\n", parts.join(", ")));
        }
    }
    text.push_str("\n\n__all__ = [\n");
    for class in &family.classes {
        text.push_str(&format!("    {:?},\n", class.name));
    }
    text.push_str("]\n");
    if !text.contains("Literal[") {
        text = text.replace("\nfrom typing import Literal\n", "");
    }
    Out { path: format!("{}.py", family.module), text }
}

#[cfg(test)]
mod tests {
    fn schema(source: &str) -> Result<String, String> {
        let item: syn::ItemConst = syn::parse_str(&format!("#[host({source})] const _: () = ();"))
            .map_err(|e| e.to_string())?;
        let entries = crate::decl::vocabulary::read(&item.attrs)?;
        super::resolve(&entries[0]).map(|family| super::emit(&family).text)
    }

    #[test]
    fn classes_keep_public_identity_inheritance_and_constructor_contracts() {
        let source = schema(
            r#"exceptions(module = "errors", path = "package.errors", classes(
            Failure(base = "RuntimeError"),
            Rejected(base = "Failure", fields(
                message(ty = "str", positional, inherited),
                task(ty = "str | None", default = None)
            ), base_args(message), repr(message, task))
        ))"#,
        )
        .unwrap();
        assert!(source.contains("class Rejected(Failure):"));
        assert!(source.contains("__module__ = \"package.errors\""));
        assert!(source.contains("def __init__(self, message: str, *, task: str | None = None)"));
        assert!(source.contains("super().__init__(message)"));
        assert!(source.contains("self.task = task"));
        assert!(!source.contains("self.message = message"));
        assert!(source.contains("{type(self).__name__}({self.message!r}, task={self.task!r})"));
    }

    #[test]
    fn normalization_runs_before_the_base_constructor() {
        let source = schema(
            r#"exceptions(module = "errors", path = "package.errors", classes(
            Failure(base = "Exception", fields(
                message(ty = "str", positional), retryable(ty = "bool | None", default = None)
            ), base_args(message), normalize(retryable(call = "normalize", args(retryable))))
        ))"#,
        )
        .unwrap();
        assert!(
            source.find("retryable = normalize(retryable)").unwrap()
                < source.find("super().__init__(message)").unwrap()
        );
    }

    #[test]
    fn qualified_bases_readonly_messages_and_empty_constructors_keep_their_contracts() {
        let source = schema(r#"exceptions(module = "errors", path = "package.errors", classes(
            Failure(base = "package.tools.ToolError"),
            Retry(base = "Exception", fields(message(ty = "str", positional, readonly)), base_args(message)),
            Stop(base = "Exception", fields(), base_args())
        ))"#).unwrap();
        assert!(source.contains("from package.tools import ToolError"));
        assert!(source.contains("class Failure(ToolError):"));
        assert!(source.contains("self._message = message"));
        assert!(
            source
                .contains("@property\n    def message(self) -> str:\n        return self._message")
        );
        assert!(source.contains("def __init__(self) -> None:\n        super().__init__()"));
        assert!(schema(r#"exceptions(module = "errors", path = "package.errors", classes(
            Failure(base = "Exception", fields(message(ty = "str", inherited, readonly)), base_args(message))
        ))"#).is_err());
    }

    #[test]
    fn incomplete_or_unknown_contracts_fail_generation() {
        for class in [
            r#"Failure(base = "Exception", typo)"#,
            r#"Failure(base = "Exception", fields(message(ty = "str")))"#,
            r#"Failure(base = "Exception", fields(message(ty = "str", positional)), base_args(missing))"#,
            r#"Failure(base = "Exception", fields(message(ty = "str", hidden)), base_args(message))"#,
        ] {
            assert!(
                schema(&format!(
                    r#"exceptions(module = "errors", path = "package.errors", classes({class}))"#
                ))
                .is_err(),
                "{class}"
            );
        }
    }
}
