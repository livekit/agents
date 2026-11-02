//! Presentation metadata uses the same attribute reader and resolved module tree as contracts.
use crate::decl::{ConstValue, Constant, Member, StrClass, Wrapper};
use crate::decl::{
    tree::{Crate, Module},
    vocabulary::{self as voc, Entry},
};

fn named(attrs: &[syn::Attribute], outer: &str) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for attr in attrs.iter().filter(|attr| attr.path().is_ident(outer)) {
        let mut attr = attr.clone();
        let syn::Meta::List(list) = &mut attr.meta else {
            return Err(format!("{outer} needs a declaration"));
        };
        list.path = syn::parse_quote!(host);
        entries.extend(voc::read(&[attr])?);
    }
    Ok(entries)
}

fn required(at: &str, entries: &[Entry], key: &str) -> Result<String, String> {
    voc::text(at, entries, key)?.ok_or_else(|| format!("{at}: missing {key}"))
}

pub fn wrapper(
    label: &str,
    attrs: &[syn::Attribute],
    core: &str,
) -> Result<Option<Wrapper>, String> {
    let entries = named(attrs, "python")?;
    let Some(wrapper) = voc::find(&entries, "wrapper") else {
        return Ok(None);
    };
    voc::only(label, &entries, &["wrapper"])?;
    let keys = &wrapper.nested;
    voc::only(label, keys, &["module", "path", "name", "base"])?;
    Ok(Some(Wrapper {
        name: required(label, keys, "name")?,
        module: required(label, keys, "module")?,
        path: voc::text(label, keys, "path")?,
        source: label.into(),
        doc: lk_agents_bindgen::doc(attrs),
        base: required(label, keys, "base")?,
        core: core.into(),
    }))
}

pub fn forward_module(label: &str, attrs: &[syn::Attribute]) -> Result<Option<String>, String> {
    let entries = named(attrs, "python")?;
    let Some(forward) = voc::find(&entries, "forward") else {
        return Ok(None);
    };
    voc::only(label, &entries, &["forward"])?;
    voc::only(label, &forward.nested, &["module"])?;
    Ok(Some(required(label, &forward.nested, "module")?))
}

pub fn members(attrs: &[syn::Attribute], outer: &str, owner: &str) -> Result<Vec<Member>, String> {
    let mut out = Vec::new();
    for entry in named(attrs, outer)? {
        if !["property", "method"].contains(&entry.key.as_str()) {
            continue;
        }
        let keys = &entry.nested;
        voc::only(owner, keys, &["name", "args", "ret", "doc", "hook"])?;
        let name = required(owner, keys, "name")?;
        let args = voc::text(owner, keys, "args")?.unwrap_or_default();
        out.push(Member {
            hook: voc::text(owner, keys, "hook")?
                .unwrap_or_else(|| format!("{}_{name}", crate::decl::resolve::snake(owner))),
            name,
            is_property: entry.key == "property",
            arguments: arguments(&args)?,
            args,
            ret: voc::text(owner, keys, "ret")?.unwrap_or_else(|| "None".into()),
            doc: voc::text(owner, keys, "doc")?.unwrap_or_default(),
        });
    }
    Ok(out)
}

/// Forward only names explicitly declared by a Python method signature. Nested annotations
/// and defaults can contain commas; these are not parameter separators.
fn arguments(signature: &str) -> Result<Vec<String>, String> {
    let mut parts = Vec::new();
    let (mut start, mut depth, mut quote, mut escape) = (0, 0, None, false);
    for (i, ch) in signature.char_indices() {
        if let Some(q) = quote {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == q {
                quote = None;
            }
        } else {
            match ch {
                '\'' | '"' => quote = Some(ch),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&signature[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
    }
    if depth != 0 || quote.is_some() {
        return Err("unbalanced method signature".into());
    }
    parts.push(&signature[start..]);
    let mut keyword = false;
    let mut names = Vec::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() || part == "/" {
            continue;
        }
        if part == "*" {
            keyword = true;
            continue;
        }
        let name = part.split([':', '=']).next().unwrap_or(part).trim();
        let bare = name.trim_start_matches('*');
        syn::parse_str::<syn::Ident>(bare)
            .map_err(|_| format!("invalid method parameter {name}"))?;
        names.push(if name.starts_with('*') || !keyword {
            name.to_owned()
        } else {
            format!("{name}={name}")
        });
        if name.starts_with('*') {
            keyword = true;
        }
    }
    Ok(names)
}

pub fn str_class(label: &str, item: &syn::ItemStruct) -> Result<Option<StrClass>, String> {
    let entries = named(&item.attrs, "python")?;
    let Some(class) = voc::find(&entries, "str_class") else {
        return Ok(None);
    };
    voc::only(label, &entries, &["str_class"])?;
    let keys = &class.nested;
    voc::only(label, keys, &["module", "path", "new", "property", "method"])?;
    let applied = |kind| -> Result<Vec<(String, String)>, String> {
        let Some(group) = voc::find(keys, kind) else {
            return Ok(Vec::new());
        };
        group
            .nested
            .iter()
            .map(|member| {
                Ok((
                    member.key.clone(),
                    voc::text(label, &group.nested, &member.key)?
                        .ok_or("an applied member names its function")?,
                ))
            })
            .collect()
    };
    Ok(Some(StrClass {
        name: item.ident.to_string(),
        module: required(label, keys, "module")?,
        path: voc::text(label, keys, "path")?,
        source: label.into(),
        doc: lk_agents_bindgen::doc(&item.attrs),
        new: required(label, keys, "new")?,
        properties: applied("property")?,
        methods: applied("method")?,
    }))
}

pub fn constants(krate: &Crate, index: usize, module: &Module) -> Result<Vec<Constant>, String> {
    let mut target = None;
    for item in &module.items {
        if let syn::Item::Const(item) = item
            && let Some(file) = voc::find(&voc::read(&item.attrs)?, "file")
        {
            if target.is_some() {
                return Err(format!("{}: duplicate file defaults", module.file));
            }
            target = Some(required(&module.file, &file.nested, "module")?);
        }
    }
    let mut out = Vec::new();
    for item in &module.items {
        let syn::Item::Const(item) = item else {
            continue;
        };
        let entries = voc::read(&item.attrs)?;
        let Some(constant) = voc::find(&entries, "constant") else {
            continue;
        };
        voc::only(&module.file, &constant.nested, &["name"])?;
        let resolved = krate.constant(index, &item.expr)?;
        let value = match &resolved {
            syn::Expr::Struct(value) => {
                let crate::decl::tree::Found::Item { module: m, item: i } =
                    krate.resolve(index, &value.path)?
                else {
                    return Err("a record constant names an ABI record".into());
                };
                let fields = value
                    .fields
                    .iter()
                    .map(|field| {
                        let syn::Member::Named(name) = &field.member else {
                            return Err("a record constant names its fields".into());
                        };
                        Ok((name.to_string(), literal(&field.expr)?))
                    })
                    .collect::<Result<_, String>>()?;
                if value.rest.is_some() {
                    return Err("a record constant lists its fields explicitly".into());
                }
                ConstValue::Record { rust: krate.extern_path(m, i)?, fields }
            }
            value => ConstValue::Literal(literal(value)?),
        };
        out.push(Constant {
            name: voc::text(&module.file, &constant.nested, "name")?
                .unwrap_or_else(|| item.ident.to_string()),
            module: target
                .clone()
                .ok_or_else(|| format!("{}: constant needs file(module = ..)", module.file))?,
            source: module.file.clone(),
            doc: lk_agents_bindgen::doc(&item.attrs),
            value,
        });
    }
    Ok(out)
}

fn literal(value: &syn::Expr) -> Result<String, String> {
    Ok(match value {
        syn::Expr::Reference(value) if value.mutability.is_none() => literal(&value.expr)?,
        syn::Expr::Array(array) => {
            let values = array.elems.iter().map(literal).collect::<Result<Vec<_>, _>>()?;
            format!("({}{})", values.join(", "), if values.len() == 1 { "," } else { "" })
        }
        syn::Expr::Lit(value) => match &value.lit {
            syn::Lit::Str(value) => format!("{:?}", value.value()),
            syn::Lit::Int(value) => value.base10_digits().into(),
            syn::Lit::Float(value) => {
                let value = value.base10_digits();
                if value.contains(['.', 'e', 'E']) { value.into() } else { format!("{value}.0") }
            }
            syn::Lit::Bool(value) => if value.value { "True" } else { "False" }.into(),
            _ => {
                return Err(
                    "a constant has a string, number, bool or literal sequence value".into()
                );
            }
        },
        syn::Expr::Path(value) if value.path.is_ident("None") => "None".into(),
        syn::Expr::Unary(value) if matches!(value.op, syn::UnOp::Neg(_)) => {
            format!("-{}", literal(&value.expr)?)
        }
        _ => return Err("a constant has a string, number, bool or literal sequence value".into()),
    })
}
