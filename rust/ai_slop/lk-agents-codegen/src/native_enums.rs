//! Native Python enum classes declared once by ABI unit enums.

use crate::decl::{Declarations, HostForm, NativeEnum, Record, Ty, vocabulary as voc};
use crate::python::Out;
use crate::surface::Stub;

fn name(entry: &voc::Entry) -> Result<String, String> {
    voc::only("native_enum", &entry.nested, &["name"])?;
    voc::text("native_enum", &entry.nested, "name")?.ok_or("native_enum needs name".into())
}

pub fn ty(entry: &voc::Entry, rust: &str) -> Result<Ty, String> {
    let name = name(entry)?;
    Ok(Ty::Record(Record {
        rust: rust.into(),
        host: HostForm::Converted(
            format!("livekit.agents._core.{name}"),
            format!("generated::native_enums::{}", crate::decl::resolve::snake(&name)),
        ),
        extract: true,
        emit: true,
        literal: false,
    }))
}

pub fn resolve(
    abi: &crate::decl::tree::Crate,
    module: usize,
    index: usize,
    item: &syn::ItemEnum,
    entry: &voc::Entry,
) -> Result<NativeEnum, String> {
    let name = name(entry)?;
    if !item.generics.params.is_empty() || item.variants.is_empty() {
        return Err(format!("{name}: native enums must be nonempty and nongeneric"));
    }
    let variants = item
        .variants
        .iter()
        .map(|variant| {
            if !matches!(variant.fields, syn::Fields::Unit) || variant.discriminant.is_some() {
                return Err(format!("{name}: native enums require implicit unit variants"));
            }
            Ok(variant.ident.to_string())
        })
        .collect::<Result<_, String>>()?;
    Ok(NativeEnum { name, rust: abi.extern_path(module, index)?, variants })
}

pub fn emit(decls: &Declarations) -> Result<(Out, Stub), String> {
    let mut text = String::from("//! GENERATED FILE -- DO NOT EDIT.\nuse pyo3::prelude::*;\n");
    let mut body = String::from("# GENERATED FILE -- DO NOT EDIT.\nfrom enum import Enum\n");
    let mut register = String::new();
    let mut names = Vec::new();
    for item in &decls.native_enums {
        let name = &item.name;
        let rust = &item.rust;
        if names.contains(name) {
            return Err(format!("duplicate native enum {name}"));
        }
        let converter = crate::decl::resolve::snake(name);
        text.push_str(&format!("#[pyclass(eq, eq_int, from_py_object, module = \"livekit.agents._core\")]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum {name} {{\n"));
        body.push_str(&format!("\nclass {name}(Enum):\n"));
        let mut read = String::new();
        let mut write = String::new();
        for variant in &item.variants {
            text.push_str(&format!("{variant},\n"));
            body.push_str(&format!("    {variant} = ...\n"));
            read.push_str(&format!("{name}::{variant} => {rust}::{variant},"));
            write.push_str(&format!("{rust}::{variant} => {name}::{variant},"));
        }
        text.push_str(&format!("}}\npub fn {converter}_from_py(obj: &Bound<'_, PyAny>) -> PyResult<{rust}> {{ Ok(match obj.extract::<{name}>()? {{ {read} }}) }}\npub fn {converter}_to_py(py: Python<'_>, value: {rust}) -> PyResult<Bound<'_, PyAny>> {{ Ok(Bound::new(py, match value {{ {write} }})?.into_any()) }}\n"));
        register.push_str(&format!("m.add_class::<{name}>()?;\n"));
        names.push(name.clone());
    }
    text.push_str(&format!(
        "pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {{ {register} Ok(()) }}\n"
    ));
    Ok((
        Out { path: "native_enums.rs".into(), text },
        Stub { out: Out { path: "_core_native_enums.pyi".into(), text: body }, names },
    ))
}

/// A host Flag uses the native newtype's public constants as its only value catalogue.
pub fn flag(
    abi: &crate::decl::tree::Crate,
    module: usize,
    index: usize,
    codec: &voc::Entry,
    declaration: &voc::Entry,
) -> Result<crate::decl::Flag, String> {
    voc::only("flag", &declaration.nested, &["module"])?;
    let generated = voc::text("flag", &declaration.nested, "module")?.ok_or("flag needs module")?;
    let host =
        voc::text("flag", &codec.nested, "host")?.ok_or("flag needs its codec host class")?;
    let (path, name) = host.rsplit_once('.').ok_or("flag host is a qualified class")?;
    if voc::text("flag", &codec.nested, "attribute")?.as_deref() != Some("value") {
        return Err("a flag codec uses its value attribute".into());
    }
    let target = crate::decl::tree::Found::Item { module, item: index };
    let mut values = Vec::new();
    for (owner, scope) in abi.modules.iter().enumerate() {
        for item in &scope.items {
            let syn::Item::Impl(implementation) = item else { continue };
            let syn::Type::Path(ty) = implementation.self_ty.as_ref() else { continue };
            if implementation.trait_.is_some() || abi.resolve(owner, &ty.path)? != target {
                continue;
            }
            for item in &implementation.items {
                let syn::ImplItem::Const(value) = item else { continue };
                if !matches!(value.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let syn::Type::Path(ty) = &value.ty else { continue };
                if !ty.path.is_ident("Self") && abi.resolve(owner, &ty.path)? != target {
                    continue;
                }
                let syn::Expr::Call(call) = &value.expr else {
                    return Err("flag constants use their native constructor".into());
                };
                let syn::Expr::Path(constructor) = call.func.as_ref() else {
                    return Err("flag constructor is a native type".into());
                };
                if !constructor.path.is_ident("Self")
                    && abi.resolve(owner, &constructor.path)? != target
                {
                    return Err("flag constants use their own native constructor".into());
                }
                let [syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(bits), .. })] =
                    &call.args.iter().cloned().collect::<Vec<_>>()[..]
                else {
                    return Err("flag values are explicit integer bits".into());
                };
                values.push((
                    value.ident.to_string(),
                    bits.base10_parse().map_err(|error| error.to_string())?,
                ));
            }
        }
    }
    if values.is_empty() {
        return Err("a flag declares public native bit constants".into());
    }
    Ok(crate::decl::Flag { name: name.into(), module: generated, path: path.into(), values })
}

pub fn flags(declarations: &Declarations) -> Vec<Out> {
    declarations.flags.iter().map(|flag| {
        let mut text = format!("# GENERATED FILE -- DO NOT EDIT.\n# Native flag constants, generated by lk-agents-codegen.\nfrom enum import Flag\n\n\nclass {}(Flag):\n", flag.name);
        for (name, value) in &flag.values { text.push_str(&format!("    {name} = {value}\n")); }
        text.push_str(&format!("\n\n{}.__module__ = {:?}\n", flag.name, flag.path));
        Out { path: format!("{}.py", flag.module), text }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::tree::Crate;

    #[test]
    fn native_enums_generate_typed_conversions_and_reject_unsupported_variants() {
        let build = |source| {
            crate::decl::resolve::declarations(
                &Crate::from_files("lk_agents_abi", &[("lib.rs", source)]).unwrap(),
            )
        };
        let (rust, stub) = emit(
            &build("#[host(native_enum(name = \"Role\"))] pub enum ProcessRole { Worker, Job }")
                .unwrap(),
        )
        .unwrap();
        assert!(rust.text.contains("eq, eq_int, from_py_object"));
        assert!(rust.text.contains("Role::Worker => ::lk_agents_abi::ProcessRole::Worker"));
        assert!(rust.text.contains("::lk_agents_abi::ProcessRole::Job => Role::Job"));
        assert_eq!(stub.names, ["Role"]);
        assert!(stub.out.text.contains("class Role(Enum)"));
        for source in [
            "#[host(native_enum(name = \"Role\"))] enum Role { Worker(u8) }",
            "#[host(native_enum(name = \"Role\"))] enum Role { Worker = 5 }",
            "#[host(native_enum(name = \"Role\"))] enum Role {}",
        ] {
            assert!(build(source).is_err());
        }
    }
}
