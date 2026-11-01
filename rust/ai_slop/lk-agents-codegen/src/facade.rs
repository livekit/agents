//! Explicit Python forwarding signatures over a facade's declared native target.

use crate::decl::{Declarations, MethodKind};
use crate::python::Out;

pub fn emit(declarations: &Declarations) -> Result<Vec<Out>, String> {
    let mut groups = std::collections::BTreeMap::<(String, String), Vec<_>>::new();
    for class in &declarations.classes {
        for method in &class.methods {
            if let Some(facade) = &method.facade {
                groups
                    .entry((facade.module.clone(), facade.class.clone()))
                    .or_default()
                    .push((class, method, facade));
            }
        }
    }
    let mut outputs = Vec::new();
    for ((module, name), methods) in groups {
        let mut imports = crate::class_stub::Imports::new(declarations);
        let mut body = format!("class {name}:\n");
        let mut targets = std::collections::BTreeMap::new();
        for (class, method, facade) in methods {
            if !matches!(method.kind, MethodKind::Call | MethodKind::Await | MethodKind::Getter) {
                return Err(format!("{}: facade needs a call, await or getter", method.rust_name));
            }
            if facade.future && method.kind != MethodKind::Await {
                return Err(format!("{}: facade future needs an await", method.rust_name));
            }
            targets.insert(facade.target.clone(), class.name.clone());
            let (mut params, mut ret) =
                crate::class_stub::signature_parts(class, method, &mut imports);
            if let Some(omit) = &facade.omit {
                let Some(parameter) = method.params.iter().find(|p| &p.name == omit) else {
                    return Err(format!("{}: facade omits no parameter {omit}", method.name));
                };
                if parameter.default.is_none() {
                    return Err(format!("{}: facade cannot omit required {omit}", method.name));
                }
                params.retain(|(p, _)| p.split(':').next() != Some(omit.as_str()));
            }
            let mut signature = vec!["self".to_owned()];
            let mut arguments = Vec::new();
            let mut star = false;
            for (parameter, keyword) in params {
                if keyword && !star {
                    signature.push("*".into());
                    star = true;
                }
                let name = parameter.split(':').next().expect("parameter name");
                arguments.push(if star { format!("{name}={name}") } else { name.to_owned() });
                signature.push(parameter);
            }
            let call = if method.kind == MethodKind::Getter {
                format!("self.{}.{}", facade.target, method.name)
            } else {
                format!("self.{}.{}({})", facade.target, method.name, arguments.join(", "))
            };
            let prefix =
                if method.kind == MethodKind::Await && !facade.future { "async " } else { "" };
            let statement = if facade.future {
                ret = format!("asyncio.Future[{ret}]");
                if call.len() + 38 <= 100 {
                    format!("return asyncio.ensure_future({call})")
                } else {
                    format!(
                        "return asyncio.ensure_future(\n            self.{}.{}(\n{}            )\n        )",
                        facade.target,
                        method.name,
                        arguments
                            .iter()
                            .map(|a| format!("                {a},\n"))
                            .collect::<String>()
                    )
                }
            } else if method.kind == MethodKind::Await {
                if facade.discard {
                    ret = "None".into();
                    format!("await {call}")
                } else {
                    format!("return await {call}")
                }
            } else {
                format!("return {call}")
            };
            let signature_line =
                format!("    {prefix}def {}({}) -> {ret}:\n", method.name, signature.join(", "));
            if method.kind == MethodKind::Getter {
                body.push_str("\n    @property\n");
                body.push_str(&signature_line);
            } else if signature_line.len() <= 101 {
                body.push_str(&format!("\n{signature_line}"));
            } else {
                body.push_str(&format!("\n    {prefix}def {}(\n", method.name));
                for parameter in signature {
                    body.push_str(&format!("        {parameter},\n"));
                }
                body.push_str(&format!("    ) -> {ret}:\n"));
            }
            crate::python::docstring(&mut body, &method.doc, 8);
            body.push_str(&format!("        {statement}\n"));
        }
        let mut header = "# GENERATED FILE -- DO NOT EDIT.\n# From #[host(facade)] declarations in lk-agents-abi.\n\nfrom __future__ import annotations\n\nimport asyncio\nfrom typing import TYPE_CHECKING\n\nfrom .. import _core\n".to_owned();
        for (module, item) in imports.entries() {
            header.push_str(&format!("from {module} import {item}\n"));
        }
        let mut declared = String::new();
        for (target, class) in targets {
            if !declared.is_empty() {
                declared.push('\n');
            }
            declared.push_str(&format!(
                "    @property\n    def {target}(self) -> _core.{class}: ...\n"
            ));
        }
        let marker = format!("class {name}:\n");
        body = body.replacen(
            &marker,
            &format!(
                "{marker}    if TYPE_CHECKING:\n\n{}",
                declared
                    .lines()
                    .map(|line| if line.is_empty() {
                        "\n".to_owned()
                    } else {
                        format!("    {line}\n")
                    })
                    .collect::<String>()
            ),
            1,
        );
        outputs.push(Out { path: format!("{module}.py"), text: format!("{header}\n\n{body}") });
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    fn declarations(method: &str) -> crate::decl::Declarations {
        crate::decl::tests::read(&format!(
            "#[crate::host(class(name = \"Native\"))] pub trait Native: Sized {{ {method} }}"
        ))
        .unwrap()
    }

    #[test]
    fn public_future_keeps_keyword_defaults_and_eager_native_call() {
        let declarations = declarations(
            r#"
            #[host(await, facade(module = "commands", class = "Commands", target = "_running", future))]
            async fn commit(&self, #[host(kw, default = 2.0)] timeout: f64) -> Result<String, crate::errors::ApiError>;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(
            output.contains("def commit(self, *, timeout: float = 2.0) -> asyncio.Future[str]")
        );
        assert!(
            output.contains("return asyncio.ensure_future(self._running.commit(timeout=timeout))")
        );
        assert!(!output.contains("async def commit"));
    }

    #[test]
    fn a_private_optional_argument_is_not_added_to_the_public_signature() {
        let declarations = declarations(
            r#"
            #[host(await, facade(module = "commands", class = "Commands", target = "_running", omit = "hold"))]
            async fn idle(&self, #[host(kw, default = None)] hold: Option<u64>) -> Result<(), crate::errors::ApiError>;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("async def idle(self) -> None"));
        assert!(output.contains("return await self._running.idle()"));
        assert!(!output.contains("hold:"));
    }

    #[test]
    fn a_required_native_argument_cannot_be_silently_dropped() {
        let declarations = declarations(
            r#"
            #[host(call, facade(module = "commands", class = "Commands", target = "_running", omit = "hold"))]
            fn idle(&self, hold: u64);
        "#,
        );
        assert!(super::emit(&declarations).err().unwrap().contains("cannot omit required hold"));
    }

    #[test]
    fn a_native_getter_is_read_as_a_property_instead_of_called() {
        let declarations = declarations(
            r#"
            #[host(getter, facade(module = "commands", class = "Commands", target = "_owner"))]
            fn usage(&self) -> String;
        "#,
        );
        let output = super::emit(&declarations).unwrap().remove(0).text;
        assert!(output.contains("@property\n    def usage(self) -> str"));
        assert!(output.contains("return self._owner.usage\n"));
        assert!(!output.contains("self._owner.usage()"));
    }
}
